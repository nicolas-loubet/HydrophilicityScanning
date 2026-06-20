"""
app/main.py — FastAPI application core.

Endpoints
---------
GET  /                                  → Renders the main SPA (index.html via Jinja2)
POST /api/upload                        → Accepts a .pdb file, runs tleap, returns metadata.
GET  /api/runs/{run_id}/file/{filename} → Serves a file from a run directory.
GET  /api/residues/map/{run_id}         → Returns residue-number → name mapping from PDB.
GET  /api/residue/2d/{res_name}         → Returns an RDKit SVG of the amino acid side chain.
POST /api/study/config                  → Saves study configuration to config.toml.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import tomli_w
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR: Path      = Path(__file__).resolve().parent        # …/app
PROJECT_ROOT: Path  = BASE_DIR.parent                        # …/hydrophobic_sim
DATA_RUNS_DIR: Path = PROJECT_ROOT.parent / "data" / "runs"

TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"
STATIC_DIR: Path    = PROJECT_ROOT / "static"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES: int      = 100 * 1024 * 1024   # 100 MB
ALLOWED_EXTENSION: str     = ".pdb"
TLEAP_TIMEOUT_SECONDS: int = 120

# ---------------------------------------------------------------------------
# Environment Modules / tleap configuration
#
# AmberTools is loaded via the HPC Environment Modules system (Lmod / Tmod).
# "module load amber" is a shell function — not an executable — so tleap
# must be invoked through a bash login shell that sources the modules
# initialisation script before issuing the load command.
#
# Override the module name with HYDROSIM_AMBER_MODULE if your cluster uses a
# versioned name such as "amber/24" or "ambertools/23".
#
# Resolution order for the modules init script:
#   1. HYDROSIM_MODULES_SH environment variable (explicit override)
#   2. Probe MODULE_INIT_PROBE_PATHS in order
#   3. Fall back to bare `tleap` already on PATH (pre-loaded shell, CI)
# ---------------------------------------------------------------------------

AMBER_MODULE_NAME: str = os.environ.get("HYDROSIM_AMBER_MODULE", "amber")

# Probe list covers the most common Lmod and Tmod / Environment Modules paths.
MODULE_INIT_PROBE_PATHS: list[str] = [
    os.environ.get("HYDROSIM_MODULES_SH", ""),       # explicit override (may be "")
    "/usr/share/lmod/lmod/init/bash",                # Lmod — Ubuntu / Debian package
    "/usr/local/lmod/lmod/init/bash",                # Lmod — custom prefix
    "/opt/lmod/lmod/init/bash",                      # Lmod — alternative prefix
    "/etc/profile.d/modules.sh",                     # Tmod / Environment Modules
    "/usr/share/modules/init/bash",                  # Environment Modules — Debian
    "/usr/local/modules/init/bash",                  # Environment Modules — custom
    "/opt/Modules/init/bash",                        # Environment Modules — RPM
]

# ---------------------------------------------------------------------------
# Amino acid side-chain SMILES (20 canonical residues)
# Represents the side chain attached to the Cα; used for 2D depiction.
# ---------------------------------------------------------------------------

AA_SIDE_CHAIN_SMILES: dict[str, str] = {
    # Full amino acid SMILES: NH2 (N-terminus) + Ca + side chain + COOH (C-terminus)
    # PRO: side chain cyclises onto backbone N (pyrrolidine ring).
    "ALA": "[NH2][C@@H](C)C(=O)O",
    "ARG": "[NH2][C@@H](CCCNC(=N)N)C(=O)O",
    "ASN": "[NH2][C@@H](CC(=O)N)C(=O)O",
    "ASP": "[NH2][C@@H](CC(=O)O)C(=O)O",
    "CYS": "[NH2][C@@H](CS)C(=O)O",
    "GLN": "[NH2][C@@H](CCC(=O)N)C(=O)O",
    "GLU": "[NH2][C@@H](CCC(=O)O)C(=O)O",
    "GLY": "[NH2]CC(=O)O",
    "HIS": "[NH2][C@@H](Cc1c[nH]cn1)C(=O)O",
    "ILE": "[NH2][C@@H]([C@@H](C)CC)C(=O)O",
    "LEU": "[NH2][C@@H](CC(C)C)C(=O)O",
    "LYS": "[NH2][C@@H](CCCCN)C(=O)O",
    "MET": "[NH2][C@@H](CCSC)C(=O)O",
    "PHE": "[NH2][C@@H](Cc1ccccc1)C(=O)O",
    "PRO": "N1CC[C@@H]1C(=O)O",
    "SER": "[NH2][C@@H](CO)C(=O)O",
    "THR": "[NH2][C@@H]([C@@H](O)C)C(=O)O",
    "TRP": "[NH2][C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "TYR": "[NH2][C@@H](Cc1ccc(O)cc1)C(=O)O",
    "VAL": "[NH2][C@@H](C(C)C)C(=O)O",
}

# Also accept common alternate residue names (protonated / variant forms)
AA_ALIASES: dict[str, str] = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "ASH": "ASP", "GLH": "GLU",
    "LYN": "LYS", "CYX": "CYS", "CYM": "CYS",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("hydrosim")

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle handler."""
    DATA_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Data directory ready: %s", DATA_RUNS_DIR)

    # Validate optional heavy dependencies at startup so missing libs fail fast.
    try:
        from rdkit import Chem  # noqa: F401
        logger.info("RDKit available.")
    except ImportError:
        logger.warning("RDKit not installed — /api/residue/2d/* will return placeholder SVGs.")

    yield
    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class StudyConfigPayload(BaseModel):
    """Payload for POST /api/study/config."""

    run_id:             str
    study_name:         str
    folder_name:        str
    original_filename:  str = ""
    pdb_to_serve:       str = ""
    total_residues:     int = 0
    selected_residues:  list[int]
    ks_factors:         list[float]
    salt_type:          str   = "NaCl"
    salt_concentration: float = 0.0
    box_shape:          str   = "box"

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9_\-]{1,80}", v):
            raise ValueError("Invalid run_id format.")
        return v

    @field_validator("ks_factors")
    @classmethod
    def _validate_ks(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("At least one Ks factor is required.")
        for f in v:
            if not (0.0 <= f <= 1.0):
                raise ValueError(f"Ks factor {f} is out of range [0, 1].")
        return v

class IonConfig(BaseModel):
    """Ion configuration submitted alongside study config."""

    salt_type:          str   = "NaCl"   # "NaCl" | "KCl" | "MgCl2"
    salt_concentration: float = 0.0      # mol/L; 0.0 = neutralize only
    box_shape:          str   = "box"    # "box" = rectangular | "oct" = truncated octahedron

    @field_validator("salt_type")
    @classmethod
    def _validate_salt(cls, v: str) -> str:
        allowed = {"NaCl", "KCl", "MgCl2"}
        if v not in allowed:
            raise ValueError(f"salt_type must be one of {allowed}.")
        return v

    @field_validator("salt_concentration")
    @classmethod
    def _validate_conc(cls, v: float) -> float:
        if v < 0:
            raise ValueError("salt_concentration must be >= 0.")
        return v

    @field_validator("box_shape")
    @classmethod
    def _validate_box(cls, v: str) -> str:
        if v not in {"box", "oct"}:
            raise ValueError('box_shape must be "box" or "oct".')
        return v



# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HydroSim — Hydrophobicity Simulation",
    version="0.3.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_run_dir(run_id: str) -> Path:
    """
    Return and validate the run directory for *run_id*.

    Raises HTTP 400 for unsafe ids, HTTP 404 when the directory is missing.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_\-]{1,80}", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id.")
    run_dir = DATA_RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run_dir


def _count_residues(pdb_path: Path) -> int:
    """Count unique residues via Cα atoms; falls back to all ATOM records."""
    seen: set[tuple[str, str]] = set()
    fallback: set[tuple[str, str]] = set()
    try:
        text = pdb_path.read_text(errors="replace")
    except OSError:
        return 0
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        chain_id  = line[21] if len(line) > 21 else "?"
        res_seq   = line[22:26].strip() if len(line) > 25 else "0"
        fallback.add((chain_id, res_seq))
        if atom_name == "CA":
            seen.add((chain_id, res_seq))
    return len(seen) if seen else len(fallback)


def _parse_residue_map(pdb_path: Path) -> dict[str, str]:
    """
    Parse a PDB file and return an ordered mapping of residue sequence number
    to three-letter residue name.  First occurrence of each sequence number wins
    (handles duplicate chain entries gracefully).
    """
    seen: dict[str, str] = {}
    try:
        text = pdb_path.read_text(errors="replace")
    except OSError:
        return seen
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        res_name = line[17:20].strip()
        res_seq  = line[22:26].strip()
        if res_seq and res_seq not in seen:
            seen[res_seq] = res_name
    return seen


def _generate_residue_svg(res_name: str) -> str:
    """
    Generate an RDKit SVG for *res_name* showing the complete amino acid:
    the backbone fragment (NH2 — Ca — COOH) together with the side chain.

    Canvas size scales with molecular complexity so small residues (GLY, ALA)
    are not stretched and large ones (TRP, ARG) are not cramped.

    Returns a valid inline SVG string styled for dark backgrounds (transparent
    canvas, light-coloured bonds and heteroatom labels).  Falls back to a
    plain-text SVG when RDKit is unavailable or the residue is unknown.
    """
    canonical = AA_ALIASES.get(res_name.upper(), res_name.upper())
    smiles    = AA_SIDE_CHAIN_SMILES.get(canonical)

    def _fallback(msg: str) -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="220">'
            '<rect width="100%" height="100%" fill="transparent"/>'
            f'<text x="150" y="110" text-anchor="middle" font-family="monospace" '
            f'font-size="13" fill="#9ca3af">{msg}</text>'
            '</svg>'
        )

    if smiles is None:
        return _fallback(f"No structure for {res_name}")

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _fallback(f"Could not parse SMILES for {res_name}")

        # Generate clean 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Scale canvas to molecular complexity so every residue fits cleanly
        heavy = mol.GetNumHeavyAtoms()
        if heavy <= 6:    w, h = 260, 200
        elif heavy <= 10: w, h = 300, 240
        elif heavy <= 14: w, h = 360, 260
        else:             w, h = 420, 290

        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts   = drawer.drawOptions()

        opts.padding             = 0.15
        opts.bondLineWidth       = 2.0
        opts.clearBackground     = False
        opts.addStereoAnnotation = True   # wedge bonds on Ca

        # Dark-theme atom colour palette — all heteroatoms visible on #0d0d11
        opts.updateAtomPalette({
            6:  (0.85, 0.85, 0.85),   # C  → light grey  (implicit, unlabelled)
            7:  (0.45, 0.75, 1.00),   # N  → sky blue
            8:  (1.00, 0.50, 0.50),   # O  → salmon
            16: (0.95, 0.90, 0.25),   # S  → yellow
            15: (0.90, 0.60, 0.20),   # P  → amber-orange
            1:  (0.75, 0.75, 0.75),   # H  → dim grey (explicit H on heteroatoms)
        })
        # Default symbol colour for unlabelled atoms and bond lines
        opts.setSymbolColour((0.85, 0.85, 0.85))

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        # Strip the XML declaration (<?xml … ?>) that RDKit prepends.
        # Inline SVG embedded in HTML5 must not carry an XML declaration.
        if svg.lstrip().startswith("<?xml"):
            svg = svg[svg.index("<svg"):]

        # Inject transparent background on the root <svg> element so the panel
        # background shows through rather than a white rectangle.
        svg = re.sub(r"(<svg)", r'\1 style="background:transparent"', svg, count=1)
        # Wedge bond fills default to black (#000000), which is invisible on dark
        # backgrounds.  Replace with light grey to match the bond/atom palette.
        svg = re.sub(r"\bfill='#000000'", "fill='#D8D8D8'", svg)
        return svg

    except ImportError:
        return _fallback("RDKit not available on this server")


# ---------------------------------------------------------------------------
# Ion residue name sets — used by both tleap script generation and
# ParmEd charge-neutralization to identify ion atoms in the topology.
# ---------------------------------------------------------------------------

SALT_ION_NAMES: dict[str, tuple[str, str]] = {
    #            positive ion   negative ion
    "NaCl":  ("Na+",  "Cl-"),
    "KCl":   ("K+",   "Cl-"),
    "MgCl2": ("Mg2+", "Cl-"),
}

# AMBER residue-name spellings that TLeap may write in the PDB / topology.
# Keys are the canonical form used in addions commands; values are all aliases.
AMBER_ION_ALIASES: dict[str, set[str]] = {
    "Na+":  {"Na+", "NA",  "SOD"},
    "Cl-":  {"Cl-", "CL",  "CLA"},
    "K+":   {"K+",  "K",   "POT"},
    "Mg2+": {"Mg2+","MG",  "MG2"},
}

# Regex for filtering ion names from TLeap stdout
_ION_NAME_RE = re.compile(
    r"^(Na\+|Cl-|K\+|Mg2\+?|MG|NA|CL)$", re.IGNORECASE
)
_TLEAP_ADDING_RE = re.compile(
    r"Adding\s+(\d+)\s+([\w\+\-]+)\s+ions?\s+to", re.IGNORECASE
)
_TLEAP_NEUTRALIZE_RE = re.compile(
    r"(\d+)\s+(Na\+|Cl-|K\+|Mg2\+|MG|NA|CL)\s+ions?\s+required\s+to\s+neutralize",
    re.IGNORECASE,
)


def _parse_ions_from_tleap_log(log: str) -> dict[str, int]:
    """
    Parse ion counts from TLeap stdout.

    Returns a dict mapping ion name → number of atoms added, e.g.
    ``{"Na+": 10, "Cl-": 10}``.
    """
    counts: dict[str, int] = {}
    for m in _TLEAP_ADDING_RE.finditer(log):
        n, ion = int(m.group(1)), m.group(2)
        if _ION_NAME_RE.match(ion):
            counts[ion] = counts.get(ion, 0) + n
    for m in _TLEAP_NEUTRALIZE_RE.finditer(log):
        n, ion = int(m.group(1)), m.group(2)
        counts[ion] = counts.get(ion, 0) + n
    return counts


def _tleap_solvate_cmd(box_shape: str) -> str:
    """
    Return the TLeap solvation command for the chosen box geometry.

    Parameters
    ----------
    box_shape : "box" or "oct"
        "box" produces a rectangular water box (``solvateBox``).
        "oct" produces a truncated octahedron (``solvateOct``), which uses
        roughly 20 % fewer water molecules for the same buffer distance,
        reducing computational cost while maintaining periodic boundary
        conditions.

    The TIP3PBOX unit is used in both cases (compatible with ff19SB).
    Buffer distance is 12 Å in both cases.
    """
    if box_shape == "oct":
        return "solvateOct sys TIP3PBOX 12.0"
    return "solvateBox sys TIP3PBOX 12.0"


def _write_tleap_script(
    run_dir: Path,
    pdb_filename: str,
    ion_cfg: "IonConfig | None" = None,
) -> Path:
    """
    Write the tleap input script (render.in) into *run_dir*.

    The script:
      1. Loads the ff19SB force field and TIP3P water model.
      2. Loads the uploaded PDB and adds missing hydrogens.
      3. Solvates the system in a TIP3P water box with a 12 Å buffer.
      4. Adds ions according to *ion_cfg*:

         - concentration == 0  → neutralize only via ``addions``.
         - concentration > 0   → neutralize first, then add salt at the
                                  requested molarity via ``addions2``.
         - Edge case: if the protein is already perfectly neutral (0 counter
           ions needed), force at least 1 positive + 1 negative ion so the
           topology always contains ions for the charge-scaling step.

      5. Saves the AmberParm topology (sys.prmtop / sys.rst7) and a PDB of
         the solvated, protonated structure (with_H.pdb).

    Parameters
    ----------
    run_dir:      Directory where render.in is written.
    pdb_filename: Name of the input PDB file (relative to run_dir).
    ion_cfg:      Ion configuration; defaults to NaCl neutralize-only if None.
    """
    if ion_cfg is None:
        from app.main import IonConfig  # local import to avoid circular ref
        ion_cfg = IonConfig()

    pos_ion, neg_ion = SALT_ION_NAMES[ion_cfg.salt_type]

    # ------------------------------------------------------------------
    # Build addions block
    #
    # Strategy:
    #   Step A — Neutralize:
    #     addions sys <pos> 0   ← TLeap adds the exact number needed
    #     addions sys <neg> 0
    #
    #   Step B — Additional salt at requested molarity (if conc > 0):
    #     addions2 sys <pos> <conc>
    #     addions2 sys <neg> <conc>
    #
    #   Edge case — already neutral protein:
    #     TLeap will add 0 ions in Step A.  We force 1 pos + 1 neg
    #     so the topology is never ion-free, which would make the
    #     ParmEd charge-redistribution step fail silently.
    # ------------------------------------------------------------------

    # Step A: neutralize
    addions_lines = [
        f"addions sys {pos_ion} 0",
        f"addions sys {neg_ion} 0",
        "# Edge-case guard: ensure at least 1 positive AND 1 negative ion",
        "# even if the protein charge was already 0.  TLeap skips addions",
        "# silently when 0 counter-ions are needed; we force 1 of each.",
        f"addions sys {pos_ion} 1",
        f"addions sys {neg_ion} 1",
    ]

    # Step B: additional molar salt
    if ion_cfg.salt_concentration > 0:
        conc = round(ion_cfg.salt_concentration, 4)
        addions_lines += [
            f"# Additional {ion_cfg.salt_type} at {conc} mol/L",
            f"addions2 sys {pos_ion} {conc}",
            f"addions2 sys {neg_ion} {conc}",
        ]

    addions_block = "\n".join(addions_lines)

    script = (
        "source leaprc.protein.ff19SB\n"
        "source leaprc.water.tip3p\n"
        f"sys = loadpdb {pdb_filename}\n"
        "savepdb sys with_H.pdb\n"
        "# Solvate with a 12 Angstrom TIP3P water buffer\n"
        + _tleap_solvate_cmd(ion_cfg.box_shape) + "\n"
        f"{addions_block}\n"
        "# Save AMBER topology and coordinates\n"
        "saveAmberParm sys sys.prmtop sys.rst7\n"
        "quit\n"
    )
    path = run_dir / "render.in"
    path.write_text(script)
    logger.debug("render.in written to %s (salt=%s conc=%.4f)",
                 path, ion_cfg.salt_type, ion_cfg.salt_concentration)
    return path


async def _run_tleap(run_dir: Path) -> tuple[int, str, str]:
    """
    Execute ``tleap -f render.in`` inside *run_dir* using the HPC Environment
    Modules system to load AmberTools.

    ``module load <amber>`` is a shell function — not an executable — so the
    invocation runs through ``bash --login -c`` which:

      1. Sources the modules initialisation script (modules.sh / lmod/init/bash)
         so that the ``module`` function becomes available in the subshell.
      2. Issues ``module load <AMBER_MODULE_NAME>``, which sets PATH / LD_LIBRARY_PATH
         so that ``tleap`` can be found.
      3. Runs ``tleap -f render.in`` with cwd=run_dir.

    Resolution order for the modules init script:
      1. HYDROSIM_MODULES_SH environment variable (explicit path)
      2. First existing file in MODULE_INIT_PROBE_PATHS
      3. Bare ``tleap`` already on PATH (pre-loaded login shell, CI)

    Environment variable overrides
    --------------------------------
    HYDROSIM_AMBER_MODULE  — module name (default: "amber")
    HYDROSIM_MODULES_SH    — absolute path to the modules bash init script

    Returns
    -------
    (return_code, stdout, stderr)

    Raises
    ------
    FileNotFoundError    – when no modules init script is found AND tleap is not
                           already on PATH.
    asyncio.TimeoutError – when execution exceeds TLEAP_TIMEOUT_SECONDS.
    """
    # ── Locate the modules initialisation script ─────────────────────────
    modules_sh: str | None = None
    for candidate in MODULE_INIT_PROBE_PATHS:
        if candidate and os.path.isfile(candidate):
            modules_sh = candidate
            break

    # ── Build the bash -c command string ─────────────────────────────────
    if modules_sh:
        # source → makes `module` available as a shell function
        # module load → sets PATH/LD_LIBRARY_PATH for AmberTools binaries
        # tleap -f render.in → runs in the cwd supplied to create_subprocess_shell
        shell_cmd = (
            f'source "{modules_sh}" && '
            f'module load {AMBER_MODULE_NAME} && '
            f'tleap -f render.in'
        )
        logger.info(
            "tleap via modules: source %s && module load %s && tleap",
            modules_sh,
            AMBER_MODULE_NAME,
        )
    else:
        # No modules init script found — assume the login shell already loaded
        # the amber module (or tleap is directly on PATH for testing).
        shell_cmd = "tleap -f render.in"
        logger.warning(
            "No modules init script found on this host. "
            "Running bare 'tleap -f render.in' — set HYDROSIM_MODULES_SH "
            "to the correct path if tleap is not on PATH."
        )

    proc = await asyncio.create_subprocess_shell(
        shell_cmd,
        cwd=str(run_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        executable="/bin/bash",   # required: module is a bash function, not a binary
    )
    stdout_b, stderr_b = await asyncio.wait_for(
        proc.communicate(),
        timeout=TLEAP_TIMEOUT_SECONDS,
    )
    return (
        proc.returncode or 0,
        stdout_b.decode(errors="replace"),
        stderr_b.decode(errors="replace"),
    )


def _sanitize_folder_name(raw: str) -> str:
    """Return a filesystem-safe, max-64-char directory name."""
    safe = re.sub(r"[^\w.\-]", "_", raw.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:64] or "run"


def _save_study_config(run_dir: Path, payload: StudyConfigPayload) -> Path:
    """Serialise the study configuration to TOML and write it to run_dir."""
    config: dict[str, Any] = {
        "study": {
            "name":        payload.study_name,
            "run_id":      payload.run_id,
            "folder_name": payload.folder_name,
            "created_at":  datetime.now(timezone.utc).isoformat(),
        },
        "protein": {
            "original_file":  payload.original_filename,
            "protonated_file": payload.pdb_to_serve,
            "total_residues": payload.total_residues,
        },
        "charge_scaling": {
            "selected_residues": payload.selected_residues,
            "ks_factors":        payload.ks_factors,
        },
        "ion_config": {
            "salt_type":          payload.salt_type,
            "salt_concentration": payload.salt_concentration,
            "box_shape":          payload.box_shape,
        },
    }
    config_path = run_dir / "config.toml"
    config_path.write_bytes(tomli_w.dumps(config).encode())
    logger.info("config.toml written to %s", config_path)
    return config_path



# ---------------------------------------------------------------------------
# Charge scaling + ParmEd ion neutralization
# ---------------------------------------------------------------------------


def _neutralize_charges_parmed(
    prmtop_path: Path,
    rst7_path: Path,
    selected_residues: list[int],
    ks_factor: float,
    salt_type: str,
    output_prmtop: Path,
    output_rst7: Path,
) -> tuple[float, float, int]:
    """
    Apply charge scaling factor *ks_factor* to the selected residues, then
    redistribute the resulting charge imbalance equally across all ion atoms
    so that the system's net charge returns exactly to 0.0.

    Algorithm
    ---------
    1. Load the AMBER topology (sys.prmtop / sys.rst7) with ParmEd.
    2. Measure Q_initial (must be ≈ 0.0 after TLeap solvation/neutralization).
    3. Scale the partial charges of every atom in *selected_residues* by
       *ks_factor*.
    4. Measure Q_scaled = Σ(all atom charges).
    5. ΔQ = Q_scaled − Q_initial.
    6. Collect all ion atoms whose residue name matches the chosen salt type.
    7. Distribute correction equally: each ion charge += −ΔQ / n_ions.
    8. Write the modified topology to *output_prmtop* / *output_rst7*.

    Examples
    --------
    Example 1 — scaling a Lysine (+1 → +0.8, ks=0.8):
        The system loses +0.2 charge  (ΔQ = −0.2).
        With 20 ions (10 Na+, 10 Cl−): correction = +0.2/20 = +0.01 per ion.

    Example 2 — scaling a Glutamate (−1 → −0.6, ks=0.6):
        The system gains +0.4 charge  (ΔQ = +0.4).
        With 20 ions: correction = −0.4/20 = −0.02 per ion.

    Returns
    -------
    (q_initial, q_scaled, n_ions)
    """
    import parmed as pmd

    struct: pmd.amber.AmberParm = pmd.load_file(str(prmtop_path), str(rst7_path))

    # Step 1 — measure initial net charge
    q_initial: float = round(sum(a.charge for a in struct.atoms), 6)
    logger.debug("ParmEd: Q_initial = %.6f", q_initial)

    # Step 2 — build a set of residue SEQUENCE NUMBERS to scale
    # ParmEd residues are 0-indexed; PDB residue numbers come from a.residue.number
    residue_set: set[int] = set(selected_residues)

    # Step 3 — apply Ks scaling
    scaled_count = 0
    for atom in struct.atoms:
        if atom.residue.number in residue_set:
            atom.charge *= ks_factor
            scaled_count += 1
    logger.debug("ParmEd: scaled %d atoms by %.4f", scaled_count, ks_factor)

    # Step 4 — measure scaled net charge
    q_scaled: float = sum(a.charge for a in struct.atoms)

    # Step 5 — excess charge
    delta_q: float = q_scaled - q_initial
    logger.debug("ParmEd: Q_scaled = %.6f  ΔQ = %.6f", q_scaled, delta_q)

    # Step 6 — collect ion atoms for this salt type
    pos_name, neg_name = SALT_ION_NAMES[salt_type]
    pos_aliases: set[str] = AMBER_ION_ALIASES.get(pos_name, {pos_name})
    neg_aliases: set[str] = AMBER_ION_ALIASES.get(neg_name, {neg_name})
    all_ion_names: set[str] = pos_aliases | neg_aliases

    ion_atoms = [a for a in struct.atoms if a.residue.name in all_ion_names]
    n_ions = len(ion_atoms)

    if n_ions == 0:
        logger.warning(
            "ParmEd: no ion atoms found for salt=%s — charge imbalance %.6f "
            "cannot be redistributed.  This usually means TLeap solvation "
            "did not run (no box in topology).",
            salt_type, delta_q,
        )
    else:
        # Step 7 — redistribute correction equally
        correction_per_ion: float = -delta_q / n_ions
        for ion in ion_atoms:
            ion.charge += correction_per_ion
        q_final = sum(a.charge for a in struct.atoms)
        logger.debug(
            "ParmEd: distributed %.6f across %d ions (%.8f each)  "
            "Q_final = %.10f",
            -delta_q, n_ions, correction_per_ion, q_final,
        )

    # Step 8 — save modified topology
    output_prmtop.parent.mkdir(parents=True, exist_ok=True)
    struct.save(str(output_prmtop), overwrite=True)
    struct.save(str(output_rst7),   overwrite=True)

    return q_initial, q_scaled, n_ions


class GenerateReplicasPayload(BaseModel):
    """Payload for POST /api/study/generate."""

    run_id:            str
    selected_residues: list[int]
    ks_factors:        list[float]
    salt_type:         str   = "NaCl"
    salt_concentration: float = 0.0
    box_shape:          str   = "box"

    @field_validator("run_id")
    @classmethod
    def _v_run_id(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9_\-]{1,80}", v):
            raise ValueError("Invalid run_id format.")
        return v

    @field_validator("ks_factors")
    @classmethod
    def _v_ks(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("At least one Ks factor is required.")
        for f in v:
            if not (0.0 <= f <= 1.0):
                raise ValueError(f"Ks factor {f} out of range [0, 1].")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", summary="Main SPA")
async def root(request: Request):
    """Render the single-page application shell."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get(
    "/api/runs/{run_id}/file/{filename}",
    summary="Serve a file from a run directory",
)
async def serve_run_file(run_id: str, filename: str) -> FileResponse:
    """Stream a PDB or other file from the run directory."""
    if not re.fullmatch(r"[a-zA-Z0-9_\-]{1,80}", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id.")
    if not re.fullmatch(r"[a-zA-Z0-9_\-\.]{1,80}", filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    file_path = DATA_RUNS_DIR / run_id / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in run '{run_id}'.")
    return FileResponse(path=str(file_path), media_type="chemical/x-pdb", filename=filename)


@app.get(
    "/api/residues/map/{run_id}",
    summary="Return residue-number to residue-name mapping for a run",
)
async def get_residue_map(run_id: str) -> JSONResponse:
    """
    Parse ``with_H.pdb`` (or the original PDB if not present) from the run
    directory and return a JSON object mapping residue sequence numbers to
    three-letter residue names.

    Example response: ``{"4": "LYS", "12": "ASP", "102": "PHE"}``
    """
    run_dir  = _resolve_run_dir(run_id)
    pdb_path = run_dir / "with_H.pdb"
    if not pdb_path.is_file():
        # Fall back to any .pdb file present
        candidates = list(run_dir.glob("*.pdb"))
        if not candidates:
            raise HTTPException(status_code=404, detail="No PDB file found in this run.")
        pdb_path = candidates[0]

    residue_map = _parse_residue_map(pdb_path)
    if not residue_map:
        raise HTTPException(status_code=422, detail="Could not parse any residues from the PDB file.")

    return JSONResponse(content={"run_id": run_id, "residues": residue_map})


@app.get(
    "/api/residue/2d/{res_name}",
    summary="Return an RDKit SVG diagram of an amino acid side chain",
)
async def get_residue_2d(res_name: str) -> Response:
    """
    Generate and return a dark-theme SVG of the requested amino acid side chain
    using RDKit.  Supports the 20 canonical residues and common AMBER variants
    (HID/HIE/HIP, CYX, etc.).
    """
    if not re.fullmatch(r"[A-Za-z0-9]{1,6}", res_name):
        raise HTTPException(status_code=400, detail="Invalid residue name.")

    svg = _generate_residue_svg(res_name.upper())
    return Response(content=svg, media_type="image/svg+xml")


@app.post(
    "/api/study/config",
    summary="Save study configuration to config.toml",
    status_code=status.HTTP_201_CREATED,
)
async def save_study_config(payload: StudyConfigPayload) -> JSONResponse:
    """
    Persist the study configuration (selected residues, Ks factors, metadata)
    as a TOML file inside the run directory.

    Returns the path of the written file relative to the data root.
    """
    run_dir     = _resolve_run_dir(payload.run_id)
    config_path = _save_study_config(run_dir, payload)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status":      "ok",
            "run_id":      payload.run_id,
            "config_path": str(config_path.relative_to(PROJECT_ROOT.parent)),
        },
    )


@app.post(
    "/api/upload",
    summary="Upload a PDB file and run tleap hydrogen addition",
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdb(
    pdb_file: UploadFile = File(..., description="Protein structure in PDB format (.pdb)"),
) -> JSONResponse:
    """
    Full pipeline:
    1. Validate extension and size.
    2. Persist the original file.
    3. Write tleap render.in script.
    4. Execute tleap asynchronously.
    5. Verify with_H.pdb output.
    6. Count residues.
    7. Return JSON metadata.
    """
    original_name: str = pdb_file.filename or "structure.pdb"
    stem: str          = Path(original_name).stem
    suffix: str        = Path(original_name).suffix.lower()

    if suffix != ALLOWED_EXTENSION:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid file type '{suffix}'. Only '{ALLOWED_EXTENSION}' files are accepted.",
        )

    contents: bytes = await pdb_file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the maximum allowed size of 100 MB.",
        )

    run_id      = uuid.uuid4().hex
    folder_name = _sanitize_folder_name(stem) or run_id
    run_dir     = DATA_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    original_dest = run_dir / original_name
    original_dest.write_bytes(contents)
    logger.info("PDB saved  run_id=%s  file=%s  bytes=%d", run_id, original_name, len(contents))

    _write_tleap_script(run_dir, original_name)

    tleap_rc, tleap_stdout, tleap_stderr = -1, "", ""
    tleap_error: str | None = None

    try:
        tleap_rc, tleap_stdout, tleap_stderr = await _run_tleap(run_dir)
        logger.info("tleap finished  rc=%d  run_id=%s", tleap_rc, run_id)
        if tleap_rc != 0:
            tleap_error = (
                f"tleap exited with code {tleap_rc}. "
                f"stderr: {tleap_stderr[:500] or '(empty)'}"
            )
    except FileNotFoundError:
        tleap_error = (
            "tleap binary not found. Install AmberTools and ensure 'tleap' is on PATH."
        )
        logger.warning("tleap not found  run_id=%s", run_id)
    except asyncio.TimeoutError:
        tleap_error = f"tleap timed out after {TLEAP_TIMEOUT_SECONDS} seconds."
        logger.error("tleap timeout  run_id=%s", run_id)

    with_h_path = run_dir / "with_H.pdb"

    if tleap_error or not with_h_path.is_file():
        if not tleap_error:
            tleap_error = "tleap ran but 'with_H.pdb' was not created."
        pdb_to_serve  = original_name
        residue_count = _count_residues(original_dest)
        tleap_ok      = False
    else:
        residue_count = _count_residues(with_h_path)
        pdb_to_serve  = "with_H.pdb"
        tleap_ok      = True

    # Parse ion counts reported by TLeap in its stdout
    ions_added: dict[str, int] = {}
    if tleap_ok and tleap_stdout:
        ions_added = _parse_ions_from_tleap_log(tleap_stdout)
    total_ions = sum(ions_added.values())

    logger.info(
        "Residues detected: %d  ions_added: %s  run_id=%s",
        residue_count, ions_added, run_id,
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status":            "ok",
            "run_id":            run_id,
            "folder_name":       folder_name,
            "original_filename": original_name,
            "pdb_to_serve":      pdb_to_serve,
            "pdb_url":           f"/api/runs/{run_id}/file/{pdb_to_serve}",
            "size_bytes":        len(contents),
            "residue_count":     residue_count,
            "tleap_ok":          tleap_ok,
            "tleap_error":       tleap_error,
            "tleap_stdout":      tleap_stdout[:2000] if tleap_stdout else None,
            "ions_added":        ions_added,
            "total_ions":        total_ions,
        },
    )

@app.post(
    "/api/study/generate",
    summary="Generate scaled charge replicas with ParmEd neutralization",
    status_code=status.HTTP_201_CREATED,
)
async def generate_replicas(payload: GenerateReplicasPayload) -> JSONResponse:
    """
    For each Ks factor in *payload.ks_factors*, produce a charge-scaled AMBER
    topology replica under ``<run_dir>/replicas/k_<ks>/``.

    Steps per replica
    -----------------
    1. Copy sys.prmtop + sys.rst7 from the run directory.
    2. Call ``_neutralize_charges_parmed`` to apply Ks scaling and redistribute
       the resulting charge imbalance across all ion atoms.
    3. Write the modified topology as ``scaled.prmtop`` and ``scaled.rst7``.
    4. Return a summary of each replica (Ks, paths, Q values, n_ions).

    Requires that TLeap has already run successfully (sys.prmtop must exist).
    If sys.prmtop is absent (e.g., TLeap was not available), returns HTTP 409.
    """
    run_dir = _resolve_run_dir(payload.run_id)
    prmtop  = run_dir / "sys.prmtop"
    rst7    = run_dir / "sys.rst7"

    if not prmtop.is_file() or not rst7.is_file():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "sys.prmtop / sys.rst7 not found in this run directory.  "
                "TLeap solvation must complete successfully before generating replicas."
            ),
        )

    replicas: list[dict] = []

    for ks in payload.ks_factors:
        # Replica directory name: k_0_8 for ks=0.8, k_1_0 for ks=1.0
        label     = f"k_{str(round(ks, 4)).replace('.', '_')}"
        replica_dir = run_dir / "replicas" / label
        replica_dir.mkdir(parents=True, exist_ok=True)

        out_prmtop = replica_dir / "scaled.prmtop"
        out_rst7   = replica_dir / "scaled.rst7"

        try:
            q_initial, q_scaled, n_ions = _neutralize_charges_parmed(
                prmtop_path       = prmtop,
                rst7_path         = rst7,
                selected_residues = payload.selected_residues,
                ks_factor         = ks,
                salt_type         = payload.salt_type,
                output_prmtop     = out_prmtop,
                output_rst7       = out_rst7,
            )
            replicas.append({
                "ks":         ks,
                "label":      label,
                "prmtop":     str(out_prmtop.relative_to(PROJECT_ROOT.parent)),
                "rst7":       str(out_rst7.relative_to(PROJECT_ROOT.parent)),
                "q_initial":  round(q_initial, 6),
                "q_scaled":   round(q_scaled, 6),
                "delta_q":    round(q_scaled - q_initial, 6),
                "n_ions":     n_ions,
                "status":     "ok",
            })
            logger.info(
                "Replica generated  label=%s  ks=%.4f  n_ions=%d  run_id=%s",
                label, ks, n_ions, payload.run_id,
            )
        except Exception as exc:
            logger.error("Replica failed  label=%s  error=%s", label, exc)
            replicas.append({
                "ks":     ks,
                "label":  label,
                "status": "error",
                "error":  str(exc),
            })

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status":   "ok",
            "run_id":   payload.run_id,
            "replicas": replicas,
        },
    )
