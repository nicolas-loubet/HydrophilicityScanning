# HydroScan — HydrophilicityScanning

**Molecular simulation tool for studying protein hydrophobicity / hydrophilicty via charge scaling in AMBER.**

## Current version: **0.0.4**

HydroScan is a single-page application that guides a researcher through an entire AMBER charge-scaling pipeline — from uploading a raw PDB structure to generating the AMBER `.in` input files for a molecular dynamics protocol — without leaving one scrolling page.

The core scientific idea: take selected residues in a protein, scale down their partial charges by a factor **k** (e.g. `1.0 → 0.8 → 0.6 → 0.4`), and study how that affects the protein's hydrophobic behavior, while keeping the whole system's net charge exactly neutral by redistributing the charge difference across the solvent's ions.

Repository: [github.com/nicolas-loubet/HydrophilicityScanning](https://github.com/nicolas-loubet/HydrophilicityScanning)

---

## Table of contents

- [Overview](#overview)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Pipeline walkthrough](#pipeline-walkthrough)
  - [Section 1 — Protein Upload](#section-1--protein-upload)
  - [Section 2 — Study Configuration & 3D Rendering](#section-2--study-configuration--3d-rendering)
  - [Section 3 — Molecular Dynamics Protocol](#section-3--molecular-dynamics-protocol)
- [Charge-neutralization algorithm](#charge-neutralization-algorithm)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

HydroScan is built as a **strict single-page application**: one HTML document, one vertical scroll, sections unlocking sequentially as the user completes each pipeline stage. There is no client-side router and no JS framework — interactivity is handled by small, self-contained vanilla-JS modules.

The backend is a thin FastAPI layer around two scientific tools:

- **AmberTools `tleap`** — protonation, solvation, and ion placement.
- **ParmEd** — post-hoc charge scaling and charge-neutrality redistribution across ions.

No database is used. All state lives in flat files (PDB, AMBER topology/coordinate files, TOML config) under a per-run directory, making every study fully inspectable and reproducible from disk.

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn | Single worker, `reload=True` for dev |
| Templating | Jinja2 | Server-side render of the base layout + single page |
| Styling | Tailwind CSS (CDN) | Custom "Cyber Amber" theme — see below |
| Fonts | Google Fonts: JetBrains Mono + DM Sans | Loaded via CDN |
| Frontend JS | Vanilla JS, IIFEs exposed on `window.*` | No React/Vue/Alpine |
| 3D viewer | NGL Viewer (CDN) | Cartoon/ball+stick/surface representations |
| 2D chemistry | RDKit (Python) | Server-rendered SVG per amino acid |
| MD engine | AmberTools (`tleap`) | Invoked via `module load amber` + subprocess |
| Charge math | ParmEd | Loads/edits/saves AMBER `prmtop`/`rst7` |
| Persistence | TOML (`tomli_w`) + flat files | No database |

### Visual theme — "Cyber Amber"

```js
colors: {
  'surface-900': '#0d0d11', // deep background
  'surface-800': '#16161f', // section containers
  'surface-700': '#23232f', // borders / inputs
  'accent':      '#f59e0b', // electric amber — buttons, highlights
  'text-dim':    '#9ca3af', // secondary text
}
```

---

## Project structure

```
hydrophobic_sim/
├── main.py                     # Uvicorn entry point (python main.py)
├── requirements.txt
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI app: routes, Pydantic models, AMBER/ParmEd logic
├── templates/
│   ├── base.html               # Layout shell: theme, fonts, header/footer, favicon
│   └── index.html              # The entire SPA: Sections 1–3 + all JS modules
└── static/
    └── icon.png

../data/runs/<run_id>/          # Created at runtime, one folder per uploaded study
```

---

## Installation

### Prerequisites

- Python 3.11+
- **AmberTools** (provides `tleap`) installed on the host and loadable via the
  [Environment Modules](https://modules.readthedocs.io/) system (`module load amber`)
- A C-toolchain is not required — RDKit and ParmEd are installed as Python wheels

### Steps

```bash
git clone https://github.com/nicolas-loubet/HydrophilicityScanning.git
cd HydrophilicityScanning

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` includes:

```
fastapi
uvicorn[standard]
jinja2
python-multipart
aiofiles
rdkit
tomli-w
parmed
```

### AmberTools / module setup

`tleap` is **not** invoked directly — it's a shell function exposed by `module load amber`, so the backend spawns a `bash --login` subshell that:

1. Sources the cluster's Environment Modules init script (`modules.sh` / Lmod's `init/bash`)
2. Runs `module load amber` (override the module name with `HYDROSIM_AMBER_MODULE` if your site uses e.g. `amber/24`)
3. Executes `tleap -f render.in`

If your modules init script isn't in one of the commonly probed locations, point to it explicitly:

```bash
export HYDROSIM_MODULES_SH=/path/to/lmod/init/bash
export HYDROSIM_AMBER_MODULE=amber        # or amber/24, ambertools/23, etc.
```

If neither AmberTools nor a modules system is available, the app still runs: TLeap failures are caught and surfaced as a non-fatal warning in the UI, falling back to the unprocessed PDB so the rest of the pipeline (residue selection, 2D viewer, MD input generation) remains usable for review.

---

## Running the app

```bash
python main.py
```

This starts Uvicorn on `http://0.0.0.0:5000` with auto-reload enabled. Open `http://localhost:5000` in a browser, or use any service for exposing it.

---

## Pipeline walkthrough

The interface is one page, divided into three sections that unlock top-to-bottom as you progress.

### Section 1 — Protein Upload

- Drag-and-drop or click-to-browse zone, strictly validating the `.pdb` extension (100 MB max).
- Upload is sent via `XMLHttpRequest` (not `fetch`) so a real upload progress bar can be shown.
- On submit, the backend:
  1. Saves the file to a new `data/runs/<run_id>/` directory (`run_id` is a UUID4 hex).
  2. Writes a TLeap script (`render.in`) that loads the structure, adds hydrogens (ff19SB), solvates it (box or truncated octahedron, TIP3P), and adds ions per the chosen salt configuration.
  3. Runs `tleap -f render.in` asynchronously and parses its stdout for the actual ion counts placed.
  4. Counts residues from the resulting structure (`with_H.pdb`) by collecting unique `(chain, resSeq)` pairs from Cα atoms.
- If TLeap fails for any reason (missing binary, timeout, non-zero exit, missing output file), the endpoint still returns `HTTP 201` with a descriptive `tleap_error` and falls back to serving the original PDB — the app never hard-crashes on a missing AMBER installation.

### Section 2 — Study Configuration & 3D Rendering

Unlocks with a smooth fade/scroll once Section 1 succeeds.

**Metadata** — "Study Name" and "Server Folder Name" inputs; the folder name is auto-suggested from the uploaded filename (sanitized to a safe slug) and can be overridden.

**3D viewer (left column)** — an [NGL Viewer](https://nglviewer.org/) instance loads the protonated structure directly from `/api/runs/{run_id}/file/{filename}`. Default representation is `cartoon` colored by secondary structure; toggle buttons switch to `ball+stick` or `surface`. Fully interactive (rotate/zoom/pan with the mouse).

**Residue selection (right column)** — type comma-separated residue numbers (e.g. `4, 12, 102`); client-side validation rejects out-of-range or malformed values and renders the valid selection as amber pill tags. A "Fetch Names" step calls `/api/residues/map/{run_id}` to resolve numbers to three-letter codes, populating a dropdown.

**2D residue viewer** — selecting a residue from the dropdown calls `/api/residue/2d/{res_name}`, which uses RDKit to render an SVG of the **full amino acid**, styled for the dark theme: light bond/atom colors, transparent background, and visible (light-gray, not black) stereochemistry wedge bonds.

**k charge-scaling matrix** — toggle chips for the default factors `1.0 / 0.8 / 0.6 / 0.4`, plus a field to add custom factors.

**Ion configuration** — salt type (`NaCl` / `KCl` / `MgCl2`), concentration in mol/L (`0` = neutralize only, no extra salt), and solvent box geometry (`Rectangular Box` or `Truncated Octahedron`, the latter using ~20% less water for the same buffer distance).

**Confirm Configuration** — persists everything (`config.toml`) and unlocks Section 3.

### Section 3 — Molecular Dynamics Protocol

A sequential, card-based builder for the AMBER minimization pipeline that will be applied identically to every **k** replica.

**Global Configuration card** — a single shared `cut` (cutoff, Å, default `12.0`) and an "Advanced Settings" accordion holding `iwrap` (default `1`).

**Pipeline builder** — pre-loaded with two Minimization stages:
- **Min-1**: `maxcyc=5000`, `ncyc=2500`, `ntwr=500`, restraints **on** (`restraint_wt=10.0`, `restraintmask="(:1-N & !:WAT & !@H=)"` where `N` is the detected residue count).
- **Min-2**: same defaults, restraints **off**.

Cards can be deleted (down to a minimum of one), and new Minimization cards can be inserted at any position via hover-activated "+" controls that appear between cards, above the first, and below the last. Stage labels (`Min-1`, `Min-2`, `Min-3`, …) automatically re-index whenever the sequence changes — labels are recomputed, not stored statically.

Each card has its own "Advanced Settings" accordion exposing `ntpr` (default `50`).

**Generate MD Inputs** — submits the ordered stage list and global config to the backend, which writes one `.in` file per stage directly into the run's root directory (shared across all **k** replicas, since the same protocol applies regardless of which residues were scaled), using zero-padded sequential naming: `01_Min.in`, `02_Min.in`, etc.

#### AMBER `&cntrl` template

Structural flags are **hardcoded server-side** and never accepted from the client:

```
imin=1, ntb=1, ntp=0, ntf=1, ntc=1, ntxo=1
```

Unrestrained stage:

```
&cntrl
imin=1, maxcyc={maxcyc}, ncyc={ncyc}, ntb=1, ntp=0, ntf=1, ntc=1,
ntpr={ntpr}, iwrap={global_iwrap}, ntwr={ntwr}, cut={global_cut}, ntxo=1,
/
```

Restrained stage (restraint block appended before the closing `/`):

```
&cntrl
imin=1, maxcyc={maxcyc}, ncyc={ncyc}, ntb=1, ntp=0, ntf=1, ntc=1,
ntpr={ntpr}, iwrap={global_iwrap}, ntwr={ntwr}, cut={global_cut}, ntxo=1,
ntr=1, restraint_wt={restraint_wt}, restraintmask="{restraintmask}",
/
```

---

## Charge-neutralization algorithm

Scaling a charged residue's partial charges by `k` breaks the system's net neutrality (a perfectly solvated/neutralized system starts at net charge `Q = 0.0`). HydroScan fixes this with ParmEd by spreading the resulting imbalance evenly across every ion atom in the topology:

1. Load the AMBER topology (`sys.prmtop` / `sys.rst7`).
2. Record `Q_initial` (≈ `0.0`, the state right after TLeap's neutralization).
3. Multiply the partial charge of every atom belonging to the selected residues by `k`.
4. Record `Q_scaled` — the new net charge after scaling.
5. Compute `ΔQ = Q_scaled − Q_initial`.
6. Collect every ion atom whose residue name matches the configured salt type (handling AMBER's alternate spellings, e.g. `Na+` / `NA` / `SOD`).
7. Add `−ΔQ / n_ions` to **every single ion's** charge, bringing the system back to exactly `0.0`.
8. Save the modified topology as a new replica (`replicas/k_<k>/scaled.prmtop` + `.rst7`).

**Worked examples** (also documented as code comments):

- Scaling a Lysine `+1 → +0.8` (`k` = 0.8) loses `+0.2` of charge (`ΔQ = −0.2`). With 20 ions in the box (10 Na⁺, 10 Cl⁻), every ion's charge is adjusted by `+0.01`.
- Scaling a Glutamate `−1 → −0.6` (`k` = 0.6) gains `+0.4` of charge (`ΔQ = +0.4`). With the same 20 ions, every ion's charge is adjusted by `−0.02`.

If TLeap reports zero counter-ions because the protein was already perfectly neutral, the backend forces at least one positive and one negative ion into the topology — without at least one ion of each sign, step 7 has nothing to redistribute onto.

---

## Configuration

Environment variables (all optional — sensible defaults are probed automatically):

| Variable | Default | Purpose |
|---|---|---|
| `HYDROSIM_AMBER_MODULE` | `amber` | Module name passed to `module load` |
| `HYDROSIM_MODULES_SH` | *(auto-probed)* | Absolute path to the Environment Modules `init/bash` script, if not found automatically |

Hardcoded constants (in `app/main.py`, change directly if needed):

| Constant | Value | Purpose |
|---|---|---|
| `MAX_UPLOAD_BYTES` | 100 MB | Upload size ceiling |
| `ALLOWED_EXTENSION` | `.pdb` | Strict upload extension check |
| `TLEAP_TIMEOUT_SECONDS` | 120 | Subprocess timeout for TLeap |

---

## License

MIT - see `LICENSE` for full text.
