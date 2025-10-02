# Copilot instructions for Modeling-Radiance

These notes give focused, actionable context so an AI coding agent can be productive immediately in this repository.

High-level architecture (big picture)
- Data ingestion: MODTRAN TAPE7 files under `data/Modtran_Unfiltering_Tape7s_*`. Primary parsers:
  - `tp7/tp7.py` — main Tape7 class that reads a .tp7, builds a scene descriptor DataFrame, computes radiances, and integrates them.
  - `matt_code/convert_tp7.py` and `tp7/fergus_code/*` contain alternate parsers and helpers; prefer `tp7/tp7.py` for canonical behavior.
- Spectral response functions (SRFs): `srfs/srfs.py` provides an `SRFS` class that loads and normalizes SRF CSVs. `filter_functions/` contains additional SRF artifacts (.sav)
- Integration & modeling: Notebooks (especially `modeling.ipynb`) assemble Tape7 outputs + SRFs, produce integrated filtered/unfiltered radiances, and train models (Random Forests mentioned in README).

Key data flow
- Read raw .tp7 -> Tape7._read_tp7() produces a numeric grid + header texts.
- Tape7._build_scene_description() parses header text into descriptor DataFrame (`describer_df`).
- Tape7._compute_radiences() computes wavelength, SW and LW radiances per run.
- Tape7._integrate_radiances() interpolates SRFs and integrates with Simpson's rule to produce filtered/unfiltered integrated radiances stored on `describer_df`.

Important implementation details & gotchas (discoverable patterns)
- Header parsing is brittle and format-dependent:
  - `_parse_metadata()` inspects first/last lines to pick `size`, `num_cols`, and `start_index`. Some files use 12 columns, others 14; `_get_column_numbers()` maps column indexes accordingly.
  - Several places assume specific header line positions (e.g. headerdata[i][10], headerdata[i][5]) — prefer using existing helper functions when possible.
- Unit conversions and ordering:
  - Frequency -> wavelength conversion uses lam = (1 / freq) * 1e4 and arrays are reversed in `_compute_radiences()` (note the `[::-1]`).
  - For Land and Deep Convective Cloud scenes the code adjusts VZA using `vza = 180 - vza`.
- SRF handling:
  - `srfs.SRFS(process_srf)` constructs an SRF table with range endpoints; Tape7 interpolates SRF onto radiance wavelengths via `np.interp` before integrating with `scipy.integrate.simpson`.

Developer workflows & commands
- Recommended environment (conda) — a pinned environment is provided in `requirements.txt` (file is in conda "--file" format). Typical setup:

```bash
# create environment (macOS arm64 recommended if using the provided file)
conda create --name mr_env --file requirements.txt
conda activate mr_env
```

- Run notebooks interactively:
  - Start Jupyter Lab or open `modeling.ipynb` to run the end-to-end pipeline and visualization.

- Quick script test (python REPL or small script):
  - Instantiate SRF and Tape7 to process a single file (use paths under `data/` and SRF CSVs under `srfs/`). The canonical constructor is `Tape7(filepath, srf_wvlns, sw_srf, lw_srf)` (see `tp7/tp7.py`).

Project conventions and patterns
- Notebooks are the primary driver for experiments and model training. Scripts under `matt_code/` and `tp7/` provide reusable programmatic access used by notebooks.
- Data-first design: many operations mutate and extend a descriptor DataFrame (`describer_df`) with integrated results; prefer changes that keep this DataFrame stable and column-named consistently.
- Avoid changing header-parsing heuristics without adding tests or sample files — many parsing branches are triggered by small differences in header text length/columns.

Integration points and external dependencies
- Heavily depends on: numpy, pandas, scipy, scikit-learn, matplotlib, jupyter. See `requirements.txt` for full list and platform hints.
- Uses local SRF artifacts in `filter_functions/` (some `.sav` files) — treat them as opaque inputs unless you need to re-generate with `matt_code/make_srf.py`.

When editing code, prioritize these places for most common changes
- `tp7/tp7.py` — change here for parsing logic, radiance calculations, or descriptor columns.
- `srfs/srfs.py` — change here for SRF loading/format adjustments.
- `modeling.ipynb` — update pipeline orchestration, model training code and visualizations.

Where to look for examples
- End-to-end usage: `modeling.ipynb` (top cells describe data sources and pipeline).
- Low-level parsing: `tp7/tp7.py` and `tp7/fergus_code/readtp7.py` to compare alternative parsing approaches.
- SRF examples: `srfs/srfs.py` and `matt_code/make_srf.py`.

If something is ambiguous
- Inspect a real `.tp7` under `data/Modtran_Unfiltering_Tape7s_*` to see which header format/column count it has before changing parsers.
- Prefer adding a small notebook cell that demonstrates a change (so CI isn't required). If you modify parsing heuristics, add a short sample test notebook or a minimal script in `matt_code/`.

If you want me to expand these rules into unit-test scaffolding, or to add small runnable examples/scripts, tell me which area (parsing, SRF handling, or modeling) and I'll add them.
