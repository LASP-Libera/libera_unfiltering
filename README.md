# Radiance Unfiltering for the Libera Mission

## What is Unfiltering?

Before reaching a satellite detector, incoming radiance passes through channel-specific optical filters that shape the measured spectral signal according to the instrument's Spectral Response Functions (SRFs). As a result, the radiances recorded by broadband radiometers are spectrally *filtered* versions of the true Earth-emitted and reflected radiances.

Recovering the corresponding unfiltered radiances — required for downstream products like Angular Distribution Models (ADMs) and Earth Radiation Budget (ERB) flux estimates — is called **unfiltering**.

This repository implements the unfiltering algorithm for the [Libera mission](https://lasp.colorado.edu/libera/), NASA's upcoming satellite to continue measuring the Earth's Radiation Budget (ERB) as a successor to CERES.

---

## The Unfiltering Problem

The Libera instrument measures radiances in four channels:

| Channel | Description |
|---|---|
| SW | Shortwave (reflected solar) |
| SSW | Split Shortwave (0.7–5 µm) |
| LW | Longwave (emitted thermal) |
| TOT | Total broadband |

The **unfiltered** radiance for each channel is defined as the integral of spectral radiance over its wavelength range. The **filtered** measurement is obtained by convolving the spectral radiance with the instrument's SRF:

```
m_f^j = ∫ S_λ^j · I_λ dλ
```

Because the SRFs are not spectrally flat, and because Earth's radiance spectra vary substantially with surface type, atmospheric state, and viewing geometry, there is no single global relationship between filtered and unfiltered radiances. This nonlinearity is what makes unfiltering non-trivial.

---

## Current Implementation: Traditional Regression-Based Unfiltering

The current implementation follows the regression-based method established for CERES (Loeb et al., 2001), which forms the operational baseline for Libera unfiltering.

### How It Works

Filtered and unfiltered radiances are related through a quadratic regression per channel:

```
m_u^CH = a_0 + a_1(m_f^CH) + a_2(m_f^CH)²
```

For the split shortwave channel, a multivariate degree-2 polynomial uses both the SSW and SW filtered measurements as predictors (7 coefficients total):

```
m_u^SSW = c_0 + c_1·1 + c_2(m_f^SSW) + c_3(m_f^SW) + c_4(m_f^SSW)² + c_5(m_f^SSW · m_f^SW) + c_6(m_f^SW)²
```

### Scene and Geometry Stratification

Because the filtered-to-unfiltered relationship depends on scene type and viewing geometry, separate regression coefficients are derived for each combination of:

**Scene types (5) × Cloud flag (2) = 10 scene/cloud strata**

| Scene | Cloud flag |
|---|---|
| Land | 0 = clear, 1 = cloudy |
| Cloudy Ocean | 0 = clear, 1 = cloudy |
| Clear Ocean | 0 = clear, 1 = cloudy |
| Snow | 0 = clear, 1 = cloudy |
| Deep Convective Cloud | always 1 (cloudy) |

Scene type and cloud flag are derived at inference time from the SCENE-ID-CAM ancillary file (TRMM `surface_type` + `cloud_fraction > 10%`).

**Angular bins (5 × 5 × 5 = 125 per scene/cloud stratum)**

All bin edges follow Loeb et al. (2001), Table 1:

| Angle | Bins | Edges (degrees) |
|---|---|---|
| Solar Zenith Angle (SZA) | 5 | 0, 22.2, 41.4, 60, 75.5, 85 |
| Viewing Zenith Angle (VZA) | 5 | 0, 15, 30, 45, 60, 90 |
| Relative Azimuth Angle (RAZ) | 5 | 0, 15, 60, 120, 165, 180 |

This gives a **coefficient cube** of shape `(5 scenes × 2 cloud × 5 SZA × 5 VZA × 5 RAZ)` = 1,250 bins. Each bin stores independent quadratic regression coefficients for each channel (7 coefficients for SW/SSW, 3 for LW/TOT). Bins with fewer than 3 MODTRAN samples are left as NaN and produce NaN output at inference time.

### Coefficient Cube Generation

Coefficients are pre-generated offline from MODTRAN radiative transfer simulations and stored in a NetCDF file under `coefficients/`.

**From MODTRAN 3.7 tape7 files (current):**

```bash
PYTHONPATH=. .venv/bin/python -c "
from prod.std.standard_method import run
run(data_dir='data/Modtran_3-7_data/', srf_dir='data/SRF/',
    srf_version='0-0-1', modtran_version='3.7')
"
```

**From MODTRAN 6 NetCDF files (pending SDC data delivery):**

```bash
PYTHONPATH=. .venv/bin/python -c "
from prod.std.standard_method import run_nc
run_nc(data_dir='<S3_PATH_FROM_SDC>', srf_dir='data/SRF/',
       srf_version='0-0-1', modtran_version='6')
"
```

The generation pipeline (`prod/std/standard_method.py`):
1. Parses all MODTRAN simulation files into a flat DataFrame (one row per run: SZA, VZA, RAZ, scene, cloud, integrated radiances)
2. Stratifies by scene × cloud × SZA bin × VZA bin × RAZ bin
3. Fits quadratic regression per bin per channel
4. Writes the full coefficient cube to `coefficients/unfiltering_coefficients_v{version}_srf-{srf}_modtran-{modtran}.nc`

At runtime, `COEFFICIENTS_FILE` env var overrides the default (latest file in `coefficients/`).

---

## Dataset

Simulated radiances are generated using MODTRAN 3.7. The dataset contains 6,195 samples across five scene types:

| Scene | Runs |
|---|---|
| Land | 3,150 |
| Snow | 1,575 |
| Clear Ocean | 735 |
| Cloudy Ocean | 420 |
| Deep Convective Cloud | 315 |

---

## Future Work: Machine Learning Unfiltering

The traditional regression approach requires explicit scene classification and maintaining hundreds of independent regression models — one per scene/geometry bin combination.

Future work in this repository will explore **tree-based machine learning** (Random Forest, gradient boosting) as a unified alternative. Rather than binning by scene and geometry, a single global model learns the continuous nonlinear mapping between filtered measurements, viewing geometry, and unfiltered radiance directly. This eliminates the need for explicit scene stratification.

**Current status: only the standard regression method is implemented. Machine learning research is ongoing.**

---

## Open Issues and Pending Work

### Blocked on SDC / External Data

| Item | Status | Blocked on |
|---|---|---|
| MODTRAN 6 coefficient generation | Code ready (`tp7/modtran6.py`, `standard_method.run_nc()`) | SDC team providing S3 path to M6 .nc files |
| `CERES_SCENE_MAP` completion | Partial (Ocean only) | Seeing full range of `CERES_TRMM_Scene_ID` values in M6 data; Land/Snow/DCC IDs unknown |
| Docker push to ECR | Script ready (`scripts/push_to_ecr.sh`) | AWS credentials setup |

### Known Science Limitations

**DCC not detectable at inference time.**
The SCENE-ID-CAM ancillary file does not include cloud optical thickness. The Deep Convective Cloud scene type is therefore never assigned at inference — those samples fall into Cloudy Ocean (if over ocean) or Land. At training time (MODTRAN 3.7), DCC is detected correctly from ICLD codes. This means the DCC coefficient bin is populated but never used in production. Resolution requires either: (a) adding cloud optical thickness to SCENE-ID-CAM, or (b) obtaining it from a separate ancillary source.

**Sparse coefficient bins produce NaN output.**
The current MODTRAN 3.7 dataset (6,195 runs) does not cover all 1,250 scene/cloud/angle bin combinations. Bins with fewer than 3 samples have no coefficients and produce NaN in the output. Approximately 73% of samples in the synthetic test data are filled; coverage with real L1B data will vary by orbit. M6 data with broader scene/geometry sampling would improve bin coverage.

**Coefficients derived from MODTRAN 3.7, not MODTRAN 6.**
MODTRAN 3.7 is the current source. M6 provides higher spectral resolution and more realistic atmospheric physics. All code is ready for M6; it is blocked only on data availability (see above).

### Infrastructure

**No real L1B test data.**
The integration tests use a synthetic L1B file with nearly-constant radiance values (~3,200 across all samples). Tests verify structural correctness but not scientific accuracy. End-to-end validation with real Libera L1B data is needed before operational deployment.

**Branch cleanup pending.**
The following local branches are fully subsumed by `docker-img-build` and can be deleted when ready: `data-file-integration`, `oa`, `rf-srf`, `fix-units`, `generate-coefficient-cube`, `ml-m6`, `scene-mapping`.

---

## Installation

For research install required packages from `requirements.txt` in the research folder:

```bash
pip install pandas numpy scikit-learn scipy
```

Some packages may require conda. If `requirements.txt` fails, install `pandas`, `numpy`, `sklearn`, and `scipy` first in that order — they pull in most other dependencies.

Notebook instructions are located in the top cell of `nb_pt1.ipynb`.

---

For questions contact Caleb Kumar.
