# Changelog — Attitude Pointing Analysis Suite
**Endurance FR RDV — GEO Rendezvous Mission**
**Rev 2 — 29 April 2026**

---

## Overview

This revision supersedes the initial pointing feasibility assessment (Rev 1, commit `2f683af`). The primary drivers for revision are: (1) consolidation of the operational mode taxonomy from three modes to two, (2) integration of propulsion event ephemerides from MATLAB simulation data packages, (3) extension of the analysis to cover thermal and operational sun-exposure constraints, and (4) restructuring of all output figures to resolve the far-range and close-range rendezvous phases as geometrically distinct regimes.

---

## 1. Mode Taxonomy Correction

**Affected files:** all scripts, `Mode3_Pointing_Analysis.md`

The prior revision incorrectly introduced a third pointing mode ("Mode 3 — Earth+Target") as a distinct operational attitude. The corrected taxonomy recognises only two operational attitude modes:

- **Mode 1** — Target+Sun: primary constraint +Z body axis aligned to the rendezvous target; secondary constraint solar array normal optimised toward the Sun.
- **Mode 2** — Nadir+Sun: primary constraint +Z body axis aligned to the local nadir (Earth centre direction); secondary constraint solar array normal optimised toward the Sun.

The simultaneous Earth–Target pointing requirement is re-cast as a **combined pointing feasibility assessment** conducted within the Mode 1 and Mode 2 attitude frameworks. No separate operational mode is defined. All internal references to "Mode 3" have been removed throughout the codebase and documentation.

The combined Earth+Target attitude — defined with +Z aligned to the target and +Y constrained to the Earth direction by projection onto the plane orthogonal to the primary boresight — is retained as a **reference attitude** for the eigenaxis slew cost computation, but is not assigned an operational mode designation.

---

## 2. Data Ingestion — Migration from CSV to MATLAB Data Packages

**Affected files:** `earth_target_separation.py`, `mode3_feasibility.py`, `mode3_slew_analysis.py`
**New dependency:** `scipy.io.loadmat`

All three analysis scripts have been migrated from flat CSV ingestion to direct loading of the MATLAB simulation data packages (`.mat`) produced by the GNC simulation tool. The data packages reside in two mode-specific directories:

```
end1_target_sunOpt/   — Mode 1 (Target+Sun) simulation outputs
end1_Nadir_sunOpt/    — Mode 2 (Nadir+Sun) simulation outputs
```

Each directory contains two temporal phases:
- `dataPackage4Thermal_RDV.mat` — far-range rendezvous phase (5,467 samples, Δt = 300 s)
- `dataPackage4Thermal_INS.mat` — inspection phase (1,154 samples, Δt = 300 s)

The `.mat` data structure (`data4thermal`) exposes the following fields consumed by the analysis suite:

| Field | Description | Usage |
|---|---|---|
| `pos_Earth2satCoG_ECI_m` | Servicer CoG position in ECI frame [m] | Earth–Target angular separation |
| `pos_Earth2tgtCoG_ECI_m` | Target CoG position in ECI frame [m] | Earth–Target angular separation |
| `quat_ECI2MRF` | Attitude quaternion, ECI → MRF (scalar-first convention) | Eigenaxis slew computation |
| `uv_sat2Earth_MRF` | Unit vector, spacecraft to Earth centre in body frame | Pre-computed pointing reference |
| `uv_sat2Target_MRF` | Unit vector, spacecraft to target in body frame | Pre-computed pointing reference |
| `angle_sunFromMRFaxes_deg` | Sun angle from each body axis (+X, +Y, +Z) [deg] | Thermal constraint analysis |
| `a_dlambda_m` | Along-track relative separation [m] | Phase boundary determination |
| `maneuvers.manEphSec_rcs` | RCS manoeuvre epoch times [s, J2000] | Propulsion event overlay |
| `maneuvers.manEphSec_pps` | PPS manoeuvre epoch times [s, J2000] | Propulsion event overlay |

**Mission epoch alignment.** The J2000 reference epoch (2000-01-01T12:00:00 UTC) is used to convert absolute manoeuvre ephemeris seconds to mission elapsed days. The mission start epoch t₀ = 2028-09-01T03:49:53 UTC is established from the first timestamp of the RDV CSV, yielding t₀ = 904,664,993 s (J2000). Mission elapsed time for each data point is reconstructed as t[i] = i × Δt / 86,400 days, consistent with the simulation propagation timestep of 300 s confirmed in `scenario_struct.timestep_propagation`.

---

## 3. New Analysis Module — Thermal and Operational Sun-Exposure Constraints

**New file:** `thermal_constraints.py`

A dedicated thermal constraints analysis module has been introduced to assess sun-exposure risk for two sensitive instruments across both operational modes. The analysis characterises the **worst thermal case** by comparing Mode 1 and Mode 2 across the full rendezvous timeline.

### 3.1 Camera Sun Exclusion

The science camera is mounted along the +Z body axis. The sun exclusion half-cone angle is set at **30° (TBC)**, corresponding to a minimum safe angular separation between the camera boresight and the Sun direction. The metric is extracted directly from the simulation output field `angle_sunFromMRFaxes_deg[2]`, which gives the instantaneous angle between the +Z body axis and the solar direction vector expressed in the body frame.

A pointing violation is defined as any epoch at which this angle falls below the exclusion threshold. The analysis reports violation percentage and minimum observed angle for each mode and each mission phase.

### 3.2 Star Tracker (STR) Sun Blinding

The STR boresight is assumed to lie along the +X body axis (TBC — to be confirmed with the platform team). The STR sun exclusion half-cone is set at **35° (TBC)**, a value representative of state-of-the-art star trackers operating in the GEO thermal environment, where solar irradiance is approximately 1,361 W/m² and scattered light exclusion requirements are more stringent than in low Earth orbit. The metric is extracted from `angle_sunFromMRFaxes_deg[0]`.

### 3.3 Outputs

| Figure | Phase | Content |
|---|---|---|
| `thermal_far.png` | Far range (−60 to −5 km) | Camera sun angle and STR sun angle vs time, Mode 1 and Mode 2 overlaid, violation shading |
| `thermal_close.png` | Close range (−5 to +1 km) | Same, zoomed to close-range phase |

### 3.4 Summary Findings (preliminary, thresholds TBC)

| Constraint | Phase | Mode 1 violation | Mode 2 violation |
|---|---|---|---|
| Camera <30° | Far range | 17.4% | 16.5% |
| Camera <30° | Close range | 9.2% | 16.5% |
| STR <35° | Far range | 0.1% | 0.0% |
| STR <35° | Close range | 0.1% | 0.4% |

**Mode 2 represents the worse thermal case for the camera in the close-range phase.** In Mode 1, the camera boresight tracks the target — a direction that drifts away from solar proximity as the along-track separation closes. In Mode 2, the camera boresight is fixed to the nadir direction, which oscillates relative to the Sun at the GEO orbital period (≈ 23.9 h) without the geometric relief afforded by target proximity. STR violations are negligible in both modes.

---

## 4. S-Band Antenna Feasibility — Confirmed Beamwidth Parameter

**Affected files:** `mode3_feasibility.py`, `mode3_slew_analysis.py`, `earth_target_separation.py`

The S-band antenna 3 dB beamwidth has been confirmed as a **60° full cone (30° half-cone)**. All prior feasibility analyses used a variable threshold table; this has been replaced by a single, physically grounded threshold:

- **Feasibility criterion:** antenna pointing residual ≤ 30° (half-cone, 3 dB)

The antenna pointing residual in Mode 1 is the minimum of the +Y and −Y antenna pointing errors:

> ε_antenna = min(ε_{+Y}, ε_{−Y}) = | 90° − θ_{E-T} |

where θ_{E-T} is the instantaneous Earth–Target angular separation. This identity holds exactly when the camera boresight (+Z) is constrained to the target direction and the remaining rotational degree of freedom about +Z is used to optimise +Y toward the Earth direction.

**Mode 2 combined pointing is analytically infeasible.** In Mode 2, the camera boresight is constrained to the nadir direction. For the camera to simultaneously point at the rendezvous target, a rotation of magnitude equal to the full Earth–Target angular separation (θ_{E-T} ∈ [39.6°, 174.9°] across the mission timeline) would be required, violating the primary nadir-pointing constraint. Mode 2 achieves 0% feasibility against the 30° half-cone criterion, confirming Mode 1 as the exclusive candidate for combined Earth+Target pointing.

---

## 5. Phase-Separated Figure Architecture

**Affected files:** `earth_target_separation.py`, `mode3_feasibility.py`, `mode3_slew_analysis.py`

All output figures have been restructured to present the far-range and close-range rendezvous phases as independent figures, replacing the single timeline plots of Rev 1. The phase boundary is determined dynamically as the first epoch at which the along-track separation a·δλ exceeds −5 km, evaluated as **day 15.1** of the mission timeline for the present simulation dataset.

| Phase | Along-track range | Mission days | Data points |
|---|---|---|---|
| Far range | −60.8 to −5.0 km | 0 – 15.1 | 4,349 |
| Close range | −5.0 to +1.1 km | 15.1 – 23.0 | 2,272 |

### 5.1 Far-Range Figures

Present the orbital geometry and pointing performance metrics over the full approach corridor. At far range, the Earth–Target angular separation remains tightly bounded within the S-band feasibility band (mean 90.2°, standard deviation 7.3°), yielding near-complete antenna feasibility (≥ 98% within 20°, 100% within 30°) with low slew cost from Mode 1 (mean 95°).

### 5.2 Close-Range Figures with 90° Crossing Analysis

The close-range figures incorporate three additional analytical elements not present in Rev 1:

**(a) 90° Crossing Periodicity.** The Earth–Target angular separation oscillates at the GEO synodic period (≈ 23.7 h in this dataset). The separation crosses 90° — the attitude-independent condition for zero antenna pointing error — at each zero-crossing of (θ_{E-T} − 90°). Crossings are identified by interpolated sign change detection and overlaid as fiducial markers on the angular separation time history. In the close-range phase, 16 crossings occur over approximately 8 days (mean inter-crossing interval ≈ 11.8 h, consistent with two crossings per orbital period). The maximum inter-crossing gap is **14.8 h**, which exceeds the 6-hour ConOps window interval. This implies that not every 6-hour communication slot will contain a 90° crossing, and antenna pointing residuals at certain window epochs may be elevated relative to the GEO half-cone threshold.

**(b) Time-to-Feasibility per ConOps Window.** For each 6-hour ConOps window boundary, the time elapsed until the next 90° crossing is computed and rendered as a bar chart. Bars coloured green indicate windows in which a 90° crossing occurs within the 25-minute allocated slot; bars coloured red (with hatching for crossings outside the 6-hour window) indicate windows requiring either a later observation opportunity or acceptance of elevated antenna pointing loss.

**(c) Feasibility Shading.** The angular region [90° − 30°, 90° + 30°] = [60°, 120°] — corresponding to antenna pointing residuals within the S-band 3 dB half-cone — is shaded on all angular separation panels as a direct visual indicator of geometrically feasible communication epochs.

---

## 6. Propulsion Event Integration

**Affected files:** all scripts
**New data source:** `maneuvers` struct within each `.mat` data package

Manoeuvre events from both the Primary Propulsion System (PPS, main engine) and Reaction Control System (RCS) are extracted from the simulation data packages and superimposed on all time-series panels as vertical fiducials:

- **PPS firings** (orange dashed): station-keeping correction pulses applied at 6-hour intervals during the far-range RDV phase; 80 events over days 0.2–13.0 (total ΔV budget ≈ 5.03 h equivalent firing time).
- **RCS firings** (blue dotted): discrete attitude control and trajectory correction manoeuvres; 20 events during the RDV phase (days 16.2–19.0) and 24 events during the inspection phase, with no PPS activity in the inspection phase.

Manoeuvre epoch times are stored in the `.mat` files as absolute seconds from the J2000 epoch and are converted to mission elapsed days using the established t₀ reference.

---

## 7. Slew Analysis — Updated Statistics

**Affected file:** `mode3_slew_analysis.py`

The eigenaxis slew angle from each operational mode to the combined Earth+Target reference attitude is computed at each propagation timestep via:

> θ_slew = arccos( (tr(R_rel) − 1) / 2 )

where R_rel = C_mode · C_combined^T is the relative Direction Cosine Matrix between the operational mode attitude and the reference combined attitude. Both C_mode (from the simulated quaternion) and C_combined (constructed geometrically from ECI position vectors) are updated at each 300 s timestep.

Updated full-timeline statistics (combined RDV + inspection phases):

| Metric | Mode 1 → Combined | Mode 2 → Combined |
|---|---|---|
| Mean slew | 97° | 135° |
| Minimum slew | 0° | 23° |
| Maximum slew | 180° | 180° |
| Mode cheaper | 63% of timeline | 37% of timeline |

These results are consistent with Rev 1 (CSV-based) to within numerical precision, validating the `.mat` data ingestion pipeline.

---

## 8. Open Items

| Item | Status | Notes |
|---|---|---|
| STR boresight axis | TBC | Assumed +X body axis; confirmation required from platform team |
| Camera sun exclusion half-cone | TBC | Assumed 30°; instrument datasheet value required |
| STR sun exclusion half-cone | TBC | Assumed 35° (GEO typical); confirmation required |
| Mode 1 window selection for close range | Open | Identify optimal 6-h window slots during days 15–23 to minimise antenna pointing residual |
| Antenna gain pattern | Open | Replace 3 dB half-cone scalar threshold with full antenna gain vs off-boresight angle characterisation when available |

---

*Analysis performed using Python 3.12 with NumPy, SciPy (`scipy.io.loadmat`), and Matplotlib. Simulation data packages generated by the GNC MATLAB simulation tool (Endurance FR RDV scenario, end-to-end run 1, Target+Sun and Nadir+Sun attitude modes).*
