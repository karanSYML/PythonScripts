# ITU Scripts

Tools for generating, fitting, propagating, and visualising GEO satellite orbits, primarily for ITU filing and interference-validation workflows.

---

## Scripts

### `generate_tle.py`
Generates a synthetic Two-Line Element (TLE) set for the OG3 mission at the 13°E GEO slot. Orbital parameters are modelled after HOTBIRD 13F (NORAD 54048). Prints the formatted TLE pair with checksum verification. No external dependencies — uses the standard library only.

---

### `geo_pvt_generator.py`
Numerically propagates a GEO satellite and writes a PVT (Position-Velocity-Time) ephemeris CSV. Force models included:
- Earth gravity field via Holmes-Featherstone (configurable degree/order)
- Sun and Moon third-body perturbations
- Solar radiation pressure (isotropic spherical model)

Propagation runs in GCRF; output is transformed to ITRF and written as `utc_iso, x_m, y_m, z_m, vx_mps, vy_mps, vz_mps`. Requires `orekit-data.zip` in the working directory.

---

### `groundtrack_to_elements.py`
Inverts a GEO ground-track description into Keplerian orbital elements suitable for use with `geo_pvt_generator.py`. Two modes:

- **Mode 1 (analytical):** Supply the figure-8 shape parameters (longitude, inclination, eccentricity, node direction, perigee direction) and get back an `OrbitConfig` directly.
- **Mode 2 (least-squares fit):** Supply a list of `(lat, lon)` waypoints; scipy fits the five free parameters and returns an `OrbitConfig`.

---

### `run_from_groundtrack.py`
Example driver that chains `groundtrack_to_elements.py` and `geo_pvt_generator.py`. Demonstrates both Mode 1 (shape specification) and Mode 2 (waypoint fitting) and writes the resulting ephemeris CSVs.

---

### `tle_generator.py`
Tkinter GUI application for generating synthetic TLE scenarios used in ground-station interference validation. Features:
- Six orbit presets: LEO Polar Swarm, MEO/GNSS-like, GEO Comsat, ISS-like LEO, Walker Delta, Molniya HEO
- Configurable satellite count, RAAN spread, mean anomaly spread, epoch, BSTAR drag
- Role assignment per satellite: *interferer*, *victim*, or *third-party*
- Card view and raw TLE block view with one-click clipboard copy

---

### `visualize_geo_pvt.py`
Reads a PVT ephemeris CSV produced by `geo_pvt_generator.py` and generates two output PNGs:
- **Summary (2×2 panel):** ground track on a lat/lon grid, longitude vs time (east-west drift), latitude vs time (north-south excursion), and altitude vs time (radial deviation from nominal GEO).
- **3D ECEF view:** orbit plotted in Earth-fixed coordinates with an Earth wireframe.

Usage: `python visualize_geo_pvt.py <ephemeris.csv>`

---

## Required Python Packages

| Package | Used by |
|---------|---------|
| `numpy` | `groundtrack_to_elements.py`, `visualize_geo_pvt.py` |
| `scipy` | `groundtrack_to_elements.py` (Mode 2 least-squares fit) |
| `pandas` | `visualize_geo_pvt.py` |
| `matplotlib` | `visualize_geo_pvt.py` |
| `orekit` | `geo_pvt_generator.py`, `run_from_groundtrack.py` |

Standard-library modules used (`math`, `csv`, `tkinter`, `random`, `datetime`, `pathlib`) require no separate installation.

### Installing Python dependencies

```bash
pip install numpy scipy pandas matplotlib orekit
```

### Orekit data file

`geo_pvt_generator.py` and `run_from_groundtrack.py` require `orekit-data.zip` to be present in the working directory (already included in this repo). The Orekit Python wrapper also requires a compatible JVM — install via:

```bash
pip install orekit          # installs the Python wrapper + bundled JVM bindings
```
