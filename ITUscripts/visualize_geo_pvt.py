"""
Visualize the GEO PVT ephemeris CSV produced by geo_pvt_generator.py.

Generates a 2x2 figure:
    - Ground track on a world map (cartopy if available, else plain lon/lat)
    - Longitude vs time      (east-west drift / libration)
    - Latitude  vs time      (north-south excursion from inclination)
    - Altitude  vs time      (radial deviation from GEO altitude)

Plus an optional 3D ECEF view of the orbit.

Usage:
    python visualize_geo_pvt.py io_sat_geo_pvt_itrf.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# WGS84
A_EARTH = 6_378_137.0
F_EARTH = 1.0 / 298.257223563
E2_EARTH = F_EARTH * (2.0 - F_EARTH)
GEO_ALT_REF = 35_786_000.0  # nominal GEO altitude above equator, m


def ecef_to_geodetic(x, y, z):
    """Vectorized ECEF -> geodetic (lat, lon, alt) on WGS84. Bowring's method."""
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    # Initial guess
    lat = np.arctan2(z, p * (1.0 - E2_EARTH))
    for _ in range(5):  # converges in ~3 iterations
        sin_lat = np.sin(lat)
        N = A_EARTH / np.sqrt(1.0 - E2_EARTH * sin_lat * sin_lat)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - E2_EARTH * N / (N + alt)))
    sin_lat = np.sin(lat)
    N = A_EARTH / np.sqrt(1.0 - E2_EARTH * sin_lat * sin_lat)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["utc_iso"])
    lat, lon, alt = ecef_to_geodetic(
        df["x_m"].to_numpy(),
        df["y_m"].to_numpy(),
        df["z_m"].to_numpy(),
    )
    df["lat_deg"] = lat
    df["lon_deg"] = lon
    df["alt_m"] = alt
    df["radius_m"] = np.linalg.norm(
        df[["x_m", "y_m", "z_m"]].to_numpy(), axis=1
    )
    df["speed_mps"] = np.linalg.norm(
        df[["vx_mps", "vy_mps", "vz_mps"]].to_numpy(), axis=1
    )
    return df


def plot_summary(df: pd.DataFrame, out_png: Path):
    fig = plt.figure(figsize=(14, 9))

    # --- Ground track ----------------------------------------------------
    ax_gt = fig.add_subplot(2, 2, 1)
    ax_gt.plot(df["lon_deg"], df["lat_deg"], ".", ms=2, color="#0066cc")
    ax_gt.plot(df["lon_deg"].iloc[0], df["lat_deg"].iloc[0],
               "o", color="green", ms=8, label="start")
    ax_gt.plot(df["lon_deg"].iloc[-1], df["lat_deg"].iloc[-1],
               "s", color="red", ms=8, label="end")
    ax_gt.set_xlim(-180, 180)
    ax_gt.set_ylim(-90, 90)
    ax_gt.set_xlabel("Longitude [deg]")
    ax_gt.set_ylabel("Latitude [deg]")
    ax_gt.set_title("Ground track (ITRF)")
    ax_gt.grid(True, alpha=0.3)
    ax_gt.legend(loc="lower left", fontsize=8)
    # Reference equator + prime meridian
    ax_gt.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_gt.axvline(0, color="k", lw=0.5, alpha=0.4)

    # --- Longitude vs time ----------------------------------------------
    ax_lon = fig.add_subplot(2, 2, 2)
    # Unwrap to expose secular drift cleanly across ±180° wrap
    lon_unwrapped = np.degrees(np.unwrap(np.radians(df["lon_deg"].to_numpy())))
    ax_lon.plot(df["utc_iso"], lon_unwrapped, color="#0066cc", lw=1)
    ax_lon.set_xlabel("UTC")
    ax_lon.set_ylabel("Longitude (unwrapped) [deg]")
    ax_lon.set_title("East-west drift")
    ax_lon.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    # --- Latitude vs time -----------------------------------------------
    ax_lat = fig.add_subplot(2, 2, 3)
    ax_lat.plot(df["utc_iso"], df["lat_deg"], color="#cc6600", lw=1)
    ax_lat.set_xlabel("UTC")
    ax_lat.set_ylabel("Latitude [deg]")
    ax_lat.set_title("North-south excursion (inclination signature)")
    ax_lat.grid(True, alpha=0.3)

    # --- Altitude vs time -----------------------------------------------
    ax_alt = fig.add_subplot(2, 2, 4)
    alt_km = df["alt_m"].to_numpy() / 1e3
    ax_alt.plot(df["utc_iso"], alt_km, color="#006633", lw=1)
    ax_alt.axhline(GEO_ALT_REF / 1e3, color="k", ls="--", lw=0.7,
                   label="nominal GEO (35 786 km)")
    ax_alt.set_xlabel("UTC")
    ax_alt.set_ylabel("Altitude above WGS84 [km]")
    ax_alt.set_title("Radial deviation")
    ax_alt.grid(True, alpha=0.3)
    ax_alt.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"GEO PVT — {df['utc_iso'].iloc[0]} → {df['utc_iso'].iloc[-1]}  "
        f"({len(df)} points)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


def plot_3d(df: pd.DataFrame, out_png: Path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth wireframe
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 36),
                       np.linspace(0, np.pi, 18))
    ex = A_EARTH * np.cos(u) * np.sin(v)
    ey = A_EARTH * np.sin(u) * np.sin(v)
    ez = A_EARTH * np.cos(v)
    ax.plot_wireframe(ex, ey, ez, color="lightblue", lw=0.4, alpha=0.6)

    ax.plot(df["x_m"], df["y_m"], df["z_m"], color="#cc0033", lw=1.2)
    ax.scatter(df["x_m"].iloc[0], df["y_m"].iloc[0], df["z_m"].iloc[0],
               color="green", s=40, label="start")
    ax.scatter(df["x_m"].iloc[-1], df["y_m"].iloc[-1], df["z_m"].iloc[-1],
               color="red", s=40, label="end")

    # Equal aspect
    r_max = float(df["radius_m"].max()) * 1.05
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)
    ax.set_xlabel("X_ITRF [m]")
    ax.set_ylabel("Y_ITRF [m]")
    ax.set_zlabel("Z_ITRF [m]")
    ax.set_title("Orbit in ITRF (Earth-fixed)")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


def print_diagnostics(df: pd.DataFrame):
    print("\n--- Quick diagnostics ---")
    print(f"  duration         : {df['utc_iso'].iloc[-1] - df['utc_iso'].iloc[0]}")
    print(f"  samples          : {len(df)}")
    print(f"  altitude  min/max: {df['alt_m'].min()/1e3:10.3f} / {df['alt_m'].max()/1e3:10.3f}  km")
    print(f"  radius    min/max: {df['radius_m'].min()/1e3:10.3f} / {df['radius_m'].max()/1e3:10.3f}  km")
    print(f"  longitude min/max: {df['lon_deg'].min():10.4f} / {df['lon_deg'].max():10.4f}  deg")
    print(f"  latitude  min/max: {df['lat_deg'].min():10.4f} / {df['lat_deg'].max():10.4f}  deg")
    print(f"  speed     min/max: {df['speed_mps'].min():10.3f} / {df['speed_mps'].max():10.3f}  m/s")
    # Sanity expectations for a clean GEO:
    #   altitude ~ 35786 km ± ~few km
    #   speed (in ITRF) much smaller than ~3.07 km/s inertial circular speed,
    #   because Earth rotation is co-moving — for a pure geostationary point,
    #   ITRF speed -> 0. Non-zero ITRF speed indicates eccentricity + inclination.


def main():
    if len(sys.argv) < 2:
        csv_path = Path("io_sat_geo_pvt_itrf.csv")
    else:
        csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    df = load_csv(csv_path)
    print_diagnostics(df)

    base = csv_path.with_suffix("")
    plot_summary(df, base.with_name(base.name + "_summary.png"))
    plot_3d(df, base.with_name(base.name + "_3d.png"))

    plt.show()


if __name__ == "__main__":
    main()
