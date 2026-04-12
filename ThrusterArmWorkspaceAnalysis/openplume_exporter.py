#!/usr/bin/env python3
"""
OpenPlume Case Exporter
========================
For every case returned by PlumePipeline.get_openplume_cases() (MARGINAL
and CAUTION), generates:

  1. A Gmsh .geo script with the complete 3D stack geometry:
       • Client bus          (OpenCASCADE Box)
       • Servicer bus        (OpenCASCADE Box, Z− docking position)
       • Solar panel ±X      (thin Box volumes, sun-tracking rotation)
       • Computational domain (Sphere, radius auto-sized to enclose panels)
       • Thruster source point + plume-direction arrow (visual aid)
       • Stack COG marker
       • Distance-field mesh refinement (coarse far-field → fine near thruster)

  2. A JSON sidecar with all OpenPlume boundary/source parameters:
       • thruster_pos_m        – thruster exit centre [x, y, z] m
       • thrust_direction      – unit vector from thruster toward COG
       • plume_direction       – exhaust direction (opposite of thrust)
       • plume_half_angle_deg  – beam divergence envelope
       • plume_cosine_exponent – n in cos^n(θ) flux model
       • stack_cog_m           – combined CG at mission start
       • Full thruster, geometry, and sweep parameters

  3. A manifest.json index of all exported cases

  4. A run_all.sh batch script that meshes every case with Gmsh

Gmsh usage
----------
    gmsh case_0000.geo -3 -o case_0000.msh   # mesh one case
    bash run_all.sh                           # mesh all cases

Boolean operations
------------------
The script defines all volumes independently.  For a conformal fluid-domain
mesh, run BooleanFragments in the Gmsh GUI or uncomment the line in the
generated script before meshing.

Coordinate system
-----------------
    X  velocity / along-track
    Y  orbit-normal / N-S
    Z  anti-earth  (+Z = nadir-up, −Z = earth-pointing)
    Client bus origin at its geometric centre.
    Servicer bus docked below client at Z− via the LAR.
"""

import os
import json
import math
import numpy as np
from typing import List, Dict, Optional

from plume_impingement_pipeline import (
    ThrusterParams, StackConfig, MaterialParams,
    CaseMatrixGenerator,
)


# ─── Constants ────────────────────────────────────────────────────────────────

PANEL_THICKNESS_M = 0.025    # model panels as thin solid boxes (25 mm)
DOMAIN_MARGIN_M   = 6.0      # extra radius beyond panel tip
DOMAIN_MIN_RADIUS = 20.0     # minimum domain sphere radius [m]


# ─── Gmsh script builder ──────────────────────────────────────────────────────

def _rotate_vol(tag: int, cx: float, cy: float, cz: float,
                angle_rad: float, axis: str = "x") -> str:
    """Return a Gmsh Rotate statement for a volume about a coordinate axis."""
    axes = {"x": "{1, 0, 0}", "y": "{0, 1, 0}", "z": "{0, 0, 1}"}
    ax_str = axes.get(axis.lower(), "{1, 0, 0}")
    return (
        "Rotate {" + ax_str +
        f", {{{cx:.6f}, {cy:.6f}, {cz:.6f}}}, "
        f"{angle_rad:.8f}" +
        "} { Volume{" + str(tag) + "}; }"
    )


def _build_gmsh_script(
    case_idx: int,
    result: Dict,
    stack: StackConfig,
    t_pos: np.ndarray,
    thrust_dir: np.ndarray,
    plume_dir: np.ndarray,
    cog: np.ndarray,
    serv_origin: np.ndarray,
    tracking_deg: float,
    thruster: ThrusterParams,
) -> str:
    """Build the complete Gmsh .geo script for one simulation case."""

    lines: List[str] = []

    def w(s: str = ""):
        lines.append(s)

    def sec(title: str):
        w(f"// {'─' * 64}")
        w(f"// {title}")
        w(f"// {'─' * 64}")

    # ── Derived geometry values ────────────────────────────────────────────
    cbx, cby, cbz = stack.client_bus_x, stack.client_bus_y, stack.client_bus_z
    sbx, sby, sbz = stack.servicer_bus_x, stack.servicer_bus_y, stack.servicer_bus_z
    span  = stack.panel_span_one_side
    pw    = stack.panel_width
    pt    = PANEL_THICKNESS_M
    track = math.radians(tracking_deg)

    # Domain sphere: large enough to enclose panels + margin
    r_domain = max(cbx / 2.0 + span + DOMAIN_MARGIN_M, DOMAIN_MIN_RADIUS)

    # Arrow tip for plume-direction visual (2 m along plume)
    arrow_tip = t_pos + 2.0 * plume_dir

    # ── File header ───────────────────────────────────────────────────────
    w("// " + "=" * 66)
    w(f"// OpenPlume Geometry — Case {case_idx:04d}")
    w(f"// Status   : {result['status']}  |  "
      f"Max erosion: {result['max_erosion_um']:.3f} µm  "
      f"({result['erosion_fraction'] * 100:.1f}% of Ag thickness)")
    w("//")
    w(f"// Thruster pos    : [{t_pos[0]:.5f}, {t_pos[1]:.5f}, {t_pos[2]:.5f}] m")
    w(f"// Thrust direction: [{thrust_dir[0]:.5f}, {thrust_dir[1]:.5f}, {thrust_dir[2]:.5f}]")
    w(f"// Plume direction : [{plume_dir[0]:.5f}, {plume_dir[1]:.5f}, {plume_dir[2]:.5f}]")
    w(f"// Stack COG       : [{cog[0]:.5f}, {cog[1]:.5f}, {cog[2]:.5f}] m")
    w(f"// Panel tracking  : {tracking_deg:.1f}°")
    w(f"// Servicer origin : [{serv_origin[0]:.5f}, {serv_origin[1]:.5f}, "
      f"{serv_origin[2]:.5f}] m  (Z-, LAR docked)")
    w("//")
    w("// Coordinate system: X=velocity, Y=orbit-normal, Z=anti-earth")
    w("// Client bus centred at origin. Servicer below (Z−) via LAR.")
    w("//")
    w(f"// Mesh with: gmsh case_{case_idx:04d}.geo -3 -o case_{case_idx:04d}.msh")
    w("// " + "=" * 66)
    w()

    w('SetFactory("OpenCASCADE");')
    w()

    # ── Characteristic mesh lengths ───────────────────────────────────────
    sec("Characteristic mesh lengths")
    w(f"lc_thruster  = 0.050;  // m — near thruster exit plane")
    w(f"lc_structure = 0.250;  // m — near spacecraft surfaces")
    w(f"lc_farfield  = {r_domain * 0.08:.3f};  // m — outer domain boundary")
    w()

    # ── Client bus ────────────────────────────────────────────────────────
    sec(f"Client bus  {cbx:.3f} × {cby:.3f} × {cbz:.3f} m  (centred at origin)")
    w(f"Box(1) = {{{-cbx/2:.5f}, {-cby/2:.5f}, {-cbz/2:.5f},  "
      f"{cbx:.5f}, {cby:.5f}, {cbz:.5f}}};")
    w()

    # ── Servicer bus ──────────────────────────────────────────────────────
    sec(f"Servicer bus  {sbx:.3f} × {sby:.3f} × {sbz:.3f} m  (Z− docking via LAR)")
    sx0 = serv_origin[0] - sbx / 2.0
    sy0 = serv_origin[1] - sby / 2.0
    sz0 = serv_origin[2] - sbz / 2.0
    w(f"Box(2) = {{{sx0:.5f}, {sy0:.5f}, {sz0:.5f},  "
      f"{sbx:.5f}, {sby:.5f}, {sbz:.5f}}};")
    w()

    # ── Solar panel +X ────────────────────────────────────────────────────
    sec(f"Solar panel +X  span={span:.2f} m, width={pw:.2f} m, "
        f"thickness={pt*1e3:.0f} mm, track={tracking_deg:.1f}°")
    px0 = cbx / 2.0                       # hinge x
    pz0 = cbz / 2.0 - pt / 2.0           # panel z-centre before rotation
    w(f"// Panel before tracking rotation (hinge at X={px0:.4f}, Z={cbz/2:.4f})")
    w(f"Box(3) = {{{px0:.5f}, {-pw/2:.5f}, {pz0:.5f},  "
      f"{span:.5f}, {pw:.5f}, {pt:.5f}}};")
    if abs(tracking_deg) > 0.01:
        w(_rotate_vol(3, px0, 0.0, cbz / 2.0, track, axis="x"))
    w()

    # ── Solar panel −X ────────────────────────────────────────────────────
    sec(f"Solar panel −X  (symmetric, same tracking rotation)")
    nx0 = -cbx / 2.0 - span              # panel −X corner x
    w(f"// Panel before tracking rotation (hinge at X={-cbx/2:.4f}, Z={cbz/2:.4f})")
    w(f"Box(4) = {{{nx0:.5f}, {-pw/2:.5f}, {pz0:.5f},  "
      f"{span:.5f}, {pw:.5f}, {pt:.5f}}};")
    if abs(tracking_deg) > 0.01:
        w(_rotate_vol(4, -cbx / 2.0, 0.0, cbz / 2.0, track, axis="x"))
    w()

    # ── Computational domain ──────────────────────────────────────────────
    sec(f"Computational domain — sphere  R = {r_domain:.2f} m")
    w(f"// Encloses full panel span ({cbx/2 + span:.1f} m) plus {DOMAIN_MARGIN_M:.0f} m margin")
    w(f"// To subtract spacecraft from domain (recommended before meshing):")
    w(f"// BooleanFragments{{ Volume{{1,2,3,4,100}}; Delete; }}{{}}  // conforming interfaces")
    w(f"Sphere(100) = {{0, 0, 0, {r_domain:.5f}}};")
    w()

    # ── Thruster source point + plume-direction arrow ─────────────────────
    sec("Thruster source geometry")
    w(f"// Thruster exit centre — OpenPlume plume source")
    w(f"Point(200) = {{{t_pos[0]:.6f}, {t_pos[1]:.6f}, {t_pos[2]:.6f}, lc_thruster}};")
    w()
    w(f"// Plume-direction arrow tip (2 m along plume, visual reference)")
    w(f"// Direction: [{plume_dir[0]:.5f}, {plume_dir[1]:.5f}, {plume_dir[2]:.5f}]")
    w(f"Point(201) = {{{arrow_tip[0]:.6f}, {arrow_tip[1]:.6f}, "
      f"{arrow_tip[2]:.6f}, lc_thruster}};")
    w(f"Line(200) = {{200, 201}};  // plume-direction arrow")
    w()

    # ── Stack COG marker ──────────────────────────────────────────────────
    w(f"// Stack centre-of-gravity at mission start")
    w(f"Point(202) = {{{cog[0]:.6f}, {cog[1]:.6f}, {cog[2]:.6f}, lc_structure}};")
    w()

    # ── Worst-case erosion panel point ────────────────────────────────────
    if "worst_point_distance_m" in result:
        w(f"// Worst-case erosion panel point (off-axis {result['worst_point_offaxis_deg']:.1f}°, "
          f"d={result['worst_point_distance_m']:.2f} m)")
        w(f"//   Max erosion = {result['max_erosion_um']:.3f} µm at this panel point")
    w()

    # ── Physical groups ───────────────────────────────────────────────────
    sec("Physical groups")
    w("// Solid spacecraft structures — assign Wall BC in OpenPlume")
    w('Physical Volume("client_bus",     1) = {1};')
    w('Physical Volume("servicer_bus",   2) = {2};')
    w('Physical Volume("solar_panel_+X", 3) = {3};')
    w('Physical Volume("solar_panel_-X", 4) = {4};')
    w()
    w("// Fluid / simulation domain (modify after BooleanFragments if used)")
    w('Physical Volume("fluid_domain", 100) = {100};')
    w()
    w("// Source + reference")
    w('Physical Point("thruster_source",  200) = {200};')
    w('Physical Point("cog_marker",       202) = {202};')
    w('Physical Line("plume_direction",   200) = {200};')
    w()

    # ── Mesh refinement fields ────────────────────────────────────────────
    sec("Mesh refinement — distance-based around thruster source")
    w("// Field 1: distance from thruster exit point")
    w("Field[1] = Distance;")
    w("Field[1].PointsList = {200};")
    w()
    w("// Field 2: threshold — fine mesh near thruster, coarse at domain edge")
    w("Field[2] = Threshold;")
    w("Field[2].InField  = 1;")
    w(f"Field[2].DistMin  = 1.000;          // m  inner radius (fine)")
    w(f"Field[2].DistMax  = {r_domain * 0.6:.3f};    // m  outer radius (coarse)")
    w(f"Field[2].SizeMin  = lc_thruster;")
    w(f"Field[2].SizeMax  = lc_farfield;")
    w()
    w("// Field 3: distance from spacecraft volumes — surface refinement")
    w("Field[3] = Distance;")
    w("Field[3].VolumesList = {1, 2, 3, 4};")
    w()
    w("Field[4] = Threshold;")
    w("Field[4].InField  = 3;")
    w(f"Field[4].DistMin  = 0.000;")
    w(f"Field[4].DistMax  = 3.000;")
    w(f"Field[4].SizeMin  = lc_structure;")
    w(f"Field[4].SizeMax  = lc_farfield;")
    w()
    w("// Field 5: take minimum (finest mesh wins)")
    w("Field[5] = Min;")
    w("Field[5].FieldsList = {2, 4};")
    w()
    w("Background Field = 5;")
    w()

    # ── Mesh settings ─────────────────────────────────────────────────────
    sec("Mesh settings")
    w("Mesh.CharacteristicLengthMin = lc_thruster;")
    w("Mesh.CharacteristicLengthMax = lc_farfield;")
    w("Mesh.Algorithm3D             = 4;  // Frontal-Delaunay 3D")
    w("Mesh.Optimize                = 1;")
    w()

    return "\n".join(lines)


# ─── JSON sidecar builder ─────────────────────────────────────────────────────

def _build_sidecar(
    case_idx: int,
    result: Dict,
    stack: StackConfig,
    t_pos: np.ndarray,
    thrust_dir: np.ndarray,
    plume_dir: np.ndarray,
    cog: np.ndarray,
    serv_origin: np.ndarray,
    tracking_deg: float,
    thruster: ThrusterParams,
    material: MaterialParams,
    geo_filename: str,
) -> Dict:
    """Build the JSON sidecar dict for one case."""
    arm, _, _ = CaseMatrixGenerator.case_to_objects(result)

    if hasattr(arm, "link1_length"):   # RoboticArmGeometry
        arm_info = {
            "type": "RoboticArm_3DOF_yaw_pitch_pitch",
            "link1_length_m": round(arm.link1_length, 5),
            "link2_length_m": round(arm.link2_length, 5),
            "reach_m": round(arm.link1_length + arm.link2_length, 5),
            "shoulder_yaw_deg": arm.shoulder_yaw_deg,
            "elbow_up": arm.elbow_up,
        }
    else:
        arm_info = {
            "type": "SingleLink_legacy",
            "arm_length_m": getattr(arm, "arm_length", None),
            "azimuth_deg":  getattr(arm, "azimuth_deg", None),
            "elevation_deg": getattr(arm, "elevation_deg", None),
        }

    # Sweep params only (strip computed result fields)
    _computed = {
        "max_erosion_um", "mean_erosion_um", "worst_point_distance_m",
        "worst_point_offaxis_deg", "worst_point_incidence_deg",
        "min_panel_distance_m", "thruster_pos_x", "thruster_pos_y",
        "thruster_pos_z", "cog_x", "cog_y", "cog_z",
        "status", "erosion_fraction", "ik_feasible", "arm_collision",
    }

    return {
        "case_id":         case_idx,
        "case_name":       f"case_{case_idx:04d}",
        "status":          result["status"],
        "erosion": {
            "max_um":              round(result["max_erosion_um"], 5),
            "mean_um":             round(result.get("mean_erosion_um", 0.0), 5),
            "fraction_of_thickness": round(result["erosion_fraction"], 6),
            "material_thickness_um": material.thickness_um,
            "worst_panel_distance_m":  round(result.get("worst_point_distance_m", 0.0), 4),
            "worst_offaxis_deg":       round(result.get("worst_point_offaxis_deg", 0.0), 3),
            "worst_incidence_deg":     round(result.get("worst_point_incidence_deg", 0.0), 3),
            "min_panel_distance_m":    round(result.get("min_panel_distance_m", 0.0), 4),
        },
        "openplume_source": {
            "thruster_pos_m":         [round(v, 6) for v in t_pos.tolist()],
            "thrust_direction":       [round(v, 6) for v in thrust_dir.tolist()],
            "plume_direction":        [round(v, 6) for v in plume_dir.tolist()],
            "beam_half_angle_deg":    thruster.beam_divergence_half_angle,
            "plume_cosine_exponent":  thruster.plume_cosine_exponent,
            "isp_s":                  thruster.isp,
            "discharge_voltage_V":    thruster.discharge_voltage,
            "mass_flow_rate_kg_s":    thruster.mass_flow_rate,
            "propellant":             thruster.propellant,
            "thrust_N":               thruster.thrust_N,
        },
        "stack": {
            "cog_m":              [round(v, 6) for v in cog.tolist()],
            "servicer_origin_m":  [round(v, 6) for v in serv_origin.tolist()],
            "client_bus_m":       [stack.client_bus_x, stack.client_bus_y, stack.client_bus_z],
            "servicer_bus_m":     [stack.servicer_bus_x, stack.servicer_bus_y, stack.servicer_bus_z],
            "panel_span_one_side_m": stack.panel_span_one_side,
            "panel_width_m":      stack.panel_width,
            "panel_tracking_deg": tracking_deg,
            "lar_offset_z_m":     stack.lar_offset_z,
            "client_mass_kg":     stack.client_mass,
            "servicer_mass_kg":   stack.servicer_mass,
        },
        "arm":          arm_info,
        "sweep_params": {k: v for k, v in result.items() if k not in _computed},
        "gmsh_script":  geo_filename,
    }


# ─── Main export function ─────────────────────────────────────────────────────

def export_openplume_cases(
    openplume_cases: List[Dict],
    output_dir: str = "openplume_cases",
    thruster: Optional[ThrusterParams]  = None,
    material: Optional[MaterialParams]  = None,
) -> List[Dict]:
    """Export every MARGINAL/CAUTION case to a Gmsh .geo script + JSON sidecar.

    Parameters
    ----------
    openplume_cases : list
        Cases from ``PlumePipeline.get_openplume_cases()``.
    output_dir : str
        Directory for all output files (created if absent).
    thruster : ThrusterParams, optional
        Thruster parameters for JSON metadata (defaults used if omitted).
    material : MaterialParams, optional
        Target material parameters (defaults used if omitted).

    Returns
    -------
    List of manifest records, one per exported case.
    """
    thruster = thruster or ThrusterParams()
    material = material or MaterialParams()

    os.makedirs(output_dir, exist_ok=True)
    manifest: List[Dict] = []

    n = len(openplume_cases)
    print(f"Exporting {n} OpenPlume case(s) → {output_dir}/")

    for idx, result in enumerate(openplume_cases):
        # ── Reconstruct StackConfig from case sweep parameters ─────────────
        arm, stack, ops = CaseMatrixGenerator.case_to_objects(result)
        serv_origin  = stack.servicer_origin_in_lar_frame()
        tracking_deg = float(result.get("panel_tracking_deg", 0.0))

        # ── Thruster position and COG from stored pipeline results ─────────
        t_pos = np.array([result["thruster_pos_x"],
                          result["thruster_pos_y"],
                          result["thruster_pos_z"]])
        cog   = np.array([result["cog_x"],
                          result["cog_y"],
                          result["cog_z"]])

        # ── Thrust and plume directions ────────────────────────────────────
        to_cog = cog - t_pos
        norm   = np.linalg.norm(to_cog)
        thrust_dir = to_cog / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        plume_dir  = -thrust_dir

        case_name = f"case_{idx:04d}"
        geo_path  = os.path.join(output_dir, f"{case_name}.geo")
        json_path = os.path.join(output_dir, f"{case_name}.json")

        # ── Write Gmsh .geo script ─────────────────────────────────────────
        script = _build_gmsh_script(
            case_idx=idx, result=result, stack=stack,
            t_pos=t_pos, thrust_dir=thrust_dir, plume_dir=plume_dir,
            cog=cog, serv_origin=serv_origin,
            tracking_deg=tracking_deg, thruster=thruster,
        )
        with open(geo_path, "w") as f:
            f.write(script)

        # ── Write JSON sidecar ─────────────────────────────────────────────
        sidecar = _build_sidecar(
            case_idx=idx, result=result, stack=stack,
            t_pos=t_pos, thrust_dir=thrust_dir, plume_dir=plume_dir,
            cog=cog, serv_origin=serv_origin,
            tracking_deg=tracking_deg, thruster=thruster, material=material,
            geo_filename=os.path.basename(geo_path),
        )
        with open(json_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        # ── Append to manifest ────────────────────────────────────────────
        manifest_entry = {
            "case_id":           idx,
            "case_name":         case_name,
            "status":            result["status"],
            "max_erosion_um":    round(result["max_erosion_um"], 4),
            "erosion_fraction":  round(result["erosion_fraction"], 5),
            "thruster_pos_m":    [round(v, 5) for v in t_pos.tolist()],
            "thrust_direction":  [round(v, 5) for v in thrust_dir.tolist()],
            "plume_direction":   [round(v, 5) for v in plume_dir.tolist()],
            "cog_m":             [round(v, 5) for v in cog.tolist()],
            "geo_file":          os.path.basename(geo_path),
            "json_file":         os.path.basename(json_path),
        }
        manifest.append(manifest_entry)

        print(f"  [{idx+1:3d}/{n}] {case_name}  "
              f"status={result['status']:<8s}  "
              f"erosion={result['max_erosion_um']:.2f} µm  "
              f"thruster=[{t_pos[0]:+.2f}, {t_pos[1]:+.2f}, {t_pos[2]:+.2f}]")

    # ── manifest.json ─────────────────────────────────────────────────────
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"n_cases": len(manifest), "cases": manifest}, f, indent=2)
    print(f"\n  manifest.json  → {manifest_path}")

    # ── run_all.sh ────────────────────────────────────────────────────────
    batch_path = os.path.join(output_dir, "run_all.sh")
    with open(batch_path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Mesh all OpenPlume cases with Gmsh\n")
        f.write("# Usage: bash run_all.sh\n\n")
        f.write("set -e\n")
        f.write(f"GMSH=$(command -v gmsh || echo gmsh)\n\n")
        for entry in manifest:
            geo = entry["geo_file"]
            msh = geo.replace(".geo", ".msh")
            f.write(f'echo "Meshing {geo}..."\n')
            f.write(f'"${{GMSH}}" {geo} -3 -o {msh} -format msh2\n\n')
        f.write('echo "Done. Generated *.msh files:"\n')
        f.write('ls -lh *.msh 2>/dev/null || echo "(none found)"\n')
    os.chmod(batch_path, 0o755)
    print(f"  run_all.sh     → {batch_path}")

    return manifest


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_demo():
    from plume_impingement_pipeline import PlumePipeline

    print("=" * 70)
    print("  OpenPlume Case Exporter — Demo")
    print("=" * 70)

    thruster = ThrusterParams()
    material = MaterialParams()

    pipeline = PlumePipeline(thruster, material)
    gen      = pipeline.generator

    # ── Small parametric sweep to produce MARGINAL/CAUTION cases ──────────
    gen.set_param_range("arm_reach_m",         np.arange(1.5, 5.5, 0.5))
    gen.set_param_range("shoulder_yaw_deg",    np.array([-30, 0, 30, 60, 90]))
    gen.set_param_range("link_ratio",          np.array([0.5]))
    gen.set_param_range("panel_span_one_side", np.array([10, 12, 15]))
    gen.set_param_range("firing_duration_s",   np.array([600, 1200]))
    gen.set_param_range("mission_duration_yr", np.array([5, 7]))
    gen.set_param_range("client_mass",         np.array([3000]))
    gen.set_param_range("servicer_mass",       np.array([400]))
    gen.set_param_range("panel_tracking_deg",  np.array([0.0]))

    fixed = {
        "link_ratio": 0.5, "client_mass": 3000.0, "servicer_mass": 400.0,
        "panel_tracking_deg": 0.0,
    }
    cases = gen.generate_reduced_matrix(
        fixed, ["arm_reach_m", "shoulder_yaw_deg",
                "panel_span_one_side", "firing_duration_s", "mission_duration_yr"])

    print(f"\nSweep: {len(cases)} cases...")
    results  = pipeline.run_sweep(cases, verbose=False)
    summary  = pipeline.summary()
    print(f"  SAFE={summary['SAFE']}  CAUTION={summary['CAUTION']}  "
          f"MARGINAL={summary['MARGINAL']}  FAIL={summary['FAIL']}")

    op_cases = pipeline.get_openplume_cases()
    if not op_cases:
        print("\n  No MARGINAL/CAUTION cases.  Increase mission duration or reduce arm reach.")
        return

    print(f"  Cases for OpenPlume simulation: {len(op_cases)}")

    out_dir  = "openplume_cases"
    manifest = export_openplume_cases(
        op_cases, output_dir=out_dir, thruster=thruster, material=material,
    )

    # ── Print a sample of the first case's JSON ────────────────────────────
    print(f"\n─── Sample JSON sidecar (case_0000) ─────────────────────────────")
    first_json = os.path.join(out_dir, "case_0000.json")
    with open(first_json) as f:
        sample = json.load(f)
    src = sample["openplume_source"]
    print(f"  thruster_pos_m    : {src['thruster_pos_m']}")
    print(f"  thrust_direction  : {src['thrust_direction']}")
    print(f"  plume_direction   : {src['plume_direction']}")
    print(f"  beam_half_angle   : {src['beam_half_angle_deg']}°")
    stk = sample["stack"]
    print(f"  cog_m             : {stk['cog_m']}")
    print(f"  servicer_origin_m : {stk['servicer_origin_m']}")

    # ── Print a snippet of the first .geo file ─────────────────────────────
    print(f"\n─── Gmsh script snippet (case_0000.geo, first 30 lines) ─────────")
    with open(os.path.join(out_dir, "case_0000.geo")) as f:
        for i, line in enumerate(f):
            if i >= 30:
                break
            print("  " + line, end="")


if __name__ == "__main__":
    import numpy as np
    run_demo()
