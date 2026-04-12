"""
urdf_generator.py — Parametric URDF generator for the Thruster Arm / Stack assembly.

Coordinate frame: LAR interface origin, +Z = nadir, −Z = anti-earth, +X = North, +Y = East.

Kinematic tree
──────────────
world (inertial)
└── LAR_interface  (fixed, origin)
    ├── client_bus             (fixed joint)
    │   ├── client_panel_north (fixed)
    │   ├── client_panel_south (fixed)
    │   ├── antenna_E1         (fixed)
    │   ├── antenna_E2         (fixed)
    │   ├── antenna_W1         (fixed)
    │   └── antenna_W2         (fixed)
    └── servicer_bus           (fixed joint, docked via LAR)
        └── thruster_arm_base  (fixed, at servicer +Z face)
            └── arm_link_1     (revolute J1 – shoulder yaw, axis Z)
                └── arm_link_2 (revolute J2 – elbow pitch, axis Y)
                    └── thruster_bracket (revolute J3 – wrist pitch, axis Y)
                        └── thruster_frame (fixed, thrust ⊥ bracket long axis)

Usage
─────
    from urdf_generator import URDFGenerator
    from plume_impingement_pipeline import StackConfig, RoboticArmGeometry

    gen = URDFGenerator(StackConfig(), RoboticArmGeometry())
    urdf_xml = gen.generate()          # returns XML string
    gen.save("output/")                # writes thruster_arm.urdf + meshes/*.stl
"""

import math
import os
import struct
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Try to import pipeline dataclasses; provide stubs if not on path
# ---------------------------------------------------------------------------
try:
    from plume_impingement_pipeline import StackConfig, RoboticArmGeometry
except ImportError:  # pragma: no cover
    StackConfig = None          # type: ignore
    RoboticArmGeometry = None   # type: ignore


# ---------------------------------------------------------------------------
# 1.  INERTIA HELPERS
# ---------------------------------------------------------------------------

def _inertia_box(mass: float, bx: float, by: float, bz: float
                 ) -> Tuple[float, float, float]:
    """Principal inertia moments (Ixx, Iyy, Izz) for a solid box."""
    return (
        mass / 12.0 * (by**2 + bz**2),
        mass / 12.0 * (bx**2 + bz**2),
        mass / 12.0 * (bx**2 + by**2),
    )


def _inertia_cylinder_x(mass: float, radius: float, length: float
                        ) -> Tuple[float, float, float]:
    """Inertia for a solid cylinder whose long axis is X.

    Ixx = ½ m r²  (spin axis)
    Iyy = Izz = m(3r² + L²)/12
    """
    ixx = 0.5 * mass * radius**2
    iyy = izz = mass * (3.0 * radius**2 + length**2) / 12.0
    return ixx, iyy, izz


def _inertia_thin_plate_xy(mass: float, lx: float, ly: float
                           ) -> Tuple[float, float, float]:
    """Inertia for a thin plate in the XY plane (normal = Z).

    Ixx = m ly²/12,  Iyy = m lx²/12,  Izz = m(lx²+ly²)/12
    """
    ixx = mass * ly**2 / 12.0
    iyy = mass * lx**2 / 12.0
    izz = mass * (lx**2 + ly**2) / 12.0
    return ixx, iyy, izz


def _inertia_parabolic_dish(mass: float, diameter: float
                            ) -> Tuple[float, float, float]:
    """Approximate inertia for a parabolic dish as a thin disc (aperture in XY plane).

    Ixx = Iyy = m R²/4,  Izz = m R²/2
    """
    R = diameter / 2.0
    ixx = iyy = mass * R**2 / 4.0
    izz = mass * R**2 / 2.0
    return ixx, iyy, izz


# ---------------------------------------------------------------------------
# 2.  PARABOLIC DISH MESH (binary STL)
# ---------------------------------------------------------------------------

def generate_parabolic_dish_stl(diameter: float, f_d_ratio: float,
                                 resolution: int = 48) -> bytes:
    """Generate a parabolic dish as binary STL bytes.

    The dish opens in the +Z direction (aperture faces +Z / nadir).
    The vertex (deepest point) is at Z = 0; the rim is at Z = depth.

    Parameters
    ──────────
    diameter   : dish diameter [m]
    f_d_ratio  : focal-length / diameter ratio
    resolution : number of radial rings and azimuthal segments

    Returns
    ───────
    Binary STL bytes (80-byte header + triangles).
    """
    R     = diameter / 2.0
    focal = f_d_ratio * diameter            # focal length  f = (f/D) * D
    # depth = D² / (16 f) = diameter / (16 * f/D)
    depth = diameter**2 / (16.0 * focal)   # depth of parabola (vertex to rim) [m]  # noqa

    n_r  = resolution           # radial steps
    n_az = resolution * 2       # azimuthal steps

    r_vals  = np.linspace(0.0, R, n_r + 1)
    az_vals = np.linspace(0.0, 2.0 * math.pi, n_az, endpoint=False)

    def _point(r: float, az: float) -> np.ndarray:
        x = r * math.cos(az)
        y = r * math.sin(az)
        z = r**2 / (4.0 * focal)
        return np.array([x, y, z], dtype=np.float32)

    triangles: list = []

    for i in range(n_r):
        r0, r1 = float(r_vals[i]), float(r_vals[i + 1])
        for j in range(n_az):
            az0 = float(az_vals[j])
            az1 = float(az_vals[(j + 1) % n_az])

            p00 = _point(r0, az0)
            p01 = _point(r0, az1)
            p10 = _point(r1, az0)
            p11 = _point(r1, az1)

            if i == 0:
                # Fan triangle from centre ring
                _add_tri(triangles, p00, p10, p11)
            else:
                _add_tri(triangles, p00, p10, p11)
                _add_tri(triangles, p00, p11, p01)

    return _triangles_to_stl(triangles)


def _add_tri(tris: list,
             v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> None:
    """Compute outward normal and append triangle."""
    e1 = v1 - v0
    e2 = v2 - v0
    n  = np.cross(e1, e2).astype(np.float32)
    nlen = float(np.linalg.norm(n))
    if nlen > 1e-12:
        n /= nlen
    tris.append((n, v0.astype(np.float32),
                     v1.astype(np.float32),
                     v2.astype(np.float32)))


def _triangles_to_stl(tris: list) -> bytes:
    """Pack list of (normal, v0, v1, v2) tuples into binary STL bytes."""
    header = b'\0' * 80
    buf = bytearray(header)
    buf += struct.pack('<I', len(tris))
    for (n, v0, v1, v2) in tris:
        buf += struct.pack('<fff', *n)
        buf += struct.pack('<fff', *v0)
        buf += struct.pack('<fff', *v1)
        buf += struct.pack('<fff', *v2)
        buf += struct.pack('<H', 0)   # attribute byte count
    return bytes(buf)


# ---------------------------------------------------------------------------
# 3.  URDF GENERATOR
# ---------------------------------------------------------------------------

class URDFGenerator:
    """Builds a parametric URDF string for the full Thruster Arm / Stack assembly.

    Parameters
    ──────────
    stack : StackConfig
        Combined servicer + client geometry and mass parameters.
    arm   : RoboticArmGeometry
        Arm link/joint parameters.
    mesh_dir : str, optional
        Directory where STL mesh files will be written and referenced in the
        URDF.  If None, antenna dishes use a cylinder placeholder geometry.
    robot_name : str
        Value of the <robot name="..."> attribute.
    """

    # Arm link tube radius [m] — used for visual/collision cylinders
    LINK_RADIUS   = 0.05
    # Panel thickness [m]
    PANEL_THICK   = 0.02
    # Dish mesh resolution
    DISH_RES      = 48
    # Half-pi constant
    _HALF_PI      = math.pi / 2.0

    def __init__(self, stack, arm,
                 mesh_dir: Optional[str] = None,
                 robot_name: str = "thruster_arm_stack"):
        self.stack      = stack
        self.arm        = arm
        self.mesh_dir   = mesh_dir
        self.robot_name = robot_name
        self._mesh_map: Dict[str, str] = {}   # antenna_name → relative STL path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> str:
        """Return the complete URDF XML string."""
        if self.mesh_dir is not None:
            self._write_mesh_files()

        parts = [self._xml_header()]
        parts += self._links()
        parts += self._joints()
        parts.append("</robot>")
        return "\n".join(parts)

    def save(self, output_dir: str, filename: str = "thruster_arm.urdf") -> str:
        """Write URDF and mesh files to *output_dir*.

        Parameters
        ──────────
        output_dir : path to output directory (created if absent)
        filename   : URDF filename

        Returns
        ───────
        Absolute path of the written URDF file.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.mesh_dir = os.path.join(output_dir, "meshes")
        os.makedirs(self.mesh_dir, exist_ok=True)

        urdf_str  = self.generate()
        urdf_path = os.path.join(output_dir, filename)
        with open(urdf_path, "w", encoding="utf-8") as fh:
            fh.write(urdf_str)

        print(f"URDF written  : {urdf_path}")
        for ant, path in self._mesh_map.items():
            print(f"  mesh {ant:4s} : {path}")
        return urdf_path

    # ------------------------------------------------------------------
    # Internal helpers — formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _f(*vals) -> str:
        """Format floats for XML attributes."""
        return " ".join(f"{v:.8g}" for v in vals)

    def _origin(self, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> str:
        return (f'<origin xyz="{self._f(*xyz)}" rpy="{self._f(*rpy)}"/>')

    def _inertia_tag(self, ixx: float, iyy: float, izz: float,
                     ixy: float = 0.0, ixz: float = 0.0, iyz: float = 0.0) -> str:
        return (f'<inertia ixx="{ixx:.6e}" ixy="{ixy:.6e}" ixz="{ixz:.6e}" '
                f'iyy="{iyy:.6e}" iyz="{iyz:.6e}" izz="{izz:.6e}"/>')

    def _inertial_block(self, mass: float,
                        ixx: float, iyy: float, izz: float,
                        cog_xyz=(0.0, 0.0, 0.0)) -> str:
        lines = [
            "    <inertial>",
            f"      {self._origin(cog_xyz)}",
            f"      <mass value=\"{mass:.6f}\"/>",
            f"      {self._inertia_tag(ixx, iyy, izz)}",
            "    </inertial>",
        ]
        return "\n".join(lines)

    @staticmethod
    def _rgba(r: float, g: float, b: float, a: float = 1.0) -> str:
        return f'<material name=""><color rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}"/></material>'

    # ------------------------------------------------------------------
    # Mesh file generation
    # ------------------------------------------------------------------

    def _write_mesh_files(self) -> None:
        """Generate and write dish STL files; populate self._mesh_map."""
        os.makedirs(self.mesh_dir, exist_ok=True)
        cfg = self.stack
        dishes = {
            "E1": cfg.antenna_diameter_east,
            "E2": cfg.antenna_diameter_east,
            "W1": cfg.antenna_diameter_west,
            "W2": cfg.antenna_diameter_west,
        }
        f_d = 0.5   # f/D ratio (fixed per spec)
        for name, diam in dishes.items():
            stl_bytes = generate_parabolic_dish_stl(diam, f_d, self.DISH_RES)
            fname = f"antenna_{name}_d{diam:.2f}.stl"
            fpath = os.path.join(self.mesh_dir, fname)
            with open(fpath, "wb") as fh:
                fh.write(stl_bytes)
            self._mesh_map[name] = fpath

    def _dish_geometry(self, ant_name: str, diameter: float) -> str:
        """Visual/collision geometry block for an antenna dish."""
        if ant_name in self._mesh_map:
            fpath = self._mesh_map[ant_name]
            # Use absolute path in URDF (portable via package:// prefix if desired)
            return (f'<geometry>'
                    f'<mesh filename="file://{fpath}" scale="1 1 1"/>'
                    f'</geometry>')
        # Fallback: flat cylinder approximation
        r = diameter / 2.0
        depth = diameter**2 / (16.0 * 0.5 * diameter)  # f/D=0.5
        # Cylinder along Z in link frame (aperture faces +Z)
        return (f'<geometry>'
                f'<cylinder radius="{r:.6f}" length="{depth:.6f}"/>'
                f'</geometry>')

    # ------------------------------------------------------------------
    # Link builders
    # ------------------------------------------------------------------

    def _link(self, name: str, visual_geo: str, col_geo: str,
              inertial: str,
              visual_origin: str = "", col_origin: str = "",
              color: str = "") -> str:
        vo = f"\n      {visual_origin}" if visual_origin else ""
        co = f"\n      {col_origin}"    if col_origin    else ""
        cl = f"\n      {color}"         if color         else ""
        return (
            f'  <link name="{name}">\n'
            f'    <visual>{vo}\n      {visual_geo}{cl}\n    </visual>\n'
            f'    <collision>{co}\n      {col_geo}\n    </collision>\n'
            f'{inertial}\n'
            f'  </link>'
        )

    def _empty_link(self, name: str) -> str:
        return f'  <link name="{name}"/>'

    def _links(self) -> list:
        """Return list of <link> XML strings for all links."""
        s   = self.stack
        arm = self.arm
        sections = []

        # ── world ──────────────────────────────────────────────────────
        sections.append(self._empty_link("world"))
        sections.append(self._empty_link("LAR_interface"))

        # ── client_bus ─────────────────────────────────────────────────
        cbx, cby, cbz = s.client_bus_x, s.client_bus_y, s.client_bus_z
        ixx, iyy, izz = _inertia_box(s.client_mass, cbx, cby, cbz)
        geo = f'<geometry><box size="{self._f(cbx, cby, cbz)}"/></geometry>'
        inertial = self._inertial_block(s.client_mass, ixx, iyy, izz)
        sections.append(self._link(
            "client_bus", geo, geo, inertial,
            color=self._rgba(0.6, 0.6, 0.8)
        ))

        # ── solar panels (North and South) ────────────────────────────
        pw  = s.panel_width
        psp = s.panel_span_one_side
        pt  = self.PANEL_THICK
        # Total plate span = bus half-width + panel span
        # Each panel modelled as a box: psp × pw × pt
        panel_mass = 50.0   # kg, approximate (not in StackConfig, use fixed value)
        ixx_p, iyy_p, izz_p = _inertia_thin_plate_xy(panel_mass, psp, pw)
        panel_geo = (f'<geometry>'
                     f'<box size="{self._f(psp, pw, pt)}"/>'
                     f'</geometry>')
        panel_inertial = self._inertial_block(panel_mass, ixx_p, iyy_p, izz_p)
        for side, name in [(1, "client_panel_north"), (-1, "client_panel_south")]:
            sections.append(self._link(
                name, panel_geo, panel_geo, panel_inertial,
                color=self._rgba(0.2, 0.6, 0.2)
            ))

        # ── antenna dishes ─────────────────────────────────────────────
        ant_centers = s.antenna_centers_in_lar_frame()
        diameters = {
            "E1": s.antenna_diameter_east, "E2": s.antenna_diameter_east,
            "W1": s.antenna_diameter_west, "W2": s.antenna_diameter_west,
        }
        for ant_name, diam in diameters.items():
            m_ant = s.antenna_mass
            ixx_a, iyy_a, izz_a = _inertia_parabolic_dish(m_ant, diam)
            dish_geo  = self._dish_geometry(ant_name, diam)
            dish_col  = (f'<geometry>'
                         f'<cylinder radius="{diam/2:.6f}" '
                         f'length="{diam**2/(16*0.5*diam):.6f}"/>'
                         f'</geometry>')
            dish_inertial = self._inertial_block(m_ant, ixx_a, iyy_a, izz_a)
            sections.append(self._link(
                f"antenna_{ant_name}", dish_geo, dish_col, dish_inertial,
                color=self._rgba(0.8, 0.8, 0.2)
            ))

        # ── servicer_bus ───────────────────────────────────────────────
        sbx, sby, sbz = s.servicer_bus_x, s.servicer_bus_y, s.servicer_bus_z
        ixx_s, iyy_s, izz_s = _inertia_box(s.servicer_mass, sbx, sby, sbz)
        svc_geo = (f'<geometry>'
                   f'<box size="{self._f(sbx, sby, sbz)}"/>'
                   f'</geometry>')
        svc_inertial = self._inertial_block(s.servicer_mass, ixx_s, iyy_s, izz_s)
        sections.append(self._link(
            "servicer_bus", svc_geo, svc_geo, svc_inertial,
            color=self._rgba(0.7, 0.4, 0.1)
        ))

        # ── thruster_arm_base (massless reference frame) ───────────────
        sections.append(self._empty_link("thruster_arm_base"))

        # ── arm_link_1 (cylinder along X) ──────────────────────────────
        L1, r1 = arm.link1_length, self.LINK_RADIUS
        ixx_l1, iyy_l1, izz_l1 = _inertia_cylinder_x(arm.link1_mass, r1, L1)
        # Cylinder default axis = Z; rotate 90° about Y to align with X
        link1_geo = (f'<geometry>'
                     f'<cylinder radius="{r1:.6f}" length="{L1:.6f}"/>'
                     f'</geometry>')
        link1_origin = self._origin(xyz=(L1/2, 0, 0), rpy=(0, self._HALF_PI, 0))
        link1_inertial = self._inertial_block(
            arm.link1_mass, ixx_l1, iyy_l1, izz_l1, cog_xyz=(L1/2, 0, 0))
        sections.append(self._link(
            "arm_link_1", link1_geo, link1_geo, link1_inertial,
            visual_origin=link1_origin, col_origin=link1_origin,
            color=self._rgba(0.9, 0.5, 0.1)
        ))

        # ── arm_link_2 ─────────────────────────────────────────────────
        L2, r2 = arm.link2_length, self.LINK_RADIUS
        ixx_l2, iyy_l2, izz_l2 = _inertia_cylinder_x(arm.link2_mass, r2, L2)
        link2_geo = (f'<geometry>'
                     f'<cylinder radius="{r2:.6f}" length="{L2:.6f}"/>'
                     f'</geometry>')
        link2_origin = self._origin(xyz=(L2/2, 0, 0), rpy=(0, self._HALF_PI, 0))
        link2_inertial = self._inertial_block(
            arm.link2_mass, ixx_l2, iyy_l2, izz_l2, cog_xyz=(L2/2, 0, 0))
        sections.append(self._link(
            "arm_link_2", link2_geo, link2_geo, link2_inertial,
            visual_origin=link2_origin, col_origin=link2_origin,
            color=self._rgba(0.9, 0.5, 0.1)
        ))

        # ── thruster_bracket ───────────────────────────────────────────
        Lb, rb = arm.bracket_length, self.LINK_RADIUS * 0.8
        ixx_b, iyy_b, izz_b = _inertia_cylinder_x(arm.bracket_mass, rb, Lb)
        brk_geo = (f'<geometry>'
                   f'<cylinder radius="{rb:.6f}" length="{Lb:.6f}"/>'
                   f'</geometry>')
        brk_origin = self._origin(xyz=(Lb/2, 0, 0), rpy=(0, self._HALF_PI, 0))
        brk_inertial = self._inertial_block(
            arm.bracket_mass, ixx_b, iyy_b, izz_b, cog_xyz=(Lb/2, 0, 0))
        sections.append(self._link(
            "thruster_bracket", brk_geo, brk_geo, brk_inertial,
            visual_origin=brk_origin, col_origin=brk_origin,
            color=self._rgba(0.8, 0.2, 0.2)
        ))

        # ── thruster_frame (massless, thrust ⊥ bracket long axis = along Z) ──
        # Small visual marker: thin box
        tf_geo = ('<geometry>'
                  '<box size="0.05 0.1 0.02"/>'
                  '</geometry>')
        tf_inertial = self._inertial_block(0.001, 1e-6, 1e-6, 1e-6)
        sections.append(self._link(
            "thruster_frame", tf_geo, tf_geo, tf_inertial,
            color=self._rgba(1.0, 0.0, 0.0)
        ))

        return sections

    # ------------------------------------------------------------------
    # Joint builders
    # ------------------------------------------------------------------

    def _joint(self, name: str, jtype: str, parent: str, child: str,
               xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0),
               axis=(0.0, 0.0, 1.0),
               lower: Optional[float] = None,
               upper: Optional[float] = None,
               effort: float = 200.0,
               velocity: float = 1.0) -> str:
        lines = [
            f'  <joint name="{name}" type="{jtype}">',
            f'    <parent link="{parent}"/>',
            f'    <child link="{child}"/>',
            f'    {self._origin(xyz, rpy)}',
        ]
        if jtype != "fixed":
            lines.append(
                f'    <axis xyz="{self._f(*axis)}"/>')
            if lower is not None and upper is not None:
                lines.append(
                    f'    <limit lower="{lower:.8f}" upper="{upper:.8f}" '
                    f'effort="{effort:.1f}" velocity="{velocity:.4f}"/>')
            lines.append(
                f'    <dynamics damping="0.1" friction="0.0"/>')
        lines.append('  </joint>')
        return "\n".join(lines)

    def _joints(self) -> list:
        """Return list of <joint> XML strings for all joints."""
        s   = self.stack
        arm = self.arm
        sections = []

        # world → LAR_interface  (fixed, identity)
        sections.append(self._joint(
            "world_to_LAR", "fixed", "world", "LAR_interface"))

        # LAR_interface → client_bus  (fixed; bus centre at [0,0,cbz/2])
        sections.append(self._joint(
            "LAR_to_client_bus", "fixed",
            "LAR_interface", "client_bus",
            xyz=(0.0, 0.0, s.client_bus_z / 2.0)))

        # client_bus → solar panels
        # Panel centre in client_bus frame:
        #   North: [ (cbx/2 + psp/2),  panel_hinge_offset_y, 0 ]
        #   South: [-(cbx/2 + psp/2),  panel_hinge_offset_y, 0 ]
        psp = s.panel_span_one_side
        px  = s.client_bus_x / 2.0 + psp / 2.0
        py  = s.panel_hinge_offset_y
        sections.append(self._joint(
            "client_bus_to_panel_north", "fixed",
            "client_bus", "client_panel_north",
            xyz=( px, py, 0.0)))
        sections.append(self._joint(
            "client_bus_to_panel_south", "fixed",
            "client_bus", "client_panel_south",
            xyz=(-px, py, 0.0)))

        # client_bus → antenna dishes
        # antenna_centers_in_lar_frame() gives LAR coords; subtract bus centre
        bus_centre_lar = np.array([0.0, 0.0, s.client_bus_z / 2.0])
        ant_centres_lar = s.antenna_centers_in_lar_frame()
        for ant_name, centre_lar in ant_centres_lar.items():
            offset = centre_lar - bus_centre_lar   # in client_bus frame
            sections.append(self._joint(
                f"client_bus_to_antenna_{ant_name}", "fixed",
                "client_bus", f"antenna_{ant_name}",
                xyz=tuple(float(v) for v in offset)))

        # LAR_interface → servicer_bus  (fixed, docked)
        svc_origin = s.servicer_origin_in_lar_frame()
        sections.append(self._joint(
            "LAR_to_servicer_bus", "fixed",
            "LAR_interface", "servicer_bus",
            xyz=tuple(float(v) for v in svc_origin)))

        # servicer_bus → thruster_arm_base  (fixed, at servicer +Z face)
        # In servicer frame: +Z face is at z = +servicer_bus_z/2
        pivot_in_svc = (
            arm.pivot_offset_x,
            arm.pivot_offset_y,
            s.servicer_bus_z / 2.0 + arm.pivot_offset_z,
        )
        sections.append(self._joint(
            "servicer_to_arm_base", "fixed",
            "servicer_bus", "thruster_arm_base",
            xyz=pivot_in_svc))

        # thruster_arm_base → arm_link_1  (J1 – shoulder yaw, revolute about Z)
        j1_lo = math.radians(arm.q0_min_deg)
        j1_hi = math.radians(arm.q0_max_deg)
        sections.append(self._joint(
            "J1_shoulder_yaw", "revolute",
            "thruster_arm_base", "arm_link_1",
            xyz=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 1.0),
            lower=j1_lo, upper=j1_hi))

        # arm_link_1 → arm_link_2  (J2 – elbow pitch, revolute about Y)
        # Joint is at the far end of link 1: x = L1
        j2_lo = math.radians(arm.q1_min_deg)
        j2_hi = math.radians(arm.q1_max_deg)
        sections.append(self._joint(
            "J2_elbow_pitch", "revolute",
            "arm_link_1", "arm_link_2",
            xyz=(arm.link1_length, 0.0, 0.0),
            axis=(0.0, 1.0, 0.0),
            lower=j2_lo, upper=j2_hi))

        # arm_link_2 → thruster_bracket  (J3 – wrist pitch, revolute about Y)
        j3_lo = math.radians(arm.q2_min_deg)
        j3_hi = math.radians(arm.q2_max_deg)
        sections.append(self._joint(
            "J3_wrist_pitch", "revolute",
            "arm_link_2", "thruster_bracket",
            xyz=(arm.link2_length, 0.0, 0.0),
            axis=(0.0, 1.0, 0.0),
            lower=j3_lo, upper=j3_hi))

        # thruster_bracket → thruster_frame  (fixed, thrust direction = +Z)
        sections.append(self._joint(
            "bracket_to_thruster_frame", "fixed",
            "thruster_bracket", "thruster_frame",
            xyz=(arm.bracket_length, 0.0, 0.0)))

        return sections

    # ------------------------------------------------------------------
    # XML header
    # ------------------------------------------------------------------

    def _xml_header(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<!-- Generated by urdf_generator.py -->\n'
            f'<!-- StackConfig + RoboticArmGeometry → parametric URDF -->\n'
            f'<!-- Coordinate frame: LAR origin, +Z=nadir, +X=North, +Y=East -->\n'
            f'<robot name="{self.robot_name}">'
        )


# ---------------------------------------------------------------------------
# 4.  CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def build_urdf(stack=None, arm=None,
               output_dir: Optional[str] = None,
               robot_name: str = "thruster_arm_stack") -> str:
    """Generate (and optionally save) the URDF for the given stack + arm.

    Parameters
    ──────────
    stack      : StackConfig (uses defaults if None)
    arm        : RoboticArmGeometry (uses defaults if None)
    output_dir : If provided, saves URDF + STL meshes to this directory
    robot_name : <robot name="..."> attribute

    Returns
    ───────
    URDF XML string.
    """
    if stack is None:
        from plume_impingement_pipeline import StackConfig
        stack = StackConfig()
    if arm is None:
        from plume_impingement_pipeline import RoboticArmGeometry
        arm = RoboticArmGeometry()

    gen = URDFGenerator(stack, arm, robot_name=robot_name)
    if output_dir is not None:
        gen.save(output_dir)
    return gen.generate()


# ---------------------------------------------------------------------------
# 5.  CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Generate parametric URDF for the Thruster Arm / Stack assembly.")
    parser.add_argument("output_dir", nargs="?", default="urdf_output",
                        help="Output directory (default: urdf_output/)")
    parser.add_argument("--yaw", type=float, default=45.0,
                        help="Shoulder yaw angle in degrees (default: 45)")
    parser.add_argument("--no-mesh", action="store_true",
                        help="Skip STL mesh generation (use cylinder placeholders)")
    args = parser.parse_args()

    from plume_impingement_pipeline import StackConfig, RoboticArmGeometry

    stack = StackConfig()
    arm   = RoboticArmGeometry(shoulder_yaw_deg=args.yaw)

    gen = URDFGenerator(stack, arm)
    if args.no_mesh:
        urdf_str = gen.generate()
        outfile  = os.path.join(args.output_dir, "thruster_arm.urdf")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(outfile, "w") as fh:
            fh.write(urdf_str)
        print(f"URDF written (no meshes): {outfile}")
    else:
        gen.save(args.output_dir)

    # Quick validation: count links and joints in generated XML
    xml = gen.generate()
    n_links  = xml.count('<link name=')
    n_joints = xml.count('<joint name=')
    print(f"Links: {n_links}   Joints: {n_joints}")
    sys.exit(0)
