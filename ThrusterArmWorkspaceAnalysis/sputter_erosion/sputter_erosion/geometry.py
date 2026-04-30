"""
geometry.py
===========

Geometric coupling between the satellite, the thruster firing direction, and
the solar-array interconnect surfaces.

Coordinate convention
---------------------
All geometry expressed in a satellite body-fixed frame:
  +X : nominal +ram direction (or any chosen reference axis)
  +Y : completes the right-handed frame
  +Z : satellite +zenith / array-normal sun-pointing axis

Thruster firing direction is a unit vector in this frame. Solar array panels
are described by their position, surface normal, and (optionally) gimbal
state. Interconnects are sub-elements on the array with their own local
geometry (pitch, exposed face, coverglass overhang).

This module implements "physical-parameter group 2" (impact conditions:
local incidence angle, plasma potential / sheath bias) and provides the
plumbing for groups 1 (plume), 3 (target), 4 (environment) to be evaluated
at the right (theta_plume, r) for each interconnect.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import numpy as np

from .plume import PlumeModel, PlumeState
from .materials import Material, get_material


# ---------------------------------------------------------------------------
# Vectors and rotations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_array(cls, a: np.ndarray) -> "Vector3":
        return cls(float(a[0]), float(a[1]), float(a[2]))

    def normalized(self) -> "Vector3":
        n = np.linalg.norm(self.to_array())
        if n < 1e-15:
            return self
        return Vector3.from_array(self.to_array() / n)

    def dot(self, other: "Vector3") -> float:
        return float(np.dot(self.to_array(), other.to_array()))

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3.from_array(self.to_array() - other.to_array())

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3.from_array(self.to_array() + other.to_array())


def angle_between(u: Vector3, v: Vector3) -> float:
    """Angle between two vectors in radians, in [0, pi]."""
    a = u.normalized().to_array()
    b = v.normalized().to_array()
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.arccos(c))


# ---------------------------------------------------------------------------
# Sheath / bias model (group 2: impact conditions)
# ---------------------------------------------------------------------------

@dataclass
class SheathModel:
    """
    Sheath acceleration of low-energy CEX ions onto biased interconnects.

    For a string voltage V_string, the most-negative interconnect sits at
    roughly -V_string + V_floating relative to local plasma potential. CEX
    ions arriving at a negative surface gain that potential drop in
    additional kinetic energy.

    For unbiased (floating) surfaces, V_floating ~ -3-5 Te in eV (GEO plasma).
    """
    string_voltage: float = 100.0       # V end-to-end array string voltage
    floating_potential: float = -15.0   # V wrt local plasma
    Te_local: float = 2.0               # eV local electron temperature

    def added_energy_eV(self, position_along_string: float = 1.0) -> float:
        """
        Extra energy [eV] gained by a CEX ion arriving at an interconnect
        located at fractional position `position_along_string` (0 = positive
        end, 1 = negative end). The most-negative end sees the largest boost.
        """
        return abs(self.floating_potential) + position_along_string * self.string_voltage


# ---------------------------------------------------------------------------
# Interconnect and solar-array geometry
# ---------------------------------------------------------------------------

@dataclass
class Interconnect:
    """
    A single interconnect (or representative interconnect strip) on a panel.

    position_local        : (x, y) position on the panel, in panel local frame [m]
    exposed_face_normal   : outward normal of the exposed sidewall/edge,
                            in the panel local frame
    material_name         : key into materials.MATERIALS (e.g. "Ag")
    coverglass_overhang   : coverglass overhang over interconnect [m] (used to
                            compute geometric shadowing of grazing-angle ions)
    string_position       : 0..1, fractional position along electrical string
                            (0 = + terminal, 1 = - terminal), for sheath model
    exposed_thickness     : initial interconnect exposed dimension [m] for
                            life-limiting calculation
    """
    position_local: Tuple[float, float]
    exposed_face_normal: Vector3
    material_name: str = "Ag"
    coverglass_overhang: float = 50e-6
    string_position: float = 0.5
    exposed_thickness: float = 25e-6   # 25 um typical Ag interconnect thickness

    def material(self) -> Material:
        return get_material(self.material_name)


@dataclass
class SolarArray:
    """
    A flat solar-array panel with origin (panel-frame -> body-frame) and
    rotation. The panel normal is the +Z_panel axis; +X_panel and +Y_panel
    span the panel surface.
    """
    origin_body: Vector3
    panel_normal_body: Vector3
    panel_x_body: Vector3
    width: float
    height: float
    interconnects: List[Interconnect] = field(default_factory=list)

    def panel_y_body(self) -> Vector3:
        n = self.panel_normal_body.normalized().to_array()
        x = self.panel_x_body.normalized().to_array()
        y = np.cross(n, x)
        return Vector3.from_array(y)

    def interconnect_position_body(self, ic: Interconnect) -> Vector3:
        """Return interconnect position in the body frame."""
        ux = self.panel_x_body.normalized().to_array()
        uy = self.panel_y_body().to_array()
        p_local = ic.position_local
        pos = (
            self.origin_body.to_array()
            + p_local[0] * ux
            + p_local[1] * uy
        )
        return Vector3.from_array(pos)

    def interconnect_normal_body(self, ic: Interconnect) -> Vector3:
        """Transform the interconnect's local exposed-face normal into body frame."""
        # The local frame is (panel_x, panel_y, panel_normal). Express the
        # interconnect normal (given in panel local coords as a Vector3) in body.
        ux = self.panel_x_body.normalized().to_array()
        uy = self.panel_y_body().to_array()
        un = self.panel_normal_body.normalized().to_array()
        n_local = ic.exposed_face_normal.to_array()
        n_body = n_local[0] * ux + n_local[1] * uy + n_local[2] * un
        return Vector3.from_array(n_body / (np.linalg.norm(n_body) + 1e-30))


# ---------------------------------------------------------------------------
# Thruster placement and orientation
# ---------------------------------------------------------------------------

@dataclass
class ThrusterPlacement:
    """
    A thruster mounted at a given body-frame position, firing in a given
    direction. The plume model defines its own internal coordinate system,
    which we map into the body frame here.

    fire_direction_body : unit vector along which the beam is centred
                          (i.e. opposite to the thrust direction).
    """
    position_body: Vector3
    fire_direction_body: Vector3
    plume: PlumeModel
    cant_angle_deg: float = 0.0   # info only; embed in fire_direction_body

    def vector_to(self, point_body: Vector3) -> Tuple[Vector3, float]:
        """Vector from thruster to point, in body frame, plus its norm [m]."""
        v = point_body - self.position_body
        r = float(np.linalg.norm(v.to_array()))
        return v, r

    def plume_polar_angle(self, point_body: Vector3) -> float:
        """
        Angle [rad] between the thruster firing axis and the line from the
        thruster to a point in the body frame. This is the "theta" in plume
        models.
        """
        v, _ = self.vector_to(point_body)
        return angle_between(self.fire_direction_body, v)

    def evaluate_plume_at(self, point_body: Vector3,
                          surface_normal_body: Vector3) -> PlumeState:
        """
        Return the local plume state at a given target point, with the
        incidence angle measured against the supplied surface normal.
        """
        _, r = self.vector_to(point_body)
        theta_plume = self.plume_polar_angle(point_body)

        # Incidence angle: angle between the local plume direction (from
        # thruster to point) and the surface inward-normal.
        v_to_point, _ = self.vector_to(point_body)
        # Inward normal = -outward; ion travels along v_to_point direction.
        v_dir = v_to_point.normalized()
        # Angle between v_dir and the surface normal
        cos_inc = abs(v_dir.dot(surface_normal_body))
        cos_inc = float(np.clip(cos_inc, -1.0, 1.0))
        # We want angle from the surface normal of the surface OUTward; ions
        # arrive moving INTO the surface, so their direction is -outward_normal,
        # and incidence angle is angle between v_dir and -outward_normal.
        v_arr = v_dir.to_array()
        n_out = surface_normal_body.normalized().to_array()
        cos_theta_inc = float(np.clip(np.dot(v_arr, -n_out), -1.0, 1.0))
        # If plume direction is from thruster outward, ions hit if cos > 0
        if cos_theta_inc <= 0:
            # surface is shadowed from this thruster (back-face)
            theta_inc = np.pi / 2  # grazing limit; will give zero yield contribution
        else:
            theta_inc = float(np.arccos(cos_theta_inc))

        return self.plume.evaluate(theta_plume, r,
                                    surface_normal_angle_rad=theta_inc)


# ---------------------------------------------------------------------------
# Full satellite assembly
# ---------------------------------------------------------------------------

@dataclass
class SatelliteGeometry:
    """
    Top-level container: thrusters + solar arrays + sheath model.

    Use `iter_targets` to walk every (thruster, array, interconnect) triplet
    that contributes to erosion at a given moment.
    """
    thrusters: List[ThrusterPlacement]
    solar_arrays: List[SolarArray]
    sheath: SheathModel = field(default_factory=SheathModel)

    def iter_targets(self):
        """
        Generator yielding (thruster, array, interconnect, point_body,
        normal_body, plume_state) for every interconnect on every array, for
        each active thruster.

        Skip pairs where the array faces away from the thruster (cos < 0).
        """
        for thr in self.thrusters:
            for arr in self.solar_arrays:
                for ic in arr.interconnects:
                    p_body = arr.interconnect_position_body(ic)
                    n_body = arr.interconnect_normal_body(ic)
                    state = thr.evaluate_plume_at(p_body, n_body)
                    if state.j_i <= 0.0:
                        continue
                    yield thr, arr, ic, p_body, n_body, state
