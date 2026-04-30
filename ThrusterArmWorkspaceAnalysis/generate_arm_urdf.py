#!/usr/bin/env python3
"""
generate_arm_urdf.py
====================
Generate a URDF for the 3-DOF thruster arm from RoboticArmGeometry /
StackConfig parameters.

Robot topology (servicer-body frame at q = [0, 0, 0])
------------------------------------------------------
  servicer_bus  (floating base, box inertia)
  └── joint_0   revolute  axis1 = [0, 0, -1]   J1 shoulder yaw
      └── link_0  (link 1: H1 → H2)
          └── joint_1   revolute  axis2 = [1, 0, 0]   J2 elbow pitch
              └── link_1  (link 2: H2 → H3)
                  └── joint_2   revolute  axis3 = [0, -0.3746, 0.9272]   J3 wrist
                      └── link_2  (bracket: H3 → nozzle)
                          └── joint_nozzle  fixed
                              └── nozzle_frame  (massless)

Inertia convention
------------------
Each arm link uses a thin-rod inertia about its CoM:
    I = m * L² / 12 · (I₃ − û û^T)
where û is the unit vector along the link from proximal to distal joint.

All tensors are expressed in the link's coordinate frame (parallel to the
servicer body frame at q = 0).  URDF <inertial><origin> places the CoM
at com_offset · û from the proximal joint.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plume_impingement_pipeline import RoboticArmGeometry, StackConfig


# ---------------------------------------------------------------------------
# Inertia helpers
# ---------------------------------------------------------------------------

def _rod_inertia_at_com(mass: float, length: float,
                         u_hat: np.ndarray) -> np.ndarray:
    """Thin-rod inertia tensor at the CoM (3×3).

    For a uniform rod of mass *mass* and length *length* along unit vector
    *u_hat*, the inertia about the CoM in any frame is:
        I = m L² / 12 · (I₃ − û û^T)
    """
    return mass * length ** 2 / 12.0 * (np.eye(3) - np.outer(u_hat, u_hat))


def _box_inertia(mass: float, bx: float, by: float, bz: float) -> np.ndarray:
    """Uniform solid-box inertia about the box centre (3×3)."""
    return np.diag([
        mass * (by ** 2 + bz ** 2) / 12.0,
        mass * (bx ** 2 + bz ** 2) / 12.0,
        mass * (bx ** 2 + by ** 2) / 12.0,
    ])


def _inertia_xml(I: np.ndarray, indent: str = "      ") -> str:
    return (
        f'{indent}<inertia '
        f'ixx="{I[0, 0]:.8g}" ixy="{I[0, 1]:.8g}" ixz="{I[0, 2]:.8g}" '
        f'iyy="{I[1, 1]:.8g}" iyz="{I[1, 2]:.8g}" izz="{I[2, 2]:.8g}"/>'
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_urdf(
    arm: RoboticArmGeometry | None = None,
    stack: StackConfig | None = None,
    output_path: str | None = None,
) -> str:
    """Return URDF as a string; write to *output_path* if given.

    Parameters
    ----------
    arm         : arm geometry (defaults to hardware-confirmed values)
    stack       : stack config (defaults to hardware-confirmed values)
    output_path : optional file path to write the URDF
    """
    if arm is None:
        arm = RoboticArmGeometry()
    if stack is None:
        stack = StackConfig()

    # ── Link offsets (TA body frame at q = 0 = servicer body frame) ──────────
    d1 = np.asarray(arm.d_h1h2, dtype=float)
    d2 = np.asarray(arm.d_h2h3, dtype=float)
    d3 = np.asarray(arm.d_h3n,  dtype=float)

    L1 = float(np.linalg.norm(d1))
    L2 = float(np.linalg.norm(d2))
    L3 = float(np.linalg.norm(d3))

    u1 = d1 / L1
    u2 = d2 / L2
    u3 = d3 / L3

    # ── CoM positions in each link frame ─────────────────────────────────────
    com1 = arm.effective_link1_com()   * u1
    com2 = arm.effective_link2_com()   * u2
    com3 = arm.effective_bracket_com() * u3

    # ── Inertia tensors at CoM (thin-rod, expressed in link frame) ───────────
    I_base = _box_inertia(stack.servicer_mass,
                          stack.servicer_bus_x,
                          stack.servicer_bus_y,
                          stack.servicer_bus_z)
    I1 = _rod_inertia_at_com(arm.link1_mass,   L1, u1)
    I2 = _rod_inertia_at_com(arm.link2_mass,   L2, u2)
    I3 = _rod_inertia_at_com(arm.bracket_mass, L3, u3)

    # ── Joint axes and pivot ──────────────────────────────────────────────────
    ax1    = np.asarray(arm.axis1, dtype=float)
    ax2    = np.asarray(arm.axis2, dtype=float)
    ax3    = np.asarray(arm.axis3, dtype=float)
    pivot  = arm.arm_pivot_in_servicer_body()          # H1 in servicer body frame

    def v3(v: np.ndarray) -> str:
        return f"{v[0]:.8g} {v[1]:.8g} {v[2]:.8g}"

    ql = np.radians   # shorthand

    lines = [
        '<?xml version="1.0"?>',
        '<robot name="thruster_arm">',
        '',
        '  <!-- ── Floating base: servicer bus ────────────────────────── -->',
        '  <link name="servicer_bus">',
        '    <inertial>',
        '      <origin xyz="0 0 0" rpy="0 0 0"/>',
        f'      <mass value="{stack.servicer_mass:.6g}"/>',
        _inertia_xml(I_base),
        '    </inertial>',
        '  </link>',
        '',
        '  <!-- ── J1: shoulder yaw  axis = [0, 0, -1] ────────────────── -->',
        '  <joint name="joint_0" type="revolute">',
        '    <parent link="servicer_bus"/>',
        '    <child link="link_0"/>',
        f'    <origin xyz="{v3(pivot)}" rpy="0 0 0"/>',
        f'    <axis xyz="{v3(ax1)}"/>',
        f'    <limit lower="{ql(arm.q0_min_deg):.6g}" upper="{ql(arm.q0_max_deg):.6g}"'
        '           effort="500" velocity="0.5"/>',
        '  </joint>',
        '',
        '  <link name="link_0">',
        '    <inertial>',
        f'      <origin xyz="{v3(com1)}" rpy="0 0 0"/>',
        f'      <mass value="{arm.link1_mass:.6g}"/>',
        _inertia_xml(I1),
        '    </inertial>',
        '  </link>',
        '',
        '  <!-- ── J2: elbow pitch  axis = [1, 0, 0] ──────────────────── -->',
        '  <joint name="joint_1" type="revolute">',
        '    <parent link="link_0"/>',
        '    <child link="link_1"/>',
        f'    <origin xyz="{v3(d1)}" rpy="0 0 0"/>',
        f'    <axis xyz="{v3(ax2)}"/>',
        f'    <limit lower="{ql(arm.q1_min_deg):.6g}" upper="{ql(arm.q1_max_deg):.6g}"'
        '           effort="500" velocity="0.5"/>',
        '  </joint>',
        '',
        '  <link name="link_1">',
        '    <inertial>',
        f'      <origin xyz="{v3(com2)}" rpy="0 0 0"/>',
        f'      <mass value="{arm.link2_mass:.6g}"/>',
        _inertia_xml(I2),
        '    </inertial>',
        '  </link>',
        '',
        '  <!-- ── J3: wrist  axis = [0, -0.3746, 0.9272] ─────────────── -->',
        '  <joint name="joint_2" type="revolute">',
        '    <parent link="link_1"/>',
        '    <child link="link_2"/>',
        f'    <origin xyz="{v3(d2)}" rpy="0 0 0"/>',
        f'    <axis xyz="{v3(ax3)}"/>',
        f'    <limit lower="{ql(arm.q2_min_deg):.6g}" upper="{ql(arm.q2_max_deg):.6g}"'
        '           effort="200" velocity="0.5"/>',
        '  </joint>',
        '',
        '  <link name="link_2">',
        '    <inertial>',
        f'      <origin xyz="{v3(com3)}" rpy="0 0 0"/>',
        f'      <mass value="{arm.bracket_mass:.6g}"/>',
        _inertia_xml(I3),
        '    </inertial>',
        '  </link>',
        '',
        '  <!-- ── Nozzle exit frame (fixed, no mass) ───────────────────── -->',
        '  <joint name="joint_nozzle" type="fixed">',
        '    <parent link="link_2"/>',
        '    <child link="nozzle_frame"/>',
        f'    <origin xyz="{v3(d3)}" rpy="0 0 0"/>',
        '  </joint>',
        '  <link name="nozzle_frame"/>',
        '',
        '</robot>',
    ]

    urdf_str = '\n'.join(lines) + '\n'

    if output_path:
        with open(output_path, 'w') as fh:
            fh.write(urdf_str)

    return urdf_str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate thruster arm URDF from hardware geometry"
    )
    parser.add_argument("--output", "-o", default="thruster_arm.urdf",
                        help="Output URDF file path (default: thruster_arm.urdf)")
    args = parser.parse_args()

    urdf = generate_urdf(output_path=args.output)
    print(f"Wrote {args.output}  ({len(urdf.splitlines())} lines)")
