# Thruster Arm Realistic Dynamics & URDF Specification

## Overview

This document specifies enhancements to the existing plasma plume impingement parametric pipeline for a satellite life-extension servicer mission. The updates add realistic mass distribution, center-of-gravity (CoG) management, and a URDF-based kinematic model for the thruster arm.

**Context:** A servicer spacecraft docks to an end-of-life GEO telecom satellite via a gripper arm on the Launch Adapter Ring (LAR). A separate thruster arm with plasma thrusters performs North-South and East-West station keeping (NSSK/EWSK). The concern is plasma plume erosion of critical surfaces, particularly silver interconnects on solar panels.

---

## 1. Coordinate Frame Conventions

**Origin:** LAR interface (mechanical connection between gripper and client satellite)

| Axis | Direction | Description |
|------|-----------|-------------|
| +Z | Nadir | Toward Earth; client extends this direction from LAR |
| −Z | Anti-Earth | Servicer sits this side of LAR |
| +X | North | Direction of North solar panel |
| +Y | East | Direction of East antenna face |

---

## 2. System Components

### 2.1 Client Satellite (GEO Comsat)

**Status:** End-of-life, dry mass only, fixed CoG

| Component | Geometry | Parameters |
|-----------|----------|------------|
| Bus | Rectangular prism | Dimensions parameterized (typical 2–3 m per side) |
| Panel North | Rectangle (thin plate) | Extends in +X direction from bus |
| Panel South | Rectangle (thin plate) | Extends in −X direction from bus |
| Antennas E1, E2 | Parabolic dishes | 2.2 m diameter, on +Y face |
| Antennas W1, W2 | Parabolic dishes | 2.5 m diameter, on −Y face |

**Antenna Mounting Details:**
- Two dishes per face (East and West)
- Stacked along X axis (North-South separation)
- Offset in +Z direction (toward Earth deck)
- All dishes point +Z (nadir)

**Assumed Antenna Parameters (adjustable):**

| Parameter | Value |
|-----------|-------|
| X separation between dish centers | 1.5 m |
| Z offset from bus centerline | +0.8 m |
| f/D ratio (focal length / diameter) | 0.5 |
| Dish depth (2.2 m dish) | 0.275 m |
| Dish depth (2.5 m dish) | 0.3125 m |
| Mass per antenna | 20 kg |

### 2.2 Servicer Satellite

**CoG behavior:** Shifts as propellant depletes over mission lifetime

| Component | Geometry | Mass |
|-----------|----------|------|
| Bus | Rectangular prism | Dry mass (parameterized) |
| Solar panels | Rectangular prism |
| Propellant | N/A (handled externally) | Parameterized fraction 0–100% |
| Gripper arm | Rigid link (cylinder) | Parameterized |

**Propellant Handling:**
- URDF contains dry mass inertia only
- Propellant mass and CoG shift computed externally
- Servicer CoG at any time:

```
r_CoG_servicer = (m_dry * r_dry + m_prop(t) * r_tank) / (m_dry + m_prop(t))
```

**Parameterized Servicer Mass Model:**

| Parameter | Treatment |
|-----------|-----------|
| Servicer dry mass | Parameterized |
| Propellant capacity | Parameterized |
| Propellant fraction | Sweep 0% to 100% |
| Tank offset from geometric center | Parameterized vector |

### 2.3 Thruster Arm

**Architecture:** 3-DoF, yaw-pitch-pitch configuration

| Joint | Type | Axis | Range | Following Link |
|-------|------|------|-------|----------------|
| J1 | Revolute (yaw/azimuth) | Z | 0° to 270° | Link 1 |
| J2 | Revolute (pitch/elbow) | Perpendicular to Link 1 | 0° to 235° | Link 2 |
| J3 | Revolute (pitch/wrist) | Perpendicular to Link 2 | −36° to +99° | Bracket |

**Link Properties:**

| Link | Length | Mass |
|------|--------|------|
| Link 1 | 1.12 m | 10 kg |
| Link 2 | 1.40 m | 10 kg |
| Bracket | 0.35 m | 3 kg |

**Total arm mass:** 23 kg  
**Total reach (extended):** 2.87 m

**Joint Zero Configuration:** Stowed (arm folded against servicer)

**Thruster Mounting:**
- Two thrusters on bracket (one redundant)
- Thrust direction: perpendicular to bracket long axis
- No gimbal; pointing controlled entirely by arm joints

**Arm Base Location:** On servicer's +Z face (facing toward LAR/client)

---

## 3. URDF Kinematic Tree

```
world (inertial frame)
│
└── LAR_interface (fixed frame, origin)
    │
    ├── client_bus (fixed joint)
    │   ├── client_panel_north (fixed)
    │   ├── client_panel_south (fixed)
    │   ├── antenna_east_1 (fixed)
    │   ├── antenna_east_2 (fixed)
    │   ├── antenna_west_1 (fixed)
    │   └── antenna_west_2 (fixed)
    │
    └── servicer_bus (fixed joint, docked via gripper)
        │
        └── thruster_arm_base (fixed to servicer +Z face)
            │
            └── arm_link_1 (revolute J1, yaw about Z)
                │
                └── arm_link_2 (revolute J2, pitch)
                    │
                    └── thruster_bracket (revolute J3, pitch)
                        │
                        └── thruster_frame (fixed, thrust perpendicular to bracket)
```

---

## 4. URDF Link Geometries

| Link | Visual/Collision Geometry | Inertial Model |
|------|---------------------------|----------------|
| LAR_interface | Reference frame only | None |
| client_bus | Box (parameterized dimensions) | Mass + full inertia tensor |
| client_panel_north | Rectangle (thin plate) | Mass + thin plate inertia |
| client_panel_south | Rectangle (thin plate) | Mass + thin plate inertia |
| antenna_east_1 | Parabolic dish (2.2 m, f/D=0.5) | 20 kg + dish inertia |
| antenna_east_2 | Parabolic dish (2.2 m, f/D=0.5) | 20 kg + dish inertia |
| antenna_west_1 | Parabolic dish (2.5 m, f/D=0.5) | 20 kg + dish inertia |
| antenna_west_2 | Parabolic dish (2.5 m, f/D=0.5) | 20 kg + dish inertia |
| servicer_bus | Box (parameterized) | Dry mass + inertia tensor |
| gripper_arm | Cylinder (rigid link) | Mass + slender rod inertia |
| arm_link_1 | Cylinder (1.12 m length) | 10 kg, distributed |
| arm_link_2 | Cylinder (1.40 m length) | 10 kg, distributed |
| thruster_bracket | Small box/cylinder (0.35 m) | 3 kg |
| thruster_frame | Reference frame only | Included in bracket mass |

**Note on Antenna Dishes:**  
Parabolic dish geometry is needed for accurate plume intersection checks. Generate as mesh (STL) or use spherical cap approximation with equivalent depth.

---

## 5. Operational Constraints

### 5.1 Station-Keeping Thrust Direction

**Thrust angle budget:** 45–50° deviation from ideal NSSK/EWSK direction is acceptable (soft budget, not hard limit)

- Most burns stay within this cone
- Can exceed occasionally to avoid critical surfaces or reduce CoG offset
- Cosine loss at 45°: ~30% penalty on effective ΔV

### 5.2 Arm Operation During Burns

- Arm is **stationary** during burns (not actively repositioning)
- Configuration selected and locked before burn initiation
- CoG is fixed for burn duration

### 5.3 Thruster Pointing

- No gimbal on thruster
- All pointing authority from arm joints
- J3 (−36° to +99°) provides fine pointing in wrist pitch plane

---

## 6. CoG Management

### 6.1 Stack CoG Calculation

The stack CoG must account for:

1. **Client satellite:** Fixed mass, fixed CoG relative to LAR
2. **Servicer satellite:** Dry mass + propellant (propellant fraction parameterized)
3. **Thruster arm:** 23 kg total, CoG position depends on joint configuration

```
r_CoG_stack = Σ(m_i * r_i) / Σ(m_i)
```

Where components include:
- Client bus, panels, antennas (fixed)
- Servicer bus dry mass (fixed relative to servicer frame)
- Servicer propellant (variable mass, fixed location)
- Arm links 1, 2, bracket (positions from forward kinematics)

### 6.2 CoG Shift Considerations

**Due to arm motion:**
- 23 kg moving up to 2.87 m from base
- For 800 kg stack: ~7 cm CoG shift at full extension
- For 3000+ kg stack: ~1.5 cm shift

**Due to propellant depletion:**
- Servicer CoG migrates from wet to dry position over mission
- Optimal arm configurations may change as propellant depletes

### 6.3 Disturbance Torque

When thrust line does not pass through stack CoG:

```
τ_disturbance = r_moment × F_thrust
```

Where `r_moment` is the perpendicular distance from thrust line to CoG.

**Strategy:** Optimize arm configuration to minimize moment arm while satisfying thrust direction and plume avoidance constraints.

---

## 7. Pipeline Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      URDF Model                         │
│   (geometry, kinematics, mass properties)               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Kinematics Engine                      │
│   - Load URDF (use urdfpy or yourdfpy)                  │
│   - Forward kinematics: (J1, J2, J3) → poses            │
│   - Extract link CoG positions for current config       │
│   - Extract surface meshes for intersection checks      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 External Mass Handler                   │
│   - Servicer propellant fraction → propellant mass      │
│   - Compute servicer CoG for current propellant state   │
│   - Combine with URDF masses for full stack CoG         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Case Matrix Generator                  │
│   Sweep parameters:                                     │
│   - J1, J2, J3 (joint angles)                           │
│   - Client mass                                         │
│   - Propellant fraction (0–100%)                        │
│   - Client geometry variants (optional)                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                Analytical Pre-Screener                  │
│   For each case, compute:                               │
│   - Thrust vector direction (from FK)                   │
│   - Angular deviation from ideal NSSK/EWSK direction    │
│   - Stack CoG position                                  │
│   - Moment arm (perpendicular distance, thrust to CoG)  │
│   - Plume cone intersection with surfaces               │
│   - Flux at critical surfaces (inverse-square model)    │
│   - Erosion estimates (sputter yield model)             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Scoring / Filtering                    │
│   Metrics:                                              │
│   - Fuel cost: f(angular deviation)                     │
│   - Disturbance torque: f(moment arm × thrust)          │
│   - Erosion risk: f(flux on critical surfaces)          │
│   Outputs:                                              │
│   - Pareto-optimal configurations                       │
│   - Feasibility flags (joint limits, angle budget)      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Heatmap Visualizer                     │
│   - Multi-dimensional dashboard                         │
│   - Selectable axes (any two parameters)                │
│   - Color by metric (fuel cost, torque, erosion risk)   │
│   - Overlay Pareto frontier                             │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Notes

### 8.1 Python Libraries

| Purpose | Recommended Library |
|---------|---------------------|
| URDF parsing | `urdfpy` or `yourdfpy` |
| Forward kinematics | `urdfpy` (built-in) or `numpy` transforms |
| Mesh handling | `trimesh` (for dish geometry, intersection checks) |
| Visualization | `plotly` for interactive heatmaps |
| Numerical | `numpy`, `scipy` |

### 8.2 URDF Generation

Options:
1. **Hand-write URDF XML** with parameterized values
2. **Generate programmatically** using `urdfpy` API
3. **Template approach:** Base URDF with placeholder values, substitute at runtime

Recommended: Option 3 — allows quick parameter sweeps without regenerating full URDF.

### 8.3 Antenna Dish Meshes

Generate parabolic dish STL files programmatically:

```python
def generate_parabolic_dish(diameter, f_d_ratio, resolution=50):
    """
    Generate vertices/faces for parabolic dish.
    depth = diameter / (16 * f_d_ratio)
    z = (x^2 + y^2) / (4 * focal_length)
    """
    # Implementation generates mesh vertices and faces
    # Export as STL for URDF collision/visual
```

### 8.4 Propellant Mass Handling

Since URDF doesn't support time-varying mass:

```python
class ServicerMassModel:
    def __init__(self, dry_mass, propellant_capacity, tank_offset):
        self.dry_mass = dry_mass
        self.propellant_capacity = propellant_capacity
        self.tank_offset = tank_offset  # vector from dry CoG to tank CoG
    
    def get_mass_and_cog(self, propellant_fraction):
        """Returns (total_mass, cog_position) for given propellant state."""
        prop_mass = propellant_fraction * self.propellant_capacity
        total_mass = self.dry_mass + prop_mass
        cog = (self.dry_mass * self.dry_cog + prop_mass * self.tank_cog) / total_mass
        return total_mass, cog
```

### 8.5 Stack CoG Computation

```python
def compute_stack_cog(urdf_model, joint_angles, servicer_mass_model, propellant_fraction, client_mass, client_cog):
    """
    Compute combined stack CoG for given configuration.
    
    1. Get arm link positions from FK
    2. Get servicer mass/CoG from propellant state
    3. Combine all masses weighted by position
    """
    # Get arm link CoGs from forward kinematics
    arm_link_cogs = urdf_model.get_link_cogs(joint_angles)
    
    # Get servicer state
    servicer_mass, servicer_cog = servicer_mass_model.get_mass_and_cog(propellant_fraction)
    
    # Sum all contributions
    total_mass = client_mass + servicer_mass + sum(arm_link_masses)
    weighted_sum = (client_mass * client_cog + 
                   servicer_mass * servicer_cog + 
                   sum(m * r for m, r in zip(arm_link_masses, arm_link_cogs)))
    
    return weighted_sum / total_mass
```

---

## 9. Outputs

### 9.1 Per-Configuration Outputs

| Output | Description | Units |
|--------|-------------|-------|
| `thrust_vector` | Direction of thrust in LAR frame | Unit vector |
| `thrust_position` | Location of thruster in LAR frame | m |
| `angle_deviation_nssk` | Angle between thrust and ideal NSSK direction | deg |
| `angle_deviation_ewsk` | Angle between thrust and ideal EWSK direction | deg |
| `stack_cog` | Combined CoG position in LAR frame | m |
| `moment_arm` | Perpendicular distance from thrust line to CoG | m |
| `disturbance_torque` | Moment arm × thrust magnitude | N·m |
| `panel_flux_north` | Plume flux at north panel | particles/cm²/s |
| `panel_flux_south` | Plume flux at south panel | particles/cm²/s |
| `antenna_flux_E1` | Plume flux at antenna E1 | particles/cm²/s |
| `antenna_flux_E2` | Plume flux at antenna E2 | particles/cm²/s |
| `antenna_flux_W1` | Plume flux at antenna W1 | particles/cm²/s |
| `antenna_flux_W2` | Plume flux at antenna W2 | particles/cm²/s |
| `erosion_risk_score` | Combined erosion risk metric | dimensionless |
| `feasibility_flag` | Joint limits satisfied, angle budget OK | boolean |

### 9.2 Aggregated Outputs

- Pareto-optimal configurations for each thrust direction (NSSK, EWSK)
- Sensitivity of optimal configuration to propellant fraction
- "Robust" configurations that work across propellant states
- Risk maps: joint space colored by erosion risk, torque, fuel cost

---

## 10. Future Extensions (Out of Scope for Now)

- Structural flexibility of arm links (vibration modes)
- Full dynamics simulation in Gazebo
- Reaction wheel momentum accumulation over burn sequences
- CEX backflow modeling for improved plume physics
- Thermal effects from plume heating

---

## Appendix A: Parameter Summary Table

| Parameter | Symbol | Baseline Value | Sweep Range |
|-----------|--------|----------------|-------------|
| Client mass | m_c | — | 1500–3000 kg |
| Client bus dimensions | — | ~2.5 m cube | Parameterized |
| Panel span (each) | — | ~15 m | Parameterized |
| Antenna diameter (E) | D_E | 2.2 m | Fixed |
| Antenna diameter (W) | D_W | 2.5 m | Fixed |
| Antenna mass | m_ant | 20 kg | Fixed |
| Servicer dry mass | m_s_dry | — | Parameterized |
| Propellant capacity | m_prop_max | — | Parameterized |
| Propellant fraction | f_prop | — | 0–100% |
| Arm link 1 length | L1 | 1.12 m | Fixed |
| Arm link 2 length | L2 | 1.40 m | Fixed |
| Bracket length | L3 | 0.35 m | Fixed |
| Arm link 1 mass | m_L1 | 10 kg | Fixed |
| Arm link 2 mass | m_L2 | 10 kg | Fixed |
| Bracket mass | m_brk | 3 kg | Fixed |
| J1 range | θ1 | — | 0–270° |
| J2 range | θ2 | — | 0–235° |
| J3 range | θ3 | — | −36° to +99° |
| Thrust angle budget | — | 45–50° | Soft limit |

---

## Appendix B: Reference Frame Diagram

```
                        +X (North)
                           ↑
                           │
                           │    ┌─────────────┐
                           │    │   Panel N   │
                           │    └─────────────┘
                           │          │
        ┌──────────────────┼──────────┼──────────────────┐
        │                  │    ┌─────┴─────┐            │
        │  [Ant W1] [W2]   │    │  Client   │   [E1] [E2]│ ──→ +Y (East)
        │                  │    │   Bus     │            │
        │                  │    └─────┬─────┘            │
        └──────────────────┼──────────┼──────────────────┘
                           │          │
                           │    ┌─────────────┐
                           │    │   Panel S   │
                           │    └─────────────┘
                           │
                           │
    ════════════════════════════════════════════════════════  ← LAR Interface (Z=0)
                           │
                           │
                     ┌─────┴─────┐
                     │ Servicer  │
                     │    Bus    │
                     │  ┌─────┐  │
                     │  │Arm  │  │  ← Thruster arm base (+Z face of servicer)
                     │  │Base │  │
                     └──┴─────┴──┘
                           │
                           ↓
                        −Z (Anti-Earth, servicer side)


        +Z (Nadir/Earth) points INTO the page above LAR
        −Z (Anti-Earth) points OUT of the page below LAR
```

---

*End of specification.*
