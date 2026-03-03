# Propellant Sloshing in Spacecraft GNC — Theory and Modelling

*Document prepared for GNC / Plant Modelling Engineers*

---

## Table of Contents

1. [Introduction to Sloshing](#1-introduction-to-sloshing)
2. [Abramson's Equivalent Mechanical Model](#2-abramsons-equivalent-mechanical-model)
3. [Equations of Motion and State Space Formulation](#3-equations-of-motion-and-state-space-formulation)
4. [Xenon - Why Sloshing Can Be Neglected](#4-xenon--why-sloshing-can-be-neglected)
5. [C₃H₆ and N₂O - Why Sloshing Must Be Considered](#5-c₃h₆-and-n₂o--why-sloshing-must-be-considered)

---

## 1. Introduction to Sloshing

### 1.1 Physical Origin

When a spacecraft carries liquid propellant in a partially filled tank, the free liquid surface can oscillate in response to any external excitation — attitude manoeuvres, thruster firings, or residual angular motion. This oscillation of the free surface, and the associated motion of the liquid mass relative to the tank walls, is called **propellant sloshing**.

The fundamental requirement for sloshing is the existence of a **free liquid surface** — a liquid-gas interface inside the tank. Without this interface, sloshing in the classical sense cannot occur.

### 1.2 Why It Matters for GNC

Sloshing is not merely a fluid dynamics curiosity. It has direct and potentially destabilising consequences for the Attitude and Orbit Control System (AOCS):

- The sloshing liquid generates **forces and torques** on the spacecraft that are not commanded and not directly sensed.
- These forces and torques act as **internal disturbances** that couple back into the attitude dynamics.
- The slosh mode introduces **lightly damped oscillatory poles** into the plant transfer function. If the AOCS control bandwidth overlaps with the slosh frequency, the controller can excite or amplify sloshing — potentially destabilising the closed loop.
- On large geostationary telecom satellites carrying several hundred kilograms of propellant at beginning of life, the sloshing mass can represent **30–50% of the total propellant mass**, making this a first-order effect rather than a minor perturbation.

The problem was extensively studied in the 1960s in the context of large liquid-fuelled rockets (Saturn V, etc.) where sloshing in propellant tanks posed a serious stability risk. The definitive reference remains:

> **Abramson, H.N. (Ed.), 1966. *The Dynamic Behavior of Liquids in Moving Containers.* NASA SP-106, NASA, Washington D.C.**

This document established the equivalent mechanical modelling framework that is still the standard tool used by spacecraft GNC engineers today.

### 1.3 Scope of This Document

This document covers:
- The Abramson equivalent mechanical model and its parameters
- The coupled spacecraft-slosh equations of motion and their state space formulation
- The specific cases of xenon, propylene (C₃H₆), and nitrous oxide (N₂O) propellants, and whether sloshing modelling is required for each

---

## 2. Abramson's Equivalent Mechanical Model

### 2.1 Core Concept

The central idea of Abramson's framework is to replace the complex three-dimensional fluid dynamics of a sloshing liquid with a simple **equivalent mechanical system** that produces the same net forces and torques on the tank walls. This makes the slosh dynamics tractable for integration into spacecraft equations of motion and control design tools.

The total propellant mass `m_total` is decomposed into two parts:

| Component | Symbol | Description |
|---|---|---|
| Fixed mass | m₀ | Moves rigidly with the tank — does not participate in sloshing |
| Sloshing mass (mode n) | mₙ | Oscillates relative to the tank — generates dynamic forces and torques |

The constraint is:

```
m₀ + Σ mₙ = m_total
```

Each sloshing mass mₙ is attached to the tank via a **spring** (spring-mass model) or a **pendulum** (pendulum model), representing the restoring force of the free surface. In practice, only the **first mode** (n=1) is retained — higher modes carry negligible mass and are heavily damped.

### 2.2 Pendulum vs Spring-Mass Representation

Both representations are physically equivalent. The choice depends on the operating condition:

**Pendulum model** — natural when a steady acceleration exists along the tank axis (e.g. during a thruster burn). The sloshing mass hangs from a pivot point inside the tank with equivalent pendulum length:

```
l₁ = g_eff / ω₁²
```

where `g_eff` is the effective acceleration (thrust / spacecraft mass) and `ω₁` is the slosh natural frequency.

**Spring-mass model** — more natural for lateral sloshing during coast phases or attitude manoeuvres, where the sloshing mass is modelled as a laterally sliding mass attached to the tank via a spring of stiffness `k = m₁ ω₁²`.

### 2.3 Model Parameters

A complete first-mode Abramson model requires the following parameters, all functions of tank geometry and fill fraction:

| Parameter | Symbol | Units | Description |
|---|---|---|---|
| Fixed mass | m₀ | kg | Rigidly attached propellant fraction |
| Sloshing mass | m₁ | kg | Dynamically active propellant fraction |
| Natural frequency | ω₁ | rad/s | First slosh mode frequency |
| Damping ratio | ζ₁ | — | Damping of first mode |
| Attachment height | h₁ | m | Height of spring pivot above tank bottom |
| Pendulum length | l₁ | m | Equivalent pendulum length (derived from ω₁) |

### 2.4 Abramson Formulas for a Cylindrical Tank

For a **cylindrical tank** of inner radius `R` filled to height `h`, the classical Abramson results for the first mode are as follows.

Define the dimensionless fill ratio:
```
ξ = h / R
```

And the first zero of the derivative of the first-order Bessel function:
```
λ₁₁ = 1.8412
```

**First mode natural frequency:**
```
ω₁² = (g_eff · λ₁₁ / R) · tanh(λ₁₁ · ξ)
```

**First mode sloshing mass fraction:**
```
m₁ / m_total = (2 / λ₁₁² · ξ) · tanh(λ₁₁ · ξ)
```

**Fixed mass:**
```
m₀ = m_total - m₁
```

**Attachment height above tank bottom:**
```
h₁ = h - (R / λ₁₁) · tanh(λ₁₁ · ξ)
```

**Equivalent pendulum length:**
```
l₁ = g_eff / ω₁²
```

These formulas are valid for the linearised free-surface potential flow assumption, which holds for small-amplitude sloshing.

### 2.5 Parameter Variation with Fill Fraction

The Abramson parameters are not constant - they change continuously as propellant is consumed over the mission. Key behaviours to note:

- **ω₁** decreases as the tank empties (lower fill → lower slosh frequency), but saturates at low fill ratios.
- **m₁** peaks at mid-range fill fractions (~40–60% full) and is small when the tank is nearly full or nearly empty. This means **mid-range fill is the worst case for sloshing perturbation**.
- **ζ₁** is approximately constant for a given tank hardware configuration, but is highly uncertain — smooth walls give ζ₁ ≈ 0.001–0.005; baffles can raise it to 0.02–0.05.

For this reason, the GNC engineer must verify stability margins across the **full fill fraction sweep** from beginning to end of mission, not just at a single operating point.

### 2.6 Coordinate Transformation

The Abramson parameters are defined relative to the tank geometry. To use them in spacecraft dynamics, the attachment point height `h₁` (measured from tank bottom) must be converted to the **spacecraft body frame**, giving the position of the slosh attachment point relative to the spacecraft centre of mass (CoM).

If the tank bottom is located at position `z_tank_bottom` from the spacecraft CoM along the thrust axis:

```
z_attach = z_tank_bottom + h₁
r_arm    = z_attach          (moment arm for torque computation)
```

The moment arm `r_arm` is what determines how much torque the sloshing mass exerts on the spacecraft.

---
# Section 2b — The Spring-Mass Model for On-Orbit Satellite Applications

## 2b.1 Why the Spring-Mass Model is More Appropriate for Satellites

The Abramson framework offers two mechanically equivalent representations of the sloshing dynamics: the **pendulum model** and the **spring-mass model**. While both reproduce the same net forces and torques on the tank, they are physically motivated by different operating environments.

The **pendulum model** is natural for launch vehicles and during thruster burn phases. In those cases a strong, well-defined axial acceleration `g_eff` acts along the tank symmetry axis, providing a clear directional restoring force analogous to gravity for a pendulum. The pendulum length `l₁ = g_eff / ω₁²` is well-defined and the physical picture is intuitive.

The **spring-mass model** is more appropriate for on-orbit satellite attitude manoeuvres, for the following reasons:

- During attitude manoeuvres the spacecraft is typically in **coast** — no main thruster firing. The effective axial acceleration is essentially zero (residual accelerations on a GEO satellite are on the order of 10⁻⁶ g). The pendulum analogy breaks down because the pendulum length would diverge.
- The restoring force acting on the free surface is dominated by **surface tension** and **free surface geometry** — not by an inertial gravity-like effect. The spring-mass model captures this as a lateral spring with stiffness `k₁ = m₁ ω₁²`, which depends on tank geometry and fill level but not on g_eff.
- The excitation driving the slosh during attitude manoeuvres is a **lateral inertial force** at the attachment point due to angular acceleration of the spacecraft — a natural fit for a laterally sliding spring-mass oscillator.
- The spring-mass formulation maps directly and cleanly into a standard mechanical second-order system, making it straightforward to incorporate into state space models and control design tools.

---

## 2b.2 Physical Description of the Spring-Mass Model

The spring-mass equivalent model for a single slosh mode consists of:

- A **fixed mass** `m₀` rigidly bolted to the tank at a fixed point — representing the portion of propellant that moves with the spacecraft without oscillation.
- A **sloshing mass** `m₁` free to translate **laterally** (perpendicular to the tank symmetry axis), connected to the tank wall via a linear spring of stiffness `k₁` and a viscous damper with coefficient `c₁`.

The sloshing mass is located at a **fixed axial position** `z₁` along the tank symmetry axis (measured from the tank bottom, equivalent to the attachment height `h₁` from the pendulum model). It can only move **laterally** — this lateral degree of freedom represents the oscillation of the free surface.

```
Tank symmetry axis (z)
        │
        │     ┌─────────────┐  ← tank wall
        │     │             │
        │     │    vapour   │
        │     │             │
z₁ ────│─────│──[m₁]──/\/──│── wall   ← sloshing mass on spring
        │     │      k₁,c₁  │
        │     │             │
        │     │   [m₀]      │  ← fixed mass (rigid)
        │     │             │
        │     └─────────────┘
        │
       CoM of spacecraft
```

The displacement of the sloshing mass relative to its equilibrium position is denoted `δ` (lateral, in metres). At equilibrium, `δ = 0` and the free surface is undisturbed.

---

## 2b.3 Spring and Damper Parameters

**Spring stiffness:**

The spring stiffness is derived directly from the sloshing mass and natural frequency:

```
k₁ = m₁ · ω₁²
```

where `ω₁` is the first mode natural frequency extracted from the Abramson tables for the given tank geometry and fill level. The spring is therefore not a free parameter — it is fully determined by `m₁` and `ω₁`.

**Damping coefficient:**

```
c₁ = 2 · ζ₁ · m₁ · ω₁
```

where `ζ₁` is the modal damping ratio. This is the parameter with the highest uncertainty — typical values are:

| Tank configuration | ζ₁ range |
|---|---|
| Smooth spherical or cylindrical wall | 0.001 – 0.005 |
| Ring baffles (one or two) | 0.01 – 0.03 |
| Multiple baffles or PMD structures | 0.03 – 0.10 |

Conservative practice is to use `ζ₁ = 0.005` (smooth wall) for worst-case stability margin analysis, and a higher value for performance analysis where baffles are confirmed.

---

## 2b.4 Forces and Torques Generated by the Sloshing Mass

The key output of the model — what the GNC engineer needs — is the **force and torque that the sloshing mass exerts on the spacecraft** as a reaction to its motion.

Consider the sloshing mass `m₁` at axial position `z₁` from spacecraft CoM, displaced laterally by `δ`. Its equation of motion in an inertial frame is:

```
m₁ · (ẍ_sc,lateral + δ̈) = −k₁ · δ − c₁ · δ̇
```

where `ẍ_sc,lateral` is the lateral acceleration of the attachment point due to spacecraft rotation.

By Newton's third law, the reaction force that the sloshing mass exerts **on the spacecraft** in the lateral direction is:

```
F_slosh = m₁ · δ̈  (lateral force on spacecraft, in body frame)
```

And since the sloshing mass is located at axial distance `r_arm = z₁` from the spacecraft CoM, it generates a **reaction torque** (about the pitch or yaw axis, depending on the lateral direction of slosh):

```
T_slosh = −m₁ · r_arm · δ̈
```

This torque acts on the spacecraft and couples directly into the attitude dynamics — it is the term that makes sloshing a GNC problem.

---

## 2b.5 Two-Dimensional Lateral Sloshing

In reality, the sloshing mass can move in **both lateral directions** simultaneously — in the body x and y axes (the two axes perpendicular to the tank symmetry z axis). For a cylindrical or spherical tank, the first mode is degenerate in azimuth: the slosh frequency is the same in any lateral direction.

This means the full spring-mass model for one tank and one mode has **two lateral degrees of freedom**:

```
δ = [δₓ, δᵧ]ᵀ
```

Each component obeys an independent scalar oscillator equation:

```
δ̈ₓ + 2ζ₁ω₁ · δ̇ₓ + ω₁² · δₓ = −r_arm · θ̈ᵧ  (slosh in x driven by pitch rate)
δ̈ᵧ + 2ζ₁ω₁ · δ̇ᵧ + ω₁² · δᵧ = −r_arm · θ̈ₓ  (slosh in y driven by roll rate)
```

where `θₓ` and `θᵧ` are the roll and pitch angles respectively. The two lateral slosh directions are decoupled from each other but each is driven by the spacecraft rotation in the perpendicular plane.

For a simplified single-axis analysis (as implemented in the Python simulation), only one lateral direction is retained. For a full 3-axis AOCS model, both are included, adding **four slosh states per tank per mode** (δₓ, δ̇ₓ, δᵧ, δ̇ᵧ).

---

## 2b.6 Full Single-Axis Equations of Motion (Spring-Mass)

For completeness, the full single-axis coupled equations are restated here in the spring-mass formulation, making the physical terms explicit.

**Spacecraft pitch equation:**

```
I_sc · θ̈ = T_ctrl + T_slosh
           = T_ctrl − m₁ · r_arm · δ̈
```

Rearranging to make `θ̈` explicit:

```
I_sc · θ̈ = T_ctrl + m₁ · r_arm · (−2ζ₁ω₁ · δ̇ − ω₁² · δ − r_arm · θ̈)

⟹  (I_sc + m₁ · r_arm²) · θ̈ = T_ctrl − m₁ · r_arm · (2ζ₁ω₁ · δ̇ + ω₁² · δ)

⟹  θ̈ = [T_ctrl + m₁ · r_arm · (2ζ₁ω₁ · δ̇ + ω₁² · δ)] / I_sc
```

Note: the term `m₁ · r_arm²` represents the additional inertia contribution of the sloshing mass when it moves with the spacecraft — it slightly increases the effective rotational inertia.

**Slosh oscillator equation:**

```
δ̈ = −r_arm · θ̈ − 2ζ₁ω₁ · δ̇ − ω₁² · δ
```

Substituting `θ̈`:

```
δ̈ = −(r_arm / I_sc) · [T_ctrl + m₁ · r_arm · (2ζ₁ω₁ · δ̇ + ω₁² · δ)]
     − 2ζ₁ω₁ · δ̇ − ω₁² · δ

δ̈ = −(r_arm / I_sc) · T_ctrl
     − (2ζ₁ω₁ + m₁ · r_arm² · 2ζ₁ω₁ / I_sc) · δ̇
     − (ω₁² + m₁ · r_arm² · ω₁² / I_sc) · δ
```

These are the explicit scalar equations that are numerically integrated in the simulation.

---

## 2b.7 State Space Matrices (Spring-Mass, Single Axis)

The state vector is:

```
x = [θ,  θ̇,  δ,  δ̇]ᵀ
```

**System matrix A:**

```
A = ⎡  0          1                    0                             0             ⎤
    ⎢  0          0          m₁·r·ω₁²/I_sc              m₁·r·2ζ₁ω₁/I_sc          ⎥
    ⎢  0          0                    0                             1             ⎥
    ⎣  0          0    −(ω₁²+m₁·r²·ω₁²/I_sc)    −(2ζ₁ω₁+m₁·r²·2ζ₁ω₁/I_sc)     ⎦
```

where `r = r_arm` for brevity.

**Input matrix B** (input: control torque `T_ctrl`):

```
B = [0,   1/I_sc,   0,   −r/I_sc]ᵀ
```

**Output matrix C** (output: attitude angle `θ`):

```
C = [1,   0,   0,   0]
```

**Direct feedthrough D:**

```
D = [0]
```

These are exactly the matrices implemented in the Python simulation script.

---

## 2b.8 Effect of Multiple Tanks

A real bipropellant satellite has at least two tanks — one for oxidiser (e.g. N₂O) and one for fuel (e.g. C₃H₆). Each tank contributes its own independent set of slosh states. For `N_tanks` tanks each modelled with the first mode in two lateral directions, the total number of slosh states added to the spacecraft rigid body model is:

```
N_slosh_states = N_tanks × 2 directions × 2 states (δ, δ̇) = N_tanks × 4
```

For a two-tank system this adds 8 slosh states, giving a total plant model of dimension 12 (4 rigid body + 8 slosh) for a full 3-axis simulation — manageable in any modern simulation environment.

Each tank has its own parameter set extracted independently via Abramson, and the fill fractions of the two tanks evolve differently depending on mixture ratio and consumption history.

---

## 2b.9 Physical Interpretation: What the Spring-Mass Model Is Saying

It is worth stepping back to understand what this model is physically capturing.

When the spacecraft rotates, the tank rotates with it. The attachment point of the sloshing mass — fixed to the tank wall — accelerates laterally. This lateral acceleration acts as a forcing input to the slosh oscillator, like pushing a pendulum from its support point. If the forcing frequency (related to the manoeuvre rate) is close to ω₁, resonance can occur and the slosh amplitude grows.

The spring represents the restoring action of the free surface: when the liquid is pushed to one side, the free surface tilts, creating a pressure gradient that pushes it back. The spring stiffness `k₁ = m₁ ω₁²` encodes how stiff this restoring action is — higher fill levels and larger tanks generally mean lower ω₁ and therefore a softer effective spring.

The damper represents viscous dissipation in the bulk liquid and at the tank walls. Without damping (ζ₁ = 0), once excited the slosh would oscillate indefinitely. In reality ζ₁ is small but non-zero even for smooth walls, and baffles increase it substantially.

The reaction torque `T_slosh = −m₁ · r_arm · δ̈` is what closes the loop back onto the spacecraft: the sloshing mass accelerating laterally inside the tank is equivalent to an unbalanced internal force, which by Newton's third law must be reacted by the tank walls and hence the spacecraft structure. If `r_arm` is large (tank far from CoM) and `m₁` is large (mid-range fill), this reaction torque can be a significant fraction of the available control torque — which is precisely why sloshing becomes a first-order concern for GNC on large propellant-loaded satellites.

---


## 3. Equations of Motion and State Space Formulation

### 3.1 Coupled Dynamics

With the Abramson equivalent mechanical model established, the sloshing mass is added as an additional degree of freedom to the spacecraft rigid body equations of motion. The following derivation is for a **single-axis linearised model** (pitch axis), with the slosh mass displaced laterally (perpendicular to the tank symmetry axis).

**State variables:**

| Variable | Description |
|---|---|
| θ | Spacecraft pitch angle [rad] |
| θ̇ | Spacecraft pitch rate [rad/s] |
| δ | Slosh mass lateral displacement [m] |
| δ̇ | Slosh mass lateral velocity [m/s] |

### 3.2 Derivation of the Coupled Equations

The spacecraft is treated as a rigid body with pitch inertia `I_sc`. The sloshing mass `m₁` is located at moment arm `r_arm` from the spacecraft CoM.

**Angular momentum of the system:**

The total angular momentum includes the rigid body term and the contribution from the sloshing mass moving relative to the CoM:

```
H_total = I_sc · θ̇ + m₁ · r_arm · δ̇
```

**Spacecraft rotational equation** (Newton-Euler, about CoM):

```
I_sc · θ̈ = T_ctrl - m₁ · r_arm · δ̈
```

The sloshing mass reaction torque `m₁ · r_arm · δ̈` acts back on the spacecraft — this is the coupling term.

**Slosh oscillator equation** (forced damped harmonic oscillator):

The sloshing mass is driven by the lateral acceleration at its attachment point due to spacecraft angular motion:

```
δ̈ + 2ζ₁ω₁ · δ̇ + ω₁² · δ = −r_arm · θ̈
```

The right-hand side is the inertial forcing: when the spacecraft rotates, the attachment point accelerates laterally, driving the slosh mass.

**Substituting** the spacecraft equation into the slosh equation to obtain explicit form:

From the spacecraft equation:
```
θ̈ = (T_ctrl + m₁ · r_arm · (2ζ₁ω₁ · δ̇ + ω₁² · δ)) / I_sc
```

Substituting back:
```
δ̈ = −r_arm · θ̈ − 2ζ₁ω₁ · δ̇ − ω₁² · δ
```

These two equations form the coupled system. Note that the slosh feeding back into the spacecraft equation appears through the terms `2ζ₁ω₁ · δ̇` and `ω₁² · δ` — the slosh restoring and damping forces create reaction torques on the spacecraft.

### 3.3 State Space Formulation

The coupled system is cast into standard state space form:

```
ẋ = A · x + B · u
y = C · x
```

with state vector:
```
x = [θ,  θ̇,  δ,  δ̇]ᵀ
```

input:
```
u = T_ctrl   (control torque)
```

output:
```
y = θ        (attitude angle, for sensor feedback)
```

**System matrix A:**

```
A = [ 0,    1,                        0,                          0        ]
    [ 0,    0,    m₁·r_arm·ω₁²/I_sc,    m₁·r_arm·2ζ₁ω₁/I_sc              ]
    [ 0,    0,                        0,                          1        ]
    [ 0,    0,    −(ω₁² + m₁·r_arm²·ω₁²/I_sc),   −(2ζ₁ω₁ + m₁·r_arm²·2ζ₁ω₁/I_sc) ]
```

**Input matrix B:**

```
B = [ 0,   1/I_sc,   0,   −r_arm/I_sc ]ᵀ
```

**Output matrix C:**

```
C = [ 1,   0,   0,   0 ]
```

### 3.4 Interpretation of the State Space Model

The eigenvalues of `A` reveal the system dynamics. There are two pairs of complex conjugate poles:

- **Rigid body poles** at the origin (double integrator — pure attitude kinematics, marginally stable, stabilised by the controller)
- **Slosh poles** at `s = −ζ₁ω₁ ± jω₁√(1−ζ₁²)` — lightly damped oscillatory poles at the slosh frequency

With ζ₁ = 0.005, the slosh poles are very lightly damped. On a Bode plot of the open-loop plant (θ/T_ctrl), these appear as a sharp resonance peak at ω₁ — potentially large in magnitude if damping is low.

### 3.5 Parameter Scheduling Over Mission Life

Because the Abramson parameters (m₁, ω₁, ζ₁, r_arm) are all functions of fill fraction, the matrices A and B are **time-varying** over the mission. The model is therefore a **Linear Parameter Varying (LPV)** system, where the scheduling variable is the current propellant mass remaining.

In the AOCS simulator, this is implemented as a **parameter-scheduled block** that looks up the current Abramson parameters from a pre-computed table (indexed by fill fraction) and updates the state matrices at each simulation time step.

### 3.6 Validity Domain

The entire formulation above is **linearised** — it assumes small slosh displacements δ relative to the tank radius R. If a large manoeuvre excites vigorous sloshing that violates this assumption, the model breaks down. In practice, for typical on-orbit AOCS manoeuvres (small station-keeping corrections, modest attitude slews), the small-angle assumption holds. For aggressive manoeuvres, rate limiting or input shaping may be required to keep slosh amplitudes within the linear regime.

### 3.7 Control Law and Closed Loop

For the simulation, a simple **PD attitude controller** is used:

```
T_ctrl = −Kp · θ − Kd · θ̇
```

This closes the loop from the attitude state back to the control torque. The closed-loop system then has four states, and the GNC engineer's task is to choose Kp and Kd such that:

- The attitude error is driven to zero with acceptable settling time and overshoot
- The closed-loop gain at the slosh frequency ω₁ is sufficiently low that the slosh mode is not excited — **frequency separation** between control bandwidth and slosh frequency
- Gain and phase margins are adequate across the full fill fraction sweep

---

## 4. Xenon - Why Sloshing Can Be Neglected

### 4.1 Storage State of Xenon on GEO Satellites

On modern geostationary telecom satellites equipped with electric propulsion (ion thrusters or Hall effect thrusters), xenon is the working propellant. It is stored in a **supercritical state** — typically at pressures of 75–300 bar and temperatures between 20°C and 50°C.

The critical point of xenon is:

| Property | Value |
|---|---|
| Critical temperature | 289.73 K (16.6°C) |
| Critical pressure | 58.4 bar |
| Critical density | ~1155 kg/m³ |

As long as the tank conditions remain simultaneously above the critical temperature **and** critical pressure, xenon exists as a single-phase supercritical fluid. There is no liquid-gas interface, no meniscus, and no free surface.

### 4.2 Why No Free Surface Means No Sloshing

Classical sloshing — as described by Abramson and modelled throughout this document — is a **free surface phenomenon**. The restoring force that gives rise to oscillation is surface tension and gravitational (or inertial) effects acting on the liquid-gas interface. Remove the interface, and the physical mechanism for sloshing disappears entirely.

A supercritical fluid fills the tank uniformly as a single homogeneous phase. Any perturbation to the tank propagates as a pressure wave through the fluid — a compressible acoustic phenomenon — not as a surface oscillation. The acoustic frequencies are orders of magnitude higher than any AOCS bandwidth, and the associated forces are negligible.

**Consequence for plant modelling:** The xenon mass can be treated as a **fixed, rigidly attached mass** in the spacecraft dynamics model. It contributes to the total inertia tensor and shifts the centre of mass, but adds no oscillatory degrees of freedom. No Abramson model is needed, and no slosh states are added to the state space.

### 4.3 The Role of Tank Heaters

The tank heater system is what guarantees the supercritical condition throughout the mission. The heaters maintain the xenon temperature above the critical temperature, ensuring that even as xenon is slowly consumed and tank pressure decreases, the thermodynamic state never crosses into the two-phase region.

For this to be a reliable assumption in the GNC model, the heater system must satisfy:

- **Redundancy:** Multiple heater circuits such that a single heater failure does not cause the xenon to cool below the critical temperature
- **Monitoring:** On-board temperature and pressure telemetry with limits that trigger safe mode if conditions approach the critical point
- **Thermal design:** Adequate insulation and heater sizing to maintain temperature over the full mission lifetime and across all orbital thermal environments (eclipse, solstice, equinox)

If these conditions are met, the plant modeller can legitimately exclude xenon sloshing from the AOCS model and simply note the assumption in the model documentation.

### 4.4 Failure Mode Caveat

If heater failure causes xenon to drop below the critical point, a two-phase liquid-vapour system with a free surface could in principle form. However, this is treated as a **declared failure mode** at system level. The spacecraft would enter safe mode before any dynamic consequence becomes a GNC stability concern. It is not modelled in the nominal AOCS simulation.

---

## 5. C₃H₆ and N₂O - Why Sloshing Must Be Considered

### 5.1 Nature of These Propellants

Propylene (C₃H₆) and nitrous oxide (N₂O) are a green bipropellant combination — less toxic than the classical MMH/NTO pair. A fundamental characteristic of both is that they are **liquefied gases** stored at **saturation conditions**: liquid and vapour coexist in thermodynamic equilibrium at ambient temperature, with the tank self-pressurised by the vapour pressure.

This is in direct contrast to xenon, which is forced into a supercritical state. For C₃H₆ and N₂O, a genuine **free liquid surface** exists inside the tank at all times during normal operation.

Key thermodynamic properties at ~20°C:

| Property | N₂O (Oxidiser) | C₃H₆ (Fuel) |
|---|---|---|
| Critical temperature | 36.4°C | 91.8°C |
| Critical pressure | 72.5 bar | 46.6 bar |
| Liquid density at 20°C | ~745 kg/m³ | ~515 kg/m³ |
| Vapour pressure at 20°C | ~50 bar | ~10 bar |
| Vapour density at 20°C | ~170 kg/m³ | ~20 kg/m³ |

Both propellants are well below their critical temperatures at typical on-orbit storage conditions — N₂O by only ~16°C, and C₃H₆ by ~72°C. Both therefore exist in a two-phase liquid-vapour state with a free surface, and Abramson's modelling framework applies as a first approximation.

### 5.2 Applying the Abramson Model

For a first-pass slosh model, the Abramson cylindrical or spherical tank formulas from Section 2 are applied using the **liquid phase density** of the respective propellant to compute fill height and total liquid mass. The same parameter extraction procedure applies:

1. Compute fill height `h` from propellant liquid mass, liquid density, and tank radius
2. Compute fill ratio `ξ = h/R`
3. Extract ω₁, m₁, m₀, h₁ from Abramson formulas
4. Convert h₁ to spacecraft body frame
5. Assemble state space model as in Section 3

This delivers a conservative, tractable first-mode equivalent model suitable for handover to the GNC engineer for stability analysis.

### 5.3 Complications Specific to Self-Pressurising Propellants

While the Abramson framework gives a useful first approximation, two physical effects are present in N₂O and C₃H₆ tanks that are **not captured by the classical model**:

**5.3.1 Non-negligible vapour density**

The Abramson model assumes the gas above the free surface has negligible density compared to the liquid — an excellent assumption for classical propellants like MMH (liquid ~880 kg/m³, vapour ~negligible) but less valid here.

For N₂O at 20°C, the vapour-to-liquid density ratio is approximately 1:4.4. This is not negligible. The dense vapour phase contributes inertia that partially opposes the free surface motion, effectively reducing the sloshing mass and shifting the slosh frequency relative to the pure liquid prediction. The classical Abramson model will therefore **overestimate the sloshing mass** and may mispredict the frequency — a conservative error from a stability margin perspective, but an error nonetheless.

**5.3.2 Thermodynamic coupling**

Because both propellants are at saturation, the free surface is in continuous thermodynamic equilibrium between liquid and vapour phases. A sloshing event that disturbs the free surface creates local pressure and temperature perturbations, which drive **evaporation or condensation** at the interface. This thermodynamic exchange damps the slosh in ways not captured by the mechanical damping ratio ζ₁, and it couples the slosh dynamics to the tank thermal model.

This effect is particularly relevant for N₂O, which has a high vapour pressure and relatively dense vapour — perturbations to the free surface equilibrium are more energetic than for C₃H₆.

### 5.4 Recommendations for Plant Modelling

Given the above, the recommended approach for N₂O and C₃H₆ tanks is:

**First pass — Abramson model with conservative assumptions:**
- Use liquid phase density to compute fill height and sloshing mass
- Apply standard cylindrical Abramson formulas for ω₁, m₁, h₁
- Use a conservatively low damping ratio (ζ₁ = 0.005, smooth wall assumption) to be pessimistic about damping
- Flag explicitly in model documentation that vapour density correction and thermodynamic coupling are not captured

**Refinement if needed:**
- For large tanks or cases where the GNC stability margins are tight, commission CFD analysis with a two-phase Volume of Fluid (VOF) solver to extract corrected equivalent model parameters
- Alternatively, fit equivalent model parameters to shake-table test data on representative partially filled tanks

**Handover documentation must state:**
- The propellant is a two-phase self-pressurising liquid — free surface exists, sloshing is relevant
- Abramson model is applied as first approximation using liquid density
- Vapour density correction not applied — results are conservative (sloshing mass likely overestimated)
- Thermodynamic coupling not modelled — additional damping beyond ζ₁ may exist in reality
- GNC engineer should apply appropriate uncertainty margins on ω₁ and ζ₁

### 5.5 Comparison Summary: Xenon vs C₃H₆ vs N₂O

| Property | Xenon (EP) | N₂O (Oxidiser) | C₃H₆ (Fuel) |
|---|---|---|---|
| Storage state | Supercritical fluid | Two-phase liquid-vapour | Two-phase liquid-vapour |
| Free surface present? | No | Yes | Yes |
| Sloshing relevant? | No | Yes | Yes |
| Abramson applicable? | No | Yes (with caveats) | Yes (with caveats) |
| Liquid density | ~1155 kg/m³ (supercritical) | ~745 kg/m³ | ~515 kg/m³ |
| Key complication | Maintain supercritical via heaters | Dense vapour phase, thermodynamic coupling | Lower vapour density, still two-phase |
| Plant model recommendation | Fixed rigid mass | First-mode Abramson + flag caveats | First-mode Abramson + flag caveats |

---

*End of document.*
