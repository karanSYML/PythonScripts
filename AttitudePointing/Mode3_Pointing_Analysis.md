# Earth+Target Pointing Feasibility Analysis

**Endurance FR RDV — GEO Rendezvous Mission**
**Sep 1–24, 2028 | ~23 days | −60 km to +1 km along-track**

---

## Context

During the far-range rendezvous, the spacecraft cycles through four attitude pointing combinations (modes). Simultaneous Earth pointing for orbit determination and target pointing for inspection, is required every 6 hours per ConOps, with a 25-minute window each time.

The spacecraft geometry constraining this problem:

- **Camera boresight**: +Z body axis (for target pointing)
- **S-band antenna**: +Y and −Y body panels (for Earth comms)
- **Camera and antenna are 90° apart** on the spacecraft body

The question becomes can we satisfy both constraints simultaneously, and what pointing errors result?

## Earth–Target Angular Separation

The first step is understanding the geometrical relationship between the two pointing targets, Earth and the rendezvous target, as seen from the spacecraft. This is purely a function of the orbit geometry and is independent of the spacecraft attitude.

At GEO, even though the along-track separation between the spacecraft and target is small (1–60 km), the Earth is straight "down". Meanwhile, the target is tens of kilometres off to the side. The angle between the two directions as seen from the spacecraft is therefore dominated by the orbital geometry and not the inter-spacecraft range.

![Earth–Target angular separation](fig1_earth_target_separation.png)

The separation oscillates between **~40° and ~175°** with a period of approximately one day (the GEO orbit period), centred around ~90°. This oscillation is driven by the spacecraft's orbital motion: the relative position of the target rotates in the LVLH frame over each orbit.

Interesting thing about 90° mean is that it happens to match the fixed angle between the camera (at Z+) and antenna (+/- Y) on the spacecraft body. 

## Antenna Pointing Feasibility

With the camera locked exactly on the target (+Z → target), the remaining degree of freedom is rotation about the camera boresight. This rotation is used to swing +Y as close to the Earth direction as possible. The residual antenna pointing error is:

**Antenna error = |90° − Earth-Target angular separation|**

which follows directly from the 90° fixed angle between camera and antenna on the body.

![Antenna pointing feasibility](fig2_antenna_feasibility.png)

The results split into two clear regimes:

**Far range (Days 0–13, −60 to −5 km):** Mode 3 is highly feasible. The worst antenna error at any 6-hour window is ~8°. The Earth-target separation stays close to 90°, meaning the geometry naturally aligns with the spacecraft hardware layout. All four daily windows work with minimal pointing loss.

**Close range (Days 14–22, −5 to +1 km):** Feasibility degrades. The antenna error at some windows climbs to 30–43°, exceeding any reasonable S-band beamwidth. During these phases, not all four 6-hour windows can simultaneously achieve acceptable antenna pointing. The ConOps may need to shift Mode 3 slots to the lower-error windows within each day, or accept degraded link margin at certain passes.

Overall feasibility across the full timeline:

| Antenna error threshold | Timeline feasibility |
|------------------------|---------------------|
| ≤ 5° | 67% |
| ≤ 10° | 76% |
| ≤ 20° | 89% |
| ≤ 30° | 93% |
| ≤ 45° | 97% |

## Optimal slew rate: Which Mode to Transition From?

Mode 3 is entered from either Mode 1 (Target+Sun) or Mode 2 (Nadir+Sun). The slew angle, the total rotation required to transition into Mode 3, determines how much of the 25-minute window is consumed by the attitude manoeuvre.

![Slew analysis](fig3_slew_analysis.png)

**Mode 1 (Target+Sun) → Mode 3 is cheaper 63% of the time.** This is expected: Mode 1 already has +Z on the target, so the transition is essentially a yaw rotation to swing +Y toward Earth while keeping the camera locked. The Sun-optimisation constraint is released, but the primary pointing axis doesn't move.

**Mode 2 (Nadir+Sun) → Mode 3 is more expensive** because +Z must be repointed from nadir to the target (~90° apart at GEO range), plus +Y must be rotated toward Earth. This is a large-angle manoeuvre.

| Metric | Mode 1 → Mode 3 | Mode 2 → Mode 3 |
|--------|-----------------|-----------------|
| Mean slew | 97° | 135° |
| Min slew | 0° | 23° |
| Max slew | 180° | 180° |

**ConOps implication:** If the spacecraft is in Mode 1 before an orbit determination window, the slew into and out of Mode 3 consumes less of the 25-minute budget. If transitioning from Mode 2, the 5-minute slew allocation in the ConOps (5 min slew + 15 min comms + 5 min slew) may be insufficient for some windows, particularly during the close-range phases.

## Key Takeaways

1. **The 90° camera-antenna geometry is well-matched to the mean Earth-target separation** at GEO, making Mode 3 feasible for most of the far-range approach without significant pointing loss.

2. **Close-range phases (< 5 km) require careful window selection.** Not all four daily 6-hour windows will achieve acceptable antenna pointing. The ConOps should identify the 2–3 best windows per day during these phases.

3. **Transition from Mode 1 is preferred** when entering Mode 3, as the slew is smaller and preserves the camera-on-target constraint throughout.

4. **The antenna boresight directions (+Y/−Y) and beamwidth** are the critical parameters for finalising Mode 3 feasibility thresholds. The analysis above uses +Y only; the −Y antenna does not improve the result for the geometries encountered in this mission.

## Open Items

- Confirm S-band antenna beamwidth (half-cone angle) to set the feasibility threshold
- Confirm camera and antenna boresight directions with the platform team
- Evaluate whether Mode windows can be shifted within the 6-hour cycle to avoid high-error periods during close-range phases
- Assess Mode 4 (DV+Sun) slew costs once the manoeuvre plan is finalised
