"""
openmdao_geo_ssa_sizing.py

V0 OpenMDAO sizing exercise:
- GEO worst-case day with a fixed eclipse block (02:00–03:10)
- SSA camera imaging + downlink sessions
- Size: solar array area A_SA [m^2], battery usable energy E_bat [Wh], prop mass m_prop [kg]
- Constraints: SOC_min >= 0.20, delta_v >= delta_v_req
- Objective: minimize total mass

Run:
  python openmdao_geo_ssa_sizing.py
"""

import numpy as np
import openmdao.api as om


# ----------------------------
# Helpers (time/mode masks)
# ----------------------------
def build_time_grid(dt_min=2, hours=24):
    n = int(hours * 60 / dt_min)
    t_min = np.arange(n) * dt_min
    t_hr = t_min / 60.0
    return t_hr, dt_min


def mask_interval(t_hr, start_hr, duration_min):
    end_hr = start_hr + duration_min / 60.0
    return (t_hr >= start_hr) & (t_hr < end_hr)


def build_masks(t_hr):
    # Eclipse: 02:00–03:10 (70 min)
    eclipse = mask_interval(t_hr, start_hr=2.0, duration_min=70).astype(float)

    # Imaging: 4 sessions/day, 10 min each at 06:00, 10:00, 14:00, 18:00
    img = np.zeros_like(t_hr, dtype=float)
    for h in (6.0, 10.0, 14.0, 18.0):
        img[mask_interval(t_hr, start_hr=h, duration_min=10)] = 1.0

    # Downlink: 2 sessions/day, 20 min each at 12:00, 20:00
    tx = np.zeros_like(t_hr, dtype=float)
    for h in (12.0, 20.0):
        tx[mask_interval(t_hr, start_hr=h, duration_min=20)] = 1.0

    # Safety: if you want to forbid imaging/TX during eclipse, uncomment:
    # img = img * (1.0 - eclipse)
    # tx = tx * (1.0 - eclipse)

    return eclipse, img, tx


# ----------------------------
# OpenMDAO Components
# ----------------------------
class LoadsComp(om.ExplicitComponent):
    """
    Computes P_load(t) given fixed schedules (masks) and mode powers.

    P_load = P_base + img_mask*(P_cam + P_adcs_img) + tx_mask*(P_tx + P_adcs_tx)
    """
    def initialize(self):
        self.options.declare("n", types=int)
        self.options.declare("P_base", default=120.0)
        self.options.declare("P_cam", default=60.0)
        self.options.declare("P_adcs_img", default=80.0)
        self.options.declare("P_tx", default=120.0)
        self.options.declare("P_adcs_tx", default=20.0)

    def setup(self):
        n = self.options["n"]
        self.add_input("img_mask", val=np.zeros(n))
        self.add_input("tx_mask", val=np.zeros(n))
        self.add_output("P_load", val=np.ones(n) * self.options["P_base"])

        # Partials: P_load depends linearly on masks
        self.declare_partials("P_load", "img_mask")
        self.declare_partials("P_load", "tx_mask")

    def compute(self, inputs, outputs):
        P_base = self.options["P_base"]
        P_img_add = self.options["P_cam"] + self.options["P_adcs_img"]
        P_tx_add = self.options["P_tx"] + self.options["P_adcs_tx"]
        outputs["P_load"] = P_base + inputs["img_mask"] * P_img_add + inputs["tx_mask"] * P_tx_add

    def compute_partials(self, inputs, partials):
        P_img_add = self.options["P_cam"] + self.options["P_adcs_img"]
        P_tx_add = self.options["P_tx"] + self.options["P_adcs_tx"]
        partials["P_load", "img_mask"] = np.ones_like(inputs["img_mask"]) * P_img_add
        partials["P_load", "tx_mask"] = np.ones_like(inputs["tx_mask"]) * P_tx_add


class SolarGenComp(om.ExplicitComponent):
    """
    Computes P_gen(t) from A_SA and a simplified GEO/EOL irradiance model.

    Outside eclipse:
      P_gen = A_SA * S * eta * f_EOL * f_misc * cos_theta
    In eclipse:
      P_gen = 0
    """
    def initialize(self):
        self.options.declare("n", types=int)
        self.options.declare("S", default=1361.0)          # W/m^2
        self.options.declare("eta", default=0.28)          # cell efficiency
        self.options.declare("f_EOL", default=0.77)
        self.options.declare("f_misc", default=0.90)
        self.options.declare("cos_theta", default=0.90)    # fixed incidence factor outside eclipse

    def setup(self):
        n = self.options["n"]
        self.add_input("A_SA", val=2.0)                    # m^2
        self.add_input("eclipse_mask", val=np.zeros(n))    # 1 in eclipse, else 0
        self.add_output("P_gen", val=np.zeros(n))          # W

        self.declare_partials("P_gen", "A_SA")
        self.declare_partials("P_gen", "eclipse_mask")

    def compute(self, inputs, outputs):
        A = inputs["A_SA"]
        eclipse = inputs["eclipse_mask"]
        k = (
            self.options["S"]
            * self.options["eta"]
            * self.options["f_EOL"]
            * self.options["f_misc"]
            * self.options["cos_theta"]
        )  # W/m^2 outside eclipse
        outputs["P_gen"] = (1.0 - eclipse) * (A * k)

    def compute_partials(self, inputs, partials):
        eclipse = inputs["eclipse_mask"]
        k = (
            self.options["S"]
            * self.options["eta"]
            * self.options["f_EOL"]
            * self.options["f_misc"]
            * self.options["cos_theta"]
        )
        partials["P_gen", "A_SA"] = (1.0 - eclipse) * k
        partials["P_gen", "eclipse_mask"] = -(inputs["A_SA"] * k) * np.ones_like(eclipse)


class BatterySOCComp(om.ExplicitComponent):
    """
    Propagates battery energy E(t) and returns SOC_min.

    Uses:
      E(0) = E_bat
      P_net = P_gen - P_load
      if P_net >= 0: charge with eta_c
      else: discharge with eta_d
      E_next = clip(E + eta_c*P_net*dt, 0, E_bat) for P_net>=0
      E_next = clip(E + (P_net/eta_d)*dt, 0, E_bat) for P_net<0
    (dt in hours, power in W => Wh)

    Note: This is piecewise and not fully smooth. For v0, use a derivative-free optimizer.
    """
    def initialize(self):
        self.options.declare("n", types=int)
        self.options.declare("dt_hr", types=float)
        self.options.declare("eta_c", default=0.95)
        self.options.declare("eta_d", default=0.95)

    def setup(self):
        n = self.options["n"]
        self.add_input("E_bat", val=1200.0)    # Wh (usable)
        self.add_input("P_gen", val=np.zeros(n))
        self.add_input("P_load", val=np.zeros(n))
        self.add_output("SOC_min", val=1.0)
        self.add_output("SOC_end", val=1.0)

        # For v0, use finite-difference for this component (piecewise + loop)
        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs):
        E_bat = float(inputs["E_bat"])
        P_gen = inputs["P_gen"]
        P_load = inputs["P_load"]
        dt = self.options["dt_hr"]
        eta_c = self.options["eta_c"]
        eta_d = self.options["eta_d"]

        E = E_bat
        soc_min = 1.0

        for i in range(len(P_gen)):
            P_net = float(P_gen[i] - P_load[i])  # W
            if P_net >= 0.0:
                dE = eta_c * P_net * dt
            else:
                dE = (P_net / eta_d) * dt  # P_net is negative
            E = E + dE
            if E > E_bat:
                E = E_bat
            elif E < 0.0:
                E = 0.0

            soc = E / E_bat if E_bat > 0 else 0.0
            if soc < soc_min:
                soc_min = soc

        outputs["SOC_min"] = soc_min
        outputs["SOC_end"] = E / E_bat if E_bat > 0 else 0.0


class MassDeltaVComp(om.ExplicitComponent):
    """
    Computes dry mass, total mass, and delta-v capability.

    m_dry = m_bus + k_SA*A_SA + k_bat*E_bat
    delta_v = Isp*g0*ln((m_dry + m_prop)/m_dry)
    """
    def initialize(self):
        self.options.declare("m_bus", default=120.0)
        self.options.declare("k_SA", default=3.5)       # kg/m^2
        self.options.declare("k_bat", default=0.008)    # kg/Wh
        self.options.declare("Isp", default=220.0)      # s
        self.options.declare("g0", default=9.80665)     # m/s^2

    def setup(self):
        self.add_input("A_SA", val=2.0)
        self.add_input("E_bat", val=1200.0)
        self.add_input("m_prop", val=5.0)

        self.add_output("m_dry", val=0.0)
        self.add_output("m_tot", val=0.0)
        self.add_output("delta_v", val=0.0)

        self.declare_partials("m_dry", ["A_SA", "E_bat"])
        self.declare_partials("m_tot", ["A_SA", "E_bat", "m_prop"])
        self.declare_partials("delta_v", ["A_SA", "E_bat", "m_prop"])

    def compute(self, inputs, outputs):
        m_bus = self.options["m_bus"]
        k_SA = self.options["k_SA"]
        k_bat = self.options["k_bat"]
        Isp = self.options["Isp"]
        g0 = self.options["g0"]

        A = float(inputs["A_SA"])
        E = float(inputs["E_bat"])
        mp = float(inputs["m_prop"])

        m_dry = m_bus + k_SA * A + k_bat * E
        m0 = m_dry + mp
        m1 = m_dry

        # Avoid invalid log (shouldn't happen with bounds, but keep safe)
        if m0 <= m1 or m1 <= 0:
            dv = 0.0
        else:
            dv = Isp * g0 * np.log(m0 / m1)

        outputs["m_dry"] = m_dry
        outputs["m_tot"] = m0
        outputs["delta_v"] = dv

    def compute_partials(self, inputs, partials):
        k_SA = self.options["k_SA"]
        k_bat = self.options["k_bat"]
        Isp = self.options["Isp"]
        g0 = self.options["g0"]

        A = float(inputs["A_SA"])
        E = float(inputs["E_bat"])
        mp = float(inputs["m_prop"])

        m_dry = self.options["m_bus"] + k_SA * A + k_bat * E
        m0 = m_dry + mp
        m1 = m_dry

        # m_dry partials
        partials["m_dry", "A_SA"] = k_SA
        partials["m_dry", "E_bat"] = k_bat

        # m_tot partials
        partials["m_tot", "A_SA"] = k_SA
        partials["m_tot", "E_bat"] = k_bat
        partials["m_tot", "m_prop"] = 1.0

        # delta_v partials (handle safe)
        if m0 <= m1 or m1 <= 0:
            partials["delta_v", "A_SA"] = 0.0
            partials["delta_v", "E_bat"] = 0.0
            partials["delta_v", "m_prop"] = 0.0
            return

        # dv = Isp*g0*ln(m0/m1), where m0=m1+mp
        # d/dm1 ln((m1+mp)/m1) = d/dm1 [ln(m1+mp) - ln(m1)] = 1/(m1+mp) - 1/m1
        # d/dmp ln((m1+mp)/m1) = 1/(m1+mp)
        dlog_dm1 = (1.0 / (m1 + mp)) - (1.0 / m1)
        dlog_dmp = 1.0 / (m1 + mp)

        coeff = Isp * g0
        ddv_dm1 = coeff * dlog_dm1
        ddv_dmp = coeff * dlog_dmp

        # chain to A_SA and E_bat via m1=m_dry
        partials["delta_v", "A_SA"] = ddv_dm1 * k_SA
        partials["delta_v", "E_bat"] = ddv_dm1 * k_bat
        partials["delta_v", "m_prop"] = ddv_dmp


# ----------------------------
# Build + run OpenMDAO problem
# ----------------------------
def main():
    # Time grid + masks
    t_hr, dt_min = build_time_grid(dt_min=2, hours=24)
    dt_hr = dt_min / 60.0
    n = len(t_hr)

    eclipse_mask, img_mask, tx_mask = build_masks(t_hr)

    # OpenMDAO model
    prob = om.Problem()
    model = prob.model

    # Design variables as IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output("A_SA", val=2.0)        # m^2
    ivc.add_output("E_bat", val=1200.0)    # Wh
    ivc.add_output("m_prop", val=6.0)      # kg

    # Fixed schedules (masks) as outputs too
    ivc.add_output("eclipse_mask", val=eclipse_mask)
    ivc.add_output("img_mask", val=img_mask)
    ivc.add_output("tx_mask", val=tx_mask)

    model.add_subsystem("ivc", ivc, promotes=["*"])

    # Loads and generation
    model.add_subsystem("loads", LoadsComp(n=n), promotes=["img_mask", "tx_mask", "P_load"])
    model.add_subsystem("solar", SolarGenComp(n=n), promotes=["A_SA", "eclipse_mask", "P_gen"])

    # Battery SOC propagation
    model.add_subsystem("battery", BatterySOCComp(n=n, dt_hr=dt_hr), promotes=["E_bat", "P_gen", "P_load", "SOC_min", "SOC_end"])

    # Mass + delta-v
    model.add_subsystem("massdv", MassDeltaVComp(), promotes=["A_SA", "E_bat", "m_prop", "m_dry", "m_tot", "delta_v"])

    # Optimization
    # Use a derivative-free optimizer because BatterySOCComp is piecewise (v0).
    prob.driver = om.ScipyOptimizeDriver(optimizer="COBYLA")
    prob.driver.options["maxiter"] = 300
    prob.driver.options["tol"] = 1e-4
    prob.driver.options["disp"] = True

    # Design variable bounds
    model.add_design_var("A_SA", lower=0.5, upper=10.0)
    model.add_design_var("E_bat", lower=200.0, upper=5000.0)
    model.add_design_var("m_prop", lower=0.5, upper=50.0)

    # Constraints
    delta_v_req = 80.0  # m/s
    model.add_constraint("SOC_min", lower=0.20)
    model.add_constraint("delta_v", lower=delta_v_req)

    # Objective: minimize total mass
    model.add_objective("m_tot")

    # Setup and run
    prob.setup()
    prob.run_driver()

    # Report
    print("\n=== Optimized Design ===")
    print(f"A_SA      = {prob.get_val('A_SA')[0]:.4f} m^2")
    print(f"E_bat     = {prob.get_val('E_bat')[0]:.2f} Wh")
    print(f"m_prop    = {prob.get_val('m_prop')[0]:.4f} kg")
    print(f"m_dry     = {prob.get_val('m_dry')[0]:.3f} kg")
    print(f"m_tot     = {prob.get_val('m_tot')[0]:.3f} kg")
    print(f"delta_v   = {prob.get_val('delta_v')[0]:.3f} m/s (req {delta_v_req:.1f})")
    print(f"SOC_min   = {prob.get_val('SOC_min')[0]:.4f} (>= 0.20)")
    print(f"SOC_end   = {prob.get_val('SOC_end')[0]:.4f}")

    # Optional: quick sanity print of average powers
    P_gen = prob.get_val("P_gen")
    P_load = prob.get_val("P_load")
    print("\n=== Power Sanity ===")
    print(f"Mean P_gen  = {np.mean(P_gen):.2f} W")
    print(f"Mean P_load = {np.mean(P_load):.2f} W")
    print(f"Max  P_load = {np.max(P_load):.2f} W")


if __name__ == "__main__":
    main()
