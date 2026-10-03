"""
Buck (step-down) DC-DC converter model.

Scope:
- Continuous Conduction Mode (CCM) only. Discontinuous Conduction Mode (DCM) is
  explicitly deferred/not modeled. Callers should verify (via `check_ccm`) that
  the simulated inductor current stays positive; if it doesn't, the switching
  results near/at zero current are not physically accurate (real converter
  would enter DCM).
- Input stage: an *ideal* voltage source Vin is treated as supplying only the
  DC/average component of the input current. The input capacitor `C_in`
  absorbs the pulsating (AC) component drawn by the switch. This is the
  standard assumption used to size/estimate input capacitor ripple, and lets
  us keep the source itself ideal (zero impedance, no dynamics) while still
  getting a non-trivial input ripple waveform on C_in.
- Output stage: standard buck averaged/switching model with ideal diode
  freewheeling (zero forward drop) and ideal switch.

States (switching model): x = [iL, vOut, vIn_cap]
    iL      - inductor current (A)
    vOut    - output capacitor / load voltage (V)
    vIn_cap - input capacitor voltage (V)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class BuckConverterParams:
    Vin: float          # nominal input (ideal source) voltage [V]
    Vout: float         # target output voltage [V]
    L: float            # inductor [H]
    C_out: float        # output capacitor [F]
    R_load: float       # load resistance [ohm]
    f_sw: float         # switching frequency [Hz]
    C_in: float = 10e-6  # input capacitor [F]

    @property
    def T_sw(self) -> float:
        return 1.0 / self.f_sw

    @property
    def D(self) -> float:
        """Ideal (lossless) CCM duty cycle D = Vout / Vin."""
        return duty_cycle_ccm(self.Vin, self.Vout)


def duty_cycle_ccm(Vin: float, Vout: float) -> float:
    """Ideal CCM buck duty cycle, D = Vout / Vin."""
    if Vin <= 0:
        raise ValueError("Vin must be > 0")
    D = Vout / Vin
    if not (0.0 < D < 1.0):
        raise ValueError(f"Computed duty cycle D={D:.3f} is outside (0,1); "
                          f"check Vin/Vout for a buck (step-down) topology.")
    return D


def steady_state(params: BuckConverterParams, D: Optional[float] = None) -> dict:
    """Nominal (average, ripple-free) operating point for the given params.

    Returns a dict with: D, Iout_avg, IL_avg, Iin_avg, dIL (pk-pk inductor
    ripple), dVout (analytic pk-pk output ripple, ESR=0 assumption), dVin
    (analytic pk-pk input cap ripple).
    """
    if D is None:
        D = params.D

    Iout_avg = params.Vout / params.R_load
    IL_avg = Iout_avg  # in CCM, average inductor current = average load current
    Iin_avg = D * Iout_avg  # average current drawn from the ideal source

    dIL = output_inductor_ripple_analytic(params, D)
    dVout = output_voltage_ripple_analytic(params, D)
    dVin = input_ripple_analytic(Iout_avg, D, params.C_in, params.f_sw)

    return {
        "D": D,
        "Iout_avg": Iout_avg,
        "IL_avg": IL_avg,
        "Iin_avg": Iin_avg,
        "dIL_pp": dIL,
        "dVout_pp": dVout,
        "dVin_pp": dVin,
    }


def output_inductor_ripple_analytic(params: BuckConverterParams, D: Optional[float] = None) -> float:
    """Peak-peak inductor current ripple: dIL = Vout*(1-D) / (L*f_sw)."""
    if D is None:
        D = params.D
    return params.Vout * (1 - D) / (params.L * params.f_sw)


def output_voltage_ripple_analytic(params: BuckConverterParams, D: Optional[float] = None) -> float:
    """Peak-peak output voltage ripple (ESR=0 assumption):
    dVout = dIL / (8 * C_out * f_sw)
    """
    dIL = output_inductor_ripple_analytic(params, D)
    return dIL / (8 * params.C_out * params.f_sw)


def input_ripple_analytic(Iout_avg: float, D: float, C_in: float, f_sw: float) -> float:
    """Peak-peak input capacitor voltage ripple (standard buck estimate):
    dVin = Iout_avg * D * (1 - D) / (C_in * f_sw)
    """
    return Iout_avg * D * (1 - D) / (C_in * f_sw)


def switch_signal(t: float, f_sw: float, D: float) -> int:
    """Ideal PWM switch state (1 = ON, 0 = OFF) for time t."""
    phase = (t * f_sw) % 1.0
    return 1 if phase < D else 0


def switching_rhs(t: float, x: np.ndarray, params: BuckConverterParams, D: float,
                   Iin_avg: Optional[float] = None) -> np.ndarray:
    """ODE right-hand side for the hard-switching (non-averaged) model.

    x = [iL, vOut, vIn_cap]
    """
    iL, vOut, vIn_cap = x
    s = switch_signal(t, params.f_sw, D)

    if Iin_avg is None:
        Iin_avg = D * (params.Vout / params.R_load)

    # Inductor: switch on -> sees (vIn_cap - vOut); switch off -> ideal diode
    # freewheels, sees (-vOut).
    diL_dt = (s * vIn_cap - vOut) / params.L

    # Output node: KCL at output cap / load.
    dvOut_dt = (iL - vOut / params.R_load) / params.C_out

    # Input node: ideal source supplies only the DC/average current Iin_avg;
    # C_in supplies/absorbs the difference between that and the pulsed draw
    # (s * iL) taken by the switch.
    dvIn_dt = (Iin_avg - s * iL) / params.C_in

    return np.array([diL_dt, dvOut_dt, dvIn_dt])


def averaged_rhs(t: float, x: np.ndarray, params: BuckConverterParams, D: float) -> np.ndarray:
    """Duty-cycle-averaged (ripple-free) small-signal-style model, useful for
    fast steady-state / control studies. x = [iL, vOut] (input cap assumed
    regulated to Vin on average).
    """
    iL, vOut = x
    diL_dt = (D * params.Vin - vOut) / params.L
    dvOut_dt = (iL - vOut / params.R_load) / params.C_out
    return np.array([diL_dt, dvOut_dt])


def simulate_switching(params: BuckConverterParams, D: Optional[float] = None,
                        t_end: Optional[float] = None, n_periods: int = 50,
                        points_per_period: int = 200,
                        x0: Optional[np.ndarray] = None,
                        Iin_avg: Optional[float] = None):
    """Simulate the hard-switching model with solve_ivp.

    Returns (t, iL, vOut, vIn_cap, D).
    """
    if D is None:
        D = params.D
    if t_end is None:
        t_end = n_periods * params.T_sw
    if x0 is None:
        x0 = np.array([params.Vout / params.R_load, params.Vout, params.Vin])

    n_points = int(n_periods * points_per_period)
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        switching_rhs, (0, t_end), x0, args=(params, D, Iin_avg),
        t_eval=t_eval, method="RK45", max_step=params.T_sw / points_per_period,
        rtol=1e-8, atol=1e-10,
    )

    iL, vOut, vIn_cap = sol.y
    return sol.t, iL, vOut, vIn_cap, D


def simulate_averaged(params: BuckConverterParams, D: Optional[float] = None,
                       t_end: Optional[float] = None, n_periods: int = 50,
                       points_per_period: int = 50,
                       x0: Optional[np.ndarray] = None):
    """Simulate the averaged (ripple-free) model. Returns (t, iL, vOut, D)."""
    if D is None:
        D = params.D
    if t_end is None:
        t_end = n_periods * params.T_sw
    if x0 is None:
        x0 = np.array([params.Vout / params.R_load, 0.0])

    n_points = int(n_periods * points_per_period)
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        averaged_rhs, (0, t_end), x0, args=(params, D),
        t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
    )
    iL, vOut = sol.y
    return sol.t, iL, vOut, D


def check_ccm(iL: np.ndarray, tail_fraction: float = 0.3, tol: float = 0.0) -> bool:
    """Returns True if the inductor current stays >= tol over the last
    `tail_fraction` of the simulation (steady-state region), i.e. the
    converter remained in CCM. DCM is not modeled, so if this returns False
    the switching-model results are not physically valid in that region.
    """
    n = len(iL)
    start = int(n * (1 - tail_fraction))
    return bool(np.all(iL[start:] >= tol))


class PIController:
    """Simple discrete PI controller for optional closed-loop duty-cycle
    regulation of Vout (updated once per switching period).
    """

    def __init__(self, kp: float, ki: float, D_min: float = 0.02, D_max: float = 0.98):
        self.kp = kp
        self.ki = ki
        self.D_min = D_min
        self.D_max = D_max
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def update(self, vout_ref: float, vout_meas: float, dt: float) -> float:
        err = vout_ref - vout_meas
        self._integral += err * dt
        D = self.kp * err + self.ki * self._integral
        D_clamped = min(max(D, self.D_min), self.D_max)
        # anti-windup: back-calculate integral if clamped
        if D != D_clamped and self.ki != 0:
            self._integral = (D_clamped - self.kp * err) / self.ki
        return D_clamped


def simulate_switching_closed_loop(params: BuckConverterParams, controller: PIController,
                                    vout_ref: float, n_periods: int = 200,
                                    points_per_period: int = 100,
                                    x0: Optional[np.ndarray] = None,
                                    Iin_avg: Optional[float] = None):
    """Closed-loop simulation: duty cycle is updated once per switching
    period by `controller`, then the hard-switching ODE is integrated over
    that single period. Returns (t, iL, vOut, vIn_cap, D_history).
    """
    if x0 is None:
        x0 = np.array([params.Vout / params.R_load, params.Vout, params.Vin])
    if Iin_avg is None:
        Iin_avg = vout_ref / params.R_load * params.D

    controller.reset()

    t_all, iL_all, vOut_all, vIn_all, D_hist = [], [], [], [], []
    t0 = 0.0
    x = x0.copy()

    for k in range(n_periods):
        D = controller.update(vout_ref, x[1], params.T_sw)
        D_hist.append(D)

        t_eval = np.linspace(t0, t0 + params.T_sw, points_per_period, endpoint=False)
        sol = solve_ivp(
            switching_rhs, (t0, t0 + params.T_sw), x, args=(params, D, Iin_avg),
            t_eval=t_eval, method="RK45", max_step=params.T_sw / points_per_period,
            rtol=1e-8, atol=1e-10,
        )

        t_all.append(sol.t)
        iL_all.append(sol.y[0])
        vOut_all.append(sol.y[1])
        vIn_all.append(sol.y[2])

        x = sol.y[:, -1]
        t0 += params.T_sw

    t = np.concatenate(t_all)
    iL = np.concatenate(iL_all)
    vOut = np.concatenate(vOut_all)
    vIn_cap = np.concatenate(vIn_all)
    return t, iL, vOut, vIn_cap, np.array(D_hist)

