
## DC-DC Converter Modeling

A Python/JupyterLab model of a **buck (step-down) DC-DC converter**, operating
in **Continuous Conduction Mode (CCM) only** (Discontinuous Conduction Mode,
DCM, is intentionally not modeled/deferred).

## Contents

- `models/dc_dc_model.py` — reusable converter model:
  - Analytic CCM steady-state and ripple formulas (inductor current, output
    voltage, input capacitor voltage).
  - A hard-switching time-domain ODE model (`solve_ivp`) with states
    `[iL, vOut, vIn_cap]`.
  - A duty-cycle-averaged (ripple-free) ODE model for quick steady-state /
    control studies.
  - A simple discrete PI controller plus a closed-loop simulation helper for
    optional closed-loop Vout regulation.
  - `check_ccm` helper to flag if a simulation run dipped into DCM territory
    (not modeled, so such results are not physically valid there).
  - Optional inductor winding resistance (`R_L`, DCR) modeled as a series
    resistance, which adds a resistive drop to both ODE models and shifts
    the duty-cycle formula away from the ideal `D = Vout / Vin`.
- `dc.ipynb` — notebook that defines converter parameters, runs the
  open-loop switching simulation, compares simulated vs. analytic ripple,
  (optionally) runs the duty-cycle-averaged model for a quick ripple-free
  view, and (optionally) demonstrates closed-loop PI regulation under a load
  step.

## Modeling assumptions

- **Ideal switch and diode** (zero on-resistance / zero forward drop).
- **Ideal voltage source at the input**: `Vin` has no series impedance or
  dynamics. To still produce a non-trivial input ripple waveform, the source
  is treated as supplying only the **average/DC** input current
  (`Iin_avg = D * Iout_avg`); the input capacitor `C_in` absorbs the
  pulsating (AC) component drawn by the switch. This is the standard
  assumption used in practice to estimate/size input capacitor ripple.
- **CCM only**: the inductor is assumed large enough (relative to
  load/frequency) that current never reaches zero. Use `check_ccm()` on
  simulation results to verify this holds for your chosen parameters.
- **Duty cycle**: `D = Vout / Vin` when the inductor is ideal (`R_L = 0`,
  the default). If a nonzero inductor DCR `R_L` is set, `D` instead accounts
  for the resistive drop: `D = (Vout + Iout_avg·R_L) / Vin`.
- **Inductor DCR (`R_L`)**: an optional series winding resistance on the
  inductor. It subtracts `iL·R_L` from the voltage applied to the output
  node in both the switch-on and freewheeling phases.

## Ripple formulas used

- Inductor current (pk-pk): `ΔIL = Vout·(1 − D) / (L·f_sw)`
- Output voltage (pk-pk, ESR = 0): `ΔVout = ΔIL / (8·C_out·f_sw)`
- Input capacitor voltage (pk-pk): `ΔVin = Iout_avg·D·(1 − D) / (C_in·f_sw)`

## Running

```powershell
pip install -r requirements.txt
jupyter lab dc.ipynb
```


