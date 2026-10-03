
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
- `dc.ipynb` — notebook that defines converter parameters, runs the
  open-loop switching simulation, compares simulated vs. analytic ripple, and
  (optionally) demonstrates closed-loop PI regulation under a load step.

## Modeling assumptions

- **Ideal switch and diode** (zero on-resistance / zero forward drop).
- **Ideal voltage source at the input**: `Vin` has no series impedance or
  dynamics. To still produce a non-trivial input ripple waveform, the source
  is treated as supplying only the **average/DC** input current
  (`Iin_avg = D * Iout_avg`); the input capacitor `C_in` absorbs the
  pulsating (AC) component drawn by the switch. This is the standard
  assumption used in practice to estimate/size input capacitor ripple.
- **CCM only**: duty cycle is computed as the ideal `D = Vout / Vin`, and the
  inductor is assumed large enough (relative to load/frequency) that current
  never reaches zero. Use `check_ccm()` on simulation results to verify this
  holds for your chosen parameters.

## Ripple formulas used

- Inductor current (pk-pk): `ΔIL = Vout·(1 − D) / (L·f_sw)`
- Output voltage (pk-pk, ESR = 0): `ΔVout = ΔIL / (8·C_out·f_sw)`
- Input capacitor voltage (pk-pk): `ΔVin = Iout_avg·D·(1 − D) / (C_in·f_sw)`

## Running

```powershell
pip install -r requirements.txt
jupyter lab dc.ipynb
```


