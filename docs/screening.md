# C3 thermal / C4 radiation / C6 slip — screening models

Status: **offline PLACEHOLDER screening**, owner P4. These exist so EVA time
windows, dose and slip risk can be *explored*; none is validated, and each
carries a validity note. They must be replaced/validated before informing
decisions. Module: `src/marswalk/physics/`.

## C3 thermal (`thermal.py`)

`surface_temp_range_k(thermal_inertia, noon_insolation_wm2, params)` → (T_min,
T_max) skin temperature (K). Peak from radiative equilibrium; day/night
amplitude shrinks with thermal inertia. **Not a validated model** — ground skin
temperature ≠ air temperature; validate against a matched quantity (MEDA ground
temperature) first. Tested for structural behaviour only (inertia damps the
swing; insolation raises the peak).

## C4 radiation (`radiation.py`)

`RadiationScreen.dose_usv(duration_h, shielding_g_cm2)` — a screening dose from a
single surface rate with exponential shielding. **Placeholder scalar**: no
GCR/SEP spectra, solar cycle or events. An owner sets a cited rate + the
conditions it holds for. Tested: dose ∝ duration, shielding reduces dose.

## C6 slip (`slip.py`)

`slip_proxy(slope_deg, params)` → slip-risk proxy in [0,1), 0 on flat, rising
with slope. **Not a probability** and not a terramechanics model — never present
it as a failure probability. Validate against negative rover cases (Spirit's
Troy, Opportunity's Purgatory, Curiosity ripple slips). Tested: monotone,
bounded, flat ≈ 0.
