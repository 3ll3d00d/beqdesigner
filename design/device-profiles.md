# Device profiles: what a filter becomes on a device, and optimising it

Both parts are **built** (2026-10-18): (a) the filter on a device, (b) optimising it for devices (beqforge 0.2.5).

## Why

A BEQ is designed in double precision, but a device recalculates it at its own internal rate and stores the coefficients
with its own precision. At low frequencies and high Q that changes the filter: on a miniDSP 2x4 HD (float32 coefficients
at 96 kHz) a 5 Hz Q 6 +12 dB peak becomes a +25 dB peak at 3.6 Hz and a dip at 6 Hz, an 18 dB error (beqforge's F2 pilot
measured the same on hardware). A filter designer should see that before publishing, and be able to correct for it.

## Device profiles (built)

A profile is a rate (48 or 96 kHz) and a coefficient storage: float32, or signed fixed point with so many integer bits
(sign included) and fractional bits (`model/device_profile.py`, Qt-free).

* **Built in**, from the devices beqdesigner exports to (`model/dsp_type.py`): the HD-class miniDSPs (2x4 HD, SHD, 88BM,
  HTx) keep float32 coefficients at 96 kHz; the earlier ones (2x4, 10x10) Q5.23 fixed point at 48 kHz. The HTP-1 and
  JRiver compute in double precision, so a filter is unchanged on them and they have no profile.
* **A person's own**, kept in Preferences (`device_profiles/custom`, a JSON list), added, edited and removed in
  *Devices > Manage profiles…*. Names are unique across built-in and own profiles.

The storage models (`Float32`, `FixedPoint`) and the high-precision magnitude calculation are beqforge's `beqoptimiser`
(`beqforge[optimiser]` is a dependency), so beqdesigner shows a device the way beqforge's optimiser models it.

## (a) The filter on a device (built)

*Devices*, beside *Filter Response?* in the main window, ticks the profiles the chart shows (`display/device_profiles`).
For each, the combined filter is recalculated at the profile's rate (`resample`), its coefficients stored with the
profile's precision (a0 is implicit), and its magnitude drawn dashed in the combined filter's colour as
"Combined on <profile>". A line under the filter table gives each profile's largest error against the filter as designed
at that rate, over 2-200 Hz (beqoptimiser's matching band), and says when the stored coefficients are unstable or cannot
be stored at all (out of a fixed-point range). Normalising the chart against *Combined* shows each device curve as its
error alone.

Code: `model/device_profile.py` (profiles, `realise()`), `FilterModel.realised()`/`device_summary()` and the curves in
`FilterModel.get_curve_data()` (`model/filter.py`), `model/device_profile_qt.py` (the menu, summary line and dialogs).
Tests: `test_device_profile.py`, `gui/test_device_profiles.py`.

## (b) Optimise for devices (built)

*Devices > Optimise filter for these devices…* (offered with a profile ticked and a peaking or shelf filter to adjust)
asks beqforge's `beqoptimiser.optimise_parameters` (0.2.5) for new frequency, Q and gain:

* **In parameter terms.** The result is still freq, Q and gain per filter, on the dialog's own grid (0.01 Hz, 0.0001,
  0.01 dB), so it saves, exports, publishes and loads like any filter: no custom coefficients, no catalogue schema.
* **Across the ticked profiles jointly.** beqforge minimises the worst error over all of them and refuses a step that
  makes any profile worse than both its original error and the 0.5 dB margin. Parameters move only a little (±1%
  frequency, ±2% Q, ±0.1 dB gain).
* **What is searched.** Peaking and shelf filters; a shelf stacked n times is n identical sections, which stay
  identical, so it comes back still stacked. Everything else (pass filters, a gain) is held at both rates but counts in
  the response. A cascade that one profile realises unstably (a 3 Hz high-pass as float32 at 96 kHz) cannot be compared,
  and the dialog says so.
* **Applying.** The dialog (run on the thread pool) shows each profile's largest error before and after and every
  parameter that changes. *Apply* is offered only when beqforge publishes the result (`replacement` or `improvement`),
  it changes something, and no ticked profile's error is larger than before; each changed filter then replaces its
  original by id (`FilterModel.save`), as an edit in the filter dialog would. There is no undo; the dialog shows the
  old values.

Code: `model/device_optimise.py` (the filter as beqforge's sections and back, `optimise_for`, `Optimisation`),
`OptimiseDialog`/`OptimiseJob` in `model/device_profile_qt.py`, the action in `app.py`. Tests:
`test_device_optimise.py`, `gui/test_device_profiles.py`.

**In the filter edit dialog** (not its small mode): the same *Devices* menu sits beside *Show Individual Filters*, the
preview chart draws the working filter on each ticked device (`Combined on <profile>`, a long dash in its colour) and the
errors are listed under the chart. *Optimise filter for these devices…* there changes the dialog's working filter only,
reloads the edit fields from it (or the next edit would write the old values back), and reaches the main window when
the dialog is saved. It is separate from the dialog's own *Optimise* row, which reduces a pass filter's biquad count.

**Packaging.** beqoptimiser's result cache identifies its implementation by hashing its own source, which a PyInstaller
app does not ship, so beqdesigner calls it with `cache=False` (a search takes 0.2-4 s). Nothing extra is bundled.
