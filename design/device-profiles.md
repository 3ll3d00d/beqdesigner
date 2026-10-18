# Device profiles: what a filter becomes on a device, and optimising it

Part (a) is **built** (2026-10-18). Part (b) is **design, not built**: it waits on beqforge (`TODO.md` P1).

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

## (b) Optimise for devices (design, not built)

Decided with the owner (2026-10-18):

* **In parameter terms.** The optimised filter is still freq, Q and gain per section, on the dialog's own grid (0.01 Hz,
  0.0001, 0.01 dB). A BEQ is saved, exported, published and loaded as parameters, so nothing downstream changes: no
  custom coefficients, no catalogue schema.
* **Across the chosen profiles jointly.** The same parameters are realised on every device, so the search minimises the
  worst error over all the profiles the designer ticks, and reports each one's error before and after.
* **The search is beqforge's.** It is handed over as beqforge TODO 27
  (`plans/parameter-space-optimisation.md`, `optimise_parameters`), beside the existing direct-coefficient `optimise()`.

beqdesigner's part, once a beqforge release has it: an *Optimise for devices…* action on the filter that calls
`optimise_parameters` with the filter's sections and the ticked profiles, shows the before/after errors per profile, and
on acceptance replaces the sections' parameters as one undoable filter edit. Sections it cannot change (pass filters, a
gain) stay as they are. A result that does not improve every chosen profile is shown, not applied.

**Packaging (b):** beqoptimiser's result cache identifies its implementation by hashing its own `core.py` and
`beq_common/biquad.py` source and ships a seed (`data/seed.json.gz`), read with `importlib.resources`. A PyInstaller app
holds bytecode only, so when (b) calls the cached `optimise` path, `beqdesigner.spec` must bundle those files
(`collect_data_files('beqoptimiser', include_py_files=True)`, likewise `beq_common`), or call it with `cache=False`.
Part (a) uses only `Float32`, `FixedPoint`, `magnitude` and `stable`, which read nothing.
