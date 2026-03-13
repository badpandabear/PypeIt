# Binospec IFU Illumination Correction Script

## Purpose

Standalone script to apply fiber-to-fiber illumination corrections to Binospec
IFU spec1d files, without building a datacube.

## Behavior

For each input spec1d file:

1. Load SpecObjs from the file.
2. Check for FLAM columns — if present, warn and skip unless `--force`.
3. Check for `ILLUMCOR` header card — if already corrected, warn and skip.
4. For each detector, load `fiber_illumination.fits` and `fiber_ref_profile.fits`.
5. For each SpecObj, map `MASKDEF_ID` to reference profile `FIB_ID` to look up
   the illumination correction factor.
6. Apply correction:
   - Divide `{BOX,OPT}_COUNTS` and `{BOX,OPT}_COUNTS_SKY` by `f_illum`
   - Multiply `{BOX,OPT}_COUNTS_IVAR` by `f_illum**2`
   - Skip fibers with `MASKDEF_ID < 0` or `f_illum < 0.1`
7. Set `ILLUMCOR = True` header card on primary HDU.
8. Write output: `spec1d_..._illumcorr.fits` (default) or overwrite (`--overwrite`).

## CLI Interface

```
pypeit_binospec_ifu_illumcorr [--overwrite] [--force] spec1d_file [spec1d_file ...]
```

- Positional: one or more spec1d file paths
- `--overwrite`: modify files in place instead of writing `_illumcorr` copies
- `--force`: apply correction even if FLAM columns exist

## Integration with binospec_ifu_cube.py

The cube builder's Step 2 (illumination correction) must check the `ILLUMCOR`
header card on input spec1d files. If set, skip the illumination correction step
to avoid double-application.

## Script Structure

- Class `BinospecIFUIllumCorr(ScriptBase)` in `pypeit/scripts/binospec_ifu_illumcorr.py`
- Entry point: `pypeit_binospec_ifu_illumcorr` in `pyproject.toml`

## Out of Scope

- Sky subtraction, wavelength resampling, cube building
- Flux calibration
- Support for non-Binospec spectrographs
