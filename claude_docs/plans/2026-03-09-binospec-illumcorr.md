# Binospec IFU Illumination Correction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a standalone script to apply fiber-to-fiber illumination corrections to Binospec IFU spec1d files, and update the cube builder to skip corrections on already-corrected files.

**Architecture:** New ScriptBase subclass that loads spec1d FITS files, looks up per-fiber illumination factors via MASKDEF_ID → FIB_ID mapping, and applies the correction directly to extraction columns (BOX/OPT COUNTS, IVAR, SKY). Uses the same `load_fiber_illumination()` and `load_fiber_ref_profile()` methods already on the Binospec spectrograph class.

**Tech Stack:** astropy.io.fits, numpy, PypeIt ScriptBase pattern

---

### Task 1: Create the illumcorr script

**Files:**
- Create: `pypeit/scripts/binospec_ifu_illumcorr.py`

**Step 1: Create the script file**

```python
"""
Apply fiber illumination correction to Binospec IFU spec1d files.

Divides each fiber's extracted spectra by its relative throughput from the
fiber illumination calibration, correcting for fiber-to-fiber sensitivity
variations.  The correction is applied to both boxcar and optimal extraction
columns (COUNTS, COUNTS_IVAR, COUNTS_SKY).

Each input file produces a corrected output file with ``_illumcorr`` appended
before the ``.fits`` extension, unless ``--overwrite`` is used to modify files
in place.

.. include:: ../include/links.rst
"""

from __future__ import annotations

import argparse

import numpy as np

from pypeit.scripts import scriptbase


class BinospecIFUIllumCorr(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width: int | None = None) -> argparse.ArgumentParser:
        parser = super().get_parser(
            description='Apply fiber illumination correction to '
                        'Binospec IFU spec1d files.',
            width=width,
            default_log_file=True)
        parser.add_argument('files', type=str, nargs='+',
                            help='One or more PypeIt spec1d FITS files')
        parser.add_argument('--overwrite', default=False, action='store_true',
                            help='Modify files in place instead of writing '
                                 'new files with _illumcorr suffix')
        parser.add_argument('--force', default=False, action='store_true',
                            help='Apply correction even if FLAM columns '
                                 'are present (flux-calibrated data)')
        return parser

    @classmethod
    def main(cls, args: argparse.Namespace) -> None:
        import os

        from astropy.io import fits

        from pypeit import log
        from pypeit.spectrographs.util import load_spectrograph

        cls.init_log(args)

        # Load spectrograph from first file
        with fits.open(args.files[0]) as hdu:
            spectrograph = load_spectrograph(hdu[0].header['PYP_SPEC'])

        for spec1d_file in args.files:
            log.info(f"Processing {os.path.basename(spec1d_file)}")
            _apply_illumcorr(spec1d_file, spectrograph,
                             overwrite=args.overwrite, force=args.force)


def _apply_illumcorr(spec1d_file: str, spectrograph, *,
                     overwrite: bool = False, force: bool = False) -> None:
    """Apply fiber illumination correction to a single spec1d file.

    Parameters
    ----------
    spec1d_file : str
        Path to the spec1d FITS file.
    spectrograph : :class:`~pypeit.spectrographs.spectrograph.Spectrograph`
        Spectrograph instance.
    overwrite : bool, optional
        If True, modify the file in place. Otherwise write a new file
        with ``_illumcorr`` appended before ``.fits``.
    force : bool, optional
        If True, apply correction even if FLAM columns exist.
    """
    import os

    from astropy.io import fits

    from pypeit import log

    with fits.open(spec1d_file) as hdul:
        # Check if already corrected
        if hdul[0].header.get('ILLUMCOR', False):
            log.warning(f"  {os.path.basename(spec1d_file)} already has "
                        f"ILLUMCOR=True, skipping")
            return

        # Check for FLAM columns (flux-calibrated data)
        has_flam = False
        for hdu in hdul[1:]:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            if any(col.startswith('OPT_FLAM') or col.startswith('BOX_FLAM')
                   for col in hdu.columns.names):
                has_flam = True
                break
        if has_flam and not force:
            log.warning(f"  {os.path.basename(spec1d_file)} contains FLAM "
                        f"columns (already flux-calibrated). Use --force to "
                        f"apply illumination correction anyway. Skipping.")
            return

        # Build illumination lookup per detector:
        # det_num -> {fib_id: f_illum_value}
        illum_lookup = {}
        for det_num in [1, 2]:
            f_illum_all = spectrograph.load_fiber_illumination(det_num)
            ref = spectrograph.load_fiber_ref_profile(det_num)
            ref_ids = ref['FIB_ID']
            lookup = {}
            for i, fid in enumerate(ref_ids):
                if i < len(f_illum_all):
                    lookup[int(fid)] = float(f_illum_all[i])
            illum_lookup[det_num] = lookup

        # Apply correction to each SpecObj HDU
        n_corrected = 0
        n_skipped = 0
        for hdu in hdul[1:]:
            if not isinstance(hdu, fits.BinTableHDU):
                continue

            # Get detector and fiber ID from HDU header
            det_str = hdu.header.get('DET', None)
            maskdef_id = hdu.header.get('MASKDEF_ID', None)
            if det_str is None or maskdef_id is None:
                continue

            det_num = int(det_str.replace('DET', ''))
            if maskdef_id < 0:
                n_skipped += 1
                continue

            # Look up illumination correction factor
            f_illum = illum_lookup.get(det_num, {}).get(maskdef_id, 1.0)
            if f_illum < 0.1:
                log.info(f"  Fiber MASKDEF_ID={maskdef_id} ({det_str}): "
                         f"f_illum={f_illum:.3f} too low, skipping")
                n_skipped += 1
                continue

            # Apply to all extraction columns
            for prefix in ['OPT', 'BOX']:
                counts_col = f'{prefix}_COUNTS'
                ivar_col = f'{prefix}_COUNTS_IVAR'
                sky_col = f'{prefix}_COUNTS_SKY'

                if counts_col in hdu.columns.names:
                    hdu.data[counts_col] /= f_illum
                if ivar_col in hdu.columns.names:
                    hdu.data[ivar_col] *= f_illum ** 2
                if sky_col in hdu.columns.names:
                    hdu.data[sky_col] /= f_illum

            n_corrected += 1

        log.info(f"  Corrected {n_corrected} fibers, skipped {n_skipped}")

        # Mark as corrected
        hdul[0].header['ILLUMCOR'] = (True, 'Fiber illumination correction applied')

        # Write output
        if overwrite:
            hdul.writeto(spec1d_file, overwrite=True)
            log.info(f"  Overwritten {os.path.basename(spec1d_file)}")
        else:
            base, ext = os.path.splitext(spec1d_file)
            outfile = f"{base}_illumcorr{ext}"
            hdul.writeto(outfile, overwrite=True)
            log.info(f"  Written {os.path.basename(outfile)}")
```

**Step 2: Verify the script can be imported**

Run: `cd /Users/tim/MMT/pypeit && python -c "from pypeit.scripts.binospec_ifu_illumcorr import BinospecIFUIllumCorr; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pypeit/scripts/binospec_ifu_illumcorr.py
git commit -m "Add script to apply fiber illumination correction to spec1d files"
```

---

### Task 2: Register the entry point

**Files:**
- Modify: `pyproject.toml:108` (near the binospec_ifu_cube entry)

**Step 1: Add entry point**

Add the following line after the `pypeit_binospec_ifu_cube` entry (line 108):

```
pypeit_binospec_ifu_illumcorr = "pypeit.scripts.binospec_ifu_illumcorr:BinospecIFUIllumCorr.entry_point"
```

**Step 2: Reinstall to register**

Run: `cd /Users/tim/MMT/pypeit && pip install -e ".[dev]"`

**Step 3: Verify entry point works**

Run: `pypeit_binospec_ifu_illumcorr --help`
Expected: Shows help text with positional `files` argument and `--overwrite`/`--force` flags.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Register pypeit_binospec_ifu_illumcorr entry point"
```

---

### Task 3: Update binospec_ifu_cube.py to check ILLUMCOR

**Files:**
- Modify: `pypeit/scripts/binospec_ifu_cube.py:568-599` (Step 2 illumination correction block)

**Step 1: Add ILLUMCOR check in the spec1d path**

In `_build_cube_from_spec1d()`, after loading the SpecObjs and before the shared Steps 2-7 block, read the `ILLUMCOR` header from the spec1d file and pass it to `_apply_steps()`. The Step 2 block (lines 568-599) should be wrapped in a check:

In `_build_cube_from_spec1d()` around line 337-340, after opening the FITS file to read the primary header for WCS, also check for ILLUMCOR:

```python
with fits.open(spec1d_file) as hdu:
    illumcor_done = hdu[0].header.get('ILLUMCOR', False)
```

Then in the Step 2 block (line 568), wrap the correction logic:

```python
# Step 2: Apply fiber-to-fiber illumination correction
if illumcor_done:
    log.info("  Illumination correction already applied (ILLUMCOR=True), "
             "skipping Step 2")
else:
    for det_name, data in det_fiber_data.items():
        # ... existing correction code ...
```

The `illumcor_done` flag needs to be passed from `_build_cube_from_spec1d()` to `_apply_steps()`. Add it as a parameter.

**Step 2: Verify the cube builder still works with uncorrected spec1d files**

Run: `cd /Users/tim/MMT/pypeit && python -c "from pypeit.scripts.binospec_ifu_cube import BinospecIFUCube; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pypeit/scripts/binospec_ifu_cube.py
git commit -m "Skip illumination correction in cube builder for pre-corrected spec1d files"
```

---

### Task 4: Test with real data

**Step 1: Run the illumcorr script on a spec1d file**

Run: `pypeit_binospec_ifu_illumcorr ~/MMT/bino_ifu/NGC_2392_G270/Science/spec1d_*.fits`

Verify:
- Output files with `_illumcorr.fits` suffix are created
- Log shows correction factors and fiber counts
- `ILLUMCOR` header card is set in output files

**Step 2: Verify idempotency**

Run: `pypeit_binospec_ifu_illumcorr ~/MMT/bino_ifu/NGC_2392_G270/Science/spec1d_*_illumcorr.fits`

Expected: Warning that files already have ILLUMCOR=True, all skipped.

**Step 3: Verify cube builder skips correction for pre-corrected files**

Run: `pypeit_binospec_ifu_cube ~/MMT/bino_ifu/NGC_2392_G270/Science/spec1d_*_illumcorr.fits`

Expected: Log shows "Illumination correction already applied, skipping Step 2"
