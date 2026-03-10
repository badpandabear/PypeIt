"""
Apply fiber-to-fiber illumination corrections to Binospec IFU spec1d files.

This script corrects for the relative throughput differences between fibers
using the illumination map derived from flat field observations.  It operates
directly on PypeIt spec1d FITS files, dividing each fiber's flux by its
relative illumination factor.

The correction should be applied *before* building datacubes with
``pypeit_binospec_ifu_cube``, which will detect the ``ILLUMCOR`` header flag
and skip its own internal illumination correction.

.. include:: ../include/links.rst
"""

from __future__ import annotations

import argparse

from pypeit.scripts import scriptbase


class BinospecIFUIllumCorr(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width: int | None = None) -> argparse.ArgumentParser:
        parser = super().get_parser(
            description='Apply fiber-to-fiber illumination corrections '
                        'to Binospec IFU spec1d files.',
            width=width,
            default_log_file=True)
        parser.add_argument('files', type=str, nargs='+',
                            help='One or more PypeIt spec1d FITS file paths')
        parser.add_argument('--overwrite', default=False, action='store_true',
                            help='Modify files in place instead of writing '
                                 '_illumcorr copies')
        parser.add_argument('--force', default=False, action='store_true',
                            help='Apply correction even if FLAM columns exist')
        return parser

    @classmethod
    def main(cls, args: argparse.Namespace) -> None:
        import os

        from astropy.io import fits

        from pypeit import log, PypeItError
        from pypeit.spectrographs.util import load_spectrograph

        cls.init_log(args)

        if len(args.files) == 0:
            raise PypeItError("No input files provided.")

        # Load spectrograph from first file header
        with fits.open(args.files[0]) as hdu:
            spectrograph = load_spectrograph(hdu[0].header['PYP_SPEC'])

        log.info(f"Spectrograph: {spectrograph.name}")

        # Build illumination lookup for each detector
        illum_lookup = _build_illum_lookup(spectrograph)

        # Process each file
        for filepath in args.files:
            log.info(f"Processing {os.path.basename(filepath)}")
            _apply_illumcorr(filepath, illum_lookup, args)


def _build_illum_lookup(spectrograph) -> dict:
    """Build a dict mapping (det_num, FIB_ID) -> illumination factor.

    Parameters
    ----------
    spectrograph : :class:`~pypeit.spectrographs.spectrograph.Spectrograph`
        Spectrograph instance.

    Returns
    -------
    dict
        Keys are ``(det_num, fib_id)`` tuples, values are float illumination
        correction factors.
    """
    import numpy as np
    from pypeit import log

    lookup = {}
    for det_num in [1, 2]:
        f_illum_all = spectrograph.load_fiber_illumination(det_num)
        ref = spectrograph.load_fiber_ref_profile(det_num)
        ref_ids = ref['FIB_ID']

        for i, fib_id in enumerate(ref_ids):
            if i < len(f_illum_all):
                lookup[(det_num, int(fib_id))] = float(f_illum_all[i])

        n_fibers = min(len(ref_ids), len(f_illum_all))
        valid = [lookup[(det_num, int(ref_ids[i]))]
                 for i in range(n_fibers)
                 if lookup[(det_num, int(ref_ids[i]))] > 0.1]
        if valid:
            log.info(f"  DET{det_num:02d}: {n_fibers} fibers, "
                     f"illum range {min(valid):.3f} - {max(valid):.3f}")

    return lookup


def _apply_illumcorr(filepath: str, illum_lookup: dict,
                     args: argparse.Namespace) -> None:
    """Apply illumination correction to a single spec1d FITS file.

    Parameters
    ----------
    filepath : str
        Path to the spec1d FITS file.
    illum_lookup : dict
        Mapping of ``(det_num, FIB_ID)`` to illumination correction factor.
    args : `argparse.Namespace`_
        Parsed command-line arguments (uses ``overwrite`` and ``force``).
    """
    import os

    from astropy.io import fits

    from pypeit import log

    with fits.open(filepath, mode='readonly') as hdu:
        # Check if already corrected
        if hdu[0].header.get('ILLUMCOR', False):
            log.warning(f"  Already illumination-corrected (ILLUMCOR=True), "
                        f"skipping")
            return

        # Check for FLAM columns (flux-calibrated data)
        if not args.force:
            for ext in hdu[1:]:
                if not isinstance(ext, fits.BinTableHDU):
                    continue
                for col in ext.columns.names:
                    if col.startswith('OPT_FLAM') or col.startswith('BOX_FLAM'):
                        log.warning(f"  FLAM columns found (flux-calibrated). "
                                    f"Use --force to apply anyway. Skipping.")
                        return

        # Apply corrections
        n_corrected = 0
        n_skipped = 0

        for ext in hdu[1:]:
            if not isinstance(ext, fits.BinTableHDU):
                continue

            hdr = ext.header

            # Parse detector number from DET keyword (e.g. 'DET01' -> 1)
            det_str = hdr.get('DET', None)
            if det_str is None:
                log.warning(f"  Extension {ext.name}: no DET keyword, skipping")
                n_skipped += 1
                continue

            try:
                det_num = int(det_str.replace('DET', ''))
            except (ValueError, AttributeError):
                log.warning(f"  Extension {ext.name}: cannot parse DET='{det_str}', "
                            f"skipping")
                n_skipped += 1
                continue

            maskdef_id = hdr.get('MASKDEF_ID', None)
            if maskdef_id is None or maskdef_id < 0:
                log.warning(f"  Extension {ext.name}: MASKDEF_ID={maskdef_id}, "
                            f"skipping")
                n_skipped += 1
                continue

            # Look up illumination factor
            f_illum = illum_lookup.get((det_num, int(maskdef_id)), None)
            if f_illum is None:
                log.warning(f"  Extension {ext.name}: FIB_ID={maskdef_id} not "
                            f"in illumination map, skipping")
                n_skipped += 1
                continue

            if f_illum < 0.1:
                log.warning(f"  Extension {ext.name}: FIB_ID={maskdef_id} "
                            f"f_illum={f_illum:.4f} too low, skipping")
                n_skipped += 1
                continue

            # Apply correction to all extraction columns that exist
            col_names = ext.columns.names
            for prefix in ['OPT', 'BOX']:
                counts_col = f'{prefix}_COUNTS'
                ivar_col = f'{prefix}_COUNTS_IVAR'
                sky_col = f'{prefix}_COUNTS_SKY'

                if counts_col in col_names:
                    ext.data[counts_col] /= f_illum
                if ivar_col in col_names:
                    ext.data[ivar_col] *= f_illum ** 2
                if sky_col in col_names:
                    ext.data[sky_col] /= f_illum

            n_corrected += 1

        log.info(f"  Corrected {n_corrected} fibers, skipped {n_skipped}")

        if n_corrected == 0:
            log.warning(f"  No fibers corrected, not writing output")
            return

        # Mark as corrected
        hdu[0].header['ILLUMCOR'] = (True, 'Fiber illumination correction applied')

        # Write output
        if args.overwrite:
            outfile = filepath
        else:
            base, ext_str = os.path.splitext(filepath)
            outfile = base + '_illumcorr' + ext_str

        hdu.writeto(outfile, overwrite=True)
        log.info(f"  Wrote {os.path.basename(outfile)}")
