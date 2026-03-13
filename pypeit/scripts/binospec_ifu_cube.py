"""
Build a datacube from Binospec IFU spec1d or spec2d files.

Unlike the general ``pypeit_coadd_datacube`` script designed for slicer-based
IFUs, this script handles the fiber-fed Binospec IFU by:

1. Reading extracted 1D fiber spectra from spec1d files, or extracting
   them directly from spec2d files
2. Sky subtracting using dedicated sky fiber spectra
3. Mapping 320 science fibers per side to sky positions
4. Combining both detectors (640 science fibers total)
5. Interpolating scattered fiber positions onto a regular spatial grid

Each input file produces a separate output datacube.

.. include:: ../include/links.rst
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

import numpy as np

from pypeit.scripts import scriptbase

if TYPE_CHECKING:
    from pypeit.spec2dobj import Spec2DObj
    from pypeit.spectrographs.spectrograph import Spectrograph


class BinospecIFUCube(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width: int | None = None) -> argparse.ArgumentParser:
        parser = super().get_parser(
            description='Build a datacube from Binospec IFU spec1d or spec2d files.',
            width=width,
            default_log_file=True)
        parser.add_argument('files', type=str, nargs='+',
                            help='One or more PypeIt spec1d or spec2d files, '
                                 'or a text file listing them (one per line)')
        parser.add_argument('-o', '--output', type=str, default=None,
                            help='Output FITS filename (only valid for a single '
                                 'input file; default: auto-generated)')
        parser.add_argument('--spatial_scale', type=float, default=0.27,
                            help='Output spatial pixel scale in arcsec (default: 0.27)')
        parser.add_argument('--no_skysub', default=False, action='store_true',
                            help='Skip sky subtraction')
        parser.add_argument('--use_fibers', default=False, action='store_true',
                            help='Use dedicated sky fibers for sky subtraction '
                                 'instead of the default spec2d skymodel '
                                 '(spec2d only; ignored for spec1d)')
        parser.add_argument('--boxcar', default=False, action='store_true',
                            help='Use boxcar extraction instead of optimal '
                                 '(profile-weighted) extraction')
        parser.add_argument('--gaussian', default=False, action='store_true',
                            help='Use Gaussian profile for optimal extraction '
                                 'instead of the default empirical profile '
                                 'measured from the flat field')
        parser.add_argument('--method', type=str, default='linear',
                            choices=['nearest', 'linear', 'cubic'],
                            help='Spatial interpolation method (default: linear)')
        return parser

    @classmethod
    def main(cls, args: argparse.Namespace) -> None:
        import os

        from astropy.io import fits

        from pypeit import log, PypeItError
        from pypeit.spectrographs.util import load_spectrograph

        cls.init_log(args)

        # ----------------------------------------------------------------
        # Parse input files
        # ----------------------------------------------------------------
        input_files: list[str] = []
        for f in args.files:
            if f.endswith('.txt'):
                with open(f, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            input_files.append(line)
            else:
                input_files.append(f)

        if len(input_files) == 0:
            raise PypeItError("No input files provided.")

        if args.output is not None and len(input_files) > 1:
            raise PypeItError("--output can only be used with a single input file.")

        # Detect file type from first file
        is_spec1d = os.path.basename(input_files[0]).startswith('spec1d')

        # Validate all files are the same type
        for f in input_files:
            bn = os.path.basename(f)
            if is_spec1d and not bn.startswith('spec1d'):
                raise PypeItError("Cannot mix spec1d and spec2d files.")
            if not is_spec1d and not bn.startswith('spec2d'):
                raise PypeItError("Cannot mix spec1d and spec2d files.")

        file_type = 'spec1d' if is_spec1d else 'spec2d'
        log.info(f"Processing {len(input_files)} {file_type} file(s)")

        # Load spectrograph and fiber layout (shared across all files)
        with fits.open(input_files[0]) as hdu:
            spectrograph = load_spectrograph(hdu[0].header['PYP_SPEC'])

        targetx, targety = spectrograph.load_sky_layout()

        # ----------------------------------------------------------------
        # Process each file into a separate datacube
        # ----------------------------------------------------------------
        builder = _build_cube_from_spec1d if is_spec1d else _build_cube
        for input_file in input_files:
            log.info(f"Building datacube for {os.path.basename(input_file)}")
            builder(input_file, args, spectrograph, targetx, targety)


def _load_flat(spec2d: Spec2DObj, det_name: str,
               log: logging.Logger) -> np.ndarray | None:
    """Load the flat field image associated with a Spec2DObj.

    Parameters
    ----------
    spec2d : :class:`~pypeit.spec2dobj.Spec2DObj`
        The spec2d object (for one detector).
    det_name : str
        Detector name, e.g. ``'DET01'``.
    log : :class:`~pypeit.pypmsgs.PypeItLogger`
        Logger instance.

    Returns
    -------
    `numpy.ndarray`_ or None
        The raw (unnormalized) pixel flat image, or None if unavailable.
    """
    from pathlib import Path
    from pypeit.flatfield import FlatImages

    if not hasattr(spec2d, 'calibs') or spec2d.calibs is None:
        return None

    calib_dir = spec2d.calibs.get('DIR')
    flat_file = spec2d.calibs.get('FLAT')
    if calib_dir is None or flat_file is None:
        return None

    flat_path = Path(calib_dir) / flat_file
    if not flat_path.exists():
        log.warning(f"    Flat field file not found: {flat_path}")
        return None

    try:
        flatimages = FlatImages.from_file(str(flat_path))
        if flatimages.pixelflat_raw is not None:
            return flatimages.pixelflat_raw
        log.warning(f"    pixelflat_raw is None in {flat_path}")
        return None
    except Exception as e:
        log.warning(f"    Error loading flat field: {e}")
        return None


def _build_empirical_profiles(flatimg: np.ndarray, slitid_img: np.ndarray,
                              spat_ids: np.ndarray, nspec: int, nspat: int,
                              log: logging.Logger) -> list[np.ndarray] | None:
    """Build empirical spatial profiles for each fiber from the flat field.

    For each fiber, the cross-sectional profile is extracted from the flat
    field image by taking the median across all spectral rows (for robust
    S/N), then normalizing to unit sum.  This captures the true fiber
    spatial profile shape without assuming a functional form.

    Parameters
    ----------
    flatimg : `numpy.ndarray`_
        Raw (unnormalized) flat field image, shape ``(nspec, nspat)``.
    slitid_img : `numpy.ndarray`_
        Slit ID image from ``slits.slit_img(pad=0)``.
    spat_ids : `numpy.ndarray`_
        Array of spatial IDs for each fiber.
    nspec : int
        Number of spectral pixels.
    nspat : int
        Number of spatial pixels.
    log : :class:`~pypeit.pypmsgs.PypeItLogger`
        Logger instance.

    Returns
    -------
    list of `numpy.ndarray`_ or None
        List of length ``nfibers``, where each element is a 1D array of
        length ``nspat`` giving the normalized profile for that fiber
        (zero outside the slit).  Returns None if profiles could not be
        built.
    """
    nfibers = len(spat_ids)
    profiles: list[np.ndarray] = []

    for i, spat_id in enumerate(spat_ids):
        # Full-width profile array for this fiber (indexed by spatial pixel)
        prof = np.zeros(nspat)
        onslit = slitid_img == spat_id
        if not np.any(onslit):
            profiles.append(prof)
            continue

        # For each spatial column that belongs to this fiber, take the
        # median flat value across all spectral rows
        cols = np.where(np.any(onslit, axis=0))[0]
        for c in cols:
            rows_on = onslit[:, c]
            vals = flatimg[rows_on, c]
            good = np.isfinite(vals) & (vals > 0)
            if np.any(good):
                prof[c] = np.median(vals[good])

        # Normalize to sum=1
        psum = np.sum(prof)
        if psum > 0:
            prof /= psum
        profiles.append(prof)

    # Sanity check: most fibers should have non-zero profiles
    n_good = sum(1 for p in profiles if np.sum(p) > 0)
    if n_good < nfibers * 0.5:
        log.warning(f"    Only {n_good}/{nfibers} fibers have valid "
                    f"empirical profiles")
        return None

    return profiles


def _build_cube_from_spec1d(spec1d_file: str, args: argparse.Namespace,
                            spectrograph: Spectrograph,
                            targetx: np.ndarray, targety: np.ndarray) -> None:
    """Build a single datacube from one spec1d file.

    Reads the already-extracted 1D fiber spectra from a PypeIt spec1d file
    and builds a datacube.  Uses OPT (optimal) extraction by default, or
    BOX (boxcar) extraction if ``--boxcar`` is specified.

    Parameters
    ----------
    spec1d_file : str
        Path to the spec1d FITS file.
    args : `argparse.Namespace`_
        Parsed command-line arguments.
    spectrograph : :class:`~pypeit.spectrographs.spectrograph.Spectrograph`
        Spectrograph instance.
    targetx : `numpy.ndarray`_
        Fiber x positions on sky (arcsec).
    targety : `numpy.ndarray`_
        Fiber y positions on sky (arcsec).
    """
    import os

    import numpy as np
    from astropy.io import fits

    from pypeit import log, PypeItError
    from pypeit.specobjs import SpecObjs

    sobjs = SpecObjs.from_fitsfile(spec1d_file)
    if sobjs.nobj == 0:
        log.warning(f"No objects in {os.path.basename(spec1d_file)}, skipping")
        return

    # Choose extraction type
    prefix = 'BOX' if args.boxcar else 'OPT'
    log.info(f"  Using {prefix} extraction from spec1d")

    # ------------------------------------------------------------------
    # Step 1: Organize fiber spectra by detector
    # ------------------------------------------------------------------
    det_fiber_data = {}

    for det_name in ['DET01', 'DET02']:
        det_sobjs = sobjs[sobjs.DET == det_name]
        if len(det_sobjs) == 0:
            log.warning(f"  No objects for {det_name}, skipping")
            continue

        nfibers = len(det_sobjs)
        log.info(f"  {det_name}: {nfibers} fibers from spec1d")

        # Read extracted spectra
        wave_key = f'{prefix}_WAVE'
        flux_key = f'{prefix}_COUNTS'
        ivar_key = f'{prefix}_COUNTS_IVAR'
        sky_key = f'{prefix}_COUNTS_SKY'

        # Determine spectral length from first object
        nspec = getattr(det_sobjs[0], wave_key).shape[0]

        fiber_flux = np.zeros((nfibers, nspec))
        fiber_ivar = np.zeros((nfibers, nspec))
        fiber_wave = np.zeros((nfibers, nspec))
        fiber_sky = np.zeros((nfibers, nspec))
        spat_ids = np.zeros(nfibers, dtype=int)

        for i, sobj in enumerate(det_sobjs):
            fiber_wave[i] = getattr(sobj, wave_key)
            fiber_flux[i] = getattr(sobj, flux_key)
            fiber_ivar[i] = getattr(sobj, ivar_key)
            sky = getattr(sobj, sky_key, None)
            if sky is not None:
                fiber_sky[i] = sky
            spat_ids[i] = sobj.SLITID

        # Get fiber metadata
        fiber_meta = spectrograph.get_fiber_metadata(
            int(det_name.replace('DET', '')), spat_ids)

        det_fiber_data[det_name] = {
            'flux': fiber_flux,
            'ivar': fiber_ivar,
            'wave': fiber_wave,
            'sky': fiber_sky,
            'fiber_meta': fiber_meta,
        }

    if len(det_fiber_data) == 0:
        log.warning(f"No detector data from "
                    f"{os.path.basename(spec1d_file)}, skipping")
        return

    # ------------------------------------------------------------------
    # Steps 2-7: shared with spec2d path
    # ------------------------------------------------------------------
    with fits.open(spec1d_file) as hdu:
        raw_hdr = hdu[0].header

    _build_cube_common(det_fiber_data, args, spectrograph,
                       targetx, targety, raw_hdr, spec1d_file,
                       sky_already_subtracted=True)


def _build_cube(spec2d_file: str, args: argparse.Namespace,
                spectrograph: Spectrograph,
                targetx: np.ndarray, targety: np.ndarray) -> None:
    """Build a single datacube from one spec2d file."""
    import os

    import numpy as np
    from astropy.io import fits

    from pypeit import log, PypeItError
    from pypeit.spec2dobj import AllSpec2DObj

    allspec = AllSpec2DObj.from_fits(spec2d_file)

    # ------------------------------------------------------------------
    # Step 1: Extract 1D fiber spectra from each detector
    # ------------------------------------------------------------------
    det_fiber_data = {}

    for det_name in ['DET01', 'DET02']:
        if det_name not in allspec.detectors:
            log.warning(f"Detector {det_name} not found in "
                        f"{os.path.basename(spec2d_file)}, skipping")
            continue

        log.info(f"  Extracting fibers from {det_name}")

        spec2d = allspec[det_name]
        sciimg = spec2d.sciimg
        skymodel = spec2d.skymodel
        ivarraw = spec2d.ivarraw
        waveimg = spec2d.waveimg
        slits = spec2d.slits
        bpmmask = spec2d.bpmmask

        nspec, nspat = sciimg.shape
        slitid_img = slits.slit_img(pad=0)
        spat_ids = slits.spat_id

        nfibers = len(spat_ids)
        log.info(f"    Found {nfibers} fiber traces")

        # Compute fiber trace centers at each spectral row
        left = slits.left_init
        right = slits.right_init
        trace_centers = (left + right) / 2.0  # (nspec, nfibers)
        trace_sigma = (right - left) / (2.0 * 2.3548)  # FWHM -> sigma

        # ----------------------------------------------------------
        # Build extraction profiles: empirical from flat or Gaussian
        # ----------------------------------------------------------
        # empirical_profiles[i] is a 1D array giving the normalized
        # spatial profile for fiber i, measured from the flat field.
        # If the flat cannot be loaded, fall back to Gaussian.
        empirical_profiles = None
        use_gaussian = args.boxcar or args.gaussian

        if not use_gaussian:
            # Try to load the flat field from calibrations
            flatimg = _load_flat(spec2d, det_name, log)
            if flatimg is not None:
                empirical_profiles = _build_empirical_profiles(
                    flatimg, slitid_img, spat_ids, nspec, nspat, log)
                if empirical_profiles is not None:
                    log.info(f"    Built empirical extraction profiles "
                             f"from flat field")
                else:
                    log.warning(f"    Failed to build empirical profiles, "
                                f"falling back to Gaussian")
            else:
                log.warning(f"    Flat field not available, "
                            f"falling back to Gaussian")

        fiber_flux = np.zeros((nfibers, nspec))
        fiber_ivar = np.zeros((nfibers, nspec))
        fiber_wave = np.zeros((nfibers, nspec))
        fiber_sky = np.zeros((nfibers, nspec))

        if args.boxcar:
            log.info(f"    Using boxcar extraction")
        elif empirical_profiles is not None:
            log.info(f"    Using optimal extraction with empirical profiles")
        else:
            log.info(f"    Using optimal extraction with Gaussian profiles")

        for i, spat_id in enumerate(spat_ids):
            onslit = slitid_img == spat_id
            if not np.any(onslit):
                continue

            # Mask bad pixels
            # bpmmask is an ImageBitMaskArray; 0 means good
            good = onslit & (bpmmask.mask == 0)

            for row in range(nspec):
                pix = good[row, :]
                if not np.any(pix):
                    continue

                fiber_wave[i, row] = np.median(waveimg[row, pix])

                if args.boxcar:
                    # Boxcar: simple sum
                    fiber_flux[i, row] = np.sum(sciimg[row, pix])
                    fiber_sky[i, row] = np.sum(skymodel[row, pix])
                    ivar_pix = ivarraw[row, pix]
                    good_ivar = ivar_pix > 0
                    if np.any(good_ivar):
                        fiber_ivar[i, row] = 1.0 / np.sum(
                            1.0 / ivar_pix[good_ivar])
                else:
                    # Optimal extraction (Horne 1986)
                    cols = np.where(pix)[0]

                    if empirical_profiles is not None:
                        # Empirical profile from flat field
                        profile = empirical_profiles[i][cols]
                    else:
                        # Gaussian profile from trace geometry
                        sig = trace_sigma[row, i]
                        if sig < 0.1:
                            sig = 1.0
                        cen = trace_centers[row, i]
                        profile = np.exp(
                            -0.5 * ((cols - cen) / sig) ** 2)

                    psum = np.sum(profile)
                    if psum <= 0:
                        continue
                    profile = profile / psum  # normalize to sum=1

                    iv = ivarraw[row, cols]
                    good_iv = iv > 0

                    if not np.any(good_iv):
                        continue

                    # Horne Eq. 8: flux = sum(P * ivar * data) / sum(P^2 * ivar)
                    denom = np.sum(profile[good_iv] ** 2 * iv[good_iv])
                    if denom <= 0:
                        continue
                    fiber_flux[i, row] = (
                        np.sum(profile[good_iv] * iv[good_iv]
                               * sciimg[row, cols[good_iv]]) / denom)
                    fiber_sky[i, row] = (
                        np.sum(profile[good_iv] * iv[good_iv]
                               * skymodel[row, cols[good_iv]]) / denom)
                    # Horne Eq. 9: var = sum(P) / sum(P^2 * ivar)
                    fiber_ivar[i, row] = denom / np.sum(profile[good_iv])

        # Get fiber metadata (IDs, names, types) from the spectrograph
        fiber_meta = spectrograph.get_fiber_metadata(
            int(det_name.replace('DET', '')), spat_ids)

        det_fiber_data[det_name] = {
            'flux': fiber_flux,
            'ivar': fiber_ivar,
            'wave': fiber_wave,
            'sky': fiber_sky,
            'slits': slits,
            'fiber_meta': fiber_meta,
        }

    if len(det_fiber_data) == 0:
        log.warning(f"No detector data extracted from "
                    f"{os.path.basename(spec2d_file)}, skipping")
        return

    # ------------------------------------------------------------------
    # Steps 2-7: shared with spec1d path
    # ------------------------------------------------------------------
    with fits.open(spec2d_file) as hdu:
        raw_hdr = hdu[0].header

    _build_cube_common(det_fiber_data, args, spectrograph,
                       targetx, targety, raw_hdr, spec2d_file,
                       sky_already_subtracted=False)


def _build_cube_common(det_fiber_data: dict, args: argparse.Namespace,
                       spectrograph: Spectrograph,
                       targetx: np.ndarray, targety: np.ndarray,
                       raw_hdr, input_file: str,
                       sky_already_subtracted: bool = False) -> None:
    """Shared steps for building a datacube from fiber spectra.

    Parameters
    ----------
    det_fiber_data : dict
        Per-detector fiber data. Keys are detector names (e.g. ``'DET01'``),
        values are dicts with keys ``'flux'``, ``'ivar'``, ``'wave'``,
        ``'sky'``, ``'fiber_meta'``.
    args : `argparse.Namespace`_
        Parsed command-line arguments.
    spectrograph : :class:`~pypeit.spectrographs.spectrograph.Spectrograph`
        Spectrograph instance.
    targetx : `numpy.ndarray`_
        Fiber x positions on sky (arcsec).
    targety : `numpy.ndarray`_
        Fiber y positions on sky (arcsec).
    raw_hdr : `astropy.io.fits.Header`_
        Primary header from the input file.
    input_file : str
        Path to the input file (for generating the output filename).
    sky_already_subtracted : bool, optional
        If True, the flux arrays are already sky-subtracted (e.g. from
        spec1d) and sky subtraction is skipped.
    """
    import os

    import numpy as np
    from astropy import units
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy import wcs
    from scipy.interpolate import griddata
    from scipy.spatial import QhullError

    from pypeit import log

    # ------------------------------------------------------------------
    # Identify sky fibers and sky-subtract
    # ------------------------------------------------------------------
    for det_name, data in det_fiber_data.items():
        nfibers = data['flux'].shape[0]

        # Derive sky mask from fiber metadata
        fiber_meta = data['fiber_meta']
        sky_mask = fiber_meta['fiber_type'] == 'SKY'
        n_sky = np.sum(sky_mask)
        n_sci = np.sum(~sky_mask)
        log.info(f"  {det_name}: {n_sky} sky fibers, {n_sci} science fibers")

        if sky_already_subtracted:
            log.info(f"  Sky already subtracted (spec1d), skipping")
        elif not args.no_skysub:
            if args.use_fibers and n_sky > 0:
                log.info(f"  Computing sky spectrum from {n_sky} sky fibers")

                # Sigma-clipped mean of sky fiber spectra
                sky_spectra = data['flux'][sky_mask]
                # Use 3-sigma clipping matching IDL resistant_mean
                sky_mean = np.zeros(data['flux'].shape[1])
                for col in range(len(sky_mean)):
                    vals = sky_spectra[:, col]
                    good = vals != 0
                    if np.sum(good) >= 3:
                        mean, _, _ = sigma_clipped_stats(
                            vals[good], sigma=3.0)
                        sky_mean[col] = mean
                    elif np.sum(good) > 0:
                        sky_mean[col] = np.mean(vals[good])

                # Subtract sky from all fibers
                data['flux'] -= sky_mean[np.newaxis, :]

                # Propagate variance (sky subtraction adds sky variance)
                sky_ivar_spectra = data['ivar'][sky_mask]
                sky_var = np.zeros(data['flux'].shape[1])
                for col in range(len(sky_var)):
                    ivals = sky_ivar_spectra[:, col]
                    good = ivals > 0
                    if np.sum(good) > 0:
                        sky_var[col] = 1.0 / np.sum(ivals[good])
                old_var = np.where(data['ivar'] > 0,
                                  1.0 / data['ivar'], 0.0)
                new_var = old_var + sky_var[np.newaxis, :]
                data['ivar'] = np.where(new_var > 0,
                                        1.0 / new_var, 0.0)
            else:
                # Default: use the per-fiber sky from PypeIt's skymodel
                log.info(f"  Subtracting sky using spec2d skymodel")
                data['flux'] -= data['sky']

        # Store masks
        data['sky_mask'] = sky_mask
        data['sci_mask'] = ~sky_mask

    # ------------------------------------------------------------------
    # Step 4: Wavelength linearization
    # ------------------------------------------------------------------
    # Find global wavelength range across all detectors
    all_waves = []
    for data in det_fiber_data.values():
        wave = data['wave']
        valid = wave > 0
        if np.any(valid):
            all_waves.extend([np.min(wave[valid]), np.max(wave[valid])])

    if len(all_waves) == 0:
        log.error("No valid wavelength data found. "
                  "Check that input files contain extracted fiber spectra.")
        return
    wave_min = min(all_waves[::2])
    wave_max = max(all_waves[1::2])

    # Use median dispersion for wavelength step
    dispersions = []
    for data in det_fiber_data.values():
        wave = data['wave']
        for i in range(wave.shape[0]):
            valid = wave[i] > 0
            if np.sum(valid) > 10:
                dw = np.diff(wave[i, valid])
                dw = dw[dw > 0]
                if len(dw) > 0:
                    dispersions.append(np.median(dw))
                break  # One fiber is enough per detector

    dwv = np.median(dispersions)
    n_wave = int(np.ceil((wave_max - wave_min) / dwv)) + 1
    wave_grid = np.linspace(wave_min, wave_min + (n_wave - 1) * dwv, n_wave)
    log.info(f"Wavelength grid: {wave_min:.1f} to {wave_grid[-1]:.1f} A, "
              f"dw={dwv:.3f} A, {n_wave} pixels")

    # Resample each fiber onto common wavelength grid
    for data in det_fiber_data.values():
        nfibers = data['flux'].shape[0]
        flux_resamp = np.zeros((nfibers, n_wave))
        ivar_resamp = np.zeros((nfibers, n_wave))

        for i in range(nfibers):
            valid = data['wave'][i] > 0
            if np.sum(valid) < 10:
                continue
            w = data['wave'][i, valid]
            f = data['flux'][i, valid]
            iv = data['ivar'][i, valid]

            # Sort by wavelength
            srt = np.argsort(w)
            w, f, iv = w[srt], f[srt], iv[srt]

            # Linear interpolation onto common grid
            in_range = (wave_grid >= w[0]) & (wave_grid <= w[-1])
            flux_resamp[i, in_range] = np.interp(wave_grid[in_range], w, f)
            ivar_resamp[i, in_range] = np.interp(wave_grid[in_range], w, iv)

        data['flux_resamp'] = flux_resamp
        data['ivar_resamp'] = ivar_resamp

    # ------------------------------------------------------------------
    # Step 5: Combine both detectors
    # ------------------------------------------------------------------
    # Map science fibers to layout file positions
    sci_flux_list = []
    sci_ivar_list = []
    layout_idx_list = []

    for det_name in sorted(det_fiber_data.keys()):
        data = det_fiber_data[det_name]
        det_num = int(det_name.replace('DET', ''))

        sci_mask = data['sci_mask']
        fiber_meta = data['fiber_meta']
        layout_indices = spectrograph.get_science_fiber_layout_indices(
            det_num, fiber_meta['fiber_id'], fiber_meta['fiber_type'])

        sci_flux = data['flux_resamp'][sci_mask]
        sci_ivar = data['ivar_resamp'][sci_mask]
        sci_layout = layout_indices[sci_mask]

        # Remove any fibers with invalid layout indices
        valid = sci_layout >= 0
        sci_flux_list.append(sci_flux[valid])
        sci_ivar_list.append(sci_ivar[valid])
        layout_idx_list.append(sci_layout[valid])

    combined_flux = np.vstack(sci_flux_list)
    combined_ivar = np.vstack(sci_ivar_list)
    combined_layout = np.concatenate(layout_idx_list)

    n_sci_fibers = combined_flux.shape[0]
    log.info(f"Combined {n_sci_fibers} science fibers from "
              f"{len(det_fiber_data)} detector(s)")

    # Trim wavelength range to where a reasonable fraction of fibers
    # have valid data (avoids degenerate interpolation at edges)
    n_valid = np.sum((combined_flux != 0) | (combined_ivar > 0), axis=0)
    min_fibers = max(10, int(0.10 * n_sci_fibers))
    good_wave = n_valid >= min_fibers
    if not np.all(good_wave):
        first = np.argmax(good_wave)
        last = n_wave - 1 - np.argmax(good_wave[::-1])
        log.info(f"Trimming wavelength range: slices {first}-{last} of "
                  f"{n_wave} (>={min_fibers} fibers required)")
        wave_grid = wave_grid[first:last + 1]
        combined_flux = combined_flux[:, first:last + 1]
        combined_ivar = combined_ivar[:, first:last + 1]
        n_wave = len(wave_grid)

    # Load fiber sky positions
    fiber_x = targetx[combined_layout]
    fiber_y = targety[combined_layout]

    # ------------------------------------------------------------------
    # Step 6: Build datacube via spatial interpolation
    # ------------------------------------------------------------------
    # IDL scaling: positions / 3.0, then offset and scale
    scl = args.spatial_scale
    x_scaled = fiber_x / 3.0
    y_scaled = fiber_y / 3.0

    # Compute grid dimensions from scaled positions
    x_min, x_max = np.min(x_scaled), np.max(x_scaled)
    y_min, y_max = np.min(y_scaled), np.max(y_scaled)

    # Add small padding
    pad = scl
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    nx = int(np.ceil((x_max - x_min) / scl)) + 1
    ny = int(np.ceil((y_max - y_min) / scl)) + 1

    log.info(f"Output cube dimensions: {nx} x {ny} x {n_wave}")

    # Build regular grid
    x_grid = np.linspace(x_min, x_min + (nx - 1) * scl, nx)
    y_grid = np.linspace(y_min, y_min + (ny - 1) * scl, ny)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')

    # Interpolate at each wavelength
    points = np.column_stack([x_scaled, y_scaled])
    cube = np.zeros((nx, ny, n_wave), dtype=np.float32)
    var_cube = np.zeros((nx, ny, n_wave), dtype=np.float32)

    log.info(f"Interpolating {n_wave} wavelength slices using "
              f"method='{args.method}'...")
    for k in range(n_wave):
        if k % 500 == 0:
            log.info(f"  Wavelength slice {k}/{n_wave}")

        flux_slice = combined_flux[:, k]
        ivar_slice = combined_ivar[:, k]

        # Only interpolate fibers with valid data
        good = (flux_slice != 0) | (ivar_slice > 0)
        if np.sum(good) < 4:
            continue

        try:
            cube[:, :, k] = griddata(
                points[good], flux_slice[good], (grid_x, grid_y),
                method=args.method, fill_value=0.0)
        except QhullError:
            # Fall back to nearest-neighbor when valid points are
            # degenerate (e.g. collinear at spectral edges)
            cube[:, :, k] = griddata(
                points[good], flux_slice[good], (grid_x, grid_y),
                method='nearest', fill_value=0.0)

        # Interpolate variance
        var_slice = np.where(ivar_slice > 0, 1.0 / ivar_slice, 0.0)
        if np.any(var_slice[good] > 0):
            try:
                var_cube[:, :, k] = griddata(
                    points[good], var_slice[good], (grid_x, grid_y),
                    method=args.method, fill_value=0.0)
            except QhullError:
                var_cube[:, :, k] = griddata(
                    points[good], var_slice[good], (grid_x, grid_y),
                    method='nearest', fill_value=0.0)

    # ------------------------------------------------------------------
    # Step 7: Build WCS and write output
    # ------------------------------------------------------------------
    # Pointing
    raval = raw_hdr.get('RA', 0.0)
    decval = raw_hdr.get('DEC', 0.0)
    try:
        coord = SkyCoord(raval, decval, unit=(units.hourangle, units.deg))
    except Exception:
        coord = SkyCoord(raval, decval, unit=(units.deg, units.deg))

    posang = raw_hdr.get('POSANG', 0.0)
    crota = np.radians(-posang)

    cdelt1 = -scl / 3600.0  # RA decreases with x, degrees
    cdelt2 = scl / 3600.0   # DEC increases with y, degrees

    cd11 = cdelt1 * np.cos(crota)
    cd12 = abs(cdelt2) * np.sign(cdelt1) * np.sin(crota)
    cd21 = -abs(cdelt1) * np.sign(cdelt2) * np.sin(crota)
    cd22 = cdelt2 * np.cos(crota)

    w = wcs.WCS(naxis=3)
    w.wcs.equinox = raw_hdr.get('EQUINOX', 2000.0)
    w.wcs.name = 'Binospec IFU'
    w.wcs.radesys = 'ICRS'
    w.wcs.cname = ['RA', 'DEC', 'Wavelength']
    w.wcs.cunit = [units.degree, units.degree, units.Angstrom]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
    w.wcs.crval = [coord.ra.degree, coord.dec.degree, wave_grid[0]]
    w.wcs.crpix = [nx / 2.0, ny / 2.0, 1.0]
    w.wcs.cd = np.array([[cd11, cd12, 0.0],
                         [cd21, cd22, 0.0],
                         [0.0, 0.0, dwv]])
    w.wcs.lonpole = 180.0
    w.wcs.latpole = 0.0

    # Build output FITS
    hdr = w.to_header()
    hdr['INSTRUME'] = 'BINOSPEC'
    hdr['TELESCOP'] = 'MMT'
    hdr['IFUMODE'] = 'FIBER'
    hdr['NFIBERS'] = (n_sci_fibers, 'Number of science fibers')
    hdr['SPATSCL'] = (scl, 'Spatial pixel scale [arcsec]')
    hdr['WAVEMIN'] = (wave_grid[0], 'Minimum wavelength [Angstrom]')
    hdr['WAVEMAX'] = (wave_grid[-1], 'Maximum wavelength [Angstrom]')
    hdr['WAVESTP'] = (dwv, 'Wavelength step [Angstrom]')
    hdr['INTERP'] = (args.method, 'Spatial interpolation method')

    # Copy useful keywords from raw header
    for key in ['OBJECT', 'EXPTIME', 'DATE-OBS', 'DISPERSE', 'FILTER']:
        if key in raw_hdr:
            hdr[key] = raw_hdr[key]

    # Output filename
    if args.output is not None:
        outfile = args.output
    else:
        base = os.path.splitext(os.path.basename(input_file))[0]
        # Handle both spec1d_ and spec2d_ prefixes
        for prefix in ['spec1d_', 'spec2d_']:
            if prefix in base:
                base = base.replace(prefix, 'cube_')
                break
        outfile = base + '.fits'

    # Transpose from numpy (nx, ny, n_wave) to FITS order (n_wave, ny, nx)
    # so that NAXIS1=nx(RA), NAXIS2=ny(DEC), NAXIS3=n_wave(WAVE)
    cube = np.transpose(cube, (2, 1, 0))
    var_cube = np.transpose(var_cube, (2, 1, 0))

    primary = fits.PrimaryHDU(header=fits.Header())
    primary.header['AUTHOR'] = 'PypeIt'
    flux_hdu = fits.ImageHDU(data=cube, header=hdr, name='FLUX')
    var_hdu = fits.ImageHDU(data=var_cube, header=hdr, name='VAR')

    hdulist = fits.HDUList([primary, flux_hdu, var_hdu])
    hdulist.writeto(outfile, overwrite=True)
    log.info(f"Wrote datacube to {outfile}")
    log.info(f"Cube shape: {cube.shape}")
