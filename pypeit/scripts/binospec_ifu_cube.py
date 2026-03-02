"""
Build a datacube from Binospec IFU spec2d files.

Unlike the general ``pypeit_coadd_datacube`` script designed for slicer-based
IFUs, this script handles the fiber-fed Binospec IFU by:

1. Extracting each fiber as a 1D spectrum from spec2d files
2. Sky subtracting using dedicated sky fiber spectra
3. Mapping 320 science fibers per side to sky positions
4. Combining both detectors (640 science fibers total)
5. Interpolating scattered fiber positions onto a regular spatial grid

.. include:: ../include/links.rst
"""

from pypeit.scripts import scriptbase


class BinospecIFUCube(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(
            description='Build a datacube from Binospec IFU spec2d files.',
            width=width,
            default_log_file=True)
        parser.add_argument('files', type=str, nargs='+',
                            help='One or more PypeIt spec2d files, or a text file '
                                 'listing spec2d files (one per line)')
        parser.add_argument('--det', type=str, nargs='+', default=['DET01', 'DET02'],
                            help='Detector(s) to process (default: DET01 DET02)')
        parser.add_argument('-o', '--output', type=str, default=None,
                            help='Output FITS filename (default: auto-generated)')
        parser.add_argument('--spatial_scale', type=float, default=0.27,
                            help='Output spatial pixel scale in arcsec (default: 0.27)')
        parser.add_argument('--no_skysub', default=False, action='store_true',
                            help='Skip sky subtraction')
        parser.add_argument('--method', type=str, default='linear',
                            choices=['nearest', 'linear', 'cubic'],
                            help='Spatial interpolation method (default: linear)')
        return parser

    @classmethod
    def main(cls, args):
        import os

        import numpy as np
        from astropy import units
        from astropy.coordinates import SkyCoord
        from astropy.io import fits
        from astropy.stats import sigma_clipped_stats
        from astropy import wcs
        from scipy.interpolate import griddata

        from pypeit import log, PypeItError
        from pypeit.spec2dobj import AllSpec2DObj
        from pypeit.spectrographs.util import load_spectrograph

        cls.init_log(args)

        # ----------------------------------------------------------------
        # Parse input files
        # ----------------------------------------------------------------
        spec2d_files = []
        for f in args.files:
            if f.endswith('.txt'):
                with open(f, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            spec2d_files.append(line)
            else:
                spec2d_files.append(f)

        if len(spec2d_files) == 0:
            raise PypeItError("No spec2d files provided.")

        log.info(f"Processing {len(spec2d_files)} spec2d file(s)")

        # Load spectrograph from first file's primary header
        with fits.open(spec2d_files[0]) as hdu:
            spectrograph = load_spectrograph(hdu[0].header['PYP_SPEC'])

        # ----------------------------------------------------------------
        # Step 1: Extract 1D fiber spectra from each detector
        # ----------------------------------------------------------------
        det_fiber_data = {}  # keyed by det name

        for det_name in args.det:
            log.info(f"Processing detector {det_name}")

            all_flux = []
            all_ivar = []
            all_wave = []
            all_sky = []

            for spec2d_file in spec2d_files:
                allspec = AllSpec2DObj.from_fits(spec2d_file)

                if det_name not in allspec.detectors:
                    log.warning(f"Detector {det_name} not found in {spec2d_file}, skipping")
                    continue

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
                log.info(f"  Found {nfibers} fiber traces in {os.path.basename(spec2d_file)}")

                # Extract each fiber via boxcar sum
                fiber_flux = np.zeros((nfibers, nspec))
                fiber_ivar = np.zeros((nfibers, nspec))
                fiber_wave = np.zeros((nfibers, nspec))
                fiber_sky = np.zeros((nfibers, nspec))

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
                        fiber_flux[i, row] = np.sum(sciimg[row, pix])
                        fiber_sky[i, row] = np.sum(skymodel[row, pix])
                        # Sum ivar: for boxcar, 1/var_sum = 1/sum(var_i)
                        ivar_pix = ivarraw[row, pix]
                        good_ivar = ivar_pix > 0
                        if np.any(good_ivar):
                            fiber_ivar[i, row] = 1.0 / np.sum(
                                1.0 / ivar_pix[good_ivar])
                        fiber_wave[i, row] = np.median(waveimg[row, pix])

                all_flux.append(fiber_flux)
                all_ivar.append(fiber_ivar)
                all_wave.append(fiber_wave)
                all_sky.append(fiber_sky)

            if len(all_flux) == 0:
                log.warning(f"No data for detector {det_name}")
                continue

            # For now, use first file's data (coadding multiple exposures is
            # a future enhancement)
            det_fiber_data[det_name] = {
                'flux': all_flux[0],
                'ivar': all_ivar[0],
                'wave': all_wave[0],
                'sky': all_sky[0],
                'slits': slits,
                'spec2d_file': spec2d_files[0],
            }

        if len(det_fiber_data) == 0:
            raise PypeItError("No detector data extracted.")

        # ----------------------------------------------------------------
        # Step 2 & 3: Identify sky fibers and sky-subtract
        # ----------------------------------------------------------------
        for det_name, data in det_fiber_data.items():
            det_num = int(det_name.replace('DET', ''))
            nfibers = data['flux'].shape[0]

            sky_mask = spectrograph.get_sky_fiber_mask(det_num, nfibers)
            n_sky = np.sum(sky_mask)
            n_sci = np.sum(~sky_mask)
            log.info(f"  {det_name}: {n_sky} sky fibers, {n_sci} science fibers")

            if not args.no_skysub and n_sky > 0:
                log.info(f"  Computing sky spectrum from {n_sky} sky fibers")

                # Sigma-clipped mean of sky fiber spectra
                sky_spectra = data['flux'][sky_mask]
                # Use 3-sigma clipping matching IDL resistant_mean
                sky_mean = np.zeros(data['flux'].shape[1])
                for col in range(len(sky_mean)):
                    vals = sky_spectra[:, col]
                    good = vals != 0
                    if np.sum(good) >= 3:
                        mean, _, _ = sigma_clipped_stats(vals[good], sigma=3.0)
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
                sci_mask = ~sky_mask
                old_var = np.where(data['ivar'] > 0, 1.0 / data['ivar'], 0.0)
                new_var = old_var + sky_var[np.newaxis, :]
                data['ivar'] = np.where(new_var > 0, 1.0 / new_var, 0.0)

            # Store masks
            data['sky_mask'] = sky_mask
            data['sci_mask'] = ~sky_mask

        # ----------------------------------------------------------------
        # Step 4: Wavelength linearization
        # ----------------------------------------------------------------
        # Find global wavelength range across all detectors
        all_waves = []
        for data in det_fiber_data.values():
            wave = data['wave']
            valid = wave > 0
            if np.any(valid):
                all_waves.extend([np.min(wave[valid]), np.max(wave[valid])])

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

        # ----------------------------------------------------------------
        # Step 5: Combine both detectors
        # ----------------------------------------------------------------
        # Map science fibers to layout file positions
        sci_flux_list = []
        sci_ivar_list = []
        layout_idx_list = []

        for det_name in sorted(det_fiber_data.keys()):
            data = det_fiber_data[det_name]
            det_num = int(det_name.replace('DET', ''))

            sci_mask = data['sci_mask']
            layout_indices = spectrograph.get_science_fiber_layout_indices(
                det_num, len(sci_mask))

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

        # Load fiber sky positions
        targetx, targety = spectrograph.load_sky_layout()

        fiber_x = targetx[combined_layout]
        fiber_y = targety[combined_layout]

        # ----------------------------------------------------------------
        # Step 6: Build datacube via spatial interpolation
        # ----------------------------------------------------------------
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

            cube[:, :, k] = griddata(
                points[good], flux_slice[good], (grid_x, grid_y),
                method=args.method, fill_value=0.0)

            # Interpolate variance
            var_slice = np.where(ivar_slice > 0, 1.0 / ivar_slice, 0.0)
            if np.any(var_slice[good] > 0):
                var_cube[:, :, k] = griddata(
                    points[good], var_slice[good], (grid_x, grid_y),
                    method=args.method, fill_value=0.0)

        # ----------------------------------------------------------------
        # Step 7: Build WCS and write output
        # ----------------------------------------------------------------
        # Get header info from first spec2d file
        first_det = sorted(det_fiber_data.keys())[0]
        first_data = det_fiber_data[first_det]
        with fits.open(first_data['spec2d_file']) as hdu:
            raw_hdr = hdu[0].header

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
            base = os.path.splitext(os.path.basename(spec2d_files[0]))[0]
            outfile = base.replace('spec2d_', 'cube_') + '.fits'

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
