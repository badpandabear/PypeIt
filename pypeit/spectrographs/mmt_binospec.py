"""
Module for MMT/BINOSPEC specific methods.

.. include:: ../include/links.rst
"""
from itertools import chain
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time
from astropy import wcs
from IPython import embed
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import median_filter

from pypeit import dataPaths
from pypeit import io
from pypeit import log
from pypeit import PypeItError
from pypeit import telescopes
from pypeit import utils
from pypeit.core import framematch
from pypeit.core import parse
from pypeit.images import detector_container
from pypeit.par import parset
from pypeit.spectrographs import spectrograph
from pypeit.spectrographs.slitmask import SlitMask


class MMTBINOSPECSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle MMT/BINOSPEC specific code
    """
    ndet = 2
    name = 'mmt_binospec'

    # Per-amplifier nonlinearity correction coefficients from IDL calibration
    # file (scicam_bino_sep2017.fits, measured 2017-09). Polynomial form:
    # C_corr = c[0] + c[1]*C + c[2]*C^2 + c[3]*C^3 + c[4]*C^4
    # Compatible with np.polynomial.polynomial.polyval.
    # Shape: (8 amplifiers, 5 coefficients for degree-4 polynomial)
    nonlinearity_coeffs = np.array([
        [0.00000000e+00, 1.00400089e+00, -1.39235362e-06, 8.31711824e-12, -1.20653479e-17],
        [0.00000000e+00, 1.00361458e+00, -1.29223833e-06, 6.93723177e-12, -9.67406255e-18],
        [0.00000000e+00, 1.00269542e+00, -9.29361806e-07, 5.97902827e-12, -2.30257302e-17],
        [0.00000000e+00, 1.00339616e+00, -8.47134521e-07, 7.92441693e-12, -4.46542834e-17],
        [0.00000000e+00, 1.00727205e+00, -1.69093388e-06, 2.07225055e-11, -1.62655178e-16],
        [0.00000000e+00, 1.00858745e+00, -2.35668901e-06, 2.40641019e-11, -1.50286358e-16],
        [0.00000000e+00, 1.00728526e+00, -1.80779473e-06, 1.73427719e-11, -1.01685780e-16],
        [0.00000000e+00, 1.00845168e+00, -2.02050567e-06, 2.97587091e-11, -2.65508521e-16],
    ])
    telescope = telescopes.MMTTelescopePar()
    camera = 'BINOSPEC'
    url = 'https://lweb.cfa.harvard.edu/mmti/binospec.html'
    header_name = 'Binospec'
    supported = True

    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # Binning
        binning = '1,1' if hdu is None else self.get_meta_value(self.get_headarr(hdu), 'binning')

        # Detector 1
        detector_dict1 = dict(
                            binning         = binning,
                            det             = 1,
                            dataext         = 1,
                            specaxis        = 0,
                            specflip        = False,
                            spatflip        = False,
                            xgap            = 0.,
                            ygap            = 0.,
                            ysize           = 1.,
                            platescale      = 0.24,
                            darkcurr        = 3.6,  #e-/pixel/hour  (=0.001 e-/pixel/s)  --  pulled from the ETC
                            saturation      = 65535.,
                            nonlinear       = 0.95,  #ToDO: To Be update
                            mincounts       = -1e10,
                            numamplifiers   = 4,
                            gain            = np.atleast_1d([1.085,1.046,1.042,0.975]),
                            ronoise         = np.atleast_1d([3.2,3.2,3.2,3.2]),
                            )
        # Detector 2
        detector_dict2 = detector_dict1.copy()
        detector_dict2.update(dict(
            det=2,
            dataext=2,
            gain=np.atleast_1d([1.028,1.115,1.047,1.045]), #ToDo: FW measures 1.115 for amp2 but 1.163 in IDL pipeline
            ronoise=np.atleast_1d([3.6,3.6,3.6,3.6])
        ))

        # Instantiate
        detector_dicts = [detector_dict1, detector_dict2]
        return detector_container.DetectorContainer(**detector_dicts[det-1])

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the PypeIt-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=1, card='RA')
        self.meta['dec'] = dict(ext=1, card='DEC')
        self.meta['target'] = dict(ext=1, card='OBJECT')
        self.meta['decker'] = dict(ext=1, card='MASK')

        self.meta['dichroic'] = dict(ext=1, card=None, default='default')
        self.meta['binning'] = dict(ext=1, card='CCDSUM', compound=True)

        self.meta['mjd'] = dict(ext=1, card='MJD')
        self.meta['exptime'] = dict(ext=1, card='EXPTIME')
        self.meta['airmass'] = dict(ext=1, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['dispname'] = dict(ext=1, card='DISPERS1')
        self.meta['idname'] = dict(ext=1, card='IMAGETYP')

        # used for arclamp
        self.meta['lampstat01'] = dict(ext=1, card='HENEAR')
        # used for flatlamp, SCRN is actually telescope status
        self.meta['lampstat02'] = dict(ext=1, card='SCRN')
        self.meta['instrument'] = dict(ext=1, card='INSTRUME')

    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        if meta_key == 'binning':
            binspatial, binspec = parse.parse_binning(headarr[1]['CCDSUM'])
            binning = parse.binning2string(binspec, binspatial)
            return binning

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return ['dispname']

    def raw_header_cards(self):
        """
        Return additional raw header cards to be propagated in
        downstream output files for configuration identification.

        The list of raw data FITS keywords should be those used to populate
        the :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.configuration_keys`
        or are used in :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.config_specific_par`
        for a particular spectrograph, if different from the name of the
        PypeIt metadata keyword.

        This list is used by :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.subheader_for_spec`
        to include additional FITS keywords in downstream output files.

        Returns:
            :obj:`list`: List of keywords from the raw data files that should
            be propagated in output files.
        """
        return ['DISPERS1']

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        par = super().default_pypeit_par()

        # Wavelengths
        # 1D wavelength solution
        par['calibrations']['wavelengths']['rms_thresh_frac_fwhm'] = 0.125
        par['calibrations']['wavelengths']['sigdetect'] = 5.
        par['calibrations']['wavelengths']['fwhm']= 4.0
        par['calibrations']['wavelengths']['lamps'] = ['ArI', 'ArII']
        par['calibrations']['wavelengths']['method'] = 'full_template'
        par['calibrations']['wavelengths']['lamps'] = ['HeI', 'NeI', 'ArI', 'ArII']

        # Tilt and slit parameters
        par['calibrations']['tilts']['tracethresh'] =  10.0
        par['calibrations']['tilts']['spat_order'] = 6
        par['calibrations']['tilts']['spec_order'] = 6
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'

        # Processing steps
        turn_off = dict(use_biasimage=False, use_darkimage=False)
        par.reset_all_processimages_par(**turn_off)

        # Extraction
        par['reduce']['skysub']['bspline_spacing'] = 0.8
        par['reduce']['extraction']['sn_gauss'] = 4.0
        ## Do not perform global sky subtraction for standard stars
        par['reduce']['skysub']['global_sky_std']  = False

        par['flexure']['spec_method'] = 'boxcar'

        # cosmic ray rejection parameters for science frames
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['standardframe']['exprng'] = [None, 100]
        par['calibrations']['arcframe']['exprng'] = [20, None]
        par['calibrations']['darkframe']['exprng'] = [20, None]
        par['scienceframe']['exprng'] = [20, None]

        # Sensitivity function parameters
        par['sensfunc']['polyorder'] = 7
        par['sensfunc']['IR']['telgridfile'] = 'TellPCA_3000_26000_R10000.fits'

        return par

    def config_specific_par(
            self,
            inp:str|list|Path|fits.Header|Table,
            inp_par:parset.ParSet|None=None
        ) -> parset.ParSet:
        """
        Modify the PypeIt parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            inp (:obj:`str`, :obj:`list`, `Path`_, `astropy.io.fits.Header`_, `astropy.table.Table`_):
                Input filename, an `astropy.io.fits.Header`_ object, or a list
                of `astropy.io.fits.Header`_ objects.  Or a row from the
                metadata table.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        # Start with instrument-wide parameters
        par = super().config_specific_par(inp, inp_par=inp_par)

        # Adjust parameters based on instrument configuration
        grating = self.get_meta_value(inp, 'dispname')
        decker = self.get_meta_value(inp, 'decker')

        # wavelengths
        match grating:
            case 'x270':
                par['calibrations']['wavelengths']['reid_arxiv'] = 'mmt_binospec_270.fits'
            case 'x600':
                par['calibrations']['wavelengths']['reid_arxiv'] = 'mmt_binospec_600.fits'
            case 'x1000':
                par['calibrations']['wavelengths']['reid_arxiv'] = 'mmt_binospec_1000.fits'

        if 'Longslit' in decker:
            # Observations use a longslit so we skip the parameters primarily
            # used for multislit data
            return par

        # Turn on the use of mask design
        par['calibrations']['slitedges']['use_maskdesign'] = True
        # Since we use the slitmask info to find the alignment boxes, I don't need `minimum_slit_length_sci`
        par['calibrations']['slitedges']['minimum_slit_length_sci'] = None
        # Sometime the added missing slits at the edge of the detector are to small to be useful.
        par['calibrations']['slitedges']['minimum_slit_length'] = 3.
        # Since we use the slitmask info to add and remove traces, 'minimum_slit_gap' may undo the matching effort.
        par['calibrations']['slitedges']['minimum_slit_gap'] = 0.
        # Lower edge_thresh works better
        par['calibrations']['slitedges']['edge_thresh'] = 10.
        # Assign RA, DEC, OBJNAME to detected objects
        par['reduce']['slitmask']['assign_obj'] = True
        # force extraction of undetected objects
        par['reduce']['slitmask']['extract_missing_objs'] = True
        # Adjust sky subtraction parameters

        # lower tilts spat_order and higher spec_order for multislits (i.e., generally not very long slits)
        par['calibrations']['tilts']['spat_order'] = 2  # Default: 3
        par['calibrations']['tilts']['spec_order'] = 5  # Default: 4
        # pca
        par['calibrations']['slitedges']['sync_predict'] = 'auto'

        par['coadd2d']['offsets'] = 'maskdef_offsets'

        return par

    def update_edgetracepar(self, par):
        """
        This method is used in :func:`pypeit.edgetrace.EdgeTraceSet.maskdesign_matching`
        to update EdgeTraceSet parameters when the slitmask design matching is not feasible
        because too few slits are present in the detector.

        Args:
            par (:class:`pypeit.par.pypeitpar.EdgeTracePar`):
                The parameters used to guide slit tracing.

        Returns:
            :class:`pypeit.par.pypeitpar.EdgeTracePar`
            The modified parameters used to guide slit tracing.
        """

        par['minimum_slit_gap'] = 0.25
        par['minimum_slit_length_sci'] = 4.5
        return par

    def bpm(self, filename, det, shape=None, msbias=None):
        """
        Generate a default bad-pixel mask.

        Loads a pre-built static BPM from the IDL pipeline calibration data
        (``badpix_binospec.fits`` + hard-coded bad columns and detector trap
        regions from ``bino_mosaic.pro``).

        Even though they are both optional, either the precise shape for
        the image (``shape``) or an example file that can be read to get
        the shape (``filename`` using :func:`get_image_shape`) *must* be
        provided.

        Args:
            filename (:obj:`str` or None):
                An example file to use to get the image shape.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (tuple, optional):
                Processed image shape
                Required if filename is None
                Ignored if filename is not None
            msbias (`numpy.ndarray`_, optional):
                Processed bias frame used to identify bad pixels

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """
        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)

        # Load and apply the static BPM from IDL pipeline calibration
        bpm_file = dataPaths.static_calibs.get_file_path(
            f'mmt_binospec/bpm_binospec_det{det}.fits.gz')
        static_bpm = fits.getdata(bpm_file)
        bpm_img |= static_bpm

        return bpm_img

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        if ftype == 'science':
            return good_exp & (fitstbl['lampstat01'] == 'off') & (fitstbl['lampstat02'] == 'stowed') & (fitstbl['exptime'] > 100.0)
        if ftype == 'standard':
            return good_exp & (fitstbl['lampstat01'] == 'off') & (fitstbl['lampstat02'] == 'stowed') & (fitstbl['exptime'] <= 100.0)
        if ftype in ['arc', 'tilt']:
            return good_exp & (fitstbl['lampstat01'] == 'on')
        if ftype in ['pixelflat', 'trace', 'illumflat']:
            return good_exp & (fitstbl['lampstat01'] == 'off') & (fitstbl['lampstat02'] == 'deployed')

        log.debug('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

    def get_slitmask(self, filename:str, det:int=1):
        """
        Parse the slitmask data from a raw file into :attr:`slitmask`, a
        :class:`~pypeit.spectrographs.slitmask.SlitMask` object.

        Parameters
        ----------
        filename : :obj:`str`
            Name of the file to read.
        det : :obj:`int`, optional
            1-indexed detector number to read the slitmask for.  Must be either
            1 or 2 for MMT/Binospec.

        Returns
        -------
        :class:`~pypeit.spectrographs.slitmask.SlitMask`
            The slitmask data read from the file. The returned object is the
            same as :attr:`slitmask`.

        Notes
        -----
        - Target-slit alignment is characterized via distances from slit edges.
        - Slit corners and on-sky positions are stored for each target.
        """
        slit_id, slit_width, slit_x, slit_y, poly_x, poly_y, \
            obj_id, obj_name, obj_ra, obj_dec, obj_mag, \
            mm_arcsec, rac, decc, posx_pa \
            = self._parse_slitmask_data(filename, det)
        
        # Number of slits
        numslits = slit_id.size

        # The polygon coordinates have a shape that is (Nslits,4), their order
        # is: [0]: bottom left, [1]: top left, [2]: top right, [3]: bottom right
        # TODO: I assume the focal plane is flipped wrt the coordinates
        # provided, which is why the code is as given below.

        # Compute projected distances from target to slit edges in arcseconds
        # left
        topdist = (slit_y - poly_y[:,0]) / mm_arcsec
        # right
        botdist = (poly_y[:,1] - slit_y) / mm_arcsec
        if det == 2:
            # flip for detector 2
            topdist, botdist = botdist, topdist
        slit_length_arcsec = topdist + botdist

        # Assemble object array: [slit_id, id, ra, dec, name, mag, mag_band, top, bot]
        # TODO: I don't know why we need to use round
        objects = np.array([
            slit_id,
            obj_id,
            obj_ra,
            obj_dec,
            obj_name,
            obj_mag,
            ['None'] * numslits,
            np.round(topdist,2),
            np.round(botdist,2)
        ], dtype=object).T

        # Compute slit centers offsets from object positions
        xcen_slit = (poly_x[:,0] + poly_x[:,3]) / 2.
        ycen_slit = (poly_y[:,0] + poly_y[:,1]) / 2.
        slit_xoff = (xcen_slit - slit_x) / mm_arcsec  # in arcseconds
        slit_yoff = (ycen_slit - slit_y) / mm_arcsec  # in arcseconds
        slit_offset = np.sqrt(slit_xoff ** 2 + slit_yoff ** 2)

        # Compute slit center RA/Dec via spherical offset from target position
        obj_coord = SkyCoord(ra=obj_ra, dec=obj_dec, unit='deg')

        # Slit position angles and sign of offset (accounting for up/down location)
        slit_pas = np.full(numslits, posx_pa, dtype=float)
        off_signs = np.ones_like(slit_pas)
        negy = slit_yoff < 0.
        off_signs[negy] = -1.

        # Compute slit center RA/Dec
        slit_ra = np.empty(numslits, dtype=float)
        slit_dec = np.empty(numslits, dtype=float)
        for i, (slit_off, obj_coo, slit_pa, off_sign) in enumerate(zip(
            slit_offset, obj_coord, slit_pas, off_signs
        )):
            slit_coord = obj_coo.directional_offset_by(
                slit_pa * units.deg, off_sign * slit_off * units.arcsec
            )
            slit_ra[i] = slit_coord.ra.deg
            slit_dec[i] = slit_coord.dec.deg

        # TODO: This ordering doesn't seem to match the documentation for
        # SlitMask, but there may be a reflection involved.
        corners = np.stack((poly_x, poly_y), axis=-1)
        corners = corners[:, [3, 0, 1, 2], :]

        # Slitmask pointing coordinates (mask center RA/Dec)
        mask_coord = SkyCoord(rac, decc, unit=('hourangle', 'deg'))

        # Construct and return the slitmask object
        self.slitmask = SlitMask(
            corners,
            slitid=slit_id,
            onsky=np.asarray([
                slit_ra, slit_dec, np.round(slit_length_arcsec, 2),
                slit_width, slit_pas
            ]).T,
            objects=objects,
            mask_radec=(mask_coord.ra.deg, mask_coord.dec.deg),
            posx_pa=posx_pa
        )

        return self.slitmask

    @staticmethod
    def _parse_slitmask_data(filename, det):

        # Open the FITS file
        hdu = io.fits_open(filename)

        # Position angle corresponding to detector +x axis (spatial direction)
        posx_pa = float(hdu[1].header['POSANG']) - 180
        if posx_pa < 0:
            posx_pa += 360.

        # Select appropriate extension for detector 1 or 2
        match det:
            case 1:
                mask_hdu = hdu[9].data[0]
            case 2:
                mask_hdu = hdu[10].data[0]
            case _:
                raise PypeItError(f'Detector number must be 1 or 2 for MMT/Binospec, not {det}.')

        targ = mask_hdu['TARGET_TYPE'] == 'TARGET'
        numslits = mask_hdu['NTARGETS']

        if np.sum(targ) != numslits:
            raise PypeItError(
                f'Expected {numslits} TARGET slits but found {np.sum(targ)} in mask design file.'
            )
        
        # NOTE: The use of np.atleast_* here is to handle the case when there is
        # only one target.

        # Slit properties
        slit_id = np.atleast_1d(mask_hdu['SLIT_ID'])[targ]          # ID number
        slit_width = np.atleast_1d(mask_hdu['SLIT_WIDTH'])[targ]    # in arcsec
        # Target positions in mm
        slit_x = np.atleast_1d(mask_hdu['SLITX'])[targ]
        slit_y = np.atleast_1d(mask_hdu['SLITY'])[targ]
        # Slit polygon coordinates in mm
        poly_x = np.atleast_2d(mask_hdu['POLY_X']).T[targ]
        poly_y = np.atleast_2d(mask_hdu['POLY_Y']).T[targ]

        # Target properties
        obj_id = np.atleast_1d(mask_hdu['TARGET_ID'])[targ]
        obj_name = np.atleast_1d(mask_hdu['TARGET_NAME'])[targ]
        obj_ra = np.atleast_1d(mask_hdu['RA'])[targ]
        obj_dec = np.atleast_1d(mask_hdu['DEC'])[targ]
        obj_mag = np.atleast_1d(mask_hdu['MAG'])[targ]

        # Scalars with the platescale and center coordinates of the slit mask
        mm_arcsec = mask_hdu['MM_PER_ARCSEC']
        rac = mask_hdu['CENTERRA']      # in hours
        decc = mask_hdu['CENTERDEC']    # in degrees

        hdu.close()

        return (
            slit_id, slit_width, slit_x, slit_y, poly_x, poly_y,
            obj_id, obj_name, obj_ra, obj_dec, obj_mag,
            mm_arcsec, rac, decc, posx_pa
        )

    def get_maskdef_slitedges(self, filename:str=None, det:int=1, debug:bool=None, 
                              binning:str=None, trc_path:str=None):
        """
        Provides the slit edges positions predicted by the slitmask design.

        This method is not defined for all spectrographs. This base-class
        method raises an exception. This may be because ``use_maskdesign``
        has been set to True for a spectrograph that does not support it.

        Parameters
        ---------- 
        filename : :obj:`str`, :obj:`list`, optional:
            Name of the file holding the mask design info or the maskfile and
            wcs_file in that order
        det : :obj:`int`, optional
            Detector number
        debug : :obj:`bool`, optional
            Flag to run in debugging mode
        trc_path : str, optional
            Path to the first trace file used to generate the trace flat
        binning : str, optional
            String with the comma-separated number of pixels binned in each
            dimension of the flat-field image.  Order must be spectral then
            spatial.

        Returns
        -------
        top_edges : :class:`numpy.ndarray`
            Predicted locations of the top edges of the slits in spatial pixel
            coordinates.
        bot_edges : :class:`numpy.ndarray`
            Predicted locations of the bottom edges of the slits in spatial pixel
            coordinates.
        sortindx : :class:`numpy.ndarray`
            Indices of the slits in the provided ``slitmask`` object that orders
            the slits from left to right, in the PypeIt orientation.
        slitmask : :class:`~pypeit.spectrographs.slitmask.SlitMask`
            Slit mask metadata read from the provided input file(s).

        Notes
        -----
        - Edges are sorted by bottom edge y-coordinate to order slits spatially.
        """
        if det is None:
            raise ValueError("A valid detector number must be provided.")

        if filename is None:
            raise ValueError("A valid slitmask filename must be provided.")

        # get the full path to the mask design file
        _maskfile = str(Path(trc_path) / filename) if not Path(filename).exists() else filename

        # check if the mask design file exists
        if not Path(_maskfile).exists():
            raise PypeItError(f'The mask design file {_maskfile} does not exist.')

        # Load slitmask information if a file is provided
        self.get_slitmask(_maskfile, det=det)

        if self.slitmask is None:
            raise ValueError("Unable to read slitmask design info. Provide a file.")

        # Open FITS file and read mask data for the correct detector
        hdu = io.fits_open(filename)
        mask_fits = hdu[9].data[0] if det == 1 else hdu[10].data[0]
        # keep only the TARGET slits
        targ = mask_fits['TARGET_TYPE'] == 'TARGET'

        # Define det buffer and mm/pixel scale factor
        # NOTE: these are hard-coded and not sure if there is a more robust way to determine them
        # slitmask offset from the detector edge in pixels
        mask_edge_off = 200
        # scale factor to convert mm to pixel. The value should be equal to
        # 1/mask_fits['MM_PER_ARCSEC']/(platescale * bin_spat), but for some reason it's not,
        # and it's also different for the two detectors
        mm_pixel = 24.555832 if det == 1 else 24.548194

        left_edges = (mask_fits['POLY_Y'][0][targ] - mask_fits['MASK_CORNERS'][1])*mm_pixel + mask_edge_off
        right_edges = (mask_fits['POLY_Y'][1][targ] - mask_fits['MASK_CORNERS'][1])*mm_pixel + mask_edge_off
        if det == 2:
            # flip and reverse for detector 2
            Nx = self.get_rawimage(filename, det)[1].shape[1]
            left_edges, right_edges = Nx - right_edges, Nx - left_edges

        # Sort slits by their bottom edge position in ascending y-coordinate
        sortindx = np.argsort(left_edges)

        # Return the slit edges, sorted indices, and slitmask object
        return left_edges.astype(float), right_edges.astype(float), sortindx, self.slitmask

    def get_rawimage(self, raw_file, det):
        """
        Read raw images and generate a few other bits and pieces
        that are key for image processing.

        Parameters
        ----------
        raw_file : :obj:`str`
            File to read
        det : :obj:`int`
            1-indexed detector to read

        Returns
        -------
        detector_par : :class:`pypeit.images.detector_container.DetectorContainer`
            Detector metadata parameters.
        raw_img : `numpy.ndarray`_
            Raw image for this detector.
        hdu : `astropy.io.fits.HDUList`_
            Opened fits file
        exptime : :obj:`float`
            Exposure time read from the file header
        rawdatasec_img : `numpy.ndarray`_
            Data (Science) section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        oscansec_img : `numpy.ndarray`_
            Overscan section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        """
        fil = utils.find_single_file(f'{raw_file}*', required=True)

        # Read
        log.info(f'Reading BINOSPEC file: {fil}')
        hdu = io.fits_open(fil)
        head1 = hdu[1].header

        # TOdO Store these parameters in the DetectorPar.
        # Number of amplifiers
        detector_par = self.get_detector_par(det if det is not None else 1, hdu=hdu)
        numamp = detector_par['numamplifiers']

        # get the x and y binning factors...
        binning = head1['CCDSUM']
        xbin, ybin = [int(ibin) for ibin in binning.split(' ')]

        # First read over the header info to determine the size of the output array...
        datasec = head1['DATASEC']
        x1, x2, y1, y2 = chain.from_iterable(parse.load_sections(datasec, fmt_iraf=False))
        nxb = x1 - 1

        # determine the output array size...
        nx = (x2 - x1 + 1) * int(numamp/2) + nxb * int(numamp/2)
        ny = (y2 - y1 + 1) * int(numamp/2)

        # allocate output array...
        array = np.zeros((nx, ny))
        rawdatasec_img = np.zeros_like(array, dtype=int)
        oscansec_img = np.zeros_like(array, dtype=int)

        if det == 1:  # A DETECTOR
            order = range(1, 5, 1)
        elif det == 2:  # B DETECTOR
            order = range(5, 9, 1)

        # insert extensions into calibration image...
        for kk, jj in enumerate(order):
            # grab complete extension...
            data, overscan, datasec, biassec = binospec_read_amp(hdu, jj)

            # insert components into output array...
            inx = data.shape[0]
            xs = inx * kk
            xe = xs + inx

            iny = data.shape[1]
            ys = iny * kk
            yn = ys + iny

            b1, b2, b3, b4 = chain.from_iterable(parse.load_sections(biassec, fmt_iraf=False))

            if kk == 0:
                array[b2:inx+b2,:iny] = data #*1.028
                rawdatasec_img[b2:inx+b2,:iny] = kk + 1
                array[:b2,:iny] = overscan
                oscansec_img[2:b2,:iny] = kk + 1
            elif kk == 1:
                array[b2+inx:2*inx+b2,:iny] = np.flipud(data) #* 1.115
                rawdatasec_img[b2+inx:2*inx+b2:,:iny] = kk + 1
                array[2*inx+b2:,:iny] = overscan
                oscansec_img[2*inx+b2:,:iny] = kk + 1
            elif kk == 2:
                array[b2+inx:2*inx+b2,iny:] = np.fliplr(np.flipud(data)) #* 1.047
                rawdatasec_img[b2+inx:2*inx+b2,iny:] = kk + 1
                array[2*inx+b2:, iny:] = overscan
                oscansec_img[2*inx+b2:, iny:] = kk + 1
            elif kk == 3:
                array[b2:inx+b2,iny:] = np.fliplr(data) #* 1.045
                rawdatasec_img[b2:inx+b2,iny:] = kk + 1
                array[:b2,iny:] = overscan
                oscansec_img[2:b2,iny:] = kk + 1

        # Need the exposure time
        exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]
        # Return, transposing array back to orient the overscan properly
        return detector_par, np.fliplr(np.flipud(array)), hdu, exptime, np.fliplr(np.flipud(rawdatasec_img)), \
               np.fliplr(np.flipud(oscansec_img))

    def bino_get_slit_region(self, filename, det=None, Nx=4096, Ny=4112, pady=0):
        """
        Compute the pixel-space rectangular regions for each slit in a Binospec mask.

        This function reads the slitmask design from a FITS file (or an already-loaded
        `SlitMask` object), converts slit and object positions from mask coordinates to
        pixel coordinates, and determines the x/y pixel boundaries for each slit on the
        detector. It returns these boundaries along with the updated slitmask object.

        Parameters
        ----------
        filename : :obj:`str`
            Path to the slitmask FITS file. Must be provided unless the slitmask
            is already loaded via `self.get_slitmask`.
        det : :obj:`int`, optional
            Detector number (1 or 2). Must be specified.
        Nx : :obj:`int`, optional
            Detector size in the x-direction (default: 4096 pixels).
        Ny : :obj:`int`, optional
            Detector size in the y-direction (default: 4112 pixels).
        pady : :obj:`float`, optional
            Additional padding (in pixels) applied to the slit boundaries (default: 0).

        Returns
        -------
        region : :obj:`list`
            A list containing:
            - slit_x_range : array of x-boundaries for each slit [Nslits, 2]
            - slit_y_range : array of y-boundaries for each slit [Nslits, 2]
            - x_slitobj_pix : array of x pixel positions for slit objects
            - y_slitobj_pix : array of y pixel positions for slit objects
        slitmask : :class:`SlitMask`
            The updated `SlitMask` object containing slit geometry and metadata.

        Notes
        -----
        - Converts mask coordinates to pixel coordinates using the appropriate scale factor.
        - Handles detector 2 by reversing slit order and applying a vertical flip.
        - Slit boundaries are clipped to remain within detector dimensions.
        """

        if det is None:
            raise ValueError("A valid detector number must be provided.")

        # Load slitmask information if a file is provided
        if filename is None:
            raise ValueError("The name of a science file should be provided")
        self.get_slitmask(filename, det=det)

        if self.slitmask is None:
            raise ValueError("Unable to read slitmask design info. Provide a file.")

        # Open FITS file and read mask data for the correct detector
        hdu = io.fits_open(filename)
        mask_fits = hdu[9].data[0] if det == 1 else hdu[10].data[0]
        numslits = len(self.slitmask.slitid)

        # Initialize arrays to hold slit x/y boundaries
        res_x = np.zeros((2, numslits))
        res_y = np.zeros((2, numslits))

        # Extract target distances from slit edges and slit widths
        topdist = np.asarray(self.slitmask.objects[:, 7], dtype=float)
        botdist = np.asarray(self.slitmask.objects[:, 8], dtype=float)
        width = np.asarray(self.slitmask.width)

        # Extract slit center positions in mask coordinates
        x_slits = np.asarray(self.slitmask.center[:, 0])
        x_obj = x_slits
        y_slits = -np.asarray(self.slitmask.center[:, 1])

        # Extract slit corner y-coordinates (for top/bottom edges)
        y_slitsh = -np.asarray(self.slitmask.corners[:, 0, 1])
        y_slitsl = -np.asarray(self.slitmask.corners[:, 2, 1])

        # Compute object y-position relative to slit center
        y_obj = y_slits + (topdist - botdist) / 2

        # Extract slit lengths and widths (in mask coordinates)
        dx_slits = self.slitmask.length
        dy_slits = width

        # Define scale factor and detector offsets
        dy0 = -200.0
        y_scl = 24.555832 if det == 1 else 24.548194

        # Extract mask corner reference point
        mask_corners = np.asarray(mask_fits['MASK_CORNERS'])
        corner_x = mask_corners[0]
        corner_y = mask_corners[1]

        # Convert slit center positions to pixel coordinates
        x_slits_pix = (x_slits - corner_x) * y_scl + Nx / 2.0
        x_slitobj_pix = (x_obj - corner_x) * y_scl + Nx / 2.0
        y_slits_pix = Ny - 1 - ((y_slits - corner_y) * y_scl) + dy0
        y_slitobj_pix = Ny - 1 - ((y_obj - corner_y) * y_scl) + dy0
        y_slitsl_pix = Ny - 1 - ((y_slitsl - corner_y) * y_scl) + dy0
        y_slitsh_pix = Ny - 1 - ((y_slitsh - corner_y) * y_scl) + dy0

        # Convert slit lengths and widths to pixel units
        dx_slits_pix = dx_slits * y_scl
        dy_slits_pix = dy_slits * y_scl

        # Loop through slits to compute pixel-space rectangular boundaries
        for i in range(numslits):
            xmin = round(x_slits_pix[i] - dx_slits_pix[i] / 2.0 - pady)
            xmax = round(x_slits_pix[i] + dx_slits_pix[i] / 2.0 - 1 + pady)
            res_x[0, i] = max(0, xmin)
            res_x[1, i] = min(Ny - 1, xmax)

            ymin = round(y_slits_pix[i] - dy_slits_pix[i] / 2.0 - pady)
            ymax = round(y_slits_pix[i] + dy_slits_pix[i] / 2.0 - 1 + pady)
            res_y[0, i] = max(0, ymin)
            res_y[1, i] = min(Ny - 1, ymax)

        # Handle detector 2: reverse slit order and flip vertically
        if det == 2:
            res_y = res_y[:, ::-1]

            # Apply vertical flip relative to detector height (Ny) and offset
            res_y_flipped = np.zeros_like(res_y)
            res_y_flipped[0, :] = -1 * (res_y[1, :] - Ny - 14)
            res_y_flipped[1, :] = -1 * (res_y[0, :] - Ny - 14)
            res_y = res_y_flipped

        # Package results and return
        slit_x_range, slit_y_range = res_x.T, res_y.T
        region = [slit_x_range, slit_y_range, x_slitobj_pix, y_slitobj_pix]

        return region, self.slitmask


    def plot_mask(self, filename, det=None, save_dir=None):
        """
        Plot the slit mask layout and target positions for one or both detectors.

        This function retrieves slit region data for a given Binospec mask and
        plots the rectangular slit outlines and target positions for detector 1,
        detector 2, or both. It is useful for visually validating mask design and
        target alignment.

        Parameters
        ----------
        filename : :obj:`str`
            Path to the mask design file (e.g., a JSON file containing slit definitions).
        det : :obj:`int` or :obj:`str`
            Specifies which detector(s) to plot. Accepts 1, 2, or 'both'.
        save_dir : :obj:`str`, optional
            If provided, the plot will be saved as a PNG in the given directory.

        Returns
        -------
        region_1 : :obj:`tuple`, optional
            Slit region and target position data for detector 1, if requested.
        region_2 : :obj:`tuple`, optional
            Slit region and target position data for detector 2, if requested.
        """

        if det is None:
            raise ValueError("A valid detector number must be provided: 1, 2, or 'both'")

        if filename is None:
            raise ValueError("A valid filename must be provided.")

        # Build save filename from FITS header
        hdu = io.fits_open(filename)
        basename = Path(filename).name
        save_filename = Path(f"plot_mask_{hdu[1].header['MASK']}_{basename}").with_suffix('.png')

        plt.rcParams.update({"font.size": 20})

        # Load slit regions depending on the selected detector(s)
        if det == 'both':
            fig, (axA, axB) = plt.subplots(ncols=2, figsize=(16, 16))
            region_1 = self.bino_get_slit_region(filename, det=1)[0]
            region_2 = self.bino_get_slit_region(filename, det=2)[0]

        elif det == 1:
            fig, axA = plt.subplots(figsize=(8, 8))
            region_1 = self.bino_get_slit_region(filename, det=1)[0]

        elif det == 2:
            fig, axB = plt.subplots(figsize=(8, 8))
            region_2 = self.bino_get_slit_region(filename, det=2)[0]

        else:
            raise ValueError("det must be 1, 2, or 'both'.")

        # Plot based on detector selection
        if det == 'both':
            _plot_region(axA, region_1, color="red", side_label="1")
            _plot_region(axB, region_2, color="green", side_label="2")
        elif det == 1:
            _plot_region(axA, region_1, color="red", side_label="1")
        elif det == 2:
            _plot_region(axB, region_2, color="green", side_label="2")

        # Save to file if directory provided
        if save_dir is not None:
            _save_dir = Path(save_dir).absolute()
            _save_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(_save_dir / save_filename)
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()

        # Return the plotted region data
        if det == 'both':
            return region_1, region_2
        elif det == 1:
            return region_1
        elif det == 2:
            return region_2


# Internal helper to draw slits and targets on a given axis
def _plot_region(ax, region, color, side_label):
    num_targets = len(region[0])
    label = f" N = {num_targets}"

    for i in range(len(region[0])):
        slit_x_range = region[0][i]
        slit_y_range = region[1][i]
        width = slit_x_range[1] - slit_x_range[0]
        height = slit_y_range[1] - slit_y_range[0]

        rect = patches.Rectangle(
            (slit_x_range[0], slit_y_range[0]),
            width,
            height,
            linewidth=1,
            edgecolor="blue",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.scatter(region[2], region[3], s=10, color=color, label=label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Detector {side_label}")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()


def clean_overscan_vector(overscan, w=9, nsig=1.0, rdnoise=4.0):
    """
    Clean a 1D overscan vector by median-filtering and interpolating
    over outliers.

    Replicates the IDL ``clean_overscan_vector`` function from
    ``bino_mosaic.pro``.

    Parameters
    ----------
    overscan : `numpy.ndarray`_
        1D overscan vector to clean.
    w : :obj:`int`, optional
        Window size for median filtering. Must be >= 3. Default is 9.
    nsig : :obj:`float`, optional
        Sigma threshold for outlier rejection. Pixels deviating from
        the median-filtered vector by more than ``nsig * rdnoise`` are
        replaced by interpolation. Default is 1.0.
    rdnoise : :obj:`float`, optional
        Read noise in ADU, used to set the outlier threshold.
        Default is 4.0.

    Returns
    -------
    clean : `numpy.ndarray`_
        Cleaned overscan vector with outliers interpolated over.
    """
    w = max(w, 3)
    m_overscan = median_filter(overscan, size=w, mode='reflect')
    bad = np.abs(overscan - m_overscan) > rdnoise * nsig
    good = ~bad
    if not np.any(bad):
        return overscan.copy()
    if not np.any(good):
        return overscan.copy()
    clean = overscan.copy()
    good_idx = np.where(good)[0]
    bad_idx = np.where(bad)[0]
    clean[bad_idx] = np.interp(bad_idx, good_idx, overscan[good_idx])
    return clean


def binospec_read_amp(inp, ext):
    """
    Read one amplifier of an MMT BINOSPEC multi-extension FITS image

    Parameters
    ----------
    inp : str, :class:`astropy.io.fits.HDUList`
        The input FITS file name or already opened HDU list.
    ext : :obj:`int`
        FITS extension to read

    Returns
    -------
    data : :class:`numpy.ndarray`
        Array with data from the data section of the image.    
    overscan : :class:`numpy.ndarray`
        Array with the overscan section of the image.
    datasec : :obj:`str`
        String with the data section in IRAF format, e.g. '[x1:x2,y1:y2]'.
    biassec : :obj:`str`
        String with the bias section in IRAF format, e.g. '[x1:x2,y1:y2]'.
    """
    # Parse input
    hdu = io.fits_open(inp) if isinstance(inp, str) else inp

    # get entire extension...
    temp = hdu[ext].data.transpose()
    nxt = temp.shape[0]
    nyt = temp.shape[1]

    # parse the DETSEC keyword to determine the size of the array.
    header = hdu[ext].header

    # parse the DATASEC keyword to determine the size of the science region (unbinned)
    datasec = header['DATASEC']

    x1, x2, y1, y2 = chain.from_iterable(parse.load_sections(datasec, fmt_iraf=False))
    datasec = f'[{x1-1}:{x2},{y1-1}:{y2}]'

    # Overscan subtraction following IDL pipeline (bino_mosaic.pro):
    # Y-axis first, then X-axis. Uses sigma-clipped mean (resistant_mean)
    # with outlier cleaning, matching IDL defaults (clean_w=9, clean_nsig=1.0).

    # Y-axis overscan: postscan rows after datasec
    if y2 < nyt:
        overscan_y = temp[:, y2:nyt]
        overscan_vec, _, _ = sigma_clipped_stats(overscan_y, sigma=3.0, axis=1)
        overscan_vec = clean_overscan_vector(overscan_vec, w=9, nsig=1.0)
        temp = temp - overscan_vec[:, None]

    # X-axis overscan: prescan + postscan columns
    overscan_x_regions = []
    if x1 > 1:
        overscan_x_regions.append(temp[0:x1-1, :])
    if x2 < nxt:
        overscan_x_regions.append(temp[x2:nxt, :])
    if len(overscan_x_regions) > 0:
        overscan_x = np.concatenate(overscan_x_regions, axis=0)
        overscan_x_vec, _, _ = sigma_clipped_stats(overscan_x, sigma=3.0, axis=0)
        overscan_x_vec = clean_overscan_vector(overscan_x_vec, w=9, nsig=1.0)
        temp = temp - overscan_x_vec[None, :]

    # Crop to datasec
    data = temp[x1-1:x2, y1-1:y2]

    # Apply per-amplifier nonlinearity correction (IDL: poly(im_cur, c_poly))
    data = np.polynomial.polynomial.polyval(
        data, MMTBINOSPECSpectrograph.nonlinearity_coeffs[ext - 1])

    # Fake overscan for PypeIt's general pipeline (effectively a no-op)
    biassec = f'[0:{x1-1},{y1-1}:{y2}]'
    xos1, xos2, yos1, yos2 = chain.from_iterable(parse.load_sections(biassec, fmt_iraf=False))
    overscan = np.zeros_like(temp[xos1:xos2, yos1:yos2])

    return data, overscan, datasec, biassec


class MMTBINOSPECIFUSpectrograph(MMTBINOSPECSpectrograph):
    """
    Child to handle MMT/BINOSPEC IFU specific code.

    The Binospec IFU is a fiber-fed integral field unit with a hexagonal
    lenslet array feeding ~360 fibers per side into the spectrograph.
    Each side has 40 dedicated sky fibers at the outermost ring of each
    sub-bundle (indices [0-7, 88-95, 176-183, 264-271, 352-359]).
    """
    name = 'mmt_binospec_ifu'
    pypeline = 'Fiber'
    supported = True

    # IFU fiber geometry constants
    # On-sky fiber pitch in arcsec (hexagonal lenslet array)
    ifu_fiber_pitch = 0.6
    # Number of fibers per side
    nfibers_a = 360
    nfibers_b = 356
    # Dedicated sky fiber indices (0-based array positions, per side,
    # outermost ring of each sub-bundle). The IDL pipeline uses these
    # indices for sky subtraction. When matching against FIB_ID in the
    # reference profile (which is 1-based), add 1.
    sky_fiber_indices_0based = np.array([
        *range(0, 8), *range(88, 96), *range(176, 184),
        *range(264, 272), *range(352, 360)
    ])
    # 1-based fiber IDs for matching against reference profile FIB_ID
    sky_fiber_ids = sky_fiber_indices_0based + 1
    # Bright sky emission lines for throughput correction (Angstroms).
    # These must be isolated enough to measure reliably with
    # continuum sidebands.  Source: Binospec IDL pipeline
    # (Chilingarian et al. 2025, arXiv:2501.01528).
    # Note: 4358.335 (Hg I) omitted -- unreliable at dark sites.
    skyline_list_ang = np.array([
        5577.34, 6300.304, 6863.951, 7340.881,
        7993.327, 8465.353, 8885.843, 9502.808
    ])

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        Adds 'decker' to the parent keys so that IFU frames are not
        grouped with MOS frames in the same configuration.

        Returns:
            :obj:`list`: List of configuration keys.
        """
        return super().configuration_keys() + ['decker']

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        Extends the parent class metadata with IFU-specific fields
        required by the Fiber pipeline (atmospheric parameters
        for DAR correction).
        """
        super().init_meta()
        # IFU-specific metadata for Fiber pipeline
        self.meta['slitwid'] = dict(card=None, compound=True)
        self.meta['obstime'] = dict(card=None, compound=True, required=False)
        self.meta['pressure'] = dict(card=None, compound=True, required=False)
        self.meta['temperature'] = dict(card=None, compound=True, required=False)
        self.meta['humidity'] = dict(card=None, compound=True, required=False)
        self.meta['parangle'] = dict(card=None, compound=True, required=False)

    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        if meta_key in ('ra', 'dec'):
            hdrstr = 'RA' if meta_key == 'ra' else 'DEC'
            return headarr[0][hdrstr]
        elif meta_key == 'exptime':
            return headarr[0]['EXPTIME']
        elif meta_key == 'slitwid':
            # IFU fiber pitch on sky, converted to degrees for WCS
            return self.ifu_fiber_pitch / 3600.0
        elif meta_key == 'obstime':
            try:
                return Time(headarr[1]['DATE-OBS'])
            except KeyError:
                log.warning("Time of observation not in header")
                return None
        elif meta_key == 'pressure':
            # MMT at ~2600m elevation, typical pressure ~730 mbar
            try:
                return headarr[1]['PRESSURE']
            except KeyError:
                log.warning("Pressure not in header - using default for MMT "
                            "elevation (730 mbar)")
                return 730.0
        elif meta_key == 'temperature':
            try:
                return headarr[1]['TEMP']
            except KeyError:
                log.warning("Temperature not in header - using default (5 deg C)")
                return 5.0
        elif meta_key == 'humidity':
            try:
                return headarr[1]['HUMID']
            except KeyError:
                log.warning("Humidity not in header - using default (20%%)")
                return 20.0
        elif meta_key == 'parangle':
            try:
                return headarr[1]['PA'] * np.pi / 180.0
            except KeyError:
                log.warning("Parallactic angle not in header - using default (0)")
                return 0.0
        else:
            return super().compound_meta(headarr, meta_key)

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Overrides the parent to ensure only IFU frames (MASK == 'IFU')
        are selected for this spectrograph.

        Args:
            ftype (:obj:`str`):
                Type of frame to check.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type ``ftype``.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        # Use parent frame typing logic, then restrict to IFU frames only
        is_type = super().check_frame_type(ftype, fitstbl, exprng=exprng)
        is_ifu = np.array([d.strip().upper() == 'IFU' for d in fitstbl['decker']])
        return is_type & is_ifu

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        par = super().default_pypeit_par()

        # IFU science frame processing
        par['scienceframe']['process']['sigclip'] = 4.0
        par['scienceframe']['process']['objlim'] = 1.5
        par['scienceframe']['process']['use_illumflat'] = False
        par['scienceframe']['process']['use_specillum'] = False
        par['scienceframe']['process']['spat_flexure_correct'] = False
        par['scienceframe']['process']['use_biasimage'] = False
        par['scienceframe']['process']['use_darkimage'] = False

        # FiberFindObjects creates one SpecObj per fiber from slit edges
        # (no peak detection needed) and handles sky subtraction in its
        # own run() method, so skip the second find and final global.
        par['reduce']['findobj']['skip_second_find'] = True
        par['reduce']['findobj']['skip_final_global'] = True

        # Sky subtraction: use joint fit across all fibers
        par['reduce']['skysub']['no_poly'] = True
        par['reduce']['skysub']['joint_fit'] = True
        # Avoid trimming edges of narrow IFU fibers (~5-6 pixels wide)
        par['reduce']['trim_edge'] = [0, 0]

        # Slit edge parameters tuned for densely-packed IFU fibers.
        # Fibers are ~7 pixels peak-to-peak with ~5-6 pixel widths and
        # inter-fiber gaps of only ~2 pixels.
        par['calibrations']['slitedges']['edge_thresh'] = 5.
        par['calibrations']['slitedges']['minimum_slit_gap'] = 0.
        # Fiber widths are ~5-6 pixels = ~1.2-1.4 arcsec at 0.24"/pix
        par['calibrations']['slitedges']['minimum_slit_length'] = 0.5
        par['calibrations']['slitedges']['pad'] = 0
        par['calibrations']['slitedges']['use_maskdesign'] = False
        # Default min_edge_side_sep=5 * fwhm_gaussian=3 = 15 pixels,
        # which merges adjacent fibers. Reduce so edges can be as close
        # as ~3 pixels apart (1 * 3 = 3 pixels).
        par['calibrations']['slitedges']['fwhm_gaussian'] = 2.0
        par['calibrations']['slitedges']['min_edge_side_sep'] = 1.0

        # Scattered light correction for science frames only.
        # Flats don't need it (used for geometry/pixel response, not flux).
        # Each science frame gets its own model fit to inter-fiber gap pixels.
        par['calibrations']['scattlight_pad'] = 5
        par['scienceframe']['process']['subtract_scattlight'] = True
        par['scienceframe']['process']['scattlight']['method'] = 'frame'

        # Flat field: no edge tweaking for fiber-fed IFU (fixed positions)
        par['calibrations']['flatfield']['tweak_slits'] = False
        par['calibrations']['flatfield']['slit_trim'] = 0
        par['calibrations']['flatfield']['slit_illum_finecorr'] = False

        # Tilts: reduce order for short fiber "slits"
        par['calibrations']['tilts']['spat_order'] = 1
        par['calibrations']['tilts']['spec_order'] = 1

        # Flexure: Binospec has active flexure control, so spectral
        # flexure correction is not needed for IFU mode
        par['flexure']['spec_method'] = 'skip'

        # Flux calibration: extinction correction is done during datacube
        # construction, not during 1D extraction
        par['sensfunc']['UVIS']['extinct_correct'] = False

        return par

    def config_specific_par(
            self,
            inp:str|list|Path|fits.Header|Table,
            inp_par:parset.ParSet|None=None
        ) -> parset.ParSet:
        """
        Modify the PypeIt parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            inp: Input filename, header, or metadata table row.
            inp_par: Parameter set. If None, use default.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: Adjusted parameters.
        """
        par = super().config_specific_par(inp, inp_par=inp_par)

        # Grating-dependent bspline spacing for sky subtraction
        # (adopted from IDL pipeline knot spacings)
        grating = self.get_meta_value(inp, 'dispname')
        match grating:
            case 'x270':
                par['reduce']['skysub']['bspline_spacing'] = 1.05
            case 'x600':
                par['reduce']['skysub']['bspline_spacing'] = 0.5
            case 'x1000':
                par['reduce']['skysub']['bspline_spacing'] = 0.35

        # Override MOS-specific settings from parent's config_specific_par
        # that are inappropriate for IFU fibers
        par['calibrations']['slitedges']['use_maskdesign'] = False
        par['calibrations']['slitedges']['minimum_slit_length'] = 0.5
        par['calibrations']['slitedges']['edge_thresh'] = 5.
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'
        par['reduce']['slitmask']['assign_obj'] = False
        par['reduce']['slitmask']['extract_missing_objs'] = False

        return par

    def get_wcs(self, hdr, slits, platescale, wave0, dwv, spatial_scale=None):
        """
        Construct a World-Coordinate System for the IFU datacube.

        Args:
            hdr (`astropy.io.fits.Header`_):
                The header of the raw frame.
            slits (:class:`~pypeit.slittrace.SlitTraceSet`):
                Slit traces.
            platescale (:obj:`float`):
                The platescale of an unbinned pixel in arcsec/pixel.
            wave0 (:obj:`float`):
                The wavelength zeropoint.
            dwv (:obj:`float`):
                Change in wavelength per spectral pixel.
            spatial_scale (:obj:`float`, optional):
                User-specified spatial scale in arcsec.

        Returns:
            `astropy.wcs.WCS`_: The world-coordinate system.
        """
        log.info("Calculating the WCS for Binospec IFU")

        # Get binning
        binspec, binspat = parse.parse_binning(self.get_meta_value([hdr], 'binning'))

        # Spatial scales
        pxscl = platescale * binspat / 3600.0  # arcsec -> degrees
        slscl = self.get_meta_value([hdr], 'slitwid')  # already in degrees

        if spatial_scale is not None:
            pxscl = spatial_scale / 3600.0

        # Typical slit length
        slitlength = int(np.round(np.median(slits.get_slitlengths(median=True))))

        # Pointing coordinates
        raval = self.get_meta_value([hdr], 'ra')
        decval = self.get_meta_value([hdr], 'dec')
        coord = SkyCoord(raval, decval, unit=(units.deg, units.deg))

        # Position angle from POSANG header keyword
        posang = hdr.get('POSANG', 0.0)
        crota = np.radians(-posang)

        # CD matrix
        cdelt1 = -slscl
        cdelt2 = pxscl
        cd11 = cdelt1 * np.cos(crota)
        cd12 = abs(cdelt2) * np.sign(cdelt1) * np.sin(crota)
        cd21 = -abs(cdelt1) * np.sign(cdelt2) * np.sin(crota)
        cd22 = cdelt2 * np.cos(crota)

        # Reference pixels (center of FOV)
        nslits = slits.nslits
        crpix1 = nslits / 2.0
        crpix2 = slitlength / 2.0
        crpix3 = 1.0

        # Create WCS
        log.info("Generating Binospec IFU WCS")
        w = wcs.WCS(naxis=3)
        w.wcs.equinox = hdr.get('EQUINOX', 2000.0)
        w.wcs.name = 'Binospec IFU'
        w.wcs.radesys = 'ICRS'
        w.wcs.cname = ['RA', 'DEC', 'Wavelength']
        w.wcs.cunit = [units.degree, units.degree, units.Angstrom]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
        w.wcs.crval = [coord.ra.degree, coord.dec.degree, wave0]
        w.wcs.crpix = [crpix1, crpix2, crpix3]
        w.wcs.cd = np.array([[cd11, cd12, 0.0],
                             [cd21, cd22, 0.0],
                             [0.0, 0.0, dwv]])
        w.wcs.lonpole = 180.0
        w.wcs.latpole = 0.0

        return w

    def get_datacube_bins(self, slitlength, minmax, num_wave):
        r"""
        Calculate the bin edges to be used when making a datacube.

        Args:
            slitlength (:obj:`int`):
                Length of the slit in pixels.
            minmax (`numpy.ndarray`_):
                An array with the minimum and maximum pixel locations on
                each slit relative to the reference location. Shape must
                be :math:`(N_{\rm slits},2)`.
            num_wave (:obj:`int`):
                Number of wavelength steps.

        Returns:
            :obj:`tuple`: Three 1D `numpy.ndarray`_ providing the
            :math:`(x,y,\lambda)` bins for datacube construction.
        """
        # Number of fiber traces (slits) is determined from minmax
        nslits = minmax.shape[0]
        ref_slit = nslits // 2
        xbins = np.arange(1 + nslits) - ref_slit - 0.5
        ybins = np.linspace(np.min(minmax[:, 0]), np.max(minmax[:, 1]),
                            1 + slitlength) - 0.5
        spec_bins = np.arange(1 + num_wave) - 0.5
        return xbins, ybins, spec_bins

    @staticmethod
    def _ifu_calib_path() -> Path:
        """Return the path to the IFU calibration data directory."""
        return Path(__file__).resolve().parent.parent / 'data' / 'spectrographs' / 'mmt_binospec'

    def load_fiber_ref_profile(self, det: int) -> fits.FITS_rec:
        """
        Load the reference fiber trace profile for fiber identification.

        The reference profile contains the expected pixel positions and
        Gaussian-Hermite profile parameters for each fiber, obtained
        from a high-quality flat field observation. This is used to
        cross-match detected fiber traces against known fiber IDs.

        Args:
            det (:obj:`int`):
                1-indexed detector number (1=side A, 2=side B).

        Returns:
            `astropy.io.fits.FITS_rec`_: Table with columns:
                FIB_ID, X, Y, SIDE, FIB_NAME, FIB_TYPE, FIB_BLOCK,
                FIB_DEAD_FLAG, TR_A0, TR_PIX, TR_SIGMA, TR_BGR,
                TR_H3, TR_H4, TR_H5, TR_H6.
        """
        ref_file = self._ifu_calib_path() / 'fiber_ref_profile.fits'
        # ext 1 = IFUTRACES_A (side A, det 1), ext 2 = IFUTRACES_B (side B, det 2)
        ext = 1 if det == 1 else 2
        with fits.open(ref_file) as hdu:
            data = hdu[ext].data.copy()
        return data

    def load_sky_layout(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the IFU fiber-to-sky position mapping.

        Returns the on-sky x,y positions (in arcsec) for all 640 fibers
        in the hexagonal IFU field of view.

        Returns:
            :obj:`tuple`:
                - targetx_asec (`numpy.ndarray`_): x positions in arcsec (640,)
                - targety_asec (`numpy.ndarray`_): y positions in arcsec (640,)
        """
        sky_file = self._ifu_calib_path() / 'bino_IFU_sky_layout.fits'
        with fits.open(sky_file) as hdu:
            data = hdu[1].data[0]
            targetx = data['TARGETX_ASEC'].copy()
            targety = data['TARGETY_ASEC'].copy()
        return targetx, targety

    def load_fiber_illumination(self, det: int) -> np.ndarray:
        """
        Load the fiber-to-fiber illumination correction (throughput map).

        Args:
            det (:obj:`int`):
                1-indexed detector number (1=side A, 2=side B).

        Returns:
            `numpy.ndarray`_: Relative illumination correction per fiber (nfibers,).
        """
        illum_file = self._ifu_calib_path() / 'fiber_illumination.fits'
        # Row 0 = side A, row 1 = side B
        row = 0 if det == 1 else 1
        with fits.open(illum_file) as hdu:
            f_illum = hdu[1].data['F_ILLUM'][row].copy()
        return f_illum

    def modify_pixelflat(self, flatimages, slits, det):
        """
        Bake fiber-to-fiber illumination correction into the pixel flat.

        Scales each fiber's region in ``pixelflat_norm`` by its relative
        throughput factor from ``fiber_illumination.fits``.  When PypeIt
        divides the science image by this modified flat, the fiber
        throughput variation is corrected along with the pixel response.

        Args:
            flatimages (:class:`~pypeit.flatfield.FlatImages`):
                Flat-field images to modify (in place).
            slits (:class:`~pypeit.slittrace.SlitTraceSet`):
                Slit traces.
            det (:obj:`int`):
                1-indexed detector number.
        """
        from pypeit import log

        if flatimages.pixelflat_norm is None:
            return

        det_num = det if isinstance(det, int) else int(det)
        f_illum_all = self.load_fiber_illumination(det_num)
        ref = self.load_fiber_ref_profile(det_num)
        ref_ids = ref['FIB_ID']

        # Map each slit to its fiber ID via spatial position matching
        spat_ids = slits.spat_id
        fiber_meta = self.get_fiber_metadata(det_num, spat_ids)

        # Build slit mask image (pixel -> spat_id)
        slitmask = slits.slit_img(pad=0)

        n_scaled = 0
        for i, spat_id in enumerate(spat_ids):
            fid = fiber_meta['fiber_id'][i]
            if fid < 0:
                continue
            idx = np.where(ref_ids == fid)[0]
            if len(idx) == 0 or idx[0] >= len(f_illum_all):
                continue
            f_illum = float(f_illum_all[idx[0]])
            if f_illum < 0.1:
                continue

            # Scale this fiber's pixels in the flat
            slit_pixels = slitmask == spat_id
            flatimages.pixelflat_norm[slit_pixels] *= f_illum
            n_scaled += 1

        log.info(f"DET{det_num:02d}: applied fiber illumination correction "
                 f"to pixel flat ({n_scaled} fibers)")

    def compute_skyline_illum(self, sciimg, waveimg, slitmask, spat_ids):
        """
        Compute per-fiber throughput correction from sky emission lines.

        For each sky line within the wavelength range, extracts boxcar
        flux per fiber, subtracts local continuum, and normalizes by
        the median across fibers.  The per-fiber correction is the
        median ratio across the 3 brightest usable lines (or fewer if
        less than 3 are available).  The correction is wavelength-
        independent; wavelength-dependent throughput variations should
        be handled by spectral flux calibration from standard stars.

        Fibers with no valid line measurements (e.g. dead fibers) are
        left uncorrected (correction = 1.0).

        Args:
            sciimg (`numpy.ndarray`_):
                2D flat-fielded science image.
            waveimg (`numpy.ndarray`_):
                Wavelength image in Angstroms.
            slitmask (`numpy.ndarray`_):
                2D slit image (pixel -> spat_id).
            spat_ids (`numpy.ndarray`_):
                Array of spatial IDs for each fiber.

        Returns:
            `numpy.ndarray`_: 2D correction image (same shape as
            sciimg).  Values > 1 for fibers brighter than median.
        """
        nfibers = len(spat_ids)

        # Determine wavelength range from the data
        valid = waveimg > 0
        if not np.any(valid):
            log.warning("No valid wavelength data; skipping skyline "
                        "illumination correction")
            return np.ones_like(sciimg)
        wmin = waveimg[valid].min()
        wmax = waveimg[valid].max()

        # Filter sky lines to those within the wavelength range with
        # enough margin for continuum estimation (50 Ang each side)
        margin = 50.0
        usable = ((self.skyline_list_ang > wmin + margin)
                  & (self.skyline_list_ang < wmax - margin))
        sky_lines = self.skyline_list_ang[usable]
        if len(sky_lines) == 0:
            log.warning("No sky lines in wavelength range "
                        f"[{wmin:.0f}, {wmax:.0f}] Ang; skipping "
                        "skyline illumination correction")
            return np.ones_like(sciimg)
        log.info(f"Measuring {len(sky_lines)} sky lines for fiber "
                 f"throughput correction")

        # Extract 1D boxcar spectrum per fiber
        nspec = sciimg.shape[0]
        fiber_flux = np.zeros((nfibers, nspec))
        fiber_wave = np.zeros((nfibers, nspec))
        for i, spat_id in enumerate(spat_ids):
            slit_pix = slitmask == spat_id
            for j in range(nspec):
                row_pix = slit_pix[j, :]
                if np.any(row_pix):
                    fiber_flux[i, j] = np.sum(sciimg[j, row_pix])
                    fiber_wave[i, j] = np.mean(waveimg[j, row_pix])

        # Measure each sky line in each fiber
        line_window = 4.0   # Angstroms half-width for line flux
        cont_inner = 8.0    # Angstroms from line center to start of
                             # continuum window
        cont_outer = 20.0   # Angstroms from line center to end of
                             # continuum window
        min_valid_fibers = max(10, nfibers // 10)

        # Shape: (n_lines, n_fibers)
        line_ratios = np.full((len(sky_lines), nfibers), np.nan)

        for k, wl in enumerate(sky_lines):
            fiber_line_flux = np.zeros(nfibers)
            for i in range(nfibers):
                wave_i = fiber_wave[i]
                flux_i = fiber_flux[i]
                good = wave_i > 0

                if not np.any(good):
                    continue

                # Line window
                in_line = good & (np.abs(wave_i - wl) <= line_window)
                # Continuum windows (blue and red sidebands)
                in_cont = (good
                           & (np.abs(wave_i - wl) >= cont_inner)
                           & (np.abs(wave_i - wl) <= cont_outer))

                if np.sum(in_line) < 2 or np.sum(in_cont) < 3:
                    continue

                cont_level = np.median(flux_i[in_cont])
                line_sum = np.sum(flux_i[in_line] - cont_level)
                fiber_line_flux[i] = line_sum

            # Normalize by median across fibers (exclude zeros/negatives)
            valid_flux = fiber_line_flux > 0
            if np.sum(valid_flux) < min_valid_fibers:
                log.warning(f"Sky line {wl:.1f} Ang: too few valid "
                            f"fibers ({np.sum(valid_flux)}), skipping")
                continue
            med_flux = np.median(fiber_line_flux[valid_flux])
            ratios = np.where(valid_flux,
                              fiber_line_flux / med_flux, np.nan)
            line_ratios[k] = ratios
            n_valid = np.sum(valid_flux)
            rmin = np.nanmin(ratios)
            rmax = np.nanmax(ratios)
            log.info(f"  {wl:.1f} Ang: {n_valid} fibers, "
                     f"range {rmin:.3f} - {rmax:.3f}")

        # Check we have at least one usable line
        usable_lines = ~np.all(np.isnan(line_ratios), axis=1)
        if not np.any(usable_lines):
            log.warning("No sky lines measured successfully; skipping "
                        "skyline illumination correction")
            return np.ones_like(sciimg)
        line_ratios = line_ratios[usable_lines]

        # Select the 3 brightest lines (by median flux across fibers)
        # for a robust per-fiber correction
        n_best = min(3, line_ratios.shape[0])
        if n_best < line_ratios.shape[0]:
            # Rank lines by number of valid fibers (proxy for brightness
            # and reliability)
            n_valid_per_line = np.sum(~np.isnan(line_ratios), axis=1)
            best_idx = np.argsort(n_valid_per_line)[-n_best:]
            line_ratios = line_ratios[best_idx]
            log.info(f"Using {n_best} best-measured lines for correction")

        # Build a single correction per fiber: median ratio across lines
        corr_2d = np.ones_like(sciimg)
        for i, spat_id in enumerate(spat_ids):
            slit_pix = slitmask == spat_id
            if not np.any(slit_pix):
                continue

            ratios_i = line_ratios[:, i]
            good_lines = ~np.isnan(ratios_i)

            if not np.any(good_lines):
                continue

            corr_val = np.median(ratios_i[good_lines])
            corr_2d[slit_pix] = corr_val

        # Safety: clip extreme corrections
        corr_2d = np.clip(corr_2d, 0.3, 3.0)

        return corr_2d

    def skyline_illum_correct(self, sciimg, waveimg, slits, slitmask):
        """
        Apply sky-line-based illumination correction.

        Corrects for throughput differences between sky fibers (bare
        fibers) and science fibers (lenslet-fed) that the dome-flat-based
        illumination correction cannot capture.  Only modifies ``sciimg``
        in place; variance is handled by the caller.

        See :meth:`compute_skyline_illum` for the algorithm.
        """
        corr = self.compute_skyline_illum(sciimg, waveimg, slitmask,
                                          slits.spat_id)
        if np.allclose(corr, 1.0):
            return corr

        # Apply: divide science image only (variance handled by caller)
        good = corr > 0.1
        sciimg[good] /= corr[good]
        log.info("Applied sky-line illumination correction "
                 f"(range {corr[good].min():.3f} - "
                 f"{corr[good].max():.3f})")
        return corr

    def get_sky_fiber_mask(self, det: int, nslits: int) -> np.ndarray:
        """
        Return a boolean mask identifying which fiber/slit indices are
        dedicated sky fibers.

        The Binospec IFU has 40 dedicated sky fibers per side, located
        at the outermost ring of each hexagonal sub-bundle. These fibers
        observe blank sky and are used for sky subtraction.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            nslits (:obj:`int`):
                Total number of detected fiber traces (slits).

        Returns:
            `numpy.ndarray`_: Boolean array of shape (nslits,), True for
            sky fibers.
        """
        nfibers = self.nfibers_a if det == 1 else self.nfibers_b
        sky_mask = np.zeros(nfibers, dtype=bool)
        valid_sky = self.sky_fiber_indices_0based[self.sky_fiber_indices_0based < nfibers]
        sky_mask[valid_sky] = True
        # If fewer traces detected than expected, truncate
        if nslits < nfibers:
            log.warning(f"Detected {nslits} fiber traces but expected {nfibers}. "
                        f"Sky fiber mask may be incomplete.")
            sky_mask = sky_mask[:nslits]
        elif nslits > nfibers:
            sky_mask = np.pad(sky_mask, (0, nslits - nfibers), constant_values=False)
        return sky_mask

    def get_science_fiber_layout_indices(self, det: int,
                                         fiber_ids: np.ndarray,
                                         fiber_types: np.ndarray) -> np.ndarray:
        """
        Map detected fibers to layout file indices using fiber IDs.

        Uses fiber IDs from :func:`get_fiber_metadata` to look up each
        fiber's name in the reference profile, then matches that name to
        the layout file entry.  This works correctly even when fibers are
        missing from the input data.

        The layout file (``bino_IFU_sky_layout.fits``) contains 640 entries
        (indices 0-319 for side A, 320-639 for side B).  Live science
        fibers from the reference profile are sorted by detector position
        and paired with live layout entries: in forward order for side A,
        and in reverse order for side B (because the two detectors produce
        mirror-image spectra).  Dead fibers (``_DEAD`` suffix) are excluded
        from both lists so they do not disrupt the pairing.

        Args:
            det (:obj:`int`):
                1-indexed detector number (1=side A, 2=side B).
            fiber_ids (`numpy.ndarray`_):
                Physical fiber IDs for each detected fiber, from
                ``fiber_meta['fiber_id']``.  Unmatched fibers have ID < 0.
            fiber_types (`numpy.ndarray`_):
                Fiber type strings (``'SCI'``, ``'SKY'``, or ``'UNKNOWN'``),
                from ``fiber_meta['fiber_type']``.

        Returns:
            `numpy.ndarray`_: Array of shape ``(nfibers,)`` with layout file
            indices (0-639) for each fiber. Sky, dead, and unmatched fibers
            are assigned -1.
        """
        ref = self.load_fiber_ref_profile(det)
        ref_ids = ref['FIB_ID']
        ref_names = np.char.strip(ref['FIB_NAME'])
        ref_pix = ref['TR_PIX']

        # Identify live science fibers in the reference profile and sort
        # by detector position (TR_PIX).
        ref_is_sci = np.array([not n.startswith('SKY') for n in ref_names])
        ref_is_live = np.array(['_DEAD' not in n for n in ref_names])
        ref_live_sci = ref_is_sci & ref_is_live
        live_sci_ids = ref_ids[ref_live_sci]
        live_sci_pix = ref_pix[ref_live_sci]
        sort_idx = np.argsort(live_sci_pix)
        live_sci_ids_sorted = live_sci_ids[sort_idx]

        # Get live layout entries for this side in layout-file order.
        sky_file = self._ifu_calib_path() / 'bino_IFU_sky_layout.fits'
        with fits.open(sky_file) as hdu:
            layout_names = [n.strip() for n in hdu[1].data[0]['TARGET_NAME']]
        start, end = (0, 320) if det == 1 else (320, 640)
        live_layout_indices = [i for i in range(start, end)
                               if '_DEAD' not in layout_names[i]]

        # Pair reference fibers (by detector position) with layout entries.
        # Side A: forward (leftmost on detector = first layout entry).
        # Side B: reversed (leftmost on detector = last layout entry)
        # because the two detectors produce mirror-image spectra.
        n_map = min(len(live_sci_ids_sorted), len(live_layout_indices))
        id_to_layout = {}
        for rank in range(n_map):
            fid = int(live_sci_ids_sorted[rank])
            if det == 1:
                id_to_layout[fid] = live_layout_indices[rank]
            else:
                id_to_layout[fid] = live_layout_indices[n_map - 1 - rank]

        nfibers = len(fiber_ids)
        layout_indices = np.full(nfibers, -1, dtype=int)

        for i in range(nfibers):
            if fiber_types[i] == 'SKY' or fiber_ids[i] < 0:
                continue
            layout_idx = id_to_layout.get(int(fiber_ids[i]), -1)
            if layout_idx >= 0:
                layout_indices[i] = layout_idx

        n_mapped = int(np.sum(layout_indices >= 0))
        log.info(f"Mapped {n_mapped} science fibers to layout indices "
                 f"(det={det})")
        return layout_indices

    def match_fibers_to_reference(self, det, detected_positions):
        """
        Cross-match detected fiber trace positions against the reference
        profile to assign physical fiber IDs.

        Uses the algorithm from the IDL pipeline (bino_ifu_fiber_id.pro):
        compute cross-correlation between the detected fiber positions
        and the reference profile, then match individual fibers using a
        distance threshold.

        Args:
            det (:obj:`int`):
                1-indexed detector number (1=side A, 2=side B).
            detected_positions (`numpy.ndarray`_):
                Detected fiber center positions in pixels at a reference
                column (e.g., center of detector).

        Returns:
            :obj:`tuple`:
                - fiber_ids (`numpy.ndarray`_): Physical fiber IDs for each
                  detected trace, or -1 if unmatched.
                - is_sky (`numpy.ndarray`_): Boolean array, True for sky fibers.
                - is_dead (`numpy.ndarray`_): Boolean array, True for dead fibers
                  in the reference that were not detected.
        """
        ref = self.load_fiber_ref_profile(det)
        ref_positions = ref['TR_PIX']
        ref_ids = ref['FIB_ID']
        ref_dead = ref['FIB_DEAD_FLAG'].astype(bool)
        nfibers = self.nfibers_a if det == 1 else self.nfibers_b

        # Compute global offset via cross-correlation of position histograms
        # (simplified version of the IDL 5-segment approach)
        nbins = 4200  # slightly larger than detector width
        ref_hist = np.zeros(nbins)
        det_hist = np.zeros(nbins)
        for pos in ref_positions[~ref_dead]:
            idx = int(np.round(pos))
            if 0 <= idx < nbins:
                ref_hist[idx] = 1.0
        for pos in detected_positions:
            idx = int(np.round(pos))
            if 0 <= idx < nbins:
                det_hist[idx] = 1.0

        # Cross-correlate to find global shift
        corr = np.correlate(det_hist, ref_hist, mode='full')
        lag = np.arange(len(corr)) - (nbins - 1)
        # Search within +/- 50 pixels
        mask = np.abs(lag) < 50
        best_lag = lag[mask][np.argmax(corr[mask])]

        # Match fibers using distance threshold (1.7 pixels, from IDL pipeline)
        dx_thr = 1.7
        shifted_ref = ref_positions + best_lag
        fiber_ids = np.full(len(detected_positions), -1, dtype=int)
        matched = np.zeros(len(ref_positions), dtype=bool)

        for i, dpos in enumerate(detected_positions):
            dists = np.abs(shifted_ref - dpos)
            dists[matched] = np.inf
            best = np.argmin(dists)
            if dists[best] < dx_thr:
                fiber_ids[i] = ref_ids[best]
                matched[best] = True

        # Determine sky fiber status from matched IDs (using 1-based fiber IDs)
        is_sky = np.isin(fiber_ids, self.sky_fiber_ids) & (fiber_ids >= 0)

        # Dead fibers: reference fibers that were not matched
        is_dead = ~matched & ~ref_dead

        log.info(f"Fiber matching: {np.sum(fiber_ids >= 0)}/{len(detected_positions)} "
                 f"detected traces matched to reference ({np.sum(is_sky)} sky fibers, "
                 f"{np.sum(is_dead)} dead fibers)")

        return fiber_ids, is_sky, is_dead

    def get_fiber_metadata(self, det, slit_spat_ids):
        """
        Map detected fiber traces to Binospec IFU fiber identifiers.

        Uses cross-correlation of detected trace positions against the
        reference fiber profile to assign physical fiber IDs and names.

        See base class for parameter and return value documentation.
        """
        ref = self.load_fiber_ref_profile(det)

        # Get detected fiber center positions (spat_id is the spatial
        # pixel position at the spectral midpoint)
        detected_positions = slit_spat_ids.astype(float)

        fiber_ids, is_sky, _ = self.match_fibers_to_reference(det, detected_positions)

        # Build name and type arrays from the reference profile
        nslits = len(slit_spat_ids)
        fiber_names = np.full(nslits, 'UNKNOWN', dtype='U20')
        fiber_types = np.full(nslits, 'UNKNOWN', dtype='U10')

        # Map matched fiber IDs back to reference table for names,
        # and derive type from FIB_NAME (SKY* = sky fiber, else science).
        # The FIB_TYPE field in the reference file is unreliable (all SKY).
        ref_ids = ref['FIB_ID']
        ref_names = np.char.strip(ref['FIB_NAME'])

        for i in range(nslits):
            if fiber_ids[i] >= 0:
                idx = np.where(ref_ids == fiber_ids[i])[0]
                if len(idx) > 0:
                    fiber_names[i] = ref_names[idx[0]]
                    fiber_types[i] = 'SKY' if ref_names[idx[0]].startswith('SKY') \
                        else 'SCI'

        n_matched = np.sum(fiber_ids >= 0)
        n_sky = np.sum(fiber_types == 'SKY')
        log.info(f"Fiber metadata: {n_matched}/{nslits} fibers identified "
                 f"({n_sky} sky, {n_matched - n_sky} science)")

        return {'fiber_id': fiber_ids,
                'fiber_name': fiber_names,
                'fiber_type': fiber_types}

