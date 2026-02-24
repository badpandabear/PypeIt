
.. include:: ../include/links.rst

.. _mmt_binospec_pipeline_comparison:

**********************************************
Binospec Pipeline Comparison: IDL vs PypeIt
**********************************************

Comparison of the Binospec IDL reduction pipeline and PypeIt approaches
to error propagation and sky subtraction, with implications for adding
IFU support to PypeIt.

.. contents:: Table of Contents
   :depth: 2
   :local:


Error Propagation
=================

PypeIt: Full Variance Model
++++++++++++++++++++++++++++

PypeIt tracks errors through every processing step using inverse variance
(``ivar``) arrays.  The variance model
(:func:`~pypeit.core.procimg.variance_model`) is:

.. math::

   V = s^2 \left[\max(0,C) + \frac{D \cdot t}{3600} + V_\mathrm{rn}
   + V_\mathrm{proc}\right] + \varepsilon^2 \max(0,c)^2

where *s* is the inverse sensitivity (1/gain), *C* is counts, *D* is dark
current, *V_rn* is read noise variance, *V_proc* is processing variance,
and *eps* is the noise floor (fractional error).

Key properties:

- **Base variance** (detector-level: read noise + dark + processing) is
  tracked separately from signal-dependent Poisson noise
- ``ivar`` arrays propagated through every step: image processing, flat
  fielding, sky subtraction, and extraction
- Optimal extraction weights computed as ``ivar * profile**2``
- Two output ivar columns: ``OPT_COUNTS_IVAR`` (full variance including
  object shot noise) and ``OPT_COUNTS_NIVAR`` (sky + readnoise only,
  for S/N estimation)
- Model variance iteratively updated during local sky subtraction when
  ``model_noise=True``
- Relevant code: ``pypeit/core/procimg.py``, ``pypeit/core/skysub.py``,
  ``pypeit/core/extract.py``

IDL Pipeline: Incomplete Error Tracking
++++++++++++++++++++++++++++++++++++++++

The IDL pipeline does not propagate errors through most processing steps:

- **No error tracking** through fiber tracing, flat fielding, wavelength
  calibration, or spectral linearization
- **Sky subtraction** uses inverse-variance weighting internally:

  .. code-block:: none

     isky = gain^2 / (gain * |sky| + rdnoise^2)

  but the resulting sky model errors are not propagated to downstream
  products

- **Wavelength calibration** stores ``wl_s_err`` (RMS per fiber) but
  never propagates it
- **Fiber extraction** via bounded least-squares
  (``bounded_least_squares.pro``) solves for fiber fluxes with
  positivity constraints but does not compute covariance
- **Only at cube building** (``bino_ifu_cube.pro``) are errors finally
  estimated:

  .. code-block:: none

     err = sqrt(flux * flat + rdnoise^2) / flat

  This is a pure Poisson + read noise assumption that ignores
  accumulated errors from all prior processing steps

- No covariance tracking between adjacent fibers (relevant since fiber
  profiles overlap)

Comparison Table
++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Processing Step
     - PypeIt
     - IDL Pipeline
   * - Raw to processed image
     - Full variance model (Poisson+RN+dark+proc)
     - None
   * - Flat fielding
     - ``ivar`` updated for flat noise
     - No error map
   * - Sky subtraction
     - B-spline weighted by ``ivar``; model ``ivar`` iteratively updated
     - Weighted internally; no error output
   * - Extraction
     - Optimal with model variance; outputs ``ivar``
     - Bounded LS; no error bars
   * - Wavelength cal
     - Propagated through resampling
     - Stores ``wl_s_err`` (unused)
   * - Cube building
     - Full ``ivar`` propagation via ``coadd3d``
     - Poisson+RN assumption only

Implication
+++++++++++

Adding Binospec IFU to PypeIt will automatically provide proper error
propagation that the IDL pipeline currently lacks.  This is particularly
important for:

- Faint emission-line science where reliable S/N estimates are critical
- Combining exposures with different conditions (proper inverse-variance
  weighting)
- Flagging unreliable spaxels from dead/weak fibers


Sky Subtraction
===============

PypeIt
++++++

PypeIt offers three sky subtraction modes, all using B-spline fitting:

**Global sky** (:func:`~pypeit.core.skysub.global_skysub`):

- 1D B-spline fit in the spectral direction with polynomial basis in the
  spatial direction
- Configurable bspline spacing (grating-dependent for Binospec IFU),
  sigma rejection (3.0), max 35 iterations
- Optional pre-fit to ``log(sky)`` for positive pixels to handle bright
  sky lines

**Local sky** (:func:`~pypeit.core.skysub.local_skysub_extract`):

- Joint sky + object B-spline modeling
- Iterative model variance when ``model_noise=True``
- Used primarily for long-slit and MOS point sources

**IFU joint fit** (:class:`~pypeit.find_objects.SlicerIFUFindObjects`):

- Fits sky model across ALL slices simultaneously
- Convolves to common spectral resolution via FWHM map
- Applies spectral flexure correction per slice before joint fit
- Optional spatial and spectral sensitivity corrections
- Currently implemented for slicer-based IFUs (KCWI, GNIRS, OSIRIS)

IDL Pipeline
++++++++++++

**MOS mode: Kelson-style 2D/3D B-spline** (``bino_create_sky_ms.pro``,
``bino_sub_sky_ms.pro``):

- 3D spline for TARGET slits (wavelength, x_mask, y_mask in focal plane
  coordinates)
- 2D spline for BOX slits
- Grating-dependent knot spacing:

  - 270 gpm: 1.05 Angstrom
  - 600 gpm: 0.50 Angstrom
  - 1000 gpm: 0.35 Angstrom

- Inverse-variance weighting: ``isky = gain^2 / (gain * |sky| + rdnoise^2)``
- Two-stage fitting for extended slits (blue/red halves separately with
  reduced polynomial orders)

**IFU mode: Dedicated sky fibers**:

- 40 dedicated sky fibers per side at hexagonal bundle edges:

  .. code-block:: none

     sky_fib_idx = [0-7, 88-95, 176-183, 264-271, 352-359]

  (8 fibers per group x 5 radial positions = outermost ring of each
  sub-bundle)

- Simple sky estimation via ``resistant_mean()`` at 2-sigma over sky
  fibers during linearization (quick mode)
- Alternative: B-spline sky model using only sky fiber data

**Sky line correction** (``bino_sub_sky_ms.pro``):

- Corrects for PSF differences between sky model and science data
- Computes ``sigdiff = sqrt(sig_obs^2 - sig_mod^2)``
- Convolves sky model with Gaussian correction kernel when
  ``sigdiff > 0.2`` pixels

Comparison Table
++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - PypeIt
     - IDL Pipeline
   * - Sky model
     - B-spline (1D spectral + poly spatial)
     - B-spline (2D/3D in wavelength + focal plane coords)
   * - IFU strategy
     - Joint fit across all slices
     - Dedicated sky fibers at bundle edges
   * - Spatial variation
     - Polynomial basis within slit
     - Full 3D spline over focal plane
   * - Variance in fit
     - Full ``ivar`` from variance model
     - Approximate gain/readnoise weighting
   * - Flexure correction
     - Per-slice spectral flexure via cross-correlation
     - Cross-correlation offset in wavelength zero-point
   * - Sky line sharpness
     - Not explicitly corrected
     - PSF difference kernel applied

Implication for Binospec IFU in PypeIt
++++++++++++++++++++++++++++++++++++++

For Binospec IFU in PypeIt, the dedicated sky-fiber approach from the IDL
pipeline should be combined with PypeIt's variance-weighted B-spline
fitting:

1. **Sky fiber identification**: Use IDL's fiber indices
   [0-7, 88-95, 176-183, 264-271, 352-359] (outermost ring of each
   hexagonal sub-bundle) to build a sky mask
2. **Joint sky fit**: Use PypeIt's ``joint_skysub()`` with B-spline
   fitting across all sky fibers, gaining proper ``ivar`` weighting
3. **Grating-dependent spacing**: Adopt IDL's bspline_spacing values per
   grating (1.05/0.5/0.35 Angstrom for 270/600/1000 gpm)
4. **Sky line correction**: Consider implementing IDL's PSF difference
   kernel approach as a future enhancement


Fiber-Specific Considerations
==============================

Fiber Tracing
+++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - PypeIt
     - IDL Pipeline
   * - Trace model
     - Edge pairs (left/right per slit)
     - Gaussian-Hermite profile per fiber (h3-h6)
   * - Detection method
     - Sobel filter + threshold on flat field
     - Peak detection + iterative profile fitting
   * - Profile model
     - Gaussian or empirical for extraction
     - 8-parameter Gaussian-Hermite or Moffat
   * - Fiber ID
     - Sequential edge numbering
     - Cross-correlation against reference profile
   * - Dead fiber handling
     - Not applicable (slit-based)
     - Interpolation from reference catalog

Extraction
++++++++++

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - PypeIt
     - IDL Pipeline
   * - Method
     - Optimal extraction (Horne 1986)
     - Bounded least-squares (positivity constraint)
   * - Cross-talk
     - Not modeled
     - Simultaneous multi-fiber solve per block
   * - Error output
     - Full ``ivar`` per pixel
     - None
   * - Profile
     - Empirical from data
     - Gaussian-Hermite from flat field

Wavelength Calibration
++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - PypeIt
     - IDL Pipeline
   * - Method
     - Full template matching + reidentification
     - Per-block Legendre polynomial (degree 3)
   * - Spatial model
     - 2D fit (spectral + spatial)
     - Independent per-fiber solution
   * - Refinement
     - Flexure correction from sky lines
     - Sky line cross-correlation adjustment
   * - Arc lamps
     - HeI, NeI, ArI, ArII
     - Same (HeNe + Ar)


Summary
=======

PypeIt provides a more rigorous statistical framework (full error
propagation, variance-weighted fitting) while the IDL pipeline has more
specialized algorithms for fiber-fed spectroscopy (Gaussian-Hermite
profiles, bounded least-squares extraction, dedicated sky fibers,
PSF-matched sky subtraction).

The optimal strategy for Binospec IFU in PypeIt is to leverage PypeIt's
infrastructure (variance model, B-spline fitting, datacube construction)
while incorporating the domain-specific knowledge from the IDL pipeline
(sky fiber layout, fiber identification via cross-correlation,
grating-specific parameters).
