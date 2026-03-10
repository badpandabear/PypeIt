
.. include:: ../include/links.rst

.. _mmt_binospec:

************
MMT Binospec
************

Overview
========

This file summarizes several instrument specific
items for the MMTO's Binospec spectrograph.

PypeIt supports Binospec in three modes:

- **Multi-Object Spectroscopy (MOS)**: ``mmt_binospec``
- **Longslit**: ``mmt_binospec`` (auto-detected from the ``MASK`` header keyword)
- **IFU**: ``mmt_binospec_ifu`` (fiber-fed integral field unit)

Wavelength Calibration
++++++++++++++++++++++

Templates were created in 2023 that cover the usable range for each of the 3 gratings. Only the bluest part of the
1000 l/mm grating's range isn't fully covered by the template, but there should be sufficient overlap for the template
to still work at the bluest settings, e.g. a central wavelength of 4000 A. Better test data is needed to confirm this.

Bad pixel mask
++++++++++++++

The bad pixels were identified from flat and bias observations taken in
2019 and need to be verified.

IFU Mode
========

The Binospec IFU is a fiber-fed integral field unit with a hexagonal
lenslet array feeding approximately 360 fibers per side into the
spectrograph.  Each side has 40 dedicated sky fibers located at the
outermost ring of each hexagonal sub-bundle, which are used for sky
subtraction.  See the `Binospec IFU instrument page
<https://www.mmto.org/instrument-suite/binospec/binospec-ifu-information/>`__
for more details on the hardware and observing modes.

IFU data are identified automatically from the FITS header keyword
``MASK = 'IFU'`` and reduced using the ``mmt_binospec_ifu`` spectrograph
class with the ``Fiber`` pypeline.

Unlike the ``SlicerIFU`` pypeline used for slicer-based IFUs, the
``Fiber`` pypeline treats each fiber as a distinct object and performs
1D spectral extraction as part of the standard pipeline run.  This
produces both spec2d and spec1d output files.

Key IFU parameters set by default:

- Joint sky fitting across all fibers using dedicated sky fibers
  (``joint_fit = True``)
- Grating-dependent B-spline spacing for sky subtraction:
  1.05 Angstrom (270 gpm), 0.5 Angstrom (600 gpm), 0.35 Angstrom
  (1000 gpm)
- Fiber edge detection tuned for densely-packed traces
  (``edge_thresh = 5``)
- Slit edge tweaking using the gradient method
- Spectral flexure correction disabled (Binospec has active flexure
  control)

For a detailed comparison of the PypeIt and IDL IFU pipeline approaches,
see :ref:`mmt_binospec_pipeline_comparison`.

Reducing IFU data
+++++++++++++++++

IFU data are reduced using the standard ``run_pypeit`` workflow.  PypeIt
will automatically detect IFU frames from the ``MASK = 'IFU'`` header
keyword.  Both detectors (``DET01`` for side A and ``DET02`` for side B)
are processed and written to separate spec2d output files.

.. code-block:: bash

   pypeit_setup -r /path/to/raw -s mmt_binospec_ifu -c all
   run_pypeit mmt_binospec_ifu_A/mmt_binospec_ifu_A.pypeit

The pipeline produces spec1d files containing one extracted spectrum per
fiber.  Each spectrum is identified by its instrument fiber name (e.g.,
``SCI1-1``, ``SKY6-1``) via cross-correlation against a reference
profile.  Both boxcar (``BOX``) and optimal Horne (1986) (``OPT``)
extractions are performed.  The spec1d files can be inspected with
``pypeit_show_1dspec``.

.. note::

   The pipeline wavelength calibration is done independently for each
   fiber, which can be time-consuming (~360 fibers per detector).
   A typical reduction with both detectors takes several hours.

Fiber illumination correction
+++++++++++++++++++++++++++++

The pipeline flat-fields at the pixel level but does not correct for
relative fiber-to-fiber throughput differences.  The dedicated
``pypeit_binospec_ifu_illumcorr`` script applies this correction
directly to spec1d files using a pre-computed illumination map
(``fiber_illumination.fits``).

.. code-block:: bash

   pypeit_binospec_ifu_illumcorr spec1d_*.fits

This divides each fiber's flux (and sky) by its relative throughput
factor and multiplies the inverse variance accordingly.  The corrected
spectra are written to new files with an ``_illumcorr`` suffix
(e.g., ``spec1d_*_illumcorr.fits``).  Use ``--overwrite`` to modify the
original files in place instead.

The script sets the header keyword ``ILLUMCOR = True`` on the output
files.  Both ``pypeit_binospec_ifu_illumcorr`` and
``pypeit_binospec_ifu_cube`` check this flag to avoid applying the
correction twice.

If the spec1d files have already been flux-calibrated (i.e., contain
``FLAM`` columns), the script will warn and skip them.  Use ``--force``
to override this check.

.. note::

   This correction should be applied *before* flux calibration.  The
   illumination correction is a spatial (fiber-to-fiber) effect, while
   flux calibration is a spectral response correction.

Producing datacubes
+++++++++++++++++++

Because the Binospec IFU is fiber-fed rather than slicer-based, the
general-purpose ``pypeit_coadd_datacube`` script (designed for slicer
IFUs like KCWI) does not produce correct results.  Instead, use the
dedicated ``pypeit_binospec_ifu_cube`` script, which accepts either
spec1d or spec2d files and builds a datacube from the fiber spectra.

**From spec1d files (recommended):**

The script reads the already-extracted 1D spectra from the pipeline's
spec1d output.  By default, optimal (``OPT``) extraction is used; pass
``--boxcar`` to use boxcar (``BOX``) extraction instead.  Sky
subtraction is already applied by the pipeline, so no additional sky
subtraction is performed.

**From spec2d files:**

The script extracts fiber spectra directly from the 2D spectral images
using optimal (Horne 1986) extraction with an empirical spatial profile
from the flat field (default), a Gaussian profile (``--gaussian``), or
boxcar summation (``--boxcar``).  Sky is subtracted using PypeIt's
per-fiber B-spline sky model (default), or optionally using the
dedicated sky fibers (``--use_fibers``).

**Shared steps (both inputs):**

1. Applies fiber-to-fiber throughput correction using a pre-computed
   illumination map (skipped if the input spec1d file already has
   ``ILLUMCOR = True``, e.g., from ``pypeit_binospec_ifu_illumcorr``)
2. Identifies sky vs. science fibers via cross-correlation against a
   reference profile
3. Resamples all fiber spectra onto a common linear wavelength grid
4. Combines both detectors (up to 640 science fibers total)
5. Maps each fiber to its on-sky position using the IFU layout
   calibration file (``bino_IFU_sky_layout.fits``)
6. Interpolates the irregularly-spaced fiber positions onto a regular
   spatial grid using ``scipy.interpolate.griddata``

.. note::

   The two Binospec detectors produce mirror-image spectra.  The script
   automatically accounts for this by reversing the fiber-to-sky mapping
   for side B so that the two halves of the IFU tile correctly.

.. note::

   The fiber-to-fiber throughput correction applied during cube building
   is separate from the pipeline's pixel-level flat-fielding.  The
   pipeline corrects pixel response but does not correct relative fiber
   throughput (``use_illumflat = False``).  The correction can also be
   applied directly to spec1d files using
   ``pypeit_binospec_ifu_illumcorr`` (see above).  Spectral response
   (flux calibration) from standard star observations is not yet
   implemented.

Basic usage
^^^^^^^^^^^

To build datacubes from spec1d files (recommended):

.. code-block:: bash

   pypeit_binospec_ifu_cube spec1d_*.fits

To build datacubes from spec2d files:

.. code-block:: bash

   pypeit_binospec_ifu_cube spec2d_*.fits

All input files must be the same type (spec1d or spec2d); mixing is not
allowed.  Each input file produces a separate output datacube named
``cube_sci_img_*.fits``.  Each cube combines both detectors and
contains ``FLUX`` and ``VAR`` extensions.

Command-line options
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pypeit_binospec_ifu_cube spec1d_*.fits [options]
   pypeit_binospec_ifu_cube spec2d_*.fits [options]

``--output FILENAME``
   Output FITS filename.  Only valid when processing a single input
   file.  Default is auto-generated from the input filename
   (e.g., ``cube_sci_img_*.fits``).

``--spatial_scale SCALE``
   Output spatial pixel scale in arcsec.  Default is 0.27, which
   matches the IDL pipeline (``scl = 0.269461``).

``--no_skysub``
   Skip sky subtraction entirely (spec2d only; ignored for spec1d
   since sky is already subtracted by the pipeline).

``--use_fibers``
   Use the 40 dedicated sky fibers per side to compute a
   sigma-clipped mean sky spectrum for subtraction (matching the IDL
   pipeline approach).  By default, the script uses PypeIt's
   per-fiber B-spline sky model from the spec2d file.  This option
   is only relevant for spec2d input; it is ignored for spec1d.

``--boxcar``
   For spec1d input: use boxcar (``BOX``) extraction columns instead
   of the default optimal (``OPT``) columns.  For spec2d input: use
   boxcar (unweighted sum) extraction instead of the default optimal
   (Horne 1986) profile-weighted extraction.

``--gaussian``
   Use a Gaussian spatial profile for optimal extraction instead of
   the default empirical profile measured from the flat field (spec2d
   only).  The Gaussian width is derived from the slit edge traces.
   This is also the automatic fallback if the flat field calibration
   file cannot be loaded.

``--method METHOD``
   Spatial interpolation method: ``nearest``, ``linear`` (default), or
   ``cubic``.  The ``linear`` method is recommended for most use cases.

Output format
^^^^^^^^^^^^^

The output FITS file contains:

- **Extension 0** (``PRIMARY``): Empty primary HDU
- **Extension 1** (``FLUX``): 3D datacube with axes
  (NAXIS1=RA, NAXIS2=DEC, NAXIS3=wavelength) and a full 3-axis WCS
  (``RA---TAN``, ``DEC--TAN``, ``WAVE``)
- **Extension 2** (``VAR``): Variance datacube with the same shape and
  WCS

The cube can be viewed directly with tools like ds9 or QFitsView.
Typical output dimensions for the default spatial scale are approximately
63 x 47 spatial pixels, with the number of wavelength pixels depending
on the grating and wavelength coverage.

.. note::

   The datacube construction can be sped up significantly by using an
   accelerated BLAS library.  On macOS (13.3+), switching to Apple's
   ``newaccelerate`` backend yielded a ~3x speedup.  See the
   `conda-forge BLAS documentation
   <https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation>`__
   for instructions on selecting a BLAS implementation with conda, e.g.:

   .. code-block:: bash

      conda install "libblas=*=*newaccelerate"

