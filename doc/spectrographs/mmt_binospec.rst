
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
class with the ``SlicerIFU`` pipeline.

Key IFU parameters set by default:

- 1D extraction is skipped (``skip_extraction = True``); extraction is
  performed during datacube construction
- Joint sky fitting across all fibers (``joint_fit = True``)
- Grating-dependent B-spline spacing for sky subtraction:
  1.05 Angstrom (270 gpm), 0.5 Angstrom (600 gpm), 0.35 Angstrom
  (1000 gpm)
- Fiber edge detection tuned for densely-packed traces
  (``edge_thresh = 5``)
- Slit edge tweaking using the gradient method

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

Producing datacubes
+++++++++++++++++++

Because the Binospec IFU is fiber-fed rather than slicer-based, the
general-purpose ``pypeit_coadd_datacube`` script (designed for slicer
IFUs like KCWI) does not produce correct results.  Instead, use the
dedicated ``pypeit_binospec_ifu_cube`` script, which implements the
fiber-based datacube construction workflow:

1. Extracts each fiber as a 1D spectrum from the spec2d files using
   optimal (Horne 1986) profile-weighted extraction with an empirical
   spatial profile measured from the flat field calibration (default),
   a Gaussian profile (``--gaussian``), or boxcar summation
   (``--boxcar``)
2. Subtracts sky using PypeIt's per-fiber B-spline sky model from the
   spec2d file (default), or optionally using the 40 dedicated sky
   fibers per side (``--use_fibers``)
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

Basic usage
^^^^^^^^^^^

To build datacubes from spec2d files:

.. code-block:: bash

   pypeit_binospec_ifu_cube spec2d_*.fits

Each input spec2d file produces a separate output datacube.  For
example, three input files will produce three cubes named
``cube_sci_img_*.fits``.  Each cube combines both detectors, applies
sky subtraction, and contains ``FLUX`` and ``VAR`` extensions.

Command-line options
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pypeit_binospec_ifu_cube spec2d_*.fits [options]

``--output FILENAME``
   Output FITS filename.  Only valid when processing a single input
   file.  Default is auto-generated from the input filename
   (e.g., ``cube_sci_img_*.fits``).

``--spatial_scale SCALE``
   Output spatial pixel scale in arcsec.  Default is 0.27, which
   matches the IDL pipeline (``scl = 0.269461``).

``--no_skysub``
   Skip sky subtraction entirely.

``--use_fibers``
   Use the 40 dedicated sky fibers per side to compute a
   sigma-clipped mean sky spectrum for subtraction (matching the IDL
   pipeline approach).  By default, the script uses PypeIt's
   per-fiber B-spline sky model from the spec2d file, which does a
   better job handling bright features in the sky background.

``--boxcar``
   Use boxcar (unweighted sum) extraction instead of the default
   optimal (Horne 1986) profile-weighted extraction.  Boxcar may be
   preferable for extended sources that do not match the fiber profile.

``--gaussian``
   Use a Gaussian spatial profile for optimal extraction instead of
   the default empirical profile measured from the flat field.  The
   Gaussian width is derived from the slit edge traces.  This is also
   the automatic fallback if the flat field calibration file cannot be
   loaded.

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

