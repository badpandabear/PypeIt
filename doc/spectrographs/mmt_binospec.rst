
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
subtraction.

IFU data are identified automatically from the FITS header keyword
``MASK = 'IFU'`` and reduced using the ``mmt_binospec_ifu`` spectrograph
class with the ``SlicerIFU`` pipeline.

Key IFU parameters set by default:

- 1D extraction is skipped (``skip_extraction = True``); datacubes are
  built directly from the 2D spectra
- Joint sky fitting across all fibers (``joint_fit = True``)
- Grating-dependent B-spline spacing for sky subtraction:
  1.05 Angstrom (270 gpm), 0.5 Angstrom (600 gpm), 0.35 Angstrom
  (1000 gpm)
- Fiber edge detection tuned for densely-packed traces
  (``edge_thresh = 5``)
- Slit edge tweaking using the gradient method

For a detailed comparison of the PypeIt and IDL IFU pipeline approaches,
see :ref:`mmt_binospec_pipeline_comparison`.

