# Fiber Superflat and 1D Sky Subtraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the crashing 2D block-slit sky subtraction with MaNGA-style 1D sky subtraction using a superflat/fiberflat calibration framework.

**Architecture:** New `FiberFlatField` subclass builds a superflat (common spectral response), per-fiber fiberflats, and a gray throughput ratio from extracted flat field fibers. `FiberFindObjects` uses these to equalize extracted 1D spectra and build a B-spline sky model that is subtracted in 1D. The block-slit architecture is preserved for calibration.

**Tech Stack:** numpy, scipy, PypeIt bspline fitting (`pypeit.core.fitting.bspline_profile`), PypeIt DataContainer system

**Spec:** `claude_docs/specs/2026-03-28-fiber-superflat-sky-subtraction-design.md`

**Execution order:** 8 (clean state) → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9 (integration test)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pypeit/flatfield.py` | Modify | Add `FiberFlatImages` DataContainer, `FiberFlatField` subclass |
| `pypeit/calibrations.py` | Modify | Override `IFUCalibrations.get_flats()` for Fiber pypeline dispatch |
| `pypeit/find_objects.py` | Modify | Rework `FiberFindObjects.run()` for 1D sky subtraction |
| `pypeit/extraction.py` | Modify | Update `FiberExtract` to pass sky model for optimal extraction |
| `pypeit/images/rawimage.py` | Modify | Skip illumflat/specillum for Fiber pypeline |
| `pypeit/tests/test_fiberflatfield.py` | Create | Unit tests for superflat/fiberflat construction |

---

### Task 1: FiberFlatImages DataContainer

Add a new DataContainer to hold fiber-specific flat field products. This is separate from the standard `FlatImages` so the existing datamodel is untouched.

**Files:**
- Modify: `pypeit/flatfield.py` (add class after `FlatImages`, around line 345)
- Test: `pypeit/tests/test_fiberflatfield.py`

- [ ] **Step 1: Write the test for FiberFlatImages I/O round-trip**

Create `pypeit/tests/test_fiberflatfield.py`:

```python
"""Tests for FiberFlatField and FiberFlatImages."""
import numpy as np
import pytest
from pathlib import Path

from pypeit.flatfield import FiberFlatImages
from pypeit.tests.tstutils import data_output_path


def test_fiberflatimages_io():
    """Test FiberFlatImages DataContainer I/O round-trip."""
    nfibers = 40
    nwave = 100

    # Create synthetic fiber flat products
    superflat = np.random.uniform(0.8, 1.2, nwave).astype(np.float64)
    superflat_wave = np.linspace(4000.0, 7000.0, nwave).astype(np.float64)
    fiberflat = np.random.uniform(0.95, 1.05, (nfibers, nwave)).astype(np.float64)
    fiber_scale_factors = np.random.uniform(0.5, 1.5, nfibers).astype(np.float64)
    throughput_ratio = 0.85
    fiber_ids = np.arange(1, nfibers + 1).astype(np.int64)
    fiber_types = np.array(['sky'] * 8 + ['science'] * 32)

    ffi = FiberFlatImages(
        superflat=superflat,
        superflat_wave=superflat_wave,
        fiberflat=fiberflat,
        fiber_scale_factors=fiber_scale_factors,
        throughput_ratio=throughput_ratio,
        fiber_ids=fiber_ids,
        fiber_types=fiber_types,
        PYP_SPEC='mmt_binospec_ifu',
    )

    # Write and read back
    ofile = Path(data_output_path('test_fiberflatimages.fits')).absolute()
    ffi.to_file(str(ofile), overwrite=True)
    _ffi = FiberFlatImages.from_file(str(ofile))

    assert np.allclose(ffi.superflat, _ffi.superflat)
    assert np.allclose(ffi.superflat_wave, _ffi.superflat_wave)
    assert np.allclose(ffi.fiberflat, _ffi.fiberflat)
    assert np.allclose(ffi.fiber_scale_factors, _ffi.fiber_scale_factors)
    assert np.isclose(ffi.throughput_ratio, _ffi.throughput_ratio)
    assert np.array_equal(ffi.fiber_ids, _ffi.fiber_ids)

    ofile.unlink()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pypeit/tests/test_fiberflatfield.py::test_fiberflatimages_io -v`
Expected: FAIL with `ImportError: cannot import name 'FiberFlatImages'`

- [ ] **Step 3: Implement FiberFlatImages**

Add to `pypeit/flatfield.py` after the `FlatImages` class (around line 345):

```python
class FiberFlatImages(datamodel.DataContainer):
    """
    DataContainer for fiber-specific flat field calibration products.

    Holds the superflat (common spectral response), per-fiber fiberflats
    (relative throughput), scale factors, and throughput ratio between
    sky and science fiber types.  These are produced by
    :class:`FiberFlatField` and consumed by
    :class:`~pypeit.find_objects.FiberFindObjects` for 1D sky
    subtraction.
    """

    version = '1.0.0'
    hdu_prefix = None

    datamodel = {
        'PYP_SPEC': dict(otype=str, descr='PypeIt spectrograph name'),
        'superflat': dict(otype=np.ndarray, atype=np.floating,
                          descr='Superflat: common spectral response shape from '
                                'B-spline fit to all fiber flat spectra'),
        'superflat_wave': dict(otype=np.ndarray, atype=np.floating,
                               descr='Wavelength grid for superflat evaluation'),
        'fiberflat': dict(otype=np.ndarray, atype=np.floating,
                          descr='Per-fiber relative throughput as a function '
                                'of wavelength, shape (nfiber, nwave)'),
        'fiber_scale_factors': dict(otype=np.ndarray, atype=np.floating,
                                    descr='Per-fiber median flux scale factors'),
        'throughput_ratio': dict(otype=float,
                                 descr='Gray throughput ratio: '
                                       'median(sky scales) / median(science scales)'),
        'fiber_ids': dict(otype=np.ndarray, atype=np.integer,
                          descr='Fiber IDs matching fiberflat rows'),
        'fiber_types': dict(otype=np.ndarray, atype=str,
                            descr='Fiber type per fiber: sky or science'),
    }

    internals = ['calib_key', 'calib_dir']

    def __init__(self, superflat=None, superflat_wave=None, fiberflat=None,
                 fiber_scale_factors=None, throughput_ratio=None,
                 fiber_ids=None, fiber_types=None, PYP_SPEC=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        _d = dict([(k, values[k]) for k in args[1:]])
        super().__init__(d=_d)

    def _bundle(self):
        return super()._bundle(ext='FIBER_FLAT')

    def set_paths(self, calib_dir, calib_key, det_str):
        self.calib_key = calib_key
        self.calib_dir = calib_dir

    def get_path(self):
        return str(Path(self.calib_dir) /
                   f'FiberFlat_{self.calib_key}_{self.PYP_SPEC}.fits')
```

Add `import inspect` at top of `flatfield.py` if not already imported.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest pypeit/tests/test_fiberflatfield.py::test_fiberflatimages_io -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pypeit/flatfield.py pypeit/tests/test_fiberflatfield.py
git commit -m "feat: add FiberFlatImages DataContainer for fiber flat products"
```

---

### Task 2: Superflat and Fiberflat Construction

Implement the core superflat/fiberflat algorithm in `FiberFlatField`. This task focuses on the 1D fiber flat construction — the 2D pixelflat is handled in Task 3.

**Files:**
- Modify: `pypeit/flatfield.py` (add `FiberFlatField` class after `FiberFlatImages`)
- Test: `pypeit/tests/test_fiberflatfield.py`

- [ ] **Step 1: Write the test for superflat construction**

Add to `pypeit/tests/test_fiberflatfield.py`:

```python
from pypeit.flatfield import FiberFlatField


def test_superflat_construction():
    """Test superflat from synthetic fiber spectra with known properties.

    Creates 10 fibers with a common spectral shape (quadratic) but different
    throughputs (scale factors 0.5-1.5). The superflat should recover the
    common spectral shape, and fiber scale factors should match the input.
    """
    np.random.seed(42)
    nfibers = 10
    nwave = 500
    wave_grid = np.linspace(4000.0, 7000.0, nwave)

    # Common spectral shape: quadratic (simulating lamp spectrum)
    true_shape = 1.0 - 0.3 * ((wave_grid - 5500.0) / 1500.0) ** 2
    true_shape /= np.median(true_shape)  # normalize to median=1

    # Per-fiber scale factors (throughputs)
    true_scales = np.linspace(0.5, 1.5, nfibers)

    # Generate fiber spectra: shape * scale + noise
    fiber_spectra = np.zeros((nfibers, nwave))
    fiber_ivar = np.zeros((nfibers, nwave))
    for i in range(nfibers):
        flux = true_scales[i] * true_shape * 10000.0  # counts
        noise = np.sqrt(flux)
        fiber_spectra[i] = flux + np.random.normal(0, 1, nwave) * noise
        fiber_ivar[i] = 1.0 / (noise ** 2)

    # Each fiber has slightly offset wavelengths (simulating different solutions)
    fiber_waves = np.zeros((nfibers, nwave))
    for i in range(nfibers):
        offset = np.random.uniform(-2.0, 2.0)  # small wavelength offset
        fiber_waves[i] = wave_grid + offset

    # Call the static method that builds superflat + fiberflat
    superflat, superflat_wave, fiberflat, scale_factors = \
        FiberFlatField.build_superflat_fiberflat(
            fiber_spectra, fiber_waves, fiber_ivar)

    # Scale factors should correlate with true scales
    # (absolute scale depends on normalization, but relative ordering preserved)
    rank_true = np.argsort(true_scales)
    rank_measured = np.argsort(scale_factors)
    assert np.array_equal(rank_true, rank_measured), \
        f"Scale factor ranking mismatch: {scale_factors}"

    # Fiberflats should be near unity (within ~5% for S/N ~ 100)
    for i in range(nfibers):
        assert np.all(np.abs(fiberflat[i] - 1.0) < 0.1), \
            f"Fiberflat {i} deviates too far from unity: " \
            f"range [{fiberflat[i].min():.3f}, {fiberflat[i].max():.3f}]"

    # Superflat should have the same shape as the input spectrum
    # Evaluate at wave_grid midpoints
    mid = nwave // 2
    quarter = nwave // 4
    # The superflat should show the quadratic shape:
    # higher at center, lower at edges
    assert superflat[mid] > superflat[quarter], \
        "Superflat should be brighter at center than at edges"


def test_throughput_ratio():
    """Test gray throughput ratio computation."""
    scale_factors = np.array([0.6, 0.7, 0.65, 0.55,  # sky fibers (indices 0-3)
                              1.0, 1.1, 0.95, 1.05, 0.98, 1.02])  # science fibers
    fiber_types = np.array(['sky'] * 4 + ['science'] * 6)

    ratio = FiberFlatField.compute_throughput_ratio(scale_factors, fiber_types)

    expected = np.median([0.6, 0.7, 0.65, 0.55]) / np.median([1.0, 1.1, 0.95, 1.05, 0.98, 1.02])
    assert np.isclose(ratio, expected, rtol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pypeit/tests/test_fiberflatfield.py::test_superflat_construction -v`
Expected: FAIL with `ImportError: cannot import name 'FiberFlatField'`

- [ ] **Step 3: Implement FiberFlatField with build_superflat_fiberflat()**

Add to `pypeit/flatfield.py` after `FiberFlatImages`:

```python
class FiberFlatField(FlatField):
    """
    Flat-field calibration for fiber-fed spectrographs.

    Builds a MaNGA-style superflat (common spectral response) and per-fiber
    fiberflats (relative throughput) from extracted flat-field fiber spectra.
    Also produces a pixel-only 2D pixelflat for pixel-to-pixel corrections.

    The superflat captures the lamp spectrum convolved with system throughput.
    The fiberflat captures per-fiber throughput variations (near unity, slowly
    varying with wavelength).  The gray throughput ratio between sky and
    science fiber types replaces fiber_illumination.fits.

    See Also
    --------
    :class:`FlatField` : Parent class for slit-based flat-field calibration.
    :class:`FiberFlatImages` : DataContainer holding fiber flat products.
    """

    @staticmethod
    def build_superflat_fiberflat(fiber_spectra, fiber_waves, fiber_ivar,
                                  bkspace=1.5):
        """
        Build superflat and per-fiber fiberflats from extracted flat spectra.

        Parameters
        ----------
        fiber_spectra : `numpy.ndarray`_
            Extracted fiber spectra, shape ``(nfiber, nwave)``.
        fiber_waves : `numpy.ndarray`_
            Wavelength arrays per fiber, shape ``(nfiber, nwave)``.
        fiber_ivar : `numpy.ndarray`_
            Inverse variance per fiber, shape ``(nfiber, nwave)``.
        bkspace : :obj:`float`, optional
            B-spline knot spacing in Angstroms for superflat fit.

        Returns
        -------
        superflat : `numpy.ndarray`_
            Superflat evaluated on ``superflat_wave``.
        superflat_wave : `numpy.ndarray`_
            Common wavelength grid for superflat.
        fiberflat : `numpy.ndarray`_
            Per-fiber relative throughput, shape ``(nfiber, nwave)``.
        scale_factors : `numpy.ndarray`_
            Per-fiber median flux scale factors.
        """
        from pypeit.core import fitting

        nfiber, nwave = fiber_spectra.shape

        # Step 1: Compute per-fiber scale factors (median flux)
        scale_factors = np.zeros(nfiber)
        for i in range(nfiber):
            good = (fiber_ivar[i] > 0) & np.isfinite(fiber_spectra[i])
            if np.any(good):
                scale_factors[i] = np.median(fiber_spectra[i, good])
            else:
                scale_factors[i] = 1.0

        # Step 2: Normalize each fiber to median=1 and combine into
        # super-sampled array sorted by wavelength
        all_wave = []
        all_flux = []
        all_ivar = []
        for i in range(nfiber):
            if scale_factors[i] <= 0:
                continue
            good = ((fiber_ivar[i] > 0) & np.isfinite(fiber_spectra[i])
                    & (fiber_waves[i] > 0))
            if not np.any(good):
                continue
            norm_flux = fiber_spectra[i, good] / scale_factors[i]
            norm_ivar = fiber_ivar[i, good] * scale_factors[i] ** 2
            all_wave.append(fiber_waves[i, good])
            all_flux.append(norm_flux)
            all_ivar.append(norm_ivar)

        all_wave = np.concatenate(all_wave)
        all_flux = np.concatenate(all_flux)
        all_ivar = np.concatenate(all_ivar)

        # Sort by wavelength
        srt = np.argsort(all_wave)
        all_wave = all_wave[srt]
        all_flux = all_flux[srt]
        all_ivar = all_ivar[srt]

        # Step 3: B-spline fit to super-sampled composite
        sset, outmask, yfit, _, exit_status = fitting.bspline_profile(
            all_wave, all_flux, all_ivar,
            np.ones((len(all_wave), 1)),
            nord=4, upper=3.0, lower=3.0,
            kwargs_bspline={'bkspace': bkspace})

        if exit_status > 1:
            msgs.warn("Superflat B-spline fit did not converge "
                      f"(exit_status={exit_status})")

        # Evaluate superflat on a common wavelength grid
        wmin = np.min(all_wave)
        wmax = np.max(all_wave)
        superflat_wave = np.linspace(wmin, wmax, nwave)
        superflat = sset.value(superflat_wave)[0].flatten()

        # Step 4: Build fiberflat per fiber
        fiberflat = np.ones((nfiber, nwave))
        for i in range(nfiber):
            if scale_factors[i] <= 0:
                continue
            good = ((fiber_ivar[i] > 0) & np.isfinite(fiber_spectra[i])
                    & (fiber_waves[i] > 0))
            if not np.any(good):
                continue
            # Evaluate superflat at this fiber's wavelengths
            sf_at_fiber = sset.value(fiber_waves[i, good])[0].flatten()
            # Fiberflat = fiber / (scale * superflat)
            expected = scale_factors[i] * sf_at_fiber
            ratio = np.ones(np.sum(good))
            pos = expected > 0
            ratio[pos] = fiber_spectra[i, good][pos] / expected[pos]

            # B-spline smooth the fiberflat to reduce noise
            ff_ivar = fiber_ivar[i, good] * expected ** 2
            ff_ivar[~pos] = 0.0
            try:
                ff_sset, _, ff_fit, _, ff_exit = fitting.bspline_profile(
                    fiber_waves[i, good], ratio, ff_ivar,
                    np.ones((np.sum(good), 1)),
                    nord=4, upper=3.0, lower=3.0,
                    kwargs_bspline={'bkspace': bkspace * 5})
                if ff_exit <= 1:
                    # Evaluate on the common superflat wavelength grid
                    # so all fiberflats share the same wavelength sampling
                    fiberflat[i] = ff_sset.value(superflat_wave)[0].flatten()
                else:
                    fiberflat[i] = np.interp(superflat_wave,
                                             fiber_waves[i, good], ratio,
                                             left=1.0, right=1.0)
            except Exception:
                fiberflat[i] = np.interp(superflat_wave,
                                         fiber_waves[i, good], ratio,
                                         left=1.0, right=1.0)

        return superflat, superflat_wave, fiberflat, scale_factors

    @staticmethod
    def compute_throughput_ratio(scale_factors, fiber_types):
        """
        Compute gray throughput ratio between sky and science fibers.

        Parameters
        ----------
        scale_factors : `numpy.ndarray`_
            Per-fiber median flux scale factors.
        fiber_types : `numpy.ndarray`_
            Fiber type per fiber: ``'sky'`` or ``'science'``.

        Returns
        -------
        :obj:`float`
            Ratio ``median(sky_scales) / median(science_scales)``.
        """
        sky_mask = np.array([t.lower() == 'sky' for t in fiber_types])
        sci_mask = ~sky_mask
        sky_scales = scale_factors[sky_mask]
        sci_scales = scale_factors[sci_mask]
        if len(sky_scales) == 0 or len(sci_scales) == 0:
            msgs.warn("Cannot compute throughput ratio: "
                      f"{np.sum(sky_mask)} sky, {np.sum(sci_mask)} science fibers")
            return 1.0
        return float(np.median(sky_scales) / np.median(sci_scales))
```

Add `from pypeit import msgs` import if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pypeit/tests/test_fiberflatfield.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pypeit/flatfield.py pypeit/tests/test_fiberflatfield.py
git commit -m "feat: add FiberFlatField with superflat/fiberflat construction"
```

---

### Task 3: FiberFlatField.run() and Pixel-Only Flat

Implement the `run()` method that orchestrates extraction of flat fibers, superflat/fiberflat construction, and production of a pixel-only 2D flat.

**Files:**
- Modify: `pypeit/flatfield.py` (add `run()` to `FiberFlatField`)

- [ ] **Step 1: Implement FiberFlatField.run()**

Add the `run()` method to `FiberFlatField`:

```python
    def run(self, doqa=False, debug=False, show=False):
        """
        Build fiber flat field calibration products.

        Produces:

        1. A pixel-only 2D pixelflat (no spectral normalization)
        2. A 1D superflat (common spectral response)
        3. Per-fiber fiberflats (relative throughput)
        4. Gray throughput ratio (sky vs science fiber aperture)

        Parameters
        ----------
        doqa : :obj:`bool`, optional
            Save QA output.
        debug : :obj:`bool`, optional
            Run in debug mode.
        show : :obj:`bool`, optional
            Show results in viewer.

        Returns
        -------
        flatImages : :class:`FlatImages`
            Standard flat images with pixel-only pixelflat_norm.
        fiber_flatimages : :class:`FiberFlatImages`
            Fiber-specific flat products (superflat, fiberflat, etc.).
        """
        from pypeit.core import extract as core_extract

        rawflat = self.rawflatimg.image
        nspec, nspat = rawflat.shape

        # --- 2D pixel-only flat ---
        # Smooth the raw flat with a large median filter to get the
        # low-frequency response (spectral shape + fiber throughput).
        # The ratio raw/smooth isolates pixel-to-pixel variations.
        from scipy.ndimage import median_filter
        smooth_flat = median_filter(rawflat, size=(1, 15))
        smooth_flat[smooth_flat <= 0] = 1.0
        pixelflat_norm = rawflat / smooth_flat
        # Clip extreme values (bad pixels)
        pixelflat_norm = np.clip(pixelflat_norm, 0.5, 1.5)
        # Set pixels outside slits to 1.0
        slitmask = self.slits.slit_img(initial=True)
        pixelflat_norm[slitmask < 0] = 1.0

        flatImages = FlatImages(
            pixelflat_raw=rawflat,
            pixelflat_norm=pixelflat_norm,
            PYP_SPEC=self.spectrograph.name,
            spat_id=self.slits.spat_id,
        )

        # --- 1D superflat + fiberflat ---
        # Get fiber block structure and positions
        blocks = self.spectrograph.get_fiber_blocks(
            self.rawflatimg.detector.det)

        # Build wavelength image for extraction
        if self.wavetilts is not None and self.wv_calib is not None:
            waveimg = self.wv_calib.build_waveimg(
                self.wavetilts, self.slits, spat_flexure=None)
        else:
            msgs.warn("No wavelength calibration available for fiber flat; "
                      "using pixel coordinates as wavelength proxy")
            waveimg = np.tile(np.arange(nspec, dtype=float)[:, None],
                             (1, nspat))

        # Extract all flat fibers to 1D
        gpm = self.rawflatimg.select_flag(invert=True) if hasattr(
            self.rawflatimg, 'select_flag') else np.ones((nspec, nspat), dtype=bool)
        ivar = self.rawflatimg.ivar if hasattr(
            self.rawflatimg, 'ivar') and self.rawflatimg.ivar is not None \
            else np.ones_like(rawflat)

        all_spectra = []
        all_waves = []
        all_ivars = []
        all_fiber_ids = []
        all_fiber_types = []

        for slit_idx in range(self.slits.nslits):
            if slit_idx >= len(blocks):
                continue
            block = blocks[slit_idx]
            fiber_centers = block['fiber_positions'].copy()
            if len(fiber_centers) == 0:
                continue

            slit_spat_id = self.slits.spat_id[slit_idx]
            thismask = slitmask == slit_spat_id
            inmask = gpm & thismask

            # Identify fibers
            fiber_meta = self.spectrograph.identify_fibers_in_block(
                self.rawflatimg.detector.det,
                slit_idx, fiber_centers)

            # Compute aperture half-widths
            spacings = np.diff(fiber_centers)
            half_spacings = np.zeros(len(fiber_centers))
            if len(spacings) > 0:
                half_spacings[0] = spacings[0] / 2.0
                half_spacings[-1] = spacings[-1] / 2.0
                half_spacings[1:-1] = np.minimum(spacings[:-1],
                                                 spacings[1:]) / 2.0
            else:
                half_spacings[0] = 3.0  # fallback

            for j, center_pix in enumerate(fiber_centers):
                trace_spat = np.full(nspec, float(center_pix))
                box_rad = half_spacings[j]

                # Boxcar extraction
                wave, flux, flux_ivar, _, _, box_gpm, _, _, _, _, _ = \
                    core_extract.extract_boxcar(
                        box_rad, trace_spat,
                        rawflat,  # extract from raw flat (no sky)
                        ivar, inmask, waveimg,
                        np.zeros_like(rawflat))  # skyimg=0

                all_spectra.append(flux)
                all_waves.append(wave)
                all_ivars.append(flux_ivar)

                fid = int(fiber_meta['fiber_id'][j]) if fiber_meta is not None else -1
                ftype = fiber_meta['fiber_type'][j] if fiber_meta is not None else 'science'
                all_fiber_ids.append(fid)
                all_fiber_types.append(ftype)

        nfiber = len(all_spectra)
        if nfiber == 0:
            msgs.warn("No fibers extracted from flat field")
            return flatImages, None

        fiber_spectra = np.array(all_spectra)
        fiber_waves = np.array(all_waves)
        fiber_ivar = np.array(all_ivars)
        fiber_ids = np.array(all_fiber_ids)
        fiber_types = np.array(all_fiber_types)

        msgs.info(f"Extracted {nfiber} fibers from flat field "
                  f"({np.sum(fiber_types == 'sky')} sky, "
                  f"{np.sum(fiber_types == 'science')} science)")

        # Build superflat and fiberflat
        superflat, superflat_wave, fiberflat, scale_factors = \
            self.build_superflat_fiberflat(
                fiber_spectra, fiber_waves, fiber_ivar)

        # Compute throughput ratio
        throughput_ratio = self.compute_throughput_ratio(
            scale_factors, fiber_types)
        msgs.info(f"Sky/science throughput ratio: {throughput_ratio:.4f}")

        fiber_flatimages = FiberFlatImages(
            superflat=superflat,
            superflat_wave=superflat_wave,
            fiberflat=fiberflat,
            fiber_scale_factors=scale_factors,
            throughput_ratio=throughput_ratio,
            fiber_ids=fiber_ids,
            fiber_types=fiber_types,
            PYP_SPEC=self.spectrograph.name,
        )

        return flatImages, fiber_flatimages
```

- [ ] **Step 2: Verify the tests still pass**

Run: `pytest pypeit/tests/test_fiberflatfield.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add pypeit/flatfield.py
git commit -m "feat: implement FiberFlatField.run() with fiber extraction and pixel-only flat"
```

---

### Task 4: Calibrations Dispatch

Wire `FiberFlatField` into the calibrations system so it is used for the Fiber pypeline.

**Files:**
- Modify: `pypeit/calibrations.py` (override `get_flats()` in `IFUCalibrations`)

- [ ] **Step 1: Read the current get_flats() method**

Read `pypeit/calibrations.py` lines 769-1024 to understand the full method.

- [ ] **Step 2: Override get_flats() in IFUCalibrations**

Add a `get_flats()` method to `IFUCalibrations` (class starts around line 1773). This method checks for the Fiber pypeline and dispatches to `FiberFlatField`, falling through to the parent for other IFU pypelines:

```python
    def get_flats(self, force=None):
        """
        Override flat generation for Fiber pypeline.

        For the Fiber pypeline, uses :class:`~pypeit.flatfield.FiberFlatField`
        to build a pixel-only 2D flat plus 1D superflat/fiberflat products.
        For SlicerIFU pypeline, falls through to the parent implementation.
        """
        if self.spectrograph.pypeline != 'Fiber':
            return super().get_flats(force=force)

        # --- Fiber pypeline flat field ---
        from pypeit.flatfield import FiberFlatField, FiberFlatImages

        # Check prerequisites
        if self.wavetilts is None or self.wv_calib is None:
            msgs.warn("Wavelength calibration required for fiber flat field")
        if self.slits is None:
            msgs.error("Slits required for flat field processing")

        # Check for existing calibrations
        calib_key = self.fitstbl.find_calib_key(self.calib_ID)

        # Build raw pixel flat image
        pixel_flat_frames = self.fitstbl.find_frame_files(
            'pixelflat', calib_ID=self.calib_ID)
        if len(pixel_flat_frames) == 0:
            msgs.warn("No pixel flat frames found")
            self.flatimages = None
            return

        raw_pixel_flat = buildimage.buildimage_fromlist(
            self.spectrograph, self.fitstbl.find_calib_key(self.calib_ID),
            pixel_flat_frames, det=self.det, par=self.par['scienceframe'],
            bpm=self.msbpm, bias=self.msbias, dark=self.msdark,
            calib_dir=self.calibrations_path, setup=self.setup)

        # Instantiate FiberFlatField
        flatField = FiberFlatField(
            raw_pixel_flat, self.spectrograph, self.par['flatfield'],
            self.slits, wavetilts=self.wavetilts, wv_calib=self.wv_calib,
            qa_path=self.qa_path, calib_key=calib_key)

        # Run
        flatImages, fiber_flatimages = flatField.run(doqa=self.write_qa)

        # Save standard flat images
        flatImages.set_paths(self.calibrations_path, 'A', calib_key,
                             self.spectrograph.get_det_str(self.det))
        flatImages.to_file(overwrite=True)
        self.flatimages = flatImages

        # Save fiber flat images
        if fiber_flatimages is not None:
            fiber_flatimages.set_paths(self.calibrations_path, calib_key,
                                       self.spectrograph.get_det_str(self.det))
            fiber_flatimages.to_file(fiber_flatimages.get_path(), overwrite=True)
            self.fiber_flatimages = fiber_flatimages
        else:
            self.fiber_flatimages = None
```

Note: The exact frame-finding calls may need adjustment based on reading the current `get_flats()` implementation. The pattern above follows the existing code structure. Read the current method carefully before implementing.

- [ ] **Step 3: Add fiber_flatimages attribute to IFUCalibrations.__init__()**

Locate `IFUCalibrations.__init__()` (or the base `Calibrations.__init__()`) and add `self.fiber_flatimages = None` to the initialization.

- [ ] **Step 4: Verify import and basic syntax**

Run: `python -c "from pypeit.calibrations import IFUCalibrations; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add pypeit/calibrations.py
git commit -m "feat: dispatch FiberFlatField from IFUCalibrations.get_flats()"
```

---

### Task 5: Skip IllumFlat/SpecIllum for Fiber Pypeline

Ensure that `apply_flat_fielding()` only applies the pixel-only flat for the Fiber pypeline, skipping illumination flat and spectral illumination correction.

**Files:**
- Modify: `pypeit/images/rawimage.py` (around line 811)

- [ ] **Step 1: Read the current flatfield() method**

Read `pypeit/images/rawimage.py` lines 811-889 to understand the current checks.

- [ ] **Step 2: Add pypeline check to skip illumflat and specillum**

In the `flatfield()` method, after the existing checks (around line 865), add a check for the Fiber pypeline. The method receives `flatimages` but does not currently have access to the pypeline. The simplest approach is to check whether `illumflat_spat_bsplines` is None (which it will be for `FiberFlatField`-produced `FlatImages`):

The current code already handles this case — if `flatimages.illumflat_spat_bsplines` is None, `fit2illumflat()` returns an array of ones (line 869-873). Similarly, if `pixelflat_spec_illum` is None, the spectral illumination is skipped (line 878-880).

Since `FiberFlatField.run()` does not populate these fields, the existing code should already skip them. **Verify this by reading the code.**

If the existing code handles it correctly, this step just needs verification, not a code change.

- [ ] **Step 3: Verify with a quick check**

Run: `python -c "
from pypeit.flatfield import FlatImages
import numpy as np
fi = FlatImages(pixelflat_norm=np.ones((10,10)), PYP_SPEC='test')
print('illumflat_spat_bsplines:', fi.illumflat_spat_bsplines)
print('pixelflat_spec_illum:', fi.pixelflat_spec_illum)
"`
Expected: Both should print `None`

- [ ] **Step 4: Commit (only if code changes were needed)**

```bash
git add pypeit/images/rawimage.py
git commit -m "fix: ensure Fiber pypeline skips illumflat and specillum in flat fielding"
```

---

### Task 6: FiberFindObjects 1D Sky Subtraction

Rework `FiberFindObjects.run()` to implement 1D sky subtraction: extract all fibers, equalize with superflat/fiberflat, build B-spline sky model from sky fibers, subtract in 1D.

**Files:**
- Modify: `pypeit/find_objects.py` (FiberFindObjects class, lines 1330-1751)

- [ ] **Step 1: Read the current FiberFindObjects code**

Read `pypeit/find_objects.py` lines 1330-1751 to understand all current methods.

- [ ] **Step 2: Rewrite FiberFindObjects.run()**

Replace the current `run()` method (lines 1348-1426) with the new 1D sky subtraction flow. The helper methods `_extract_sky_fibers`, `_build_sky_model`, and `_project_sky_to_2d` are replaced by inline logic:

```python
    def run(self, std_trace=None, show_peaks=False, show_skysub_fit=False):
        """
        Primary code flow for fiber object finding with 1D sky subtraction.

        Extracts all fibers from the un-subtracted 2D image, equalizes using
        superflat/fiberflat calibration products, builds a B-spline sky model
        from equalized sky fiber spectra, and subtracts sky in 1D.

        Parameters
        ----------
        std_trace : `astropy.table.Table`_, optional
            Ignored for fiber reductions.
        show_peaks : :obj:`bool`, optional
            Ignored for fiber reductions.
        show_skysub_fit : :obj:`bool`, optional
            Ignored for fiber reductions.

        Returns
        -------
        initial_sky : `numpy.ndarray`_
            2D sky model image (for diagnostics; sky subtraction is done in 1D).
        sobjs_obj : :class:`~pypeit.specobjs.SpecObjs`
            List of SpecObjs with sky-subtracted BOX_COUNTS.
        """
        from pypeit.core import fitting, extract as core_extract
        from pypeit.flatfield import FiberFlatImages

        # Build the wavelength image
        if self.waveimg is None:
            if self.wv_calib is None:
                raise PypeItError("Wavelength calibration required for "
                                  "fiber sky subtraction.")
            log.info("Generating wavelength image for fiber sky subtraction")
            self.waveimg = self.wv_calib.build_waveimg(
                self.tilts, self.slits, spat_flexure=self.spat_flexure_shift)

        # Load fiber flat calibration products
        fiber_flatimages = self._load_fiber_flatimages()

        # Create SpecObjs for all fibers (sky + science)
        self.reduce_bpm = self.reduce_bpm_init.copy()
        sobjs_obj, self.nobj = self.find_objects(
            self.sciImg.image, self.sciImg.ivar,
            std_trace=std_trace, show=self.findobj_show,
            show_peaks=show_peaks)

        if len(sobjs_obj) == 0:
            log.warning("No fiber objects found")
            return np.zeros_like(self.sciImg.image), sobjs_obj

        # Extract all fibers via boxcar from the un-subtracted image
        nspec = self.sciImg.image.shape[0]
        gpm = self.sciImg.select_flag(invert=True)
        slitmask = self.slits.slit_img(initial=True)

        for sobj in sobjs_obj:
            slit_spat_id = sobj.SLITID
            thismask = slitmask == slit_spat_id
            inmask = gpm & thismask

            sobj.extract_boxcar(
                self.sciImg.image, self.sciImg.ivar, inmask,
                self.waveimg, np.zeros_like(self.sciImg.image),
                base_var=self.sciImg.base_var,
                count_scale=self.sciImg.img_scale,
                noise_floor=self.sciImg.noise_floor)

        # Equalize all fibers and build sky model
        initial_sky = self._fiber_skysub(sobjs_obj, fiber_flatimages)

        return initial_sky, sobjs_obj

    def _load_fiber_flatimages(self):
        """
        Load FiberFlatImages calibration products from disk.

        Returns
        -------
        fiber_flatimages : :class:`~pypeit.flatfield.FiberFlatImages` or None
        """
        from pypeit.flatfield import FiberFlatImages

        # Try to find the fiber flat file in the calibrations directory
        if hasattr(self, 'caliBrate') and self.caliBrate is not None:
            if hasattr(self.caliBrate, 'fiber_flatimages'):
                return self.caliBrate.fiber_flatimages

        # Try loading from calibrations path
        if hasattr(self, 'calibrations_path') and self.calibrations_path is not None:
            import glob
            pattern = str(Path(self.calibrations_path) / 'FiberFlat_*.fits')
            files = glob.glob(pattern)
            if len(files) > 0:
                return FiberFlatImages.from_file(files[0])

        log.warning("No FiberFlatImages found; sky subtraction will proceed "
                    "without fiber throughput equalization")
        return None

    def _fiber_skysub(self, sobjs, fiber_flatimages):
        """
        Perform 1D sky subtraction on extracted fiber spectra.

        Equalizes all fibers using superflat/fiberflat, builds a super-sampled
        B-spline sky model from sky fibers, and subtracts from all fibers.

        Parameters
        ----------
        sobjs : :class:`~pypeit.specobjs.SpecObjs`
            All extracted fiber SpecObjs with BOX_COUNTS populated.
        fiber_flatimages : :class:`~pypeit.flatfield.FiberFlatImages` or None
            Fiber flat calibration products.

        Returns
        -------
        sky_2d : `numpy.ndarray`_
            Diagnostic 2D sky image (reconstructed from per-fiber sky models).
        """
        from pypeit.core import fitting
        from pypeit.bspline import bspline

        nspec, nspat = self.sciImg.image.shape

        # Match each SpecObj to its fiber flat entry
        corrections = self._compute_equalization(sobjs, fiber_flatimages)

        # Equalize all fiber spectra
        for i, sobj in enumerate(sobjs):
            if sobj.BOX_COUNTS is None:
                continue
            corr = corrections[i]  # 1D array, per wavelength pixel
            if corr is not None:
                sobj.BOX_COUNTS = sobj.BOX_COUNTS / corr
                sobj.BOX_COUNTS_IVAR = sobj.BOX_COUNTS_IVAR * corr ** 2
                sobj.BOX_COUNTS_SKY = sobj.BOX_COUNTS_SKY / corr \
                    if sobj.BOX_COUNTS_SKY is not None else None

        # Identify sky fibers
        sky_indices = []
        for i, sobj in enumerate(sobjs):
            name = sobj.MASKDEF_OBJNAME
            if name is not None and str(name).upper().startswith('SKY'):
                sky_indices.append(i)

        log.info(f"Building sky model from {len(sky_indices)} sky fibers "
                 f"out of {len(sobjs)} total")

        if len(sky_indices) == 0:
            log.warning("No sky fibers found; skipping sky subtraction")
            return np.zeros((nspec, nspat))

        # Combine equalized sky fiber spectra into super-sampled array
        all_wave = []
        all_flux = []
        all_ivar = []
        for idx in sky_indices:
            sobj = sobjs[idx]
            if sobj.BOX_COUNTS is None or sobj.BOX_WAVE is None:
                continue
            good = (sobj.BOX_MASK if sobj.BOX_MASK is not None
                    else np.ones(len(sobj.BOX_WAVE), dtype=bool))
            good &= (sobj.BOX_WAVE > 0) & np.isfinite(sobj.BOX_COUNTS)
            all_wave.append(sobj.BOX_WAVE[good])
            all_flux.append(sobj.BOX_COUNTS[good])
            all_ivar.append(sobj.BOX_COUNTS_IVAR[good])

        all_wave = np.concatenate(all_wave)
        all_flux = np.concatenate(all_flux)
        all_ivar = np.concatenate(all_ivar)

        # Sort by wavelength
        srt = np.argsort(all_wave)
        all_wave = all_wave[srt]
        all_flux = all_flux[srt]
        all_ivar = all_ivar[srt]

        # Smooth inverse-variance weights (MaNGA technique to avoid Poisson bias)
        from scipy.ndimage import uniform_filter1d
        smooth_ivar = uniform_filter1d(all_ivar, size=100)
        # Use tighter smoothing near bright lines (where ivar varies rapidly)
        ivar_gradient = np.abs(np.gradient(all_ivar))
        line_mask = ivar_gradient > 3.0 * np.median(ivar_gradient)
        if np.any(line_mask):
            smooth_ivar_narrow = uniform_filter1d(all_ivar, size=3)
            smooth_ivar[line_mask] = smooth_ivar_narrow[line_mask]

        # B-spline fit
        bsp = self.par['reduce']['skysub']['bspline_spacing']
        log.info(f"Fitting B-spline sky model with bkspace={bsp} A "
                 f"to {len(all_wave)} pixels")

        try:
            sset, outmask, yfit, _, exit_status = fitting.bspline_profile(
                all_wave, all_flux, smooth_ivar,
                np.ones((len(all_wave), 1)),
                nord=4, upper=3.0, lower=3.0,
                kwargs_bspline={'bkspace': bsp})
        except Exception as e:
            log.warning(f"B-spline sky model fit failed: {e}")
            return np.zeros((nspec, nspat))

        if exit_status > 1:
            log.warning(f"Sky model B-spline fit did not converge "
                        f"(exit_status={exit_status})")

        n_rejected = np.sum(~outmask)
        log.info(f"Sky model: {n_rejected}/{len(outmask)} pixels rejected "
                 f"({100.0 * n_rejected / max(len(outmask), 1):.1f}%)")

        # Subtract sky from all fibers (both sky and science)
        sky_2d = np.zeros((nspec, nspat))
        for i, sobj in enumerate(sobjs):
            if sobj.BOX_WAVE is None or sobj.BOX_COUNTS is None:
                continue
            good = (sobj.BOX_WAVE > 0)
            sky_spec = np.zeros_like(sobj.BOX_COUNTS)
            sky_spec[good] = sset.value(sobj.BOX_WAVE[good])[0].flatten()

            # Subtract sky from equalized spectrum
            sobj.BOX_COUNTS_SKY = sky_spec.copy()
            sobj.BOX_COUNTS = sobj.BOX_COUNTS - sky_spec

            # Add sky model variance (small contribution)
            # For now, just use the existing ivar unchanged since
            # sky model variance is typically negligible

            # Reconstruct 2D sky for diagnostics: scale sky back to
            # detector counts using the fiber's correction
            corr = corrections[i]
            if corr is not None:
                sky_detector = sky_spec * corr
            else:
                sky_detector = sky_spec
            # Write into 2D image at fiber position
            trace = sobj.TRACE_SPAT
            box_r = sobj.BOX_R_PIX
            for row in range(nspec):
                col_lo = max(0, int(trace[row] - box_r))
                col_hi = min(nspat, int(trace[row] + box_r) + 1)
                if col_hi > col_lo and row < nspec:
                    sky_2d[row, col_lo:col_hi] = sky_detector[row]

        return sky_2d

    def _compute_equalization(self, sobjs, fiber_flatimages):
        """
        Compute per-fiber equalization corrections from FiberFlatImages.

        For each SpecObj, looks up its fiber ID in the FiberFlatImages
        and computes ``correction = scale_factor * superflat * fiberflat``.

        Parameters
        ----------
        sobjs : :class:`~pypeit.specobjs.SpecObjs`
            Extracted SpecObjs.
        fiber_flatimages : :class:`~pypeit.flatfield.FiberFlatImages` or None

        Returns
        -------
        corrections : :obj:`list`
            List of 1D correction arrays (or None if no match), one per sobj.
        """
        from pypeit.bspline import bspline as bspline_cls

        corrections = [None] * len(sobjs)
        if fiber_flatimages is None:
            return corrections

        sf_wave = fiber_flatimages.superflat_wave
        sf_vals = fiber_flatimages.superflat
        fiberflat = fiber_flatimages.fiberflat
        scale_factors = fiber_flatimages.fiber_scale_factors
        fiber_ids = fiber_flatimages.fiber_ids

        for i, sobj in enumerate(sobjs):
            fid = sobj.MASKDEF_ID
            if fid is None or fid < 0:
                continue
            # Find this fiber in the flat products
            idx = np.where(fiber_ids == fid)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]

            wave = sobj.BOX_WAVE
            if wave is None:
                continue

            # Interpolate superflat to this fiber's wavelength grid
            sf_interp = np.interp(wave, sf_wave, sf_vals, left=1.0, right=1.0)

            # Get fiberflat for this fiber (already on extraction wavelength grid)
            ff = fiberflat[idx]
            # Interpolate fiberflat if wavelength grids differ
            if len(ff) != len(wave):
                # Use the stored wavelength grid
                ff_wave = sf_wave  # fiberflat uses same grid as superflat
                ff = np.interp(wave, ff_wave, ff, left=1.0, right=1.0)

            # Total correction
            corr = scale_factors[idx] * sf_interp * ff
            corr[corr <= 0] = 1.0  # avoid division by zero
            corrections[i] = corr

        n_matched = sum(1 for c in corrections if c is not None)
        log.info(f"Equalization: matched {n_matched}/{len(sobjs)} fibers "
                 f"to flat field products")

        return corrections
```

- [ ] **Step 3: Remove obsolete helper methods**

Delete the following methods from `FiberFindObjects` as they are no longer used:
- `_extract_sky_fibers()` (lines 1428-1512)
- `_build_sky_model()` (lines 1514-1604)
- `_project_sky_to_2d()` (lines 1606-1631)

- [ ] **Step 4: Add necessary imports at top of find_objects.py**

Add these imports if not already present:

```python
from pathlib import Path
from scipy.ndimage import uniform_filter1d
```

- [ ] **Step 5: Verify syntax**

Run: `python -c "from pypeit.find_objects import FiberFindObjects; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add pypeit/find_objects.py
git commit -m "feat: rework FiberFindObjects for 1D sky subtraction with superflat equalization"
```

---

### Task 7: FiberExtract Updates

Update `FiberExtract.local_skysub_extract()` to work with the new 1D sky subtraction — it no longer needs to do its own sky subtraction but should perform optimal extraction with the sky model informing the noise weighting.

**Files:**
- Modify: `pypeit/extraction.py` (FiberExtract class, lines 957-1085)

- [ ] **Step 1: Read the current FiberExtract code**

Read `pypeit/extraction.py` lines 957-1085.

- [ ] **Step 2: Update local_skysub_extract()**

The key change: the `global_sky` parameter now represents the 2D diagnostic sky image from `FiberFindObjects`. The boxcar extraction results are already populated with sky-subtracted counts from the 1D subtraction. Optimal extraction should use the sky model for noise weighting.

Modify the method to:
1. Use the existing boxcar results from `FiberFindObjects` (already sky-subtracted and equalized)
2. For optimal extraction, build a per-fiber 2D sky model from `BOX_COUNTS_SKY` and pass it to the extraction

```python
    def local_skysub_extract(self, global_sky, sobjs, bkg_redux_global_sky=None,
                             spat_pix=None, model_noise=True,
                             show_resids=False, show_profile=False, show=False):
        """
        Extract fiber spectra from block-slits.

        Boxcar extraction results are already populated by FiberFindObjects
        (with 1D sky subtraction applied). This method performs optimal
        extraction using flat-derived empirical profiles, with the sky model
        informing the noise weighting.

        Parameters
        ----------
        global_sky : `numpy.ndarray`_
            2D sky model image from FiberFindObjects (diagnostic).
        sobjs : :class:`~pypeit.specobjs.SpecObjs`
            SpecObjs with BOX_COUNTS already sky-subtracted.
        bkg_redux_global_sky : `numpy.ndarray`_, optional
            Not used for fiber extraction.
        spat_pix : `numpy.ndarray`_, optional
            Not used for fiber extraction.
        model_noise : :obj:`bool`, optional
            If True, use model-based noise estimate.
        show_resids : :obj:`bool`, optional
            Show residuals.
        show_profile : :obj:`bool`, optional
            Show extraction profile.
        show : :obj:`bool`, optional
            Show results.

        Returns
        -------
        skymodel : `numpy.ndarray`_
        bkg_redux_skymodel : `numpy.ndarray`_ or None
        objmodel : `numpy.ndarray`_
        ivarmodel : `numpy.ndarray`_
        outmask : `numpy.ndarray`_
        sobjs : :class:`~pypeit.specobjs.SpecObjs`
        """
        # Initialize output images
        skymodel = global_sky.copy()
        bkg_redux_skymodel = None
        objmodel = np.zeros_like(self.sciImg.image)
        ivarmodel = self.sciImg.ivar.copy()
        outmask = self.sciImg.select_flag(invert=True)

        # Image with sky subtracted for optimal extraction
        imgminsky = self.sciImg.image - global_sky

        slitmask = self.slits.slit_img(initial=True)
        gpm = self.sciImg.select_flag(invert=True)

        # Build empirical profiles from flat field (if available)
        flatimg = None
        if self.flatimg is not None:
            flatimg = self.flatimg

        for sobj in sobjs:
            if sobj.BOX_COUNTS is None:
                continue

            slit_spat_id = sobj.SLITID
            thismask = slitmask == slit_spat_id
            inmask = gpm & thismask

            # Optimal extraction using the sky-subtracted image
            # and the global sky for noise weighting
            try:
                sobj.extract_optimal(
                    imgminsky, self.sciImg.ivar, inmask,
                    self.waveimg, global_sky, thismask,
                    flatimg=flatimg,
                    base_var=self.sciImg.base_var,
                    count_scale=self.sciImg.img_scale,
                    noise_floor=self.sciImg.noise_floor)
            except Exception as e:
                log.warning(f"Optimal extraction failed for fiber "
                            f"{sobj.MASKDEF_OBJNAME}: {e}")

        # Apply post-extraction throughput corrections if needed
        # (In the new design, equalization is done pre-sky-subtraction,
        # so no post-extraction correction is needed)

        return skymodel, bkg_redux_skymodel, objmodel, ivarmodel, outmask, sobjs
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "from pypeit.extraction import FiberExtract; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pypeit/extraction.py
git commit -m "feat: update FiberExtract for 1D sky subtraction with optimal extraction"
```

---

### Task 8: Clean Up Obsolete Code

Remove methods and code paths that are superseded by the new approach.

**Files:**
- Modify: `pypeit/spectrographs/mmt_binospec.py`
- Modify: `pypeit/find_objects.py` (if not already cleaned in Task 6)

- [ ] **Step 1: Discard unstaged changes in mmt_binospec.py**

The unstaged changes in `mmt_binospec.py` partially revert to the old per-fiber approach. Discard them to start from the committed block-slit HEAD:

```bash
git checkout -- pypeit/spectrographs/mmt_binospec.py
```

- [ ] **Step 2: Discard unstaged changes in find_objects.py**

Similarly, discard any unstaged changes that conflict with the Task 6 implementation:

```bash
git checkout -- pypeit/find_objects.py
```

Note: Do this BEFORE applying Task 6 changes, not after.

- [ ] **Step 3: Mark apply_throughput_corrections as no-op for block-slit mode**

In `pypeit/spectrographs/mmt_binospec.py`, update `apply_throughput_corrections()` (line 1847) to log that throughput correction is now handled by the superflat/fiberflat and return without modifying sobjs:

```python
    def apply_throughput_corrections(self, sobjs, det):
        """Throughput corrections handled by FiberFlatField superflat/fiberflat.

        This method is a no-op when using the superflat-based calibration.
        Fiber throughput equalization is applied to extracted spectra before
        sky subtraction in FiberFindObjects.
        """
        log.info("Throughput corrections handled by superflat/fiberflat; "
                 "skipping post-extraction correction")
        return
```

- [ ] **Step 4: Commit**

```bash
git add pypeit/spectrographs/mmt_binospec.py pypeit/find_objects.py
git commit -m "fix: clean up obsolete sky subtraction and throughput correction code"
```

---

### Task 9: Integration Test

Run the full pipeline on real Binospec IFU data to verify the implementation works end-to-end.

**Files:**
- No code changes — this is a test run

- [ ] **Step 1: Ensure clean state**

```bash
cd /Users/tim/MMT/pypeit
git status
git diff --stat
```

Verify no unstaged changes remain.

- [ ] **Step 2: Clear old calibration outputs**

```bash
cd /Users/tim/MMT/bino_ifu/JADES_1031022
# Remove old calibrations to force regeneration
rm -rf Calibrations/
rm -rf Science/
```

- [ ] **Step 3: Run the pipeline**

```bash
cd /Users/tim/MMT/bino_ifu/JADES_1031022
run_pypeit jades_1031022.pypeit 2>&1 | tee run_superflat.log
```

Monitor the log for:
- `FiberFlatField` messages (superflat construction, throughput ratio)
- `FiberFindObjects` messages (fiber extraction, sky model fitting)
- Any crashes or warnings

- [ ] **Step 4: Check for success**

```bash
# Check if pipeline completed
tail -50 run_superflat.log

# Check for calibration products
ls -la Calibrations/FiberFlat_*.fits
ls -la Calibrations/Flat_*.fits

# Check for science outputs
ls -la Science/spec1d_*.fits
```

- [ ] **Step 5: Document results**

Note the runtime, any warnings, and whether the pipeline completed. Compare sky subtraction quality with previous runs if possible.

---

## Task Ordering

Tasks 1-3 build the flat field infrastructure. Task 4 wires it into calibrations. Task 5 verifies the flat application. Task 6 implements sky subtraction. Task 7 updates extraction. **Task 8 should be done FIRST** (discard unstaged changes) to ensure a clean starting point, then Tasks 1-7 in order, then Task 9 for integration testing.

**Recommended execution order:** 8 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9
