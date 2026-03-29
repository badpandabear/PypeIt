"""Tests for FiberFlatField and FiberFlatImages."""
import numpy as np
import pytest
from pathlib import Path

from pypeit.flatfield import FiberFlatField, FiberFlatImages
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


def test_superflat_construction():
    """Test superflat from synthetic fiber spectra with known properties.

    Creates 8 science fibers and 2 sky fibers.  Science fibers have similar
    throughput (~1.0); sky fibers have 3x throughput.  The superflat is built
    from science fibers only, so science fiberflats should be near 1.0 and
    sky fiberflats should be near 3.0.
    """
    np.random.seed(42)
    n_sci = 8
    n_sky = 2
    nfibers = n_sci + n_sky
    nwave = 500
    wave_grid = np.linspace(4000.0, 7000.0, nwave)

    # Common spectral shape: quadratic (simulating lamp spectrum)
    true_shape = 1.0 - 0.3 * ((wave_grid - 5500.0) / 1500.0) ** 2

    # Science fibers: throughput ~1.0 with small variations
    # Sky fibers: throughput ~3.0
    true_scales = np.ones(nfibers)
    true_scales[:n_sci] = np.random.uniform(0.9, 1.1, n_sci)
    true_scales[n_sci:] = 3.0  # sky fibers
    fiber_types = np.array(['science'] * n_sci + ['sky'] * n_sky)

    # Generate fiber spectra
    fiber_spectra = np.zeros((nfibers, nwave))
    fiber_ivar = np.zeros((nfibers, nwave))
    for i in range(nfibers):
        flux = true_scales[i] * true_shape * 10000.0
        noise = np.sqrt(np.abs(flux)) + 1.0
        fiber_spectra[i] = flux + np.random.normal(0, 1, nwave) * noise
        fiber_ivar[i] = 1.0 / (noise ** 2)

    # Each fiber has slightly offset wavelengths
    fiber_waves = np.zeros((nfibers, nwave))
    for i in range(nfibers):
        offset = np.random.uniform(-2.0, 2.0)
        fiber_waves[i] = wave_grid + offset

    superflat, superflat_wave, fiberflat = \
        FiberFlatField.build_superflat_fiberflat(
            fiber_spectra, fiber_waves, fiber_ivar, fiber_types)

    # Science fiber fiberflats should be near 1.0
    for i in range(n_sci):
        med = np.median(fiberflat[i])
        assert 0.8 < med < 1.2, \
            f"Science fiber {i} fiberflat median={med:.3f}, expected ~1.0"

    # Sky fiber fiberflats should be near 3.0 (the throughput ratio)
    for i in range(n_sci, nfibers):
        med = np.median(fiberflat[i])
        assert 2.5 < med < 3.5, \
            f"Sky fiber {i} fiberflat median={med:.3f}, expected ~3.0"

    # Superflat should show the quadratic shape
    mid = nwave // 2
    quarter = nwave // 4
    assert superflat[mid] > superflat[quarter], \
        "Superflat should be brighter at center than at edges"


def test_throughput_ratio():
    """Test gray throughput ratio computation."""
    scale_factors = np.array([0.6, 0.7, 0.65, 0.55,
                              1.0, 1.1, 0.95, 1.05, 0.98, 1.02])
    fiber_types = np.array(['sky'] * 4 + ['science'] * 6)

    ratio = FiberFlatField.compute_throughput_ratio(scale_factors, fiber_types)

    expected = np.median([0.6, 0.7, 0.65, 0.55]) / np.median([1.0, 1.1, 0.95, 1.05, 0.98, 1.02])
    assert np.isclose(ratio, expected, rtol=1e-10)
