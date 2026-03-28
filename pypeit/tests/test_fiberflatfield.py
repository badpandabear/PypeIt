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
    """Test superflat from synthetic fiber spectra with known properties."""
    np.random.seed(42)
    nfibers = 10
    nwave = 500
    wave_grid = np.linspace(4000.0, 7000.0, nwave)

    # Common spectral shape: quadratic (simulating lamp spectrum)
    true_shape = 1.0 - 0.3 * ((wave_grid - 5500.0) / 1500.0) ** 2
    true_shape /= np.median(true_shape)

    # Per-fiber scale factors (throughputs)
    true_scales = np.linspace(0.5, 1.5, nfibers)

    # Generate fiber spectra: shape * scale + noise
    fiber_spectra = np.zeros((nfibers, nwave))
    fiber_ivar = np.zeros((nfibers, nwave))
    for i in range(nfibers):
        flux = true_scales[i] * true_shape * 10000.0
        noise = np.sqrt(flux)
        fiber_spectra[i] = flux + np.random.normal(0, 1, nwave) * noise
        fiber_ivar[i] = 1.0 / (noise ** 2)

    # Each fiber has slightly offset wavelengths
    fiber_waves = np.zeros((nfibers, nwave))
    for i in range(nfibers):
        offset = np.random.uniform(-2.0, 2.0)
        fiber_waves[i] = wave_grid + offset

    superflat, superflat_wave, fiberflat, scale_factors = \
        FiberFlatField.build_superflat_fiberflat(
            fiber_spectra, fiber_waves, fiber_ivar)

    # Scale factors should correlate with true scales
    rank_true = np.argsort(true_scales)
    rank_measured = np.argsort(scale_factors)
    assert np.array_equal(rank_true, rank_measured), \
        f"Scale factor ranking mismatch: {scale_factors}"

    # Fiberflats should be near unity
    for i in range(nfibers):
        assert np.all(np.abs(fiberflat[i] - 1.0) < 0.1), \
            f"Fiberflat {i} deviates too far from unity: " \
            f"range [{fiberflat[i].min():.3f}, {fiberflat[i].max():.3f}]"

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
