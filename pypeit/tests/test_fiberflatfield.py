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
