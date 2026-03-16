"""
Tests for Binospec overscan subtraction and nonlinearity correction.
"""
import numpy as np
import pytest

from pypeit.spectrographs.mmt_binospec import clean_overscan_vector


class TestCleanOverscanVector:
    """Tests for clean_overscan_vector."""

    def test_clean_no_outliers(self):
        """A smooth vector should be returned unchanged."""
        rng = np.random.default_rng(42)
        vec = 1000.0 + rng.normal(scale=1.0, size=100)
        cleaned = clean_overscan_vector(vec)
        np.testing.assert_array_equal(cleaned, vec)

    def test_clean_single_outlier(self):
        """A single large outlier should be interpolated over."""
        vec = np.full(100, 1000.0)
        vec[50] = 2000.0  # outlier: deviation = 1000 >> nsig*rdnoise = 4.0
        cleaned = clean_overscan_vector(vec)
        # Outlier should be replaced with interpolated value (~1000)
        assert abs(cleaned[50] - 1000.0) < 1.0
        # Non-outlier pixels should be unchanged
        assert cleaned[0] == 1000.0
        assert cleaned[99] == 1000.0

    def test_clean_edge_outlier(self):
        """Outliers at edges should be handled (extrapolation)."""
        vec = np.full(50, 500.0)
        vec[0] = 1500.0  # outlier at left edge
        vec[49] = 1500.0  # outlier at right edge
        cleaned = clean_overscan_vector(vec)
        assert abs(cleaned[0] - 500.0) < 1.0
        assert abs(cleaned[49] - 500.0) < 1.0

    def test_clean_constant_vector(self):
        """A constant vector with strict threshold should be unchanged."""
        vec = np.full(20, 500.0)
        # Even with nsig=0, constant vector has zero deviation
        # from its median-filtered version, so nothing is rejected
        cleaned = clean_overscan_vector(vec, nsig=0.0)
        np.testing.assert_array_equal(cleaned, vec)

    def test_clean_custom_params(self):
        """Custom window and nsig should be respected."""
        vec = np.full(20, 100.0)
        vec[10] = 200.0
        # With nsig=100, threshold = 100*4 = 400, so 200-100=100 < 400
        # Outlier should NOT be cleaned
        cleaned = clean_overscan_vector(vec, w=5, nsig=100.0)
        assert cleaned[10] == 200.0
