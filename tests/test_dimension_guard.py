"""Tests for observable dimension guard (PR2 §0.1)."""

import logging

import numpy as np
import pytest

from uq.observables import _align_variant_dimensions


class TestAlignVariantDimensions:
    """Test _align_variant_dimensions handles mismatched widths."""

    def test_uniform_dimensions_passthrough(self):
        """When all variants have the same width, no truncation occurs."""
        Y_list = [np.random.randn(5) for _ in range(3)]
        Y_ts = [np.random.randn(10, 5) for _ in range(3)]
        names = [f"obs_{i}" for i in range(5)]

        Y_agg, out_names = _align_variant_dimensions(Y_list, Y_ts, names)

        assert Y_agg.shape == (3, 5)
        assert len(out_names) == 5
        # Timeseries unchanged
        assert all(ts.shape[1] == 5 for ts in Y_ts)

    def test_mismatched_strict_raises(self):
        """In strict mode (default), dimension mismatch raises ValueError."""
        Y_list = [np.random.randn(5), np.random.randn(4)]
        Y_ts = [np.random.randn(10, 5), np.random.randn(10, 4)]
        names = [f"obs_{i}" for i in range(5)]

        with pytest.raises(ValueError, match="dimension mismatch"):
            _align_variant_dimensions(Y_list, Y_ts, names, strict=True)

    def test_mismatched_nonstrict_truncated(self, caplog):
        """In non-strict mode, truncate to minimum and warn."""
        Y_list = [
            np.random.randn(5),
            np.random.randn(4),  # shorter
            np.random.randn(5),
        ]
        Y_ts = [
            np.random.randn(10, 5),
            np.random.randn(10, 4),
            np.random.randn(10, 5),
        ]
        names = [f"obs_{i}" for i in range(5)]

        with caplog.at_level(logging.WARNING, logger="uq.observables"):
            Y_agg, out_names = _align_variant_dimensions(Y_list, Y_ts, names, strict=False)

        assert Y_agg.shape == (3, 4)
        assert len(out_names) == 4
        assert all(ts.shape[1] == 4 for ts in Y_ts)
        assert "dimension mismatch" in caplog.text.lower()

    def test_single_variant(self):
        """Single variant — no comparison needed."""
        Y_list = [np.random.randn(7)]
        Y_ts = [np.random.randn(20, 7)]
        names = [f"obs_{i}" for i in range(7)]

        Y_agg, out_names = _align_variant_dimensions(Y_list, Y_ts, names)

        assert Y_agg.shape == (1, 7)
        assert len(out_names) == 7

    def test_severely_mismatched_nonstrict(self, caplog):
        """Large mismatch (e.g., 4300 vs 4280 genes) truncates correctly in non-strict."""
        Y_list = [
            np.random.randn(4300),
            np.random.randn(4280),
            np.random.randn(4300),
        ]
        Y_ts = [
            np.random.randn(50, 4300),
            np.random.randn(50, 4280),
            np.random.randn(50, 4300),
        ]
        names = [f"gene_{i}" for i in range(4300)]

        with caplog.at_level(logging.WARNING, logger="uq.observables"):
            Y_agg, out_names = _align_variant_dimensions(Y_list, Y_ts, names, strict=False)

        assert Y_agg.shape == (3, 4280)
        assert len(out_names) == 4280
        assert "4280" in caplog.text
        assert "4300" in caplog.text

    def test_values_preserved_nonstrict(self):
        """Verify that truncation preserves the first N columns, not random ones."""
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = np.array([10.0, 20.0, 30.0])
        Y_list = [v1, v2]
        Y_ts = [
            np.arange(15).reshape(3, 5).astype(float),
            np.arange(9).reshape(3, 3).astype(float),
        ]
        names = ["a", "b", "c", "d", "e"]

        Y_agg, out_names = _align_variant_dimensions(Y_list, Y_ts, names, strict=False)

        assert Y_agg.shape == (2, 3)
        np.testing.assert_array_equal(Y_agg[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(Y_agg[1], [10.0, 20.0, 30.0])
        assert out_names == ["a", "b", "c"]
        assert Y_ts[0].shape == (3, 3)
        assert Y_ts[1].shape == (3, 3)
