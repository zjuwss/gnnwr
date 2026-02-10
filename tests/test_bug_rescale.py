"""
Tests for rescale() logic fix.

The bug: when x=None (or y=None), the old if/elif pattern
    if x is not None and x_scale_info is not None: ...
    elif x_scale_info is None: raise ValueError
would incorrectly raise ValueError when x is None and x_scale_info is also None,
because the first condition is False (x is None) and the elif triggers.
The fix uses nested ifs: first check if x is not None, then check scale_info.
"""

import numpy as np
import pandas as pd
import pytest

from gnnwr.datasets import baseDataset, predictDataset


def _make_base_dataset():
    data = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0, 4.0],
        'y': [10.0, 20.0, 30.0, 40.0],
        'id': [0, 1, 2, 3]
    })
    return baseDataset(data, x_column=['x1'], y_column=['y'], id_column=['id'])


# ── minmax_scale tests ──

def test_minmax_rescale_y_only():
    """rescale(None, y) should succeed when x_scale_info is None."""
    ds = _make_base_dataset()
    ds.scale_fn = "minmax_scale"
    ds.x_scale_info = None
    ds.y_scale_info = {"min": np.array([10.0]), "max": np.array([40.0])}

    y_scaled = np.array([[0.0], [0.5], [1.0]])
    x_out, y_out = ds.rescale(None, y_scaled)
    assert x_out is None
    np.testing.assert_allclose(y_out, [[10.0], [25.0], [40.0]])


def test_minmax_rescale_x_only():
    """rescale(x, None) should succeed when y_scale_info is None."""
    ds = _make_base_dataset()
    ds.scale_fn = "minmax_scale"
    ds.x_scale_info = {"min": np.array([1.0]), "max": np.array([4.0])}
    ds.y_scale_info = None

    x_scaled = np.array([[0.0], [0.5], [1.0]])
    x_out, y_out = ds.rescale(x_scaled, None)
    assert y_out is None
    np.testing.assert_allclose(x_out, [[1.0], [2.5], [4.0]])


def test_minmax_rescale_both():
    ds = _make_base_dataset()
    ds.scale_fn = "minmax_scale"
    ds.x_scale_info = {"min": np.array([1.0]), "max": np.array([4.0])}
    ds.y_scale_info = {"min": np.array([10.0]), "max": np.array([40.0])}

    x_scaled = np.array([[0.0], [1.0]])
    y_scaled = np.array([[0.0], [1.0]])
    x_out, y_out = ds.rescale(x_scaled, y_scaled)
    np.testing.assert_allclose(x_out, [[1.0], [4.0]])
    np.testing.assert_allclose(y_out, [[10.0], [40.0]])


# ── standard_scale tests ──

def test_standard_rescale_y_only():
    """rescale(None, y) with standard_scale should succeed when x_scale_info is None."""
    ds = _make_base_dataset()
    ds.scale_fn = "standard_scale"
    ds.x_scale_info = None
    ds.y_scale_info = {"mean": np.array([25.0]), "var": np.array([100.0])}

    y_scaled = np.array([[0.0], [1.0], [-1.0]])
    x_out, y_out = ds.rescale(None, y_scaled)
    assert x_out is None
    # y = y_scaled * sqrt(var) + mean = y_scaled * 10 + 25
    np.testing.assert_allclose(y_out, [[25.0], [35.0], [15.0]])


def test_standard_rescale_x_only():
    """rescale(x, None) with standard_scale should succeed when y_scale_info is None."""
    ds = _make_base_dataset()
    ds.scale_fn = "standard_scale"
    ds.x_scale_info = {"mean": np.array([2.5]), "var": np.array([1.0])}
    ds.y_scale_info = None

    x_scaled = np.array([[0.0], [1.0]])
    x_out, y_out = ds.rescale(x_scaled, None)
    assert y_out is None
    np.testing.assert_allclose(x_out, [[2.5], [3.5]])


def test_standard_rescale_both():
    ds = _make_base_dataset()
    ds.scale_fn = "standard_scale"
    ds.x_scale_info = {"mean": np.array([2.5]), "var": np.array([4.0])}
    ds.y_scale_info = {"mean": np.array([25.0]), "var": np.array([100.0])}

    x_scaled = np.array([[0.0], [1.0]])
    y_scaled = np.array([[0.0], [1.0]])
    x_out, y_out = ds.rescale(x_scaled, y_scaled)
    np.testing.assert_allclose(x_out, [[2.5], [4.5]])
    np.testing.assert_allclose(y_out, [[25.0], [35.0]])


# ── predictDataset tests ──

def test_predict_dataset_minmax_rescale_y_only():
    data = pd.DataFrame({'x1': [1.0, 2.0, 3.0, 4.0]})
    scale_info = [
        {"min": np.array([1.0]), "max": np.array([4.0])},
        {"min": np.array([10.0]), "max": np.array([40.0])}
    ]
    ds = predictDataset(data=data, x_column=['x1'],
                        process_fn="minmax_scale", scale_info=scale_info)

    y_scaled = np.array([[0.0], [1.0]])
    x_out, y_out = ds.rescale(None, y_scaled)
    assert x_out is None
    np.testing.assert_allclose(y_out, [[10.0], [40.0]])


def test_predict_dataset_standard_rescale_y_only():
    data = pd.DataFrame({'x1': [1.0, 2.0, 3.0, 4.0]})
    scale_info = [
        {"mean": np.array([2.5]), "var": np.array([1.0])},
        {"mean": np.array([25.0]), "var": np.array([100.0])}
    ]
    ds = predictDataset(data=data, x_column=['x1'],
                        process_fn="standard_scale", scale_info=scale_info)

    y_scaled = np.array([[0.0], [1.0]])
    x_out, y_out = ds.rescale(None, y_scaled)
    assert x_out is None
    np.testing.assert_allclose(y_out, [[25.0], [35.0]])


# ── error cases ──

def test_rescale_raises_when_x_given_but_scale_info_missing():
    ds = _make_base_dataset()
    ds.scale_fn = "minmax_scale"
    ds.x_scale_info = None
    ds.y_scale_info = {"min": np.array([10.0]), "max": np.array([40.0])}

    with pytest.raises(ValueError, match="x scale info"):
        ds.rescale(np.array([[0.5]]), None)


def test_standard_rescale_raises_when_y_given_but_scale_info_missing():
    ds = _make_base_dataset()
    ds.scale_fn = "standard_scale"
    ds.x_scale_info = {"mean": np.array([2.5]), "var": np.array([1.0])}
    ds.y_scale_info = None

    with pytest.raises(ValueError, match="y scale info"):
        ds.rescale(None, np.array([[0.5]]))
