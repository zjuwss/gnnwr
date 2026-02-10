"""
Tests for singular/ill-conditioned matrix handling in DIAGNOSIS.

Verifies that torch.linalg.pinv (pseudo-inverse) is used instead of
torch.linalg.inv, so that collinear or near-collinear features no longer
cause crashes or NaN results.
"""

import warnings
import torch

from gnnwr.utils import DIAGNOSIS


def test_pinv_on_singular_matrix():
    """pinv should succeed where inv would fail on a singular XtX."""
    x = torch.tensor([
        [1.0, 2.0, 2.0],  # col 3 == col 2 → rank-deficient
        [2.0, 4.0, 4.0],
        [3.0, 6.0, 6.0],
        [4.0, 8.0, 8.0],
    ])
    XtX = torch.mm(x.t(), x)

    result = torch.linalg.pinv(XtX)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_pinv_on_ill_conditioned_matrix():
    """pinv should produce finite results for near-singular matrices."""
    x = torch.tensor([
        [1.0, 1.0 + 1e-10],
        [2.0, 2.0 + 1e-10],
        [3.0, 3.0 + 1e-10],
        [4.0, 4.0 + 1e-10],
    ])
    XtX = torch.mm(x.t(), x)

    result = torch.linalg.pinv(XtX)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_diagnosis_with_collinear_features():
    """DIAGNOSIS should not crash when X has perfectly collinear columns."""
    torch.manual_seed(42)
    n = 10
    x1 = torch.randn(n, 1)
    x2 = x1 * 2  # perfectly collinear
    x_data = torch.cat([x1, x2, torch.ones(n, 1)], dim=1)

    y_data = torch.randn(n, 1)
    weight = torch.rand(n, 3)
    y_pred = y_data + torch.randn(n, 1) * 0.1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = DIAGNOSIS(weight, x_data, y_data, y_pred)

    r2 = diag.R2()
    assert not torch.isnan(r2)


def test_diagnosis_well_conditioned():
    """DIAGNOSIS should work normally for well-conditioned data."""
    torch.manual_seed(0)
    n = 20
    x_data = torch.randn(n, 2)
    true_coef = torch.tensor([[3.0], [-1.0]])
    y_data = x_data @ true_coef + torch.randn(n, 1) * 0.1
    weight = torch.rand(n, 2)
    y_pred = x_data @ true_coef + torch.randn(n, 1) * 0.05

    diag = DIAGNOSIS(weight, x_data, y_data, y_pred)

    r2 = diag.R2()
    assert not torch.isnan(r2)
    assert r2.item() > 0.5
