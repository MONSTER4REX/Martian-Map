import numpy as np
import pytest

from marswalk.science.size_frequency import (
    cumulative_number_density,
    expected_count_at_least,
    fit_exponential_sfd,
)
from marswalk.science.value_raster import science_value, top_targets


def test_sfd_recovers_rate():
    rng = np.random.default_rng(0)
    lam = 2.0  # 1/m
    d = rng.exponential(scale=1.0 / lam, size=4000)
    fit = fit_exponential_sfd(d, area_m2=100.0)
    assert fit.lambda_per_m == pytest.approx(lam, rel=0.15)
    assert fit.r2 > 0.95
    # cumulative density is decreasing in D
    assert cumulative_number_density(fit, 0.5) > cumulative_number_density(fit, 2.0)


def test_sfd_expected_count_scales_with_area():
    rng = np.random.default_rng(1)
    d = rng.exponential(scale=0.5, size=2000)
    fit = fit_exponential_sfd(d, area_m2=50.0)
    c1 = expected_count_at_least(fit, 1.0, 50.0)
    c2 = expected_count_at_least(fit, 1.0, 100.0)
    assert c2 == pytest.approx(2 * c1, rel=1e-9)


def test_sfd_needs_enough_data():
    with pytest.raises(ValueError):
        fit_exponential_sfd([0.3], area_m2=10.0)


def test_science_value_weighted_blend():
    a = np.array([[0.0, 1.0], [0.0, 1.0]])  # normalizes to itself
    b = np.array([[1.0, 1.0], [0.0, 0.0]])
    val = science_value({"a": (a, 1.0), "b": (b, 1.0)})
    # cell (0,0): (0*1 + 1*1)/2 = 0.5 ; cell (0,1): (1+1)/2 = 1
    assert val[0, 0] == pytest.approx(0.5)
    assert val[0, 1] == pytest.approx(1.0)
    assert val[1, 0] == pytest.approx(0.0)


def test_science_value_nan_handling():
    a = np.array([[np.nan, 1.0], [0.0, 1.0]])
    b = np.array([[np.nan, 0.0], [0.0, 1.0]])
    val = science_value({"a": (a, 1.0), "b": (b, 1.0)})
    assert np.isnan(val[0, 0])  # all layers NaN here
    assert np.isfinite(val[0, 1])  # at least one layer present


def test_top_targets_with_separation():
    val = np.zeros((5, 5))
    val[0, 0] = 1.0
    val[0, 1] = 0.9  # adjacent to the top cell
    val[4, 4] = 0.8
    picks = top_targets(val, k=2, min_separation_cells=2)
    cells = [c for c, _ in picks]
    assert cells[0] == (0, 0)
    assert (0, 1) not in cells  # too close to (0,0)
    assert (4, 4) in cells
