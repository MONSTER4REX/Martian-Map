import numpy as np

from marswalk.data.error_model import (
    DEMErrorModel,
    empirical_variogram,
    ensemble,
    fit_exponential_variogram,
    perturbed_grid,
    sample_error,
)
from marswalk.planner.grid import Grid
from marswalk.terrain.synthetic import punch_nodata, random_smooth_terrain


def _flat(n=64, cs=2.0):
    return Grid(np.zeros((n, n)), cs)


def test_error_variance_matches_sill():
    g = _flat()
    model = DEMErrorModel(sill_m2=4.0, range_m=10.0)
    rng = np.random.default_rng(0)
    err = sample_error(g, model, rng)
    # Large field -> sample variance near the sill (loose tolerance).
    assert 2.0 < err.var() < 7.0  # ~4.0


def test_error_is_spatially_correlated():
    g = _flat()
    model = DEMErrorModel(sill_m2=4.0, range_m=12.0)
    err = sample_error(g, model, np.random.default_rng(1))
    near = np.corrcoef(err[:, :-1].ravel(), err[:, 1:].ravel())[0, 1]
    far = np.corrcoef(err[:, :-20].ravel(), err[:, 20:].ravel())[0, 1]
    assert near > 0.5
    assert near > far  # correlation decays with distance


def test_ensemble_reproducible_and_preserves_nodata():
    g = punch_nodata(_flat(16), [(3, 3), (4, 4)])
    model = DEMErrorModel(sill_m2=1.0, range_m=6.0)
    a = ensemble(g, model, 3, seed=7)
    b = ensemble(g, model, 3, seed=7)
    for ga, gb in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.nan_to_num(ga.elevation), np.nan_to_num(gb.elevation))
        assert np.isnan(ga.elevation[3, 3]) and np.isnan(ga.elevation[4, 4])
    # different seeds differ
    c = ensemble(g, model, 1, seed=8)[0]
    assert not np.allclose(np.nan_to_num(a[0].elevation), np.nan_to_num(c.elevation))


def test_variogram_rises_then_plateaus():
    g = _flat(80)
    model = DEMErrorModel(sill_m2=4.0, range_m=14.0)
    err = sample_error(g, model, np.random.default_rng(2))
    lags, gamma = empirical_variogram(Grid(err, g.cellsize), n_lags=20, max_lag_cells=25)
    assert gamma[0] < gamma[-1]  # grows with lag
    # near-plateau: last few lags are similar
    assert abs(gamma[-1] - gamma[-3]) < 0.5 * gamma[-1]


def test_fit_recovers_exponential_params():
    # Validate the fitter on data drawn from its own model (exponential), so
    # we test the fit, not the field's Gaussian-vs-exponential mismatch.
    true = DEMErrorModel(sill_m2=4.0, range_m=14.0, nugget_m2=0.5)
    h = np.linspace(2.0, 60.0, 20)
    gamma = true.nugget_m2 + true.sill_m2 * (1.0 - np.exp(-h / true.range_m))
    gamma = gamma + np.random.default_rng(0).normal(0, 0.02, h.shape)
    fit = fit_exponential_variogram(h, gamma)
    assert 2.0 < fit.sill_m2 < 8.0        # ~4
    assert 7.0 < fit.range_m < 28.0       # ~14 (within ~2x)


def test_fit_on_field_is_positive_and_reasonable():
    # On the Gaussian-covariance field the exponential fit is a proxy; just
    # require a positive fit that tracks the empirical curve (low RMSE).
    g = _flat(80)
    model = DEMErrorModel(sill_m2=4.0, range_m=14.0)
    err = sample_error(g, model, np.random.default_rng(3))
    lags, gamma = empirical_variogram(Grid(err, g.cellsize), max_lag_cells=25)
    fit = fit_exponential_variogram(lags, gamma)
    assert fit.sill_m2 > 0 and fit.range_m > 0
    pred = fit.nugget_m2 + fit.sill_m2 * (1.0 - np.exp(-lags / fit.range_m))
    rmse = float(np.sqrt(np.mean((pred - gamma) ** 2)))
    assert rmse < 0.25 * float(np.nanmax(gamma))  # fit tracks the curve


def test_perturbed_grid_changes_elevation():
    g = random_smooth_terrain(32, 32, seed=0, relief_m=5.0)
    model = DEMErrorModel(sill_m2=1.0, range_m=8.0)
    pg = perturbed_grid(g, model, np.random.default_rng(0))
    assert pg.shape == g.shape
    assert not np.allclose(pg.elevation, g.elevation)
