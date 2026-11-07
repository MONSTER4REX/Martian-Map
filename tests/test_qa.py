import numpy as np

from marswalk.data.qa import run_dtm_qa
from marswalk.planner.grid import Grid
from marswalk.terrain.synthetic import punch_nodata, random_smooth_terrain


def test_clean_flat_has_no_flags():
    g = random_smooth_terrain(20, 20, relief_m=0.0, seed=0)
    rep, masks = run_dtm_qa(g)
    assert rep.nodata_count == 0
    assert rep.slope_spike_count == 0
    assert rep.step_edge_count == 0
    assert rep.seam_columns == [] and rep.seam_rows == []


def test_nodata_counted():
    g = random_smooth_terrain(10, 10, relief_m=0.0, seed=0)
    g = punch_nodata(g, [(1, 1), (2, 2), (3, 3)])
    rep, masks = run_dtm_qa(g)
    assert rep.nodata_count == 3
    assert rep.nodata_fraction == 3 / 100
    assert masks["nodata"].sum() == 3


def test_box_step_flagged():
    z = np.zeros((20, 20))
    z[:, 10:] += 10.0  # box step of 10 m > step_m default 5
    rep, masks = run_dtm_qa(Grid(z, 2.0))
    assert rep.step_edge_count > 0
    assert masks["step_edge"][:, 9:11].any()


def test_seam_detected_below_step_threshold():
    z = np.zeros((20, 20))
    z[:, 12:] += 3.0  # consistent 3 m offset: a seam, but below the 5 m step_m
    rep, masks = run_dtm_qa(Grid(z, 2.0), step_m=5.0, seam_m=2.0)
    assert 12 in rep.seam_columns
    assert rep.step_edge_count == 0  # 3 m < 5 m so not flagged as a box-step


def test_slope_spike_flagged():
    z = np.zeros((11, 11))
    z[5, 5] = 300.0  # a single tall spike -> near-vertical slope at neighbours
    rep, masks = run_dtm_qa(Grid(z, 2.0), slope_spike_deg=80.0)
    assert rep.slope_spike_count > 0
