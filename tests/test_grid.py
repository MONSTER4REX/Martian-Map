import numpy as np
import pytest

from marswalk.planner.grid import Grid


def test_basic_properties():
    g = Grid(np.zeros((3, 4)), cellsize=2.0)
    assert g.shape == (3, 4)
    assert g.nrows == 3 and g.ncols == 4
    assert g.cellsize == 2.0
    assert g.in_bounds((2, 3))
    assert not g.in_bounds((3, 0))
    assert not g.in_bounds((0, -1))


def test_nan_is_unknown():
    a = np.zeros((2, 2))
    a[0, 1] = np.nan
    g = Grid(a, 1.0)
    assert g.is_known((0, 0))
    assert not g.is_known((0, 1))
    assert not g.is_known((5, 5))  # out of bounds
    assert g.known_mask.sum() == 3


@pytest.mark.parametrize(
    "arr, cs",
    [
        (np.zeros((3,)), 1.0),  # not 2-D
        (np.zeros((0, 0)), 1.0),  # empty
        (np.zeros((2, 2)), 0.0),  # bad cellsize
        (np.zeros((2, 2)), -1.0),
        (np.zeros((2, 2)), np.inf),
    ],
)
def test_invalid_construction(arr, cs):
    with pytest.raises(ValueError):
        Grid(arr, cs)
