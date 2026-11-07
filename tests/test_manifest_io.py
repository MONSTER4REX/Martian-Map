import importlib.util

import numpy as np
import pytest

from marswalk.data.io import WindowSpec, array_to_grid, bounds_to_window, read_cog_window
from marswalk.data.manifest import LayerRecord, Manifest, VerifyStatus
from marswalk.planner.layers import LayerKind


def _layer(**kw):
    base = dict(
        id="hirise_dtm",
        title="HiRISE DTM",
        kind=LayerKind.INFORMATIONAL,
        product_type="DEM",
        source_url="s3://astrogeo-ard/mars/mro/hirise/controlled/dtm",
        native_res_m=1.0,
        effective_res_m=2.0,
        units="m",
        license="NASA public (UNVERIFIED)",
    )
    base.update(kw)
    return LayerRecord(**base)


def test_manifest_roundtrip(tmp_path):
    m = Manifest(layers=[_layer(), _layer(id="themis", product_type="thermal_inertia",
                                         native_res_m=100.0, effective_res_m=100.0,
                                         units="SI tiu")])
    p = tmp_path / "manifest.json"
    m.save(p)
    loaded = Manifest.load(p)
    assert loaded.aoi == "Jezero"
    assert {layer.id for layer in loaded.layers} == {"hirise_dtm", "themis"}
    assert loaded.unverified() == ["hirise_dtm", "themis"]


def test_effective_finer_than_native_rejected():
    with pytest.raises(ValueError, match="false detail"):
        _layer(native_res_m=10.0, effective_res_m=2.0)


def test_coarse_layer_flagging():
    m = Manifest(layers=[
        _layer(id="dtm", native_res_m=1.0, effective_res_m=2.0),
        _layer(id="themis", native_res_m=100.0, effective_res_m=100.0),
    ])
    assert m.coarser_than(10.0) == ["themis"]


def test_verified_status():
    layer = _layer(status=VerifyStatus.VERIFIED)
    m = Manifest(layers=[layer])
    assert m.unverified() == []


def test_bounds_to_window_math():
    # north-up transform: origin (1000, 5000), 2 m pixels, y decreasing.
    transform = (2.0, 0.0, 1000.0, 0.0, -2.0, 5000.0)
    ws = bounds_to_window(transform, (1000.0, 4980.0, 1020.0, 5000.0))
    assert isinstance(ws, WindowSpec)
    assert ws.col_off == 0 and ws.row_off == 0
    assert ws.width == 10 and ws.height == 10


def test_array_to_grid_nodata():
    a = np.array([[1.0, -9999.0], [2.0, 3.0]])
    g = array_to_grid(a, 2.0, nodata=-9999.0)
    assert np.isnan(g.elevation[0, 1])
    assert g.elevation[0, 0] == 1.0


@pytest.mark.skipif(
    importlib.util.find_spec("rasterio") is not None,
    reason="rasterio installed; skip the missing-dependency path",
)
def test_read_cog_window_without_rasterio_raises():
    with pytest.raises(RuntimeError, match="rasterio"):
        read_cog_window("s3://bucket/x.tif", window=WindowSpec(0, 0, 4, 4))
