"""Smoke test for the offline mission figure renderer.

Skips when matplotlib is absent (it is an optional `viz` extra). Asserts only
that the renderer runs end-to-end and writes a non-trivial PNG; it does not
check pixels.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib (viz extra) not installed")
def test_render_mission_figure(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import render_mission_figure as rmf
    finally:
        sys.path.pop(0)

    out = tmp_path / "fig.png"
    result = rmf.render(out, stability_n=3, seed=0)
    assert result == out
    assert out.exists()
    # a real PNG with content, not an empty stub
    header = out.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 5000
