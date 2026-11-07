import pytest

from marswalk.physics.radiation import RadiationScreen
from marswalk.physics.slip import SlipParams, slip_proxy
from marswalk.physics.thermal import ThermalParams, surface_temp_range_k


def test_thermal_inertia_damps_swing():
    lo_ti = surface_temp_range_k(50.0, 500.0)
    hi_ti = surface_temp_range_k(800.0, 500.0)
    swing_lo = lo_ti[1] - lo_ti[0]
    swing_hi = hi_ti[1] - hi_ti[0]
    assert swing_hi < swing_lo  # higher inertia -> smaller day/night swing
    assert lo_ti[0] < lo_ti[1]
    assert all(t > 0 for t in lo_ti + hi_ti)  # Kelvin, positive


def test_thermal_peak_rises_with_insolation():
    cool = surface_temp_range_k(200.0, 200.0)
    hot = surface_temp_range_k(200.0, 600.0)
    assert hot[1] > cool[1]


def test_thermal_params_have_validity_note():
    assert "validate" in ThermalParams().validity.lower()


def test_radiation_dose_scales_and_shields():
    screen = RadiationScreen(surface_dose_rate_usv_per_h=20.0, attenuation_length_g_cm2=30.0)
    d_unshielded = screen.dose_usv(6.0, 0.0)
    assert d_unshielded == pytest.approx(120.0)
    assert screen.dose_usv(12.0, 0.0) == pytest.approx(2 * d_unshielded)
    assert screen.dose_usv(6.0, 30.0) < d_unshielded  # shielding reduces dose
    assert "PLACEHOLDER" in screen.validity


def test_radiation_rejects_negatives():
    with pytest.raises(ValueError):
        RadiationScreen().dose_usv(-1.0)


def test_slip_proxy_monotone_bounded():
    assert slip_proxy(0.0) == pytest.approx(0.0)
    assert 0.0 <= slip_proxy(10.0) < slip_proxy(25.0) < 1.0
    assert slip_proxy(90.0) < 1.0
    # steeper char slope -> lower risk at a given slope
    gentle = slip_proxy(15.0, SlipParams(char_slope_deg=30.0))
    steep = slip_proxy(15.0, SlipParams(char_slope_deg=10.0))
    assert steep > gentle
