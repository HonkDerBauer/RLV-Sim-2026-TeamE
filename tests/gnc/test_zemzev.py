"""ZEM/ZEV guidance: analytic law, time-to-go, saturation and the estimated-state contract."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from core.interfaces import Command
from gnc.context import GncContext
from gnc.guidance.zemzev import (
    ZemZevGuidance,
    build,
    optimal_tgo,
    profiles_feasible,
    project_achievable,
    segment_profile_metrics,
    zem_zev,
    zem_zev_accel,
    zem_zev_cost,
    zem_zev_profile,
)

G = np.array([0.0, 0.0, 9.80665])
NOMINAL = {
    "thrust_vac": 981000.0,
    "throttle_min": 0.4,
    "isp": 311.0,
    "g0": 9.80665,
    "gravity_ned": G,
    "m_dry": 25800.0,
    "p_sea": 101325.0,
    "nozzle_exit_area": 0.9,
    "scale_height": 8500.0,
}


def _fly_unconstrained(r0, v0, r_f, v_f, tgo0, dt=1e-3):
    """Closed-loop double integrator with the ZEM/ZEV law re-evaluated every step."""
    r, v = np.array(r0, float), np.array(v0, float)
    n = int(round(tgo0 / dt))
    for k in range(n - 1):
        tgo = tgo0 - k * dt
        zem, zev = zem_zev(r, v, r_f, v_f, G, tgo)
        a = zem_zev_accel(zem, zev, tgo) + G
        v = v + a * dt
        r = r + v * dt
    return r, v


def test_unconstrained_law_reaches_terminal_state():
    r_f, v_f = np.zeros(3), np.array([0.0, 0.0, 0.5])
    r, v = _fly_unconstrained([40.0, -25.0, -300.0], [2.0, -1.0, 30.0], r_f, v_f, 12.0)
    assert np.linalg.norm(r - r_f) < 0.05
    assert np.linalg.norm(v - v_f) < 0.05


def test_open_loop_profile_integrates_to_zem_and_zev():
    zem, zev = zem_zev([10.0, 5.0, -200.0], [1.0, 0.0, 20.0], np.zeros(3), np.zeros(3), G, 9.0)
    tau, a = zem_zev_profile(zem, zev, 9.0, n=4001)
    dtau = tau[1] - tau[0]
    dv = np.trapezoid(a, dx=dtau, axis=0)
    dr = np.trapezoid((9.0 - tau)[:, None] * a, dx=dtau, axis=0)
    np.testing.assert_allclose(dv, zev, atol=1e-6)
    np.testing.assert_allclose(dr, zem, atol=1e-5)


def test_cost_matches_integral_of_squared_profile():
    r, v = np.array([30.0, 0.0, -250.0]), np.array([0.0, 2.0, 35.0])
    zem, zev = zem_zev(r, v, np.zeros(3), np.zeros(3), G, 7.0)
    tau, a = zem_zev_profile(zem, zev, 7.0, n=4001)
    energy = 0.5 * np.trapezoid(np.sum(a * a, axis=1), tau)
    assert zem_zev_cost(r, v, np.zeros(3), np.zeros(3), G, 7.0, gamma=0.0) == pytest.approx(
        energy, rel=1e-6
    )


@pytest.mark.parametrize("gamma", [1.0, 20.0, 200.0])
def test_optimal_tgo_quartic_matches_brute_force(gamma):
    r, v = np.array([25.0, -10.0, -300.0]), np.array([1.0, 0.5, 25.0])
    t_star = optimal_tgo(r, v, np.zeros(3), np.zeros(3), G, gamma, 0.5, 60.0)
    grid = np.linspace(0.5, 60.0, 60001)
    costs = [zem_zev_cost(r, v, np.zeros(3), np.zeros(3), G, t, gamma) for t in grid]
    assert t_star == pytest.approx(grid[int(np.argmin(costs))], abs=2e-3)


def test_segment_metrics_are_exact():
    rng = np.random.default_rng(3)
    a0, af = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    p_min, p_max = segment_profile_metrics(a0, af)
    s = np.linspace(0.0, 1.0, 20001)
    mags = np.linalg.norm(a0[:, None, :] + s[None, :, None] * (af - a0)[:, None, :], axis=2)
    np.testing.assert_allclose(p_min, mags.min(axis=1), atol=1e-6)
    np.testing.assert_allclose(p_max, mags.max(axis=1), atol=1e-12)


def test_profile_feasibility_cone_and_bounds():
    up = np.array([[0.0, 0.0, -15.0]])
    tilted = np.array([[15.0 * np.sin(np.deg2rad(20)), 0.0, -15.0 * np.cos(np.deg2rad(20))]])
    ok, _ = profiles_feasible(up, up, 10.0, 20.0, np.deg2rad(15))
    assert ok[0]
    ok, _ = profiles_feasible(up, tilted, 10.0, 20.0, np.deg2rad(15))
    assert not ok[0]
    ok, _ = profiles_feasible(up, up, 16.0, 20.0, np.deg2rad(15))
    assert not ok[0]


def test_saturation_keeps_vertical_priority():
    tilt = np.deg2rad(15)
    thrust, flags = project_achievable(np.array([4.0e5, 0.0, -5.0e5]), 3.0e5, 9.0e5, tilt)
    assert flags["sat_tilt"]
    assert -thrust[2] == pytest.approx(5.0e5)  # up component preserved
    lat = np.hypot(thrust[0], thrust[1])
    assert lat == pytest.approx(5.0e5 * np.tan(tilt))

    thrust, flags = project_achievable(np.array([0.0, 0.0, -1.2e6]), 3.0e5, 9.0e5, tilt)
    assert flags["sat_high"] and np.linalg.norm(thrust) == pytest.approx(9.0e5)

    thrust, flags = project_achievable(np.array([0.0, 0.0, 1.0e5]), 3.0e5, 9.0e5, tilt)
    assert flags["sat_down"]
    np.testing.assert_allclose(thrust, [0.0, 0.0, -3.0e5])


def _state(r, v, m=29500.0):
    return SimpleNamespace(r_ned=np.asarray(r, float), v_ned=np.asarray(v, float), m=m)


def test_guidance_coasts_then_ignites_within_limits():
    g = ZemZevGuidance(NOMINAL, np.array([1.0]), 4)
    mem = g.initial_memory()
    r, v, t, dt = np.array([0.0, 0.0, -300.0]), np.array([0.0, 0.0, 25.0]), 0.0, 0.02
    ignited_at = None
    while t < 10.0 and r[2] < 0.0:
        cmd, mem = g.track(_state(r, v), t, mem)
        thrust = np.asarray(cmd.thrust_vector)
        if ignited_at is None and np.linalg.norm(thrust) > 0.0:
            ignited_at = t
        if ignited_at is not None and mem["zz_freeze"] is None and np.linalg.norm(thrust) > 0:
            mag = np.linalg.norm(thrust)
            lo, hi = g._bounds(-r[2])
            assert lo - 1e-6 <= mag <= hi + 1e-6
        a = thrust / 29500.0 + G
        v, r, t = v + a * dt, r + (v + a * dt) * dt, t + dt
    assert ignited_at is not None and ignited_at > 0.5  # coasted first (latest-feasible ignition)
    assert mem["zz_ignition_reason"] == "latest_feasible"
    assert abs(r[2]) < 1.0 and np.linalg.norm(v) < 1.5


def test_guidance_uses_only_the_estimate():
    """Identical estimates give identical commands; the law has no other state input."""
    g = ZemZevGuidance(NOMINAL, np.array([1.0]), 4, ignition_lookahead_s=0.0)
    x_hat = _state([5.0, -3.0, -150.0], [0.5, 0.2, 45.0])
    cmd_a, _ = g.track(x_hat, 3.0, g.initial_memory())
    cmd_b, _ = g.track(x_hat, 3.0, g.initial_memory())
    assert isinstance(cmd_a, Command)
    np.testing.assert_array_equal(cmd_a.thrust_vector, cmd_b.thrust_vector)
    shifted = _state([5.0, -3.0, -140.0], [0.5, 0.2, 45.0])
    cmd_c, _ = g.track(shifted, 3.0, g.initial_memory())
    assert not np.allclose(cmd_a.thrust_vector, cmd_c.thrust_vector)


def test_build_from_config():
    cfg = OmegaConf.load(Path(__file__).parents[2] / "configs/gnc/guidance/zemzev.yaml")
    ctx = GncContext(cfg=cfg, nominal=NOMINAL, active_engines=np.array([1.0]), phase=4)
    g = build(ctx)
    assert isinstance(g, ZemZevGuidance)
    assert g.tgo_mode == "fixed"
    with pytest.raises(ValueError):
        ZemZevGuidance(NOMINAL, np.array([1.0]), 4, tgo_mode="bogus")
