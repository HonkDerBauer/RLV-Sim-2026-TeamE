"""ZEM/ZEV analytical feedback guidance for powered descent (3-DOF translational model).

Point-mass model in the landing NED frame with constant planning gravity ``g``::

    r' = v,   v' = a_T + g,   a_T = T / m

Zero-effort miss / velocity (the terminal error if the thrust acceleration were zero for the
remaining time-to-go ``t``)::

    ZEM = r_f - (r + v t + 1/2 g t^2)
    ZEV = v_f - (v + g t)

The energy-optimal control for ``min 1/2 int |a_T|^2`` with fixed ``t`` and hard terminal position
and velocity is linear in time; its value at the current instant is the ZEM/ZEV feedback law::

    a_T = 6 ZEM / t^2 - 2 ZEV / t

Gravity compensation is contained in ZEM/ZEV. Re-evaluating the law at every control tick from
the navigation estimate closes the loop.

Time-to-go
----------
* ``fixed``: a terminal epoch ``t_f`` is committed at ignition and ``t = t_f - t_now``.
* ``optimal``: ``t`` re-solved every tick from ``min_t J(t) = Gamma t + 1/2 int |a_T|^2``. With
  ``P(t) = t^3 J(t)``, ``dJ/dt = 0`` is the quartic ``t P'(t) - 3 P(t) = 0`` with leading
  coefficient ``Gamma + 1/2 |g|^2`` (D'Souza, 1997).

Ignition and the throttle floor
-------------------------------
A single Merlin at 40 % throttle accelerates the landing-mass vehicle at more than 1 g, so the
vehicle cannot hover. An unconstrained law ignited too early asks for less than the floor, is
clipped to it, stops above the pad and climbs away. The engine therefore stays off until a
time-to-go exists whose *entire* open-loop optimal profile (linear in time) fits inside the thrust
magnitude bounds and tilt cone with margins. The shortest such ``t`` is committed (least gravity
loss). Saturation of the closed-loop command onto the achievable set is still applied every tick.

Only ``x_hat`` (navigation estimate) and nominal vehicle constants are used; no truth states.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.polynomial import Polynomial

from core.frames import dcm_from_quat
from gnc.guidance.flip_coast import pointing_from
from gnc.guidance.base import attitude_command, thrust_aligned_quat, thrust_command, upright_quat

_UP = np.array([0.0, 0.0, -1.0])  # NED up


# ---------------------------------------------------------------------------------------------
# Pure guidance math (unit-tested; no vehicle state)
# ---------------------------------------------------------------------------------------------


def zem_zev(r, v, r_f, v_f, g, tgo):
    """Zero-effort miss and zero-effort velocity for constant gravity ``g``."""
    r, v, r_f, v_f, g = (np.asarray(x, float) for x in (r, v, r_f, v_f, g))
    zem = r_f - (r + v * tgo + 0.5 * g * tgo * tgo)
    zev = v_f - (v + g * tgo)
    return zem, zev


def zem_zev_accel(zem, zev, tgo):
    """Energy-optimal thrust acceleration ``6 ZEM/t^2 - 2 ZEV/t`` (gravity already in ZEM/ZEV)."""
    return 6.0 * np.asarray(zem, float) / (tgo * tgo) - 2.0 * np.asarray(zev, float) / tgo


def zem_zev_lag(r, v, a_act, r_f, v_f, g, tgo, tau: float):
    """ZEM/ZEV with a first-order actuator lag ``a' = (u - a) / tau``.

    The zero-effort drift now includes the acceleration already realized (``a_act``), which
    decays as ``a_act exp(-s/tau)`` when the command is zeroed. ``tau = 0`` gives :func:`zem_zev`.
    """
    zem, zev = zem_zev(r, v, r_f, v_f, g, tgo)
    if tau <= 0.0:
        return zem, zev
    a_act = np.asarray(a_act, float)
    one_m_e = -np.expm1(-tgo / tau)  # 1 - exp(-t/tau), accurate for small t/tau
    return zem - a_act * tau * (tgo - tau * one_m_e), zev - a_act * tau * one_m_e


def lag_gramian(tgo: float, tau: float):
    """Impulse response ``h(t) = [t - tau(1-E), 1-E]`` and Gramian ``G = int_0^t h h^T``.

    ``h`` maps a unit command impulse through the lag onto terminal position and velocity
    (E = exp(-t/tau)). Closed form; ``tau -> 0`` recovers the double-integrator values
    ``h = [t, 1]``, ``G = [[t^3/3, t^2/2], [t^2/2, t]]``.
    """
    T = float(tgo)
    if tau <= 0.0:
        return np.array([T, 1.0]), np.array([[T**3 / 3.0, T**2 / 2.0], [T**2 / 2.0, T]])
    E = np.exp(-T / tau)
    om = -np.expm1(-T / tau)  # 1 - E
    om2 = -np.expm1(-2.0 * T / tau)  # 1 - E^2
    h = np.array([T - tau * om, om])
    g11 = ((T - tau) ** 3 + tau**3) / 3.0 - 2.0 * tau**2 * T * E + 0.5 * tau**3 * om2
    g12 = 0.5 * T * T - tau * T + tau**2 * om + tau * T * E - 0.5 * tau**2 * om2
    g22 = T - 2.0 * tau * om + 0.5 * tau * om2
    return h, np.array([[g11, g12], [g12, g22]])


def lag_gramian_accel(tgo: float, tau: float):
    """As :func:`lag_gramian` with the terminal acceleration added as a third output.

    ``h3(t) = E/tau`` (a unit command impulse's effect on the lagged acceleration). Requires
    ``tau > 0``: without the lag the acceleration is not a state.
    """
    if tau <= 0.0:
        raise ValueError("terminal-acceleration constraint needs tau > 0")
    T = float(tgo)
    E = np.exp(-T / tau)
    om = -np.expm1(-T / tau)
    om2 = -np.expm1(-2.0 * T / tau)
    h2, G2 = lag_gramian(T, tau)
    h = np.array([h2[0], h2[1], E / tau])
    G = np.zeros((3, 3))
    G[:2, :2] = G2
    G[0, 2] = G[2, 0] = -T * E + 0.5 * tau * om2
    G[1, 2] = G[2, 1] = om - 0.5 * om2
    G[2, 2] = om2 / (2.0 * tau)
    return h, G


def zem_zev_lag_accel(zem, zev, tgo, tau: float, zea=None):
    """Minimum-energy command through a first-order lag: ``u = h(t)^T G(t)^-1 [ZEM, ZEV(, ZEA)]``.

    This is the controllability-Gramian solution of ``min 1/2 int u^2`` for ``r' = v, v' = a,
    a' = (u - a)/tau`` with hard terminal position and velocity (and, if ``zea`` is given, hard
    terminal acceleration; ZEA = a_f - a exp(-t/tau)). ``tau = 0`` without ``zea`` reproduces
    ``6 ZEM/t^2 - 2 ZEV/t`` exactly.
    """
    if zea is None:
        h, G = lag_gramian(tgo, tau)
        k = np.linalg.solve(G, h)  # gains on (ZEM, ZEV); G symmetric
        return k[0] * np.asarray(zem, float) + k[1] * np.asarray(zev, float)
    h, G = lag_gramian_accel(tgo, tau)
    k = np.linalg.solve(G, h)
    return (
        k[0] * np.asarray(zem, float)
        + k[1] * np.asarray(zev, float)
        + k[2] * np.asarray(zea, float)
    )


def zem_zev_profile(zem, zev, tgo, n: int = 21):
    """Open-loop optimal thrust acceleration a(tau), tau in [0, tgo], sampled at ``n`` points.

    The profile is linear: ``a(0) = 6 ZEM/t^2 - 2 ZEV/t`` and ``a(t) = -6 ZEM/t^2 + 4 ZEV/t``.
    Returns ``(tau, a)`` with shapes (n,) and (n, 3).
    """
    a0 = zem_zev_accel(zem, zev, tgo)
    af = -6.0 * np.asarray(zem, float) / (tgo * tgo) + 4.0 * np.asarray(zev, float) / tgo
    s = np.linspace(0.0, 1.0, n)
    return s * tgo, a0[None, :] + s[:, None] * (af - a0)[None, :]


def zem_zev_cost(r, v, r_f, v_f, g, tgo, gamma: float = 0.0) -> float:
    """Cost ``J(t) = Gamma t + 1/2 int |a_T|^2``.

    For the linear optimal profile: ``J = Gamma t + 2|ZEV|^2/t - 6 ZEV.ZEM/t^2 + 6|ZEM|^2/t^3``.
    """
    zem, zev = zem_zev(r, v, r_f, v_f, g, tgo)
    t = float(tgo)
    return float(gamma * t + 2.0 * zev @ zev / t - 6.0 * zev @ zem / t**2 + 6.0 * zem @ zem / t**3)


def optimal_tgo(r, v, r_f, v_f, g, gamma: float, t_lo: float, t_hi: float) -> float:
    """Time-to-go minimizing ``J(t)`` on ``[t_lo, t_hi]`` via the stationarity quartic.

    ``M(t) = dr - v t - 1/2 g t^2`` and ``V(t) = dv - g t`` are vector polynomials;
    ``P(t) = t^3 J = Gamma t^4 + 2 t^2 |V|^2 - 6 t V.M + 6 |M|^2``, and
    ``J' = 0 <=> t P' - 3 P = 0`` (a quartic).
    Candidates are the real roots inside the interval plus both endpoints.
    """
    dr = np.asarray(r_f, float) - np.asarray(r, float)
    dv = np.asarray(v_f, float) - np.asarray(v, float)
    v = np.asarray(v, float)
    g = np.asarray(g, float)
    t = Polynomial([0.0, 1.0])
    M = [Polynomial([dr[i], -v[i], -0.5 * g[i]]) for i in range(3)]
    V = [Polynomial([dv[i], -g[i]]) for i in range(3)]
    vv = sum((Vi * Vi for Vi in V), Polynomial([0.0]))
    vm = sum((Vi * Mi for Vi, Mi in zip(V, M, strict=True)), Polynomial([0.0]))
    mm = sum((Mi * Mi for Mi in M), Polynomial([0.0]))
    P = gamma * t**4 + 2.0 * t**2 * vv - 6.0 * t * vm + 6.0 * mm
    stationary = t * P.deriv() - 3.0 * P
    cands = [t_lo, t_hi]
    for root in stationary.roots():
        if abs(root.imag) < 1e-9 and t_lo < root.real < t_hi:
            cands.append(float(root.real))
    costs = [zem_zev_cost(r, v, r_f, v_f, g, c, gamma) for c in cands]
    return float(cands[int(np.argmin(costs))])


def _in_cone(a, tilt_max: float):
    """Row-wise: inside the upward tilt cone (strictly upward component)."""
    up = a @ _UP
    lat = np.linalg.norm(a - up[..., None] * _UP, axis=-1)
    return (up > 0.0) & (lat <= up * np.tan(tilt_max))


def segment_profile_metrics(a0, af):
    """Exact max and min |a| over linear profiles ``a(s) = a0 + s (af - a0)``, s in [0, 1].

    Row-wise for (K, 3) inputs. The max is at an endpoint (norm is convex); the min is the
    distance from the origin to the segment.
    """
    a0, af = np.atleast_2d(a0), np.atleast_2d(af)
    d = af - a0
    dd = np.einsum("ij,ij->i", d, d)
    s = np.clip(-np.einsum("ij,ij->i", a0, d) / np.maximum(dd, 1e-12), 0.0, 1.0)
    a_min = np.linalg.norm(a0 + s[:, None] * d, axis=1)
    a_max = np.maximum(np.linalg.norm(a0, axis=1), np.linalg.norm(af, axis=1))
    return a_min, a_max


def profiles_feasible(a0, af, a_lo, a_hi, tilt_max: float):
    """Exact feasibility of linear thrust-acceleration profiles (row-wise).

    The tilt cone and the max-magnitude ball are convex, so the endpoints decide them; the
    non-convex floor uses the segment's closest point to the origin.
    """
    p_min, p_max = segment_profile_metrics(a0, af)
    cone = _in_cone(np.atleast_2d(a0), tilt_max) & _in_cone(np.atleast_2d(af), tilt_max)
    return cone & (p_min >= a_lo) & (p_max <= a_hi), p_max


def project_achievable(t_cmd, rho_min: float, rho_max: float, tilt_max: float):
    """Saturate a NED thrust command with VERTICAL PRIORITY onto the achievable set.

    Achievable set: magnitude in [rho_min, rho_max], direction within ``tilt_max`` of up. The up
    component is kept (capped at rho_max); the lateral component is clipped to the tilt cone and
    to the magnitude left over, so a lateral divert never steals vertical braking. A command below
    the floor is scaled up uniformly. Returns ``(thrust, flags)`` naming the active limits.
    """
    t_cmd = np.asarray(t_cmd, float)
    flags = {"sat_low": False, "sat_high": False, "sat_tilt": False, "sat_down": False}
    s = float(t_cmd @ _UP)
    if s <= 0.0:
        flags["sat_down"] = True
        return rho_min * _UP, flags
    if s > rho_max:
        flags["sat_high"] = True
        s = rho_max
    lat = t_cmd - float(t_cmd @ _UP) * _UP
    lat_n = float(np.linalg.norm(lat))
    lat_cap = min(s * np.tan(tilt_max), np.sqrt(max(rho_max * rho_max - s * s, 0.0)))
    if lat_n > lat_cap:
        if lat_cap < s * np.tan(tilt_max):
            flags["sat_high"] = True
        else:
            flags["sat_tilt"] = True
        lat = lat * (lat_cap / lat_n)
    out = s * _UP + lat
    mag = float(np.linalg.norm(out))
    if mag < rho_min:
        flags["sat_low"] = True
        out = out * (rho_min / mag)
    return out, flags


# ---------------------------------------------------------------------------------------------
# Guidance law (framework contract: __call__ / track -> (Command, memory))
# ---------------------------------------------------------------------------------------------


class ZemZevGuidance:
    """Closed-loop ZEM/ZEV powered-descent guidance with feasibility-based ignition."""

    def __init__(
        self,
        nominal: dict,
        active_engines,
        phase: int,
        *,
        target_ned=(0.0, 0.0, 0.0),
        target_vel_ned=(0.0, 0.0, 0.0),
        tgo_mode: str = "fixed",
        gamma: float = 1.0,
        tilt_max_deg: float = 15.0,
        ignition_tilt_max_deg: float | None = None,
        margin_max: float = 0.15,
        margin_min: float = 0.10,
        tgo_min_s: float = 2.0,
        tgo_max_s: float = 60.0,
        tgo_grid_s: float = 0.25,
        tgo_freeze_s: float = 0.5,
        tgo_lateral_floor_s: float = 8.0,
        lateral_vnull_tgo_s: float = 2.5,
        lateral_vnull_tau_s: float = 1.5,
        terminal_tilt_max_deg: float = 15.0,
        lateral_lag_tau_s: float = 0.0,
        lateral_upright_terminal: bool = False,
        coast_attitude: str = "upright",
        fixed_tf_s: float | None = None,
        ignition_lookahead_s: float = 0.5,
        log_every: int = 1,
    ):
        if tgo_mode not in ("fixed", "optimal"):
            raise ValueError(f"tgo_mode must be 'fixed' or 'optimal', got {tgo_mode!r}")
        if not (0.0 <= margin_max < 1.0 and margin_min >= 0.0):
            raise ValueError("margins must satisfy 0 <= margin_max < 1 and margin_min >= 0")
        if not (0.0 < tgo_min_s < tgo_max_s and tgo_grid_s > 0.0 and tgo_freeze_s >= 0.0):
            raise ValueError("require 0 < tgo_min_s < tgo_max_s, tgo_grid_s > 0, tgo_freeze_s >= 0")
        self.phase = int(phase)
        self.active = np.asarray(active_engines, float)
        n_act = float(self.active.sum())
        self.n_active = n_act
        self.rho_max = n_act * float(nominal["thrust_vac"])
        self.rho_min = float(nominal["throttle_min"]) * self.rho_max
        self.alpha = 1.0 / (float(nominal["isp"]) * float(nominal["g0"]))  # mdot = alpha |T|
        self.g = np.asarray(nominal.get("gravity_ned", [0.0, 0.0, nominal["g0"]]), dtype=float)
        self.m_dry = float(nominal["m_dry"])
        self.p_sea = float(nominal["p_sea"])
        self.nozzle_exit_area = float(nominal["nozzle_exit_area"])
        self.scale_height = float(nominal["scale_height"])
        self.r_f = np.asarray(target_ned, float)
        self.v_f = np.asarray(target_vel_ned, float)
        self.tgo_mode = tgo_mode
        self.gamma = float(gamma)
        self.tilt_max = float(np.deg2rad(tilt_max_deg))
        # Ignition planning cone: a divert must fit a tighter cone than the attitude cap, which
        # leaves tilt headroom for feedback and lengthens the burn when a lateral divert is needed.
        tp = tilt_max_deg if ignition_tilt_max_deg is None else ignition_tilt_max_deg
        if not 0.0 < tp <= tilt_max_deg:
            raise ValueError("ignition_tilt_max_deg must be in (0, tilt_max_deg]")
        self.tilt_plan = float(np.deg2rad(tp))
        self.margin_max = float(margin_max)
        self.margin_min = float(margin_min)
        self.tgo_min = float(tgo_min_s)
        self.tgo_max = float(tgo_max_s)
        self.tgo_grid = np.arange(self.tgo_min, self.tgo_max + 1e-9, float(tgo_grid_s))
        self.tgo_freeze = float(tgo_freeze_s)
        self.tgo_lat_floor = float(tgo_lateral_floor_s)
        self.lat_tau = float(lateral_vnull_tau_s)
        self.lat_vnull_tgo = float(lateral_vnull_tgo_s)
        self.tilt_terminal = float(np.deg2rad(terminal_tilt_max_deg))
        if lateral_lag_tau_s < 0.0:
            raise ValueError("lateral_lag_tau_s must be >= 0")
        self.lag_tau = float(lateral_lag_tau_s)
        self.lag_upright = bool(lateral_upright_terminal) and self.lag_tau > 0.0
        if coast_attitude not in ("upright", "retrograde"):
            raise ValueError(f"coast_attitude must be 'upright' or 'retrograde', got {coast_attitude!r}")
        self.coast_retrograde = coast_attitude == "retrograde"
        self.fixed_tf = None if fixed_tf_s is None else float(fixed_tf_s)
        self.ignition_lookahead = float(ignition_lookahead_s)
        self.log_every = max(int(log_every), 1)
        self.q_up = upright_quat()

    # -- memory ----------------------------------------------------------------------------
    def initial_memory(self) -> dict:
        return {
            "_zemzev_phase_owner": self.phase,
            "zz_ignited": False,
            "zz_t_ignite": None,
            "zz_tf": None,  # committed terminal epoch [s] (fixed mode / freeze bookkeeping)
            "zz_tf0": None,  # terminal epoch committed at ignition [s]
            "zz_ignition_reason": None,
            "zz_cutoff": False,
            "zz_freeze": None,  # open-loop linear profile latched at freeze entry
            "zz_ticks": 0,
            "zz_sat_ticks": 0,
            "zz_thrust_mag": 0.0,  # last commanded |T| [N] (realized lateral accel estimate)
            "zemzev_log": [],
            "t_ignition": None,
        }

    def _memory(self, memory: dict) -> dict:
        mem = dict(memory)
        if mem.get("_zemzev_phase_owner") != self.phase:
            log = list(mem.get("zemzev_log", []))
            mem.update(self.initial_memory())
            mem["zemzev_log"] = log
        return mem

    # -- vehicle limits --------------------------------------------------------------------
    def _bounds(self, h: float) -> tuple[float, float]:
        """Delivered thrust bounds after the nozzle back-pressure debit at altitude ``h``."""
        p_amb = self.p_sea * np.exp(-max(h, 0.0) / self.scale_height)
        bp = self.n_active * p_amb * self.nozzle_exit_area
        return max(self.rho_min - bp, 1e-3), max(self.rho_max - bp, 1e-3)

    def _accel_bounds(self, h: float, m0: float, tgo: float) -> tuple[float, float]:
        """Conservative thrust-acceleration bounds over a burn of length ``tgo``.

        Max: start mass (acceleration capability only grows as propellant burns).
        Min: worst-case end mass after burning at full thrust (the floor only grows).
        """
        rho_min, rho_max = self._bounds(h)
        m_end = max(m0 - self.alpha * self.rho_max * tgo, self.m_dry)
        return rho_min / m_end, rho_max / m0

    # -- ignition --------------------------------------------------------------------------
    def _candidate_profiles(
        self, r, v, m, h, candidates, margins: bool = True, lateral: bool = True
    ):
        """Vectorized open-loop profiles and margined bounds for every candidate time-to-go.

        ``lateral=False`` keeps only the vertical channel (lateral ZEM/ZEV zeroed): used to time
        the ignition when the lateral divert is beyond the tilt cone and must be best-effort.
        """
        T = np.asarray(candidates, float)
        dr = self.r_f - r
        zem = dr[None, :] - v[None, :] * T[:, None] - 0.5 * self.g[None, :] * (T * T)[:, None]
        zev = (self.v_f - v)[None, :] - self.g[None, :] * T[:, None]
        if not lateral:
            zem[:, :2] = 0.0
            zev[:, :2] = 0.0
        a0 = 6.0 * zem / (T * T)[:, None] - 2.0 * zev / T[:, None]
        af = -6.0 * zem / (T * T)[:, None] + 4.0 * zev / T[:, None]
        rho_min, rho_max = self._bounds(h)
        m_end = np.maximum(m - self.alpha * self.rho_max * T, self.m_dry)
        a_lo = rho_min / m_end
        a_hi = np.full_like(T, rho_max / m)
        if margins:
            a_lo = a_lo * (1.0 + self.margin_min)
            a_hi = a_hi * (1.0 - self.margin_max)
        return T, a0, af, a_lo, a_hi

    def _feasible_tgo(self, r, v, m, h, candidates, lateral: bool = True) -> float | None:
        """Shortest candidate time-to-go whose open-loop profile fits the margined limits."""
        if len(candidates) == 0:
            return None
        T, a0, af, a_lo, a_hi = self._candidate_profiles(r, v, m, h, candidates, lateral=lateral)
        ok, _ = profiles_feasible(a0, af, a_lo, a_hi, self.tilt_plan)
        idx = np.flatnonzero(ok)
        return float(T[idx[0]]) if idx.size else None

    def _urgent(self, r, v, m, h) -> bool:
        """Past the ignition window: vertical stopping distance at full (margined) thrust >= h."""
        rho_min, rho_max = self._bounds(h)
        a_net = rho_max * (1.0 - self.margin_max) / m - float(self.g[2])
        vz = float(v[2] - self.v_f[2])  # NED: positive = descending toward the target
        dz = float(self.r_f[2] - r[2])  # remaining descent [m]
        if vz <= 0.0:
            return False
        return a_net <= 0.0 or vz * vz / (2.0 * a_net) >= dz

    def _best_effort_tgo(self, r, v, m, h) -> float:
        """Late ignition: vertical-channel time-to-go least violating the unmargined limits.

        Past the window the vertical arrest has priority; saturation keeps the up component
        and clips the lateral divert to the tilt cone.
        """
        T, a0, af, a_lo, a_hi = self._candidate_profiles(
            r, v, m, h, self.tgo_grid, margins=False, lateral=False
        )
        p_min, p_max = segment_profile_metrics(a0, af)
        upward = (a0 @ _UP > 0.0) & (af @ _UP > 0.0)
        violation = (
            np.maximum(p_max - a_hi, 0.0)
            + np.maximum(a_lo - p_min, 0.0)
            + np.where(upward, 0.0, 1e3)
        )
        return float(T[int(np.argmin(violation))])

    def _ignition(self, r, v, m, h, t) -> tuple[float | None, str]:
        """Decide whether to ignite now; returns (tgo, reason) or (None, reason) to keep coasting.

        Latest-feasible policy: keep coasting while the state one look-ahead step later (ballistic
        coast) still admits a margined feasible profile; otherwise ignite with the shortest
        feasible time-to-go now. Past the window, ignite with a best-effort time-to-go.
        """
        if self.fixed_tf is not None:
            tgo_fix = self.fixed_tf - t
            if tgo_fix < self.tgo_min:
                return (self._best_effort_tgo(r, v, m, h), "fixed_tf_missed")
            ok = self._feasible_tgo(r, v, m, h, [tgo_fix])
            if ok is not None:
                return ok, "fixed_tf"
            if self._urgent(r, v, m, h):
                return tgo_fix, "fixed_tf_urgent"
            return None, "coast"
        tgo_now = self._feasible_tgo(r, v, m, h, self.tgo_grid)
        if tgo_now is None:
            # The full 3-D divert does not fit the tilt cone. If the vertical channel alone is
            # feasible, ignite NOW (waiting only shrinks the divert authority) with the
            # vertically feasible time-to-go whose 3-D profile violates the cone least; the
            # lateral correction is then flown best-effort under vertical-priority saturation.
            tgo_div = self._divert_tgo(r, v, m, h)
            if tgo_div is not None:
                return tgo_div, "divert_limited"
            if self._urgent(r, v, m, h):
                return self._best_effort_tgo(r, v, m, h), "urgent"
            return None, "coast"
        dt = self.ignition_lookahead
        if dt <= 0.0:
            return tgo_now, "first_feasible"
        r_n = r + v * dt + 0.5 * self.g * dt * dt
        v_n = v + self.g * dt
        if self._feasible_tgo(r_n, v_n, m, -float(r_n[2]), self.tgo_grid) is not None:
            return None, "coast"
        return tgo_now, "latest_feasible"

    def _divert_tgo(self, r, v, m, h) -> float | None:
        """Vertically feasible time-to-go minimizing the 3-D profile's tilt-cone violation."""
        T, a0v, afv, a_lo, a_hi = self._candidate_profiles(r, v, m, h, self.tgo_grid, lateral=False)
        ok, _ = profiles_feasible(a0v, afv, a_lo, a_hi, self.tilt_max)
        if not ok.any():
            return None
        _, a0, af, _, _ = self._candidate_profiles(r, v, m, h, self.tgo_grid)

        def tilt_ratio(a):
            up = np.maximum(a @ _UP, 1e-6)
            return np.linalg.norm(a - up[:, None] * _UP, axis=1) / up

        ratio = np.maximum(tilt_ratio(a0), tilt_ratio(af))
        ratio = np.where(ok, ratio, np.inf)
        return float(T[int(np.argmin(ratio))])

    # -- commands --------------------------------------------------------------------------
    def _engine_off(self, mem, x_hat=None):
        """Engine-off command. Before ignition, ``coast_attitude="retrograde"`` holds the
        engines into the flow (zero angle of attack) instead of the upright attitude."""
        q_ref = self.q_up
        if self.coast_retrograde and x_hat is not None and not mem["zz_cutoff"]:
            v = np.asarray(x_hat.v_ned, float)
            if np.linalg.norm(v) > 1.0:
                q_ref = pointing_from(x_hat.q_bn, -v)
        return attitude_command(self.phase, q_ref, np.zeros_like(self.active)), mem

    def _thrust(self, thrust_ned, mem):
        thrust_ned = np.asarray(thrust_ned, float)
        q = thrust_aligned_quat(thrust_ned, self.tilt_max)
        return thrust_command(self.phase, thrust_ned, q, self.active), mem

    @staticmethod
    def _lateral_accel_estimate(x_hat, m: float, mem) -> np.ndarray:
        """Realized horizontal thrust acceleration: last commanded |T|/m along the ESTIMATED
        body axis (``x_hat.q_bn``). Zero before the first thrust command or without attitude."""
        q = getattr(x_hat, "q_bn", None)
        mag = float(mem.get("zz_thrust_mag", 0.0))
        if q is None or mag <= 0.0:
            return np.zeros(3)
        bx = np.asarray(dcm_from_quat(np.asarray(q, float)))[:, 0]
        a = (mag / m) * bx
        a[2] = 0.0
        return a

    def _log(self, mem, rec):
        mem["zz_ticks"] = int(mem["zz_ticks"]) + 1
        if (int(mem["zz_ticks"]) - 1) % self.log_every == 0:
            mem["zemzev_log"].append(rec)

    def _step(self, x_hat, t, memory):
        t0 = time.perf_counter()
        mem = self._memory(memory)
        r = np.asarray(x_hat.r_ned, float)
        v = np.asarray(x_hat.v_ned, float)
        m = float(np.asarray(x_hat.m))
        h = -float(r[2])
        t = float(t)

        if mem["zz_cutoff"]:
            return self._engine_off(mem)

        if not mem["zz_ignited"]:
            tgo0, reason = self._ignition(r, v, m, h, t)
            if tgo0 is None:
                self._log(mem, self._record(t, "coast", None, None, None, None, None, {}, t0))
                return self._engine_off(mem, x_hat)
            mem.update(
                zz_ignited=True,
                zz_t_ignite=t,
                zz_tf=t + tgo0,
                zz_tf0=t + tgo0,
                zz_ignition_reason=reason,
                t_ignition=t,
            )

        tgo_fixed = float(mem["zz_tf"]) - t
        if tgo_fixed <= 0.0:
            mem["zz_cutoff"] = True
            self._log(mem, self._record(t, "cutoff", 0.0, None, None, None, None, {}, t0))
            return self._engine_off(mem)

        if self.tgo_mode == "fixed" or tgo_fixed <= self.tgo_freeze:
            tgo = tgo_fixed
        else:
            # Re-optimized time-to-go, bounded so the burn cannot be extended indefinitely.
            tgo = optimal_tgo(
                r, v, self.r_f, self.v_f, self.g, self.gamma, self.tgo_freeze, self.tgo_max
            )
            mem["zz_tf"] = t + tgo

        zem, zev = zem_zev(r, v, self.r_f, self.v_f, self.g, tgo)
        if tgo <= self.tgo_freeze:
            # The 1/t^2 gains are singular as t -> 0: latch the open-loop optimal (linear)
            # vertical profile at freeze entry and fly it out without feedback. Lateral thrust
            # is zero in this window so the vehicle touches down upright.
            if mem["zz_freeze"] is None:
                t_fr = max(tgo, 1e-3)
                a0 = zem_zev_accel(zem, zev, t_fr)
                af = -6.0 * zem / t_fr**2 + 4.0 * zev / t_fr
                a0[:2] = 0.0
                af[:2] = 0.0
                mem["zz_freeze"] = {
                    "t_end": t + t_fr,
                    "T": t_fr,
                    "a0": a0.tolist(),
                    "af": af.tolist(),
                }
            fz = mem["zz_freeze"]
            s = float(np.clip(1.0 - (fz["t_end"] - t) / fz["T"], 0.0, 1.0))
            a_cmd = (1.0 - s) * np.asarray(fz["a0"]) + s * np.asarray(fz["af"])
            mode = "freeze"
        else:
            a_cmd = zem_zev_accel(zem, zev, tgo)
            mode = "burn"
            # Lateral thrust is produced by tilting the body through the attitude loop (settling
            # ~4/(zeta wn) s). Keep the lateral loop well below that bandwidth: floor the lateral
            # time-to-go, which caps the 6/t^2 and 2/t gains.
            t_lat = max(tgo, self.tgo_lat_floor)
            if self.lag_tau > 0.0:
                # Lag-aware lateral channel: the tilt (and so the lateral acceleration) follows
                # the command with time constant tau. Plan from the acceleration already
                # realized, estimated from the navigation attitude and the commanded |T|.
                a_act = self._lateral_accel_estimate(x_hat, m, mem)
                zem_l, zev_l = zem_zev_lag(r, v, a_act, self.r_f, self.v_f, self.g, t_lat, self.lag_tau)
                # Optional ZEA: zero lateral acceleration (= upright attitude) at touchdown.
                zea = -a_act * np.exp(-t_lat / self.lag_tau) if self.lag_upright else None
                a_cmd[:2] = zem_zev_lag_accel(zem_l, zev_l, t_lat, self.lag_tau, zea)[:2]
                mode = "burn_lag"
            elif tgo < self.tgo_lat_floor:
                zem_l, zev_l = zem_zev(r, v, self.r_f, self.v_f, self.g, t_lat)
                a_cmd[:2] = zem_zev_accel(zem_l, zev_l, t_lat)[:2]
                mode = "burn_lat_capped"
            if tgo < self.lat_vnull_tgo:
                # Endgame: ZEV-only lateral velocity nulling inside a small terminal tilt cone, so
                # the slow attitude loop is already near upright at contact. A residual position
                # error is accepted; lateral velocity and tilt at contact are not.
                a_lat = zev[:2] / self.lat_tau if self.lat_tau > 0.0 else np.zeros(2)
                cap = max(-float(a_cmd[2]), 0.0) * np.tan(self.tilt_terminal)
                n_lat = float(np.linalg.norm(a_lat))
                if n_lat > cap:
                    a_lat = a_lat * (cap / n_lat)
                a_cmd[:2] = a_lat
                mode = "burn_lat_vnull"
        rho_min, rho_max = self._bounds(h)
        thrust, flags = project_achievable(m * a_cmd, rho_min, rho_max, self.tilt_max)
        if any(flags.values()):
            mem["zz_sat_ticks"] = int(mem["zz_sat_ticks"]) + 1
        mem["zz_thrust_mag"] = float(np.linalg.norm(thrust))
        self._log(mem, self._record(t, mode, tgo, zem, zev, a_cmd, thrust / m, flags, t0))
        return self._thrust(thrust, mem)

    @staticmethod
    def _record(t, mode, tgo, zem, zev, a_cmd, a_sat, flags, t0):
        def vec(x):
            return None if x is None else [float(c) for c in np.asarray(x, float)]

        return {
            "t": float(t),
            "mode": mode,
            "tgo": None if tgo is None else float(tgo),
            "zem": vec(zem),
            "zev": vec(zev),
            "a_cmd": vec(a_cmd),
            "a_sat": vec(a_sat),
            **{k: bool(val) for k, val in flags.items()},
            "wall_ms": (time.perf_counter() - t0) * 1e3,
        }

    def __call__(self, x_hat, t, memory):
        """Guidance tick (slow rate)."""
        return self._step(x_hat, t, memory)

    def track(self, x_hat, t, memory):
        """Fast control-rate tick: the feedback law is cheap, so re-evaluate it every tick."""
        return self._step(x_hat, t, memory)


def build(ctx):
    """Factory named by configs/gnc/guidance/zemzev*.yaml (ctx: gnc.context.GncContext)."""
    gd = ctx.cfg
    return ZemZevGuidance(
        ctx.nominal,
        ctx.active_engines,
        ctx.phase,
        target_ned=tuple(float(x) for x in gd.zemzev_target_ned),
        target_vel_ned=tuple(float(x) for x in gd.zemzev_target_vel_ned),
        tgo_mode=str(gd.zemzev_tgo_mode),
        gamma=float(gd.zemzev_gamma),
        tilt_max_deg=float(gd.attitude_tilt_max_deg),
        ignition_tilt_max_deg=(
            None
            if gd.get("zemzev_ignition_tilt_max_deg") is None
            else float(gd.zemzev_ignition_tilt_max_deg)
        ),
        margin_max=float(gd.zemzev_margin_max),
        margin_min=float(gd.zemzev_margin_min),
        tgo_min_s=float(gd.zemzev_tgo_min_s),
        tgo_max_s=float(gd.zemzev_tgo_max_s),
        tgo_grid_s=float(gd.zemzev_tgo_grid_s),
        tgo_freeze_s=float(gd.zemzev_tgo_freeze_s),
        tgo_lateral_floor_s=float(gd.zemzev_tgo_lateral_floor_s),
        lateral_vnull_tgo_s=float(gd.zemzev_lateral_vnull_tgo_s),
        lateral_vnull_tau_s=float(gd.zemzev_lateral_vnull_tau_s),
        terminal_tilt_max_deg=float(gd.zemzev_terminal_tilt_max_deg),
        lateral_lag_tau_s=float(gd.get("zemzev_lateral_lag_tau_s", 0.0)),
        lateral_upright_terminal=bool(gd.get("zemzev_lateral_upright_terminal", False)),
        coast_attitude=str(gd.get("zemzev_coast_attitude", "upright")),
        fixed_tf_s=None if gd.zemzev_fixed_tf_s is None else float(gd.zemzev_fixed_tf_s),
        ignition_lookahead_s=float(gd.zemzev_ignition_lookahead_s),
        log_every=int(gd.zemzev_log_every),
    )
