"""Falcon-9-style full-mission guidance: ascent -> boostback -> landing (stage-1 only).

Copy to ``gnc/guidance/falcon9_mission.py`` and ``falcon9_mission.yaml`` to
``configs/gnc/guidance/``; run with::

    python scripts/run_single.py scenario=<f9_mission> gnc/guidance=falcon9_mission

How it plugs in
---------------
The framework only knows ONE configurable "powered" slot (``law._target_`` -> ``build(ctx)``),
wrapped by ``_PoweredCommand``. This law fills that slot but runs the WHOLE mission itself with
its own latching phase machine, so the scenario must start in the powered phase
(``scenario.start_phase=powered``, ``powered_brake=false``): the stock scheduler then never
leaves that slot. A full ``Command`` returned by the law passes through the adapter unchanged,
so each phase chooses its own engine set (the stock scheduler would only give us the landing set).

Phase sequence (latching, forward only)::

    ASCENT -> MECO_COAST -> FLIP -> BOOSTBACK -> COAST -> ENTRY_BURN -> AERO_DESCENT
           -> LANDING_BURN -> LANDED

Reused stock laws: ``ReentryBurnGuidance`` (entry burn), ``AeroDescentGuidance`` (coast / aero).

KNOWN SIMPLIFICATIONS (this is a template, not a flight-validated design):
  * Ascent is an open-loop pitch kick followed by a zero-AoA gravity turn; no closed-loop
    orbital targeting. Tune ``kick_*`` / ``meco_*`` until MECO conditions look right.
  * The boostback predictor is the constant-gravity, no-drag ballistic projection also used in
    boostback.py -- a targeting diagnostic, not a landing predictor.
  * The landing burn is a constant-deceleration vertical profile + horizontal PD, not
    fuel-optimal (see the convex laws for that).
  * Stage 2 / payload are not simulated: model them as mass in the plant params and handle
    separation as a mass step in the scenario if you need it.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation

from core.interfaces import Phase
from gnc.guidance.aero_descent import AeroDescentGuidance
from gnc.guidance.base import (
    attitude_command,
    thrust_aligned_quat,
    thrust_command,
    upright_quat,
)
from gnc.guidance.reentry_burn import ReentryBurnGuidance

_UP = np.array([0.0, 0.0, -1.0])  # NED up


class MissionPhase(IntEnum):
    # New tags (20+) avoid clashing with core.interfaces.Phase and the 10-17 boostback tags.
    # ENTRY_BURN / AERO_DESCENT / LANDING_BURN commands carry the stock Phase tags instead.
    ASCENT = 20
    MECO_COAST = 21
    FLIP = 22
    BOOSTBACK = 23
    COAST = 24
    ENTRY_BURN = 25
    AERO_DESCENT = 26
    LANDING_BURN = 27
    LANDED = 28


ORDER = list(MissionPhase)


def _opt(value):
    return None if value is None else float(value)


class Falcon9Mission:
    def __init__(self, nominal: dict, params, mission, tilt_max_rad: float,
                 aero_gain: float, aero_tilt_deg: float):
        m = mission
        positions = np.asarray(params.engines.positions, dtype=float)
        self.n_eng = len(positions)
        self.m_asc = self._mask("ascent", m.ascent_engines, positions)
        self.m_boost = self._mask("boostback", m.boostback_engines, positions)
        self.m_entry = self._mask("entry", m.entry_engines, positions)
        self.m_land = self._mask("landing", m.landing_engines, positions)
        self.m_off = np.zeros(self.n_eng)

        # Vehicle constants (from gnc.build.default_nominal).
        self.t1 = float(nominal["thrust_vac"])
        self.thr_min = float(nominal["throttle_min"])
        self.p_sea = float(nominal["p_sea"])
        self.a_e = float(nominal["nozzle_exit_area"])
        self.rho0 = float(nominal["rho0"])
        self.atm = nominal["atmosphere"]  # (rho, a_sound)(h)
        self.m_dry = float(nominal["m_dry"])
        self.g_ned = np.asarray(nominal["gravity_ned"], dtype=float)
        self.g_mag = float(np.linalg.norm(self.g_ned))
        self.g0 = float(nominal["g0"])

        self.tilt_max = float(tilt_max_rad)
        self.boost_tilt_max = np.deg2rad(float(m.boostback_tilt_max_deg))

        # Ascent
        self.azimuth_deg = float(m.launch_azimuth_deg)
        self.t_kick = float(m.t_kick_s)
        self.kick_dur = float(m.kick_duration_s)
        self.kick_angle_deg = float(m.kick_angle_deg)
        self.v_turn_min = float(m.v_turn_min_mps)
        self.maxq_thr = float(m.maxq_throttle)
        self.maxq_on = float(m.maxq_on_pa)
        self.maxq_off = float(m.maxq_off_pa)
        self.accel_limit_g = float(m.accel_limit_g)
        self.meco_prop = _opt(m.meco_propellant_kg)
        self.meco_speed = _opt(m.meco_speed_mps)
        if self.meco_prop is None and self.meco_speed is None:
            raise ValueError("set meco_propellant_kg and/or meco_speed_mps")
        self.sep_delay = float(m.separation_delay_s)

        # Flip + boostback
        self.flip_timeout = float(m.flip_timeout_s)
        self.flip_att_tol = np.deg2rad(float(m.flip_att_tol_deg))
        self.flip_rate_tol = float(m.flip_rate_tol_rad_s)
        self.boost_elev = np.deg2rad(float(m.boostback_elevation_deg))
        if np.pi / 2 - self.boost_elev > self.boost_tilt_max:
            raise ValueError("boostback_elevation_deg too low for boostback_tilt_max_deg")
        self.boost_throttle = float(m.boostback_throttle)
        self.boost_timeout = float(m.boostback_burn_timeout_s)
        self.boost_min_prop = float(m.boostback_min_propellant_kg)
        self.target = np.asarray([float(c) for c in m.target_ne_m], dtype=float)
        self.proj_alt = float(m.projection_altitude_m)
        self.proj_tol = float(m.projection_tol_m)

        # Entry burn / aero
        self.entry_h = float(m.entry_altitude_m)
        self.entry_end_h = float(m.entry_end_altitude_m)
        self.entry_end_v = float(m.entry_end_speed_mps)
        self._entry = ReentryBurnGuidance(float(self.m_entry.sum()) * self.t1, self.m_entry)
        self._aero = AeroDescentGuidance(self.m_off, aero_gain, aero_tilt_deg)

        # Landing
        self.margin = float(m.landing_margin)
        self.design_thr = float(m.landing_design_throttle)
        self.a_dec = float(m.landing_decel_mps2)
        self.v_td = float(m.touchdown_speed_mps)
        self.h_cut = float(m.cutoff_altitude_m)
        self.h_backstop = float(m.landing_backstop_altitude_m)
        self.kv = float(m.landing_kv)
        wn, zeta = float(m.landing_wn_rad_s), float(m.landing_zeta)
        self.kp, self.kd = wn**2, 2.0 * zeta * wn
        self.land_tilt = np.deg2rad(float(m.landing_tilt_max_deg))

        self._guards = [
            self._g_meco,
            self._g_separated,
            self._g_flip_done,
            self._g_boostback_cut,
            self._g_entry_start,
            self._g_entry_end,
            self._g_landing_start,
            self._g_touchdown,
        ]
        self._laws = {
            MissionPhase.ASCENT: self._ascent,
            MissionPhase.MECO_COAST: self._meco_coast,
            MissionPhase.FLIP: self._flip,
            MissionPhase.BOOSTBACK: self._boostback,
            MissionPhase.COAST: self._aero_law,
            MissionPhase.ENTRY_BURN: self._entry_law,
            MissionPhase.AERO_DESCENT: self._aero_law,
            MissionPhase.LANDING_BURN: self._landing,
            MissionPhase.LANDED: self._landed,
        }
        assert len(self._guards) == len(ORDER) - 1

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mask(name, idx, positions):
        n = len(positions)
        idx = [int(i) for i in idx]
        if not idx or len(set(idx)) != len(idx) or any(not 0 <= i < n for i in idx):
            raise ValueError(f"{name}: engine indices must be unique and in [0, {n})")
        mask = np.zeros(n)
        mask[idx] = 1.0
        if not np.allclose(positions[idx, 1:].sum(axis=0), 0.0, atol=1e-6):
            raise ValueError(f"{name}: engine set must be laterally symmetric (no net torque)")
        return mask

    def _net(self, mask, throttle, h):
        """Net thrust [N] of an engine set: vacuum thrust x throttle minus the p_amb*A_e debit.
        (Check the convention against gnc/allocation before trusting absolute numbers.)"""
        n = float(mask.sum())
        rho, _ = self.atm(h)
        p_amb = self.p_sea * rho / self.rho0 if self.rho0 > 0 else 0.0
        return max(n * self.t1 * throttle - n * p_amb * self.a_e, 1.0)

    def _projection(self, x):
        """Constant-gravity, no-drag ballistic projection to ``projection_altitude_m`` (descending)."""
        g = float(self.g_ned[2])
        r, v = np.asarray(x.r_ned, float), np.asarray(x.v_ned, float)
        disc = v[2] ** 2 + 2 * g * (-self.proj_alt - r[2])
        if g <= 0 or disc <= 0:
            return None
        tof = (-v[2] + np.sqrt(disc)) / g
        if tof <= 0:
            return None
        return float(tof), r[:2] + tof * v[:2]

    def _boostback_dir(self, x):
        proj = self._projection(x)
        if proj is None:
            return None
        tof, _ = proj
        v_des = (self.target - np.asarray(x.r_ned, float)[:2]) / tof
        dv = v_des - np.asarray(x.v_ned, float)[:2]
        n = float(np.linalg.norm(dv))
        if n < 1e-6:
            return None
        hdir = dv / n
        c, s = np.cos(self.boost_elev), np.sin(self.boost_elev)
        return np.array([hdir[0] * c, hdir[1] * c, -s])  # slightly upward, horizontal part toward the pad

    def _cmd(self, phase, vec, mask, tilt_cap):
        return thrust_command(int(phase), vec, thrust_aligned_quat(vec, tilt_cap), mask)

    # ------------------------------------------------------------------ guards
    def _g_meco(self, x, t, mem):
        if self.meco_prop is not None and float(x.m) <= self.m_dry + self.meco_prop:
            return True
        return self.meco_speed is not None and float(np.linalg.norm(x.v_ned)) >= self.meco_speed

    def _g_separated(self, x, t, mem):
        return t - mem["f9_t0"] >= self.sep_delay

    def _g_flip_done(self, x, t, mem):
        if t - mem["f9_t0"] >= self.flip_timeout:
            return True
        d = self._boostback_dir(x)
        if d is None:
            return False
        bx = Rotation.from_quat(np.asarray(x.q_bn, float), scalar_first=True).apply([1.0, 0.0, 0.0])
        err = float(np.arccos(np.clip(bx @ d, -1.0, 1.0)))
        return err <= self.flip_att_tol and float(np.linalg.norm(x.w_body)) <= self.flip_rate_tol

    def _g_boostback_cut(self, x, t, mem):
        proj = self._projection(x)
        if proj is None:
            return True
        miss = float(np.linalg.norm(proj[1] - self.target))
        return (
            miss <= self.proj_tol
            or float(x.m) <= self.m_dry + self.boost_min_prop
            or t - mem["f9_t0"] >= self.boost_timeout
        )

    def _g_entry_start(self, x, t, mem):
        return float(x.v_ned[2]) > 0.0 and float(-x.r_ned[2]) <= self.entry_h

    def _g_entry_end(self, x, t, mem):
        return (
            float(-x.r_ned[2]) <= self.entry_end_h
            or float(np.linalg.norm(x.v_ned)) <= self.entry_end_v
        )

    def _g_landing_start(self, x, t, mem):
        h, vz = float(-x.r_ned[2]), float(x.v_ned[2])
        if vz <= 0.0:
            return False
        if h <= self.h_backstop:
            return True
        _, t_max = self._thrust_range(self.m_land, h)
        a_net = max(self.design_thr * t_max / float(x.m) - self.g_mag, 0.5)
        return h <= self.margin * vz**2 / (2.0 * a_net)  # stopping distance + margin

    def _g_touchdown(self, x, t, mem):
        return float(-x.r_ned[2]) <= self.h_cut or float(x.v_ned[2]) < -0.5  # low, or climbing

    def _thrust_range(self, mask, h):
        n = float(mask.sum())
        t_max = self._net(mask, 1.0, h)
        t_min = max(self._net(mask, self.thr_min, h), 0.0)
        return t_min, max(t_max, t_min + 1.0) if n else (0.0, 1.0)

    # ------------------------------------------------------------------ phase laws
    def _ascent(self, x, t, mem):
        ta = t - mem["f9_t0"]
        h = float(-x.r_ned[2])
        v = np.asarray(x.v_ned, float)
        speed = float(np.linalg.norm(v))
        az = np.deg2rad(self.azimuth_deg)
        horiz = np.array([np.cos(az), np.sin(az), 0.0])
        if ta < self.t_kick:
            d = _UP
        else:
            frac = min((ta - self.t_kick) / self.kick_dur, 1.0)
            tilt = np.deg2rad(self.kick_angle_deg) * frac
            d = np.cos(tilt) * _UP + np.sin(tilt) * horiz
            if frac >= 1.0 and speed > self.v_turn_min and -v[2] > 0.0:
                d = v / speed  # zero-AoA gravity turn

        rho, _ = self.atm(h)
        qbar = 0.5 * rho * speed**2
        if qbar >= self.maxq_on:
            mem["f9_maxq"] = True
        elif qbar <= self.maxq_off:
            mem["f9_maxq"] = False
        thr = self.maxq_thr if mem.get("f9_maxq") else 1.0
        n_t1 = float(self.m_asc.sum()) * self.t1
        thr = min(thr, self.accel_limit_g * self.g0 * float(x.m) / n_t1)  # late-burn g limit
        thr = float(np.clip(thr, self.thr_min, 1.0))

        vec = d * self._net(self.m_asc, thr, h)
        cmd = self._cmd(MissionPhase.ASCENT, vec, self.m_asc, self.tilt_max)
        mem["f9_q_hold"] = cmd.q_ref
        return cmd

    def _meco_coast(self, x, t, mem):
        return attitude_command(int(MissionPhase.MECO_COAST), mem["f9_q_hold"], self.m_off)

    def _flip(self, x, t, mem):
        d = self._boostback_dir(x)
        q = mem["f9_q_hold"] if d is None else thrust_aligned_quat(d, self.boost_tilt_max)
        mem["f9_q_hold"] = q
        return attitude_command(int(MissionPhase.FLIP), q, self.m_off)

    def _boostback(self, x, t, mem):
        d = self._boostback_dir(x)
        if d is None:
            return attitude_command(int(MissionPhase.BOOSTBACK), mem["f9_q_hold"], self.m_off)
        h = float(-x.r_ned[2])
        vec = d * self._net(self.m_boost, self.boost_throttle, h)
        cmd = self._cmd(MissionPhase.BOOSTBACK, vec, self.m_boost, self.boost_tilt_max)
        mem["f9_q_hold"] = cmd.q_ref
        return cmd

    def _aero_law(self, x, t, mem):
        return self._aero(x, t, {})[0]  # stateless, engines off

    def _entry_law(self, x, t, mem):
        return self._entry(x, t, {})[0]  # stateless, full retro thrust

    def _landing(self, x, t, mem):
        h = float(-x.r_ned[2])
        v = np.asarray(x.v_ned, float)
        m = float(x.m)
        e_h = np.asarray(x.r_ned, float)[:2] - self.target
        vz_ref = np.sqrt(self.v_td**2 + 2.0 * self.a_dec * max(h - self.h_cut, 0.0))  # +down
        a_cmd = np.array(
            [*(-self.kp * e_h - self.kd * v[:2]), self.kv * (vz_ref - v[2])]
        )
        thrust = m * (a_cmd - self.g_ned)  # T = m (a - g)
        thrust[2] = min(thrust[2], -1.0)  # always push upward
        lat_cap = np.tan(self.land_tilt) * abs(thrust[2])
        lat = float(np.linalg.norm(thrust[:2]))
        if lat > lat_cap:
            thrust[:2] *= lat_cap / lat
        t_min, t_max = self._thrust_range(self.m_land, h)
        mag = float(np.linalg.norm(thrust))
        thrust *= float(np.clip(mag, t_min, t_max)) / max(mag, 1e-9)
        return self._cmd(Phase.POWERED_DESCENT, thrust, self.m_land, self.land_tilt)

    def _landed(self, x, t, mem):
        return attitude_command(int(MissionPhase.LANDED), upright_quat(), self.m_off)

    # ------------------------------------------------------------------ contract
    def initial_memory(self) -> dict:
        # Keys are prefixed: ScheduledGuidance merges every phase law's memory into one dict.
        return {"f9_idx": 0, "f9_t0": None, "f9_q_hold": upright_quat(), "f9_maxq": False}

    def __call__(self, x_hat, t, memory):
        mem = dict(memory)
        if "f9_idx" not in mem:
            mem.update(self.initial_memory())
        if mem["f9_t0"] is None:
            mem["f9_t0"] = float(t)
        k = int(mem["f9_idx"])
        while k < len(self._guards) and self._guards[k](x_hat, t, mem):
            k += 1  # latching: forward only
            mem["f9_idx"], mem["f9_t0"] = k, float(t)
        mem["f9_idx"] = k
        cmd = self._laws[ORDER[k]](x_hat, t, mem)
        return cmd, mem


def build(ctx):
    """Factory named by configs/gnc/guidance/falcon9_mission.yaml (ctx: gnc.context.GncContext)."""
    cfg = ctx.cfg
    return Falcon9Mission(
        ctx.nominal,
        ctx.params,
        cfg.mission,
        tilt_max_rad=float(np.deg2rad(float(cfg.attitude_tilt_max_deg))),
        aero_gain=float(cfg.aero_steer_gain),
        aero_tilt_deg=float(cfg.aero_tilt_max_deg),
    )
