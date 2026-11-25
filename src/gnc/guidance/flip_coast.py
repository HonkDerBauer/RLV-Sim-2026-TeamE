"""Post-separation coast: engine off, RCS flip to the engine-first (retrograde) attitude.

After stage separation the booster flies nose-first. Before the entry burn it must turn so the
engines face the velocity vector. This law commands that attitude while coasting (body +x along
-v_hat: the nose points away from the direction of travel), with all engines off. On the
ascending branch the retrograde direction points below the horizon, so the reference is not
capped to the upper hemisphere.

The cold-gas RCS is weak, so a step command of ~150 deg would saturate it, spin the booster up and
waste gas braking. The reference is therefore a "carrot" at most ``step_deg`` ahead of the
current (estimated) body axis; with the PD loop this gives a bounded, roughly constant flip rate.
"""

from __future__ import annotations

import numpy as np

from core.frames import dcm_from_quat, quat_from_axis_angle, quat_mul
from core.interfaces import Phase
from gnc.guidance.base import attitude_command, upright_quat

_UP = np.array([0.0, 0.0, -1.0])  # NED up
_NORTH = np.array([1.0, 0.0, 0.0])


def pointing_quat(direction):
    """Body-to-NED quaternion turning body +x from NED-up onto ``direction`` (shortest rotation).

    Valid over the whole sphere (unlike :func:`gnc.guidance.base.thrust_aligned_quat`, which is
    capped to the upper hemisphere). Straight down uses a rotation about north.
    """
    d = np.asarray(direction, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return upright_quat()
    d = d / n
    c = float(np.clip(d @ _UP, -1.0, 1.0))
    axis = np.cross(_UP, d)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        if c > 0.0:
            return upright_quat()
        axis, s = _NORTH, 1.0
    return quat_mul(quat_from_axis_angle(axis / s, float(np.arctan2(s, c))), upright_quat())


def pointing_from(q_now, direction, max_angle: float | None = None):
    """Reference attitude rotating the CURRENT body +x toward ``direction`` (NED) by the shortest
    rotation, optionally by at most ``max_angle`` [rad]. Keeps the current roll about body +x.
    """
    q_now = np.asarray(q_now, float)
    C = np.asarray(dcm_from_quat(q_now))
    bx = C[:, 0]
    d = np.asarray(direction, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return q_now
    d = d / n
    axis = np.cross(bx, d)
    s = float(np.linalg.norm(axis))
    angle = float(np.arctan2(s, float(bx @ d)))
    if s < 1e-9:
        if bx @ d > 0.0:
            return q_now
        axis, s = C[:, 2], 1.0  # antiparallel: rotate about body +z
    if max_angle is not None:
        angle = min(angle, float(max_angle))
    return quat_mul(quat_from_axis_angle(axis / s, angle), q_now)


class FlipCoastGuidance:
    """Engine-off coast holding the retrograde (engine-first) attitude via a rate-profiled flip.

    Far from the target the law commands a body RATE (q_ref = current attitude, so the PD loop
    becomes a rate loop): ``omega = min(rate_max, sqrt(2 * accel * theta))`` about the shortest
    rotation axis, i.e. spin up, coast at ``rate_max`` and brake so as to arrive at rest. Cold-gas
    cost scales with the rate (spin-up + braking impulse), so a low ``rate_max`` saves gas. Within
    ``capture_deg`` the law hands over to plain attitude hold on the retrograde direction.
    """

    def __init__(
        self,
        active_engines,
        rate_max_deg_s: float = 0.7,
        accel_deg_s2: float = 0.03,
        capture_deg: float = 3.0,
    ):
        if not (rate_max_deg_s > 0.0 and accel_deg_s2 > 0.0 and capture_deg > 0.0):
            raise ValueError("flip rate, accel and capture angle must be positive")
        self.active = np.zeros_like(np.asarray(active_engines, float))
        self.rate_max = float(np.deg2rad(rate_max_deg_s))
        self.accel = float(np.deg2rad(accel_deg_s2))
        self.capture = float(np.deg2rad(capture_deg))

    def __call__(self, x_hat, t, memory):
        v = np.asarray(x_hat.v_ned, float)
        q = np.asarray(x_hat.q_bn, float)
        if np.linalg.norm(v) <= 1.0:
            return attitude_command(int(Phase.FLIP_COAST), q, self.active), memory
        d = -v / np.linalg.norm(v)
        C = np.asarray(dcm_from_quat(q))
        bx = C[:, 0]
        axis = np.cross(bx, d)
        s = float(np.linalg.norm(axis))
        theta = float(np.arctan2(s, float(bx @ d)))
        if theta <= self.capture:
            q_ref = pointing_from(q, d)
            return attitude_command(int(Phase.FLIP_COAST), q_ref, self.active), memory
        if s < 1e-9:  # antiparallel: rotate about body +z
            axis, s = C[:, 2], 1.0
        rate = min(self.rate_max, float(np.sqrt(2.0 * self.accel * theta)))
        w_ref = C.T @ (rate * axis / s)  # body frame
        return attitude_command(int(Phase.FLIP_COAST), q, self.active, w_ref=w_ref), memory
