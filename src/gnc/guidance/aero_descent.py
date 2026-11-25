"""Aero-descent attitude guidance with engines off."""

from __future__ import annotations

import numpy as np

from core.frames import quat_from_euler
from core.interfaces import Phase
from gnc.guidance.base import attitude_command, upright_quat
from gnc.guidance.flip_coast import pointing_from


class AeroDescentGuidance:
    """Engine-off descent attitude.

    ``attitude="upright"`` (default) holds body-x vertical, optionally steered by the crude
    horizontal-velocity law. ``attitude="retrograde"`` holds body-x along -v_hat (zero angle of
    attack, engines into the flow): the natural attitude on a slanted post-entry trajectory.
    """

    def __init__(
        self, active_engines, steer_gain: float, max_tilt_deg: float, attitude: str = "upright"
    ):
        if attitude not in ("upright", "retrograde"):
            raise ValueError(f"attitude must be 'upright' or 'retrograde', got {attitude!r}")
        self.active = active_engines  # all-off (engine off in the aero phase)
        self.q_up = upright_quat()
        self.steer_gain = steer_gain
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.attitude = attitude

    def __call__(self, x_hat, t, memory):
        if self.attitude == "retrograde":
            v = np.asarray(x_hat.v_ned, float)
            q_ref = pointing_from(x_hat.q_bn, -v) if np.linalg.norm(v) > 1.0 else self.q_up
            return attitude_command(int(Phase.AERO_DESCENT), q_ref, self.active), memory
        if self.steer_gain == 0.0:
            return attitude_command(int(Phase.AERO_DESCENT), self.q_up, self.active), memory
        # crude steer: tilt nose to oppose horizontal velocity (north->pitch, east->roll proxy)
        vh = x_hat.v_ned[:2]
        tilt = np.clip(-self.steer_gain * vh, -self.max_tilt, self.max_tilt)
        q_ref = quat_from_euler(tilt[1], np.pi / 2 + tilt[0], 0.0)
        return attitude_command(int(Phase.AERO_DESCENT), q_ref, self.active), memory
