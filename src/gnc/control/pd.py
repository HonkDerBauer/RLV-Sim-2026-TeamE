"""PD attitude controller that fills Command.tau_body."""

from __future__ import annotations

import numpy as np

from core.frames import dcm_from_quat, quat_conj, quat_mul


def attitude_error(q_hat, q_ref):
    """Body-frame attitude error vector (shortest rotation q_hat -> q_ref)."""
    qe = quat_mul(quat_conj(q_hat), q_ref)
    return np.sign(qe[0] + 1e-12) * qe[1:4]


def pointing_error(q_hat, q_ref):
    """Body-frame error of the roll axis only: shortest rotation of body +x onto the reference
    body +x, as ``sin(theta/2) * axis`` (same scaling as :func:`attitude_error`).

    The roll component is identically zero, so an axisymmetric vehicle is not rolled to match the
    reference's (arbitrary) roll angle. Antiparallel axes rotate about body +z.
    """
    d = np.asarray(dcm_from_quat(q_hat)).T @ np.asarray(dcm_from_quat(q_ref))[:, 0]
    axis = np.array([0.0, -d[2], d[1]])  # cross([1, 0, 0], d)
    s = float(np.linalg.norm(axis))
    theta = float(np.arctan2(s, d[0]))
    if s < 1e-9:
        return np.zeros(3) if d[0] > 0.0 else np.array([0.0, 0.0, 1.0])
    return np.sin(0.5 * theta) * axis / s


class PDController:
    def __init__(self, kp_att, kd_att, roll_free: bool = False):
        """``kp_att``/``kd_att`` are scalars or per-axis [roll, pitch, yaw] gains (broadcast).

        ``roll_free``: control only the pointing of body +x (roll angle free, roll rate damped).
        """
        self.kp = np.asarray(kp_att)
        self.kd = np.asarray(kd_att)
        self.roll_free = bool(roll_free)

    def initial_memory(self) -> dict:
        return {}

    def __call__(self, x_hat, command, memory):
        err = pointing_error if self.roll_free else attitude_error
        e = err(x_hat.q_bn, command.q_ref)
        # Rate FEEDFORWARD: damp the rate ERROR (w - w_ref), so a rotating attitude reference is
        # tracked instead of fought. w_ref = 0 (hold) recovers the plain regulator.
        tau = self.kp * e - self.kd * (x_hat.w_body - command.w_ref)  # elementwise per-axis
        return command.replace(tau_body=tau), memory


def build(ctx):
    """Factory named by configs/gnc/control/pd.yaml (ctx: gnc.context.GncContext).

    Sizes per-axis gains from the config (wn_axis, zeta) and the wet inertia:
    Kp = I_wet * wn^2, Kd = 2 * zeta * I_wet * wn."""
    I = np.asarray(ctx.params.mass.I_wet)  # [Ixx, Iyy, Izz]
    wn = np.asarray(list(ctx.cfg.wn_axis), dtype=float)
    zeta = float(ctx.cfg.zeta)
    return PDController(
        I * wn**2, 2.0 * zeta * I * wn, roll_free=bool(ctx.cfg.get("roll_free", False))
    )
