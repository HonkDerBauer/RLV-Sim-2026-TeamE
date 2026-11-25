"""Build and run a configured episode without prescribing a particular GNC method."""

from omegaconf import open_dict

from plant.dynamics import build_plant, ic_from_scenario, initial_state
from sim.loop import run_simulation
from sim.open_loop import PrescribedInputs


def run_configured(cfg, *, progress=False, record_telemetry=True):
    bundle = build_plant(cfg)
    state = initial_state(ic_from_scenario(cfg.scenario, cfg.seed), bundle.params)
    h = cfg.scenario.horizon
    mode = str(cfg.scenario.get("input_mode", "closed_loop"))
    common = dict(
        t_max=float(h.t_max),
        plant_dt=float(cfg.plant_dt),
        log_dt=float(h.log_dt),
        progress=progress,
        record_telemetry=record_telemetry,
    )
    if mode in ("prescribed", "mission"):
        from plant.separation import Separation

        separation = Separation(cfg) if cfg.scenario.get("separation") is not None else None
        inputs = PrescribedInputs(
            cfg.scenario.inputs, bundle.params, float(h.t_max), separation=separation
        )
        mission = None
        if mode == "mission":
            from gnc.guidance.boostback import BoostbackMission

            if separation is None:
                raise ValueError("mission requires separation")
            if max(inputs.rows) != next(k for k, row in inputs.rows.items() if row["separate"]):
                raise ValueError("mission separation must be the last prescribed event")
            if inputs.rows[max(inputs.rows)]["fin_deployment"] != 0:
                raise ValueError("mission requires stowed fins at separation")
            if cfg.plant.actuator.model != "realistic":
                raise ValueError("mission requires realistic actuation")
            mission = BoostbackMission(cfg.scenario.mission, separation.bundle, float(h.input_dt))
        result = run_simulation(
            bundle,
            state,
            None,
            None,
            ctrl_dt=float(h.input_dt),
            prescribed=inputs,
            mission=mission,
            **common,
        )
    elif mode == "launch_to_landing":
        from plant.separation import Separation

        if cfg.scenario.get("separation") is None:
            raise ValueError("launch_to_landing requires a separation")
        separation = Separation(cfg)
        inputs = PrescribedInputs(
            cfg.scenario.inputs, bundle.params, float(h.t_max), separation=separation
        )
        last = inputs.rows[max(inputs.rows)]
        if not last["separate"]:
            raise ValueError("launch_to_landing: separation must be the last prescribed event")
        result = run_simulation(
            bundle,
            state,
            None,
            None,
            ctrl_dt=float(h.ctrl_dt),
            nav_dt=h.nav_dt,
            guidance_dt=h.guidance_dt,
            delay_ticks=cfg.gnc.delay_ticks,
            success_pos_tol=cfg.scenario.success.pos_tol,
            success_speed_tol=cfg.scenario.success.speed_tol,
            success_tilt_tol_deg=cfg.scenario.success.tilt_tol_deg,
            prescribed=inputs,
            handoff=GncHandoff(separation),
            **common,
        )
    elif mode == "closed_loop":
        if cfg.scenario.get("separation") is not None:
            raise ValueError("stage separation currently requires prescribed execution")
        # Prescribed runs have no dependency on constructing GNC or a navigation module.
        from gnc.build import build_gnc, build_nav

        result = run_simulation(
            bundle,
            state,
            build_gnc(cfg, bundle.params),
            build_nav(cfg, state, float(h.nav_dt)),
            ctrl_dt=float(h.ctrl_dt),
            nav_dt=h.nav_dt,
            guidance_dt=h.guidance_dt,
            delay_ticks=cfg.gnc.delay_ticks,
            success_pos_tol=cfg.scenario.success.pos_tol,
            success_speed_tol=cfg.scenario.success.speed_tol,
            success_tilt_tol_deg=cfg.scenario.success.tilt_tol_deg,
            **common,
        )
    else:
        raise ValueError(
            f"unknown input_mode {mode!r}; use prescribed, mission, launch_to_landing or closed_loop"
        )
    return bundle, result


class GncHandoff:
    """Builds the descent GNC stack and navigation for the separated booster at handoff.

    GNC nominal constants come from the separated vehicle (``separation.bundle.params``). The
    ascent navigation is not simulated: the descent navigation is initialized with a declared
    prior at the handoff state (plus the configured ``prior_*_offset`` errors), exactly as a
    stand-alone landing scenario uses its initial condition.
    """

    def __init__(self, separation):
        self.cfg = separation.cfg
        self.params = separation.bundle.params

    def build(self, state, aux):
        from gnc.build import build_gnc, build_nav

        cfg = self.cfg.copy()
        with open_dict(cfg):
            cfg.scenario.ic.r_ned = [float(x) for x in state.r_ned]
            cfg.scenario.ic.v_ned = [float(x) for x in state.v_ned]
        nav_dt = cfg.scenario.horizon.get("nav_dt")
        return build_gnc(cfg, self.params), build_nav(cfg, None, nav_dt)
