# RLV-Sim Development TODO — Project 3: Guidance Algorithms

Status as of 2026-11-23 (ZEM/ZEV first version added). Scope: [Project 3](../materials/RSLV_GNC_Project3.pdf). Develop closed-loop
guidance that drives the vehicle from its **estimated** state (Project 2 navigation) to the desired
terminal position and velocity, using two approaches:

1. **Analytical feedback guidance:** ZEM/ZEV with time-to-go estimation, gravity compensation and saturation.
2. **Optimization-based powered-descent guidance:** constrained trajectory optimization (LCvx / SCvx).

Hard rule from the brief: guidance consumes only the navigation estimate `x_hat`. No hidden truth
states inside a guidance law.

Grading weights: formulation 15% · ZEM/ZEV 20% · optimization 25% · sim/nav integration 10% ·
verification and robustness 15% · technical reasoning 10% · report/presentation 5%.

---

## Already Implemented (Projects 1–2 and existing guidance)

### Plant and simulator
- [x] Native C++ 6-DOF Falcon 9 first-stage plant (RK4, 1 ms) with mass properties, multi-engine thrust, TVC, grid fins, RCS, slosh, actuator models (ideal / simplified / realistic) and faults.
- [x] Environment: uniform and inverse-square gravity, spherical rotating Earth, USSA/exponential atmosphere, steady/Dryden/von Kármán wind.
- [x] Scenarios: `pdg` (fixed-Earth landing baseline), `project2_landing` (rotating Earth), the four prescribed `project2_*` phases, ascent/separation/boostback and reentry-to-landing experiments.
- [x] Independent physics verification, timestep convergence and conservation checks; see the [verification report](verification/physics_implementation_report_en.md).
- [x] Hydra configs with swappable `gnc/{guidance,control,navigation,allocation,scheduler}`, plus a Monte Carlo runner (`scripts/run_montecarlo.py`, `run=montecarlo`).
- [x] WebGL replay viewer and frame explorer.
- [x] Public CRS-11 flight comparison (opt-in); see the [report](verification/public_flight_comparison_report_ko.md).

### Navigation (Project 2)
- [x] Synthetic IMU / GNSS / radar-altimeter suite with noise, bias, scale factor, dropouts, outliers and scheduled outages (`gnc/navigation=synthetic_eskf`).
- [x] Strapdown INS and 15-state aided ESKF with innovation gating, NEES/NIS diagnostics and ±3σ plots (`scripts/analyze_navigation.py`).
- [x] INS-only vs. ESKF error comparison and robustness campaign (`report_assets/navigation_robustness.py`).

### Guidance and control
- [x] Guidance contract `__call__(x_hat, t, memory) -> thrust_ned | Command` plus a copy-me template ([template.py](../src/gnc/guidance/template.py)).
- [x] Phase scheduler (altitude / guard) and phase laws: reentry burn, aero-descent steering, boostback.
- [x] **LCvx** powered-descent guidance ([lcvx.py](../src/gnc/guidance/convex/lcvx.py)): lossless convexification SOCP, thrust magnitude bounds, terminal position/velocity constraints, minimum fuel.
- [x] **SCvx** drag-aware successive convexification around an LCvx free-tf initialization ([scvx.py](../src/gnc/guidance/convex/scvx.py), native solver in `src/native/scvx.cpp`).
- [x] Free-final-time line search with grid fallback; optional fixed terminal epoch (`convex_touchdown_time_s`).
- [x] Stretch constraints already present: tilt cone (`attitude_tilt_max_deg`), glide slope, thrust-magnitude rate, optional slew rate, upright terminal node, plan-side drag.
- [x] Optional online replanning (`convex_replan`) and terminal slack mode.
- [x] Solver backends Clarabel and QOCO; per-solve `solve_time_sec` and wall time recorded.
- [x] Plan-tracking PD control (`convex_tracking_k_r/k_v`), attitude PD and engine/fin/RCS allocation.
- [x] Existing tests: convex feasibility, convex guidance, guidance telemetry, SCvx mission semantics.

---

## Project 3 Work

### 1. Guidance Problem Formulation (15%)
- [ ] Write the 3-DOF translational guidance model: point mass, NED frame, gravity (nominal planning value incl. centrifugal term on rotating Earth), mass depletion `ṁ = −‖T‖/(Isp g0)`, optional drag.
- [ ] Define the terminal condition (target position and velocity, touchdown tolerance) and the constraint set (T_min ≤ ‖T‖ ≤ T_max, tilt, glide slope, throttle rate).
- [ ] Draw the GNC architecture: truth plant → sensors → ESKF → `x_hat` → guidance → control/allocation → plant. Mark exactly where truth is and is not allowed.
- [ ] Select the Project 3 reference scenario(s): `pdg` and/or `project2_landing`, with the initial state, fuel budget and target documented.

### 2. ZEM/ZEV Guidance (20%) — first version implemented 2026-11-23
Module [zemzev.py](../src/gnc/guidance/zemzev.py), config [zemzev.yaml](../configs/gnc/guidance/zemzev.yaml), tests [test_zemzev.py](../tests/gnc/test_zemzev.py). Run: `python scripts/run_single.py plant/actuator=ideal gnc/guidance=zemzev gnc/navigation=synthetic_eskf`.
- [x] `ZEM = r_f − (r + v·t_go + ½g·t_go²)`, `ZEV = v_f − (v + g·t_go)`, `a = 6·ZEM/t_go² − 2·ZEV/t_go`, `T = m·a` (gravity compensation inside ZEM/ZEV). Evaluated at the fast control rate (`track`), not only at the 1 Hz guidance tick.
- [x] Time-to-go: fixed final time (`t_go = t_f − t`, t_f committed at ignition or set with `zemzev_fixed_tf_s`) and energy-optimal t_go (`zemzev_tgo_mode=optimal`; quartic `t·P′ − 3P = 0` from `J = Γt + ½∫|a|²`).
- [x] Ignition logic for the throttle floor (T_min/m > g: no hover). Coast until the latest time a margined, exactly checked linear open-loop profile fits the thrust bounds and tilt cone; divert-limited and urgent fallbacks.
- [x] Saturation with vertical priority (keep the up component, clip lateral to the tilt cone and the leftover magnitude); saturation flags logged.
- [x] Terminal singularity: below `t_go < 0.5 s` fly the latched open-loop vertical profile, zero lateral thrust.
- [x] Telemetry: `zemzev_log.json` per run (t_go, ZEM, ZEV, a_cmd, a_sat, saturation flags, wall time per call).
- [x] Unit tests: unconstrained double integrator hits the target, profile/cost identities, quartic t_go vs brute force, exact segment metrics, saturation, estimate-only contract, config build.
- [ ] **Lateral divert vs. attitude-loop bandwidth (open).** Lateral thrust comes from tilting the body. The PD attitude loop settles in about 4 s and lags the command by about 2 s, so high lateral ZEM/ZEV gains overshoot. Current mitigation: lateral t_go floor of 8 s (gain cap) plus lateral velocity nulling below 2.5 s. Results with ideal actuators, 300 m / 25 m/s start: nominal soft (0.23 m/s); nominal + ESKF soft (2.7 m, 0.43 m/s); 25 m and 72 m lateral offsets land **hard** (1.5–4 m/s, 8–13° tilt). Options: (a) earlier, longer burn when a lateral divert is needed (ignition rule that accounts for lateral settling time), (b) faster attitude loop (Project 4), (c) attitude-lag-aware lateral command, (d) restrict the evaluated dispersion envelope and document it.
- [ ] Evaluate `tgo_mode=optimal` and first-feasible ignition in 6-DOF (point-mass prototype: optimal t_go and late ignition each save fuel; first-feasible ignition costs about 50 % more fuel).
- [ ] Optional: tunable gains (k_r, k_v) generalization and a comparison with the 6/−2 optimal gains.
- [ ] Derive ZEM/ZEV from the unconstrained energy-optimal control problem (for the report).

### 3. Optimization-Based Guidance (25%)
- [ ] Demonstrate the **required baseline**: LCvx with a fixed terminal time (`convex_touchdown_time_s`), thrust bounds and hard terminal constraints. Verify it works from the estimated state.
- [ ] Document the LCvx formulation: change of variables (σ, z = ln m), relaxed thrust constraint, losslessness argument, discretization (ZOH, N nodes), cost (minimum fuel).
- [ ] Present the advanced variants already implemented: free final time, glide slope, tilt, thrust rate, drag via SCvx (trust region, virtual control, convergence).
- [ ] Online replanning under navigation error (`convex_replan: true`): check stability, replan rate and solve-time budget.
- [ ] Record solver status, iterations and solve time per call; report the infeasibility/fallback behavior.
- [ ] Remove stale backups (`guidance.py.bak`, `*.before_*`, `lcvx.py.bak`) and root-level `*.patch` / `*.diff` once confirmed unused.

### 4. Simulator and Navigation Integration (10%)
- [ ] Run all Project 3 cases with `gnc/navigation=synthetic_eskf` (the default `identity` passes truth through).
- [ ] LCvx with ESKF currently crashes (9.4 m/s) because its default lateral tracking gains (`k_r=1, k_v=4`) exceed the attitude-loop bandwidth; use the `project2_landing` lateral gains (`[0.1,0.1,1]`, `[0.6,0.6,4]`) for a fair comparison.
- [ ] Add a test that guidance commands are unchanged when the truth state is perturbed with `x_hat` held fixed.
- [ ] Audit `ctx.params` / `ctx.nominal` usage in guidance: only nominal model values, never live plant truth.
- [ ] Provide a Project 3 config preset (e.g. `+experiment=project3_<law>`) so each run is a one-line command.

### 5. Verification and Robustness (15%)
Run every case for ZEM/ZEV, LCvx (fixed tf), LCvx (free tf) and SCvx:
- [ ] Nominal: convergence to the terminal state.
- [ ] Initial-condition dispersion: position/velocity Monte Carlo (`scenario.dispersion.enable=true`, `run=montecarlo`).
- [ ] Navigation error: ESKF nominal, GNSS outage near touchdown, GNSS outlier, increased IMU bias.
- [ ] Model mismatch: mass, thrust (`fault=engine_underperform`), drag, gravity, realistic actuators (lag).
- [ ] Optional: wind (`environment/wind=dryden`), transport delay (`gnc.delay_ticks`).
- [ ] Write a campaign script (e.g. `scripts/run_guidance_campaign.py`) that runs the matrix with fixed seeds and writes metrics to `results/`.

### 6. Required Plots and Metrics
- [ ] Trajectory (3D and ground track), position and velocity histories.
- [ ] Guidance acceleration and thrust magnitude/direction (tilt) histories with limits.
- [ ] ZEM and ZEV histories.
- [ ] Terminal position and velocity error (tables plus Monte Carlo scatter/histograms).
- [ ] Propellant use / control effort and constraint histories (thrust bounds, tilt, glide slope).
- [ ] Computation time per guidance call (distribution per law).
- [ ] Generator script `report_assets/generate_guidance.py`, in the same style as the navigation assets.

### 7. Report and Presentation
- [ ] Manual-style technical report: formulation, architecture, derivations, implementation, results, robustness and sensitivity.
- [ ] Comparison of the two approaches: optimality (fuel), accuracy, robustness to navigation/model error, constraint handling, compute cost.
- [ ] Assumptions, limitations and data sources.
- [ ] Final question: the most likely failure causes on a real RLV (e.g. ideal-thrust-tracking assumption vs. throttle lag, navigation bias near touchdown, reachability under single-engine T/W, solver infeasibility) and how to redesign the algorithm.
- [ ] Reproduction guide: commands for every figure and table.
- [ ] 12–15 min presentation.
- [ ] Prepare the guidance interface for Project 4 (control and integrated GNC).

---

## Maintenance Principles

- Keep generated JSON/CSV/NPZ, plots and resolved run configs under ignored `results/`; keep source, authored YAML, tests, reference snapshots and manual reports in Git.
- Add new guidance laws as swappable modules via `configs/gnc/guidance/<name>.yaml` + `build(ctx)`; reuse the existing runners, telemetry and replay.
- Guidance consumes only `x_hat` and nominal parameters; truth is used only by the plant, sensor generation and error scoring.
- Preserve existing functionality and test coverage; run `pytest` and `ruff` before committing.
