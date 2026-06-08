# Track 2 Bonus Final Report

## Method

The submitted controller uses the required two-level interface. The low-level Go2
policy is a PPO command-tracking checkpoint fine-tuned from the HW1-style policy.
Its actor uses only the normal `state` observation (`policy_obs_key="state"`,
48 dimensions) and outputs the standard 12-dimensional joint action. The critic
may use `privileged_state` during PPO training, but privileged observations are
not used by the runtime actor.

The high-level planner is a learned MLP policy. It maps the official 5D track
observation to normalized velocity commands that are converted to
`vx`, `vy`, and `yaw_rate` for the low-level controller. The network has two
hidden layers of 64 units.

## Training

The high-level MLP was trained with supervised behavior cloning, also referred
to here as teacher-student distillation. A turn-in-apex racing-line teacher was
used only offline to generate labels from normal high-level observations. The
final submitted planner config is clean: at evaluation time it loads only the
MLP weights and does not call the hand-written teacher, racing-line formula, or
extra if-else correction logic.

The final low-level checkpoint is:

```text
artifacts/lowlevel_highspeed_finetune_v7_3p0_bridge_2m/best_checkpoint
```

The final high-level planner is:

```text
artifacts/highlevel_mlp_distill_turnin_apex_3p0_v1/final_clean_mlp_planner_config.json
artifacts/highlevel_mlp_distill_turnin_apex_3p0_v1/best_planner_weights.npz
```

## Result

The final clean local official evaluation completed the 200 m track without a
fall or boundary violation:

```text
finish_time: 67.34 s
lap_completion: 1.0
valid_distance_m: 200.0
mean_progress_speed: 2.9700 m/s
min_boundary_margin_m: 0.1906 m
composite_score: 0.7923
```

## Notes

Several faster hand-tuned or higher-speed low-level variants were tested, but
they either failed to complete the track or became unstable near turn entry and
exit. The submitted version was selected because it is learned at the high level,
uses the official track geometry and observation interface, and reliably
completes the lap at about 67 seconds.
