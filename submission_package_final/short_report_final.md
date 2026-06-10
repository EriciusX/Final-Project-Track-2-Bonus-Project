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

Before training the MLP, we built a non-learnable turn-in-apex racing-line
teacher. This teacher was hand-designed rather than learned: it used the
official track coordinates to place the target line outside on straights,
inside near turn apexes, and back outside on turn exit, then converted heading
and lateral error into `vx`, `vy`, and `yaw_rate` commands. Its parameters,
including lookahead, inside/outside offset, straight speed, and curve speed,
were selected by local sweep experiments.

The high-level MLP was then trained with supervised distillation plus
rollout-state augmentation. First, the non-learnable teacher labeled sampled
official 5D track observations. Then, an intermediate MLP was rolled out on the
track, the states it actually visited were collected, and the same teacher
re-labeled those rollout states. The final training set combined the sampled
states and the rollout-visited states. This is not a full iterative DAgger
loop; it is a conservative teacher-student distillation setup augmented with
states visited by previous rollouts. The final submitted planner config is
clean: at evaluation time it loads only the MLP weights and does not call the
hand-written teacher, racing-line formula, or extra if-else correction logic.

The final low-level checkpoint is:

```text
artifacts/lowlevel_highspeed_finetune_v9_5p0_turnmix_gpu_500k/best_checkpoint
```

The final high-level planner is:

```text
artifacts/highlevel_mlp_dagger_straight5p0_curve4p5_yawplus2_v9test/best_planner_config.json
artifacts/highlevel_mlp_dagger_straight5p0_curve4p5_yawplus2_v9test/best_planner_weights.npz
```

## Result

The final clean local official evaluation completed the 200 m track without a
fall or boundary violation:

```text
finish_time: 52.00 s
lap_completion: 1.0
valid_distance_m: 200.0
mean_progress_speed: 3.8462 m/s
min_boundary_margin_m: 0.4462 m
composite_score: 0.8282
```

## Notes

Several faster curve-speed variants were tested. The selected setting uses a
4.6 m/s straight target and a 4.0 m/s curve target because it was the fastest
clean-worktree full-lap result selected for submission robustness. More
aggressive settings such as 5.0/4.5 could complete in some local runs but were
not selected because clean reproduction could fall before completing the lap.
