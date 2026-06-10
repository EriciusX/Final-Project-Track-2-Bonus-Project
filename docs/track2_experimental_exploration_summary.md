# Track 2 Experimental Exploration Summary

## Objective

The goal was to build a Track 2 controller that can complete the official
200 m oval track as fast as possible while still following the required
hierarchical interface:

- Low-level Go2 locomotion policy: normal `state` observation to 12 joint actions.
- High-level planner: official 5D track observation to `[vx, vy, yaw_rate]`.
- No direct joint trajectory playback.
- No actor use of privileged observations.
- No modification of the official track geometry.

The final submitted version prioritizes clean reproducibility over the single
fastest unstable local run.

## Baseline And Early Observations

We started from the original HW1-style Go2 PPO locomotion checkpoint and the
starter high-level planner. The initial behavior could complete conservative
paths, but the speed was limited and the route was not close to a racing line.

Early tests showed two important bottlenecks:

- Shorter geometric paths were not automatically faster because the low-level
  policy could become unstable when commanded to turn too aggressively.
- The low-level policy tracked forward velocity better than yaw rate, so high
  speed turns were the main limiting case.

## Racing-Line Planning Exploration

We explored racing-line behavior before training the final MLP planner. The
desired line was:

- Use the outside of the straight before turn entry.
- Move inward through the turn.
- Approach an apex near the inside boundary.
- Return outward on turn exit.

Several target-line parameters were tested:

- `safety_racing_line_amplitude_norm`
- straight outside offset
- turn inside offset
- start blend distance
- straight and curve lookahead distance
- heading bias and yaw bias limits
- optional boundary and speed safety limits

We tested safe, mild, normal, aggressive, and visual racing-line amplitudes.
More aggressive lines looked closer to a racing line, but they also reduced
stability margin. In particular, too much inside targeting or too much
lookahead could pull the robot inward too early and produce poor turn entry.

## Lookahead And Heading Experiments

A large part of the exploration focused on how far ahead the high-level planner
should target:

- Long lookahead at the start helped avoid abruptly cutting from the start point
  to the outer line.
- Shorter curve lookahead made the dog follow local geometry more tightly.
- Longer curve lookahead helped anticipate turn entry, but too much lookahead
  could point the robot too far inward.

We also tested heading-angle constraints and dead zones:

- Straight deadband allowed the dog to keep going straight when it was already
  close enough to the target line.
- Heading command clipping around 30 degrees was tested to avoid overly sharp
  corrections.
- Some of these constraints improved visual smoothness but did not consistently
  improve final lap time.

## High-Level Learned Planner

The assignment requires a learnable final method, so the final high-level
planner was changed from hand-written racing-line logic to an MLP.

The final high-level planner:

- Takes the official 5D track observation.
- Outputs normalized commands converted to `vx`, `vy`, and `yaw_rate`.
- Uses a 64 x 64 MLP.
- Loads learned weights at evaluation time.
- Does not call the hand-written racing-line teacher during evaluation.

The MLP was trained with supervised distillation plus rollout-state
augmentation. A turn-in-apex racing-line teacher was used offline to generate
labels from normal high-level observations and states collected from previous
rollouts. This should not be described as a full multi-round DAgger loop,
because we did not run repeated iterative cycles of rollout, expert relabeling,
dataset aggregation, retraining, and rollout again. The teacher was only used
for data generation, not as the runtime controller.

## Low-Level Speed Fine-Tuning

We also explored low-level PPO fine-tuning to increase the range of commands
the Go2 policy can track.

The final v9 low-level training used a high-speed turn-mix command distribution.
The stage 2 command range covered:

- `vx`: `2.6` to `5.0 m/s`
- `vy`: `-0.04` to `0.04 m/s`
- `yaw`: `-0.85` to `0.85 rad/s`

The command mixture had four main regions:

| Profile | Weight | Velocity Range | Yaw Range | Purpose |
|---|---:|---:|---:|---|
| High-speed straight | 25% | `4.6 - 5.0` | `0.00 - 0.12` | Sprint capability |
| Fast moderate turn | 35% | `4.0 - 4.7` | `0.18 - 0.45` | Track-speed turning |
| Slower sharp turn | 30% | `2.6 - 3.8` | `0.45 - 0.85` | Larger yaw authority |
| Transition | 10% | `3.4 - 4.6` | `0.00 - 0.25` | Smooth intermediate commands |

Single-command tests showed:

- `vx=5.0` could run stably for 10 seconds.
- The measured forward speed saturated around `4.4 - 4.5 m/s`.
- High-speed yaw tracking was still weak compared with commanded yaw.

This means v9 improved straight-line robustness, but did not fully solve the
high-speed curve bottleneck.

## Speed Sweep Experiments

We tested multiple straight and curve target speeds using the learned MLP
planner and v9 low-level policy.

Representative results:

| Low-Level | Straight Target | Curve Target | Result |
|---|---:|---:|---|
| v8 | `4.6` | `4.0` | Around `50.90 s`, completed |
| v9 | `4.6` | `4.0` | Stable, clean reproducible, around `52.00 s` |
| v9 | `5.0` | `4.5` | Fast local run around `50.34 s`, but not robust in clean reproduction |
| v9 | `5.0` | `4.6` | Could fail early in some runs |
| v9 | `5.0` | `4.7` | Completed in one sweep, but slower than `4.5` and less stable |
| v9 | `5.0` | `4.8` | Completed in one sweep, but slower than `4.5` |
| v9 | `5.0` | `4.9` | Fell early |
| v9 | `5.0` | `5.0` | Fell before completing the lap |

The fastest local run was not selected because clean-worktree reproduction
showed that it could fall. The final submitted planner uses the more robust
`4.6 / 4.0` speed envelope.

## Failure Modes Observed

The main failure modes were:

- Falling during or near turn entry when curve target speed was too high.
- Heading/yaw under-tracking at high forward speed.
- Cutting inward too aggressively when the racing-line target or lookahead was
  too strong.
- Faster straight speed not always improving lap time because it made the next
  turn less stable.
- Local runs at the extreme speed envelope were not always reproducible from a
  clean checkout.

These failures suggested that the practical bottleneck is not only maximum
straight-line velocity, but the combined ability to track yaw while maintaining
body stability at high forward speed.

## Final Submitted Version

The final GitHub submission uses:

- Low-level policy: v9 high-speed turn-mix PPO checkpoint.
- High-level planner: learned MLP planner.
- Planner speed envelope:
  - straight max speed: `4.6 m/s`
  - curve speed: `4.0 m/s`
- Track geometry:
  - length: `200 m`
  - turn radius: `18.25 m`
  - half width: `2.0 m`

Clean local evaluation from `submission_package_final`:

```text
valid_distance_m: 200.0
finish_time: 52.00 s
fall: false
boundary_violation: false
min_boundary_margin_m: 0.4462 m
composite_score: 0.8282
```

## Reproducibility Check

Before finalizing the GitHub submission, we checked clean reproduction. This
surfaced two issues:

- The Go2 mesh assets were missing from Git because they were ignored.
- The most aggressive `5.0 / 4.5` planner could fail in clean reproduction.

Both were addressed:

- The required `go2_pg_env/xmls/assets/*.obj` mesh files were added to Git.
- The final package was switched to the stable `4.6 / 4.0` planner.

The uploaded version is therefore selected for reliable full-lap completion and
assignment compliance, not just the fastest single local run.

## Report Summary

In report form, the main contribution can be described as:

> We explored both low-level speed scaling and learned high-level racing-line
> planning. The low-level Go2 PPO policy was fine-tuned with a high-speed
> turn-mix command distribution, while the high-level planner was distilled into
> a learned MLP from a turn-in-apex racing-line teacher. Aggressive speed
> settings achieved faster local runs but were less reproducible. The final
> submission uses a stable v9 low-level checkpoint and a learned MLP planner
> that completes the official 200 m track without falling or leaving the
> boundary.
