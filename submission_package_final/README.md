# Final Track 2 Submission Package

This directory contains the final learned MLP Track 2 submission.

Run from the repository root:

```bash
python run_track_bonus.py \
  --checkpoint-dir submission_package_final/best_checkpoint \
  --config submission_package_final/course_config.json \
  --planner-config submission_package_final/planner_config.json \
  --output-dir artifacts/repro_from_submission_package \
  --entry-name learned_mlp_v9_fast \
  --duration-seconds 120 \
  --no-render
```

Expected clean evaluation result:

```text
valid_distance_m: 200.0
finish_time: 52.00 s
fall: false
boundary_violation: false
```
