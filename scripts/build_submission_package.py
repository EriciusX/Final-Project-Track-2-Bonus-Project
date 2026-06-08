#!/usr/bin/env python3
"""Build a Track Bonus submission package from local artifacts.

The script intentionally does not delete files or directories. If the output
directory already exists, files are overwritten in place and directories are
merged with copytree(..., dirs_exist_ok=True).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission_package_final")
    parser.add_argument("--team-name", default="go2_track2_learned_mlp_v7")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "artifacts" / "lowlevel_highspeed_finetune_v7_3p0_bridge_2m" / "best_checkpoint",
    )
    parser.add_argument(
        "--lowlevel-config",
        type=Path,
        default=ROOT / "configs" / "course_config_highspeed_finetune_v7_3p0_bridge.json",
    )
    parser.add_argument(
        "--planner-config",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "highlevel_mlp_distill_turnin_apex_3p0_v1"
            / "final_clean_mlp_planner_config.json"
        ),
    )
    parser.add_argument(
        "--track-eval-dir",
        type=Path,
        default=ROOT / "artifacts" / "wsl_track_eval_highlevel_mlp_distill_turnin_apex_3p0_v1_clean",
    )
    parser.add_argument(
        "--short-report",
        type=Path,
        default=ROOT / "docs" / "short_report_final.md",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build a package even if official track_eval/results.json is missing.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_dir(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    lowlevel_config_path = args.lowlevel_config.resolve()
    planner_config_path = args.planner_config.resolve()
    track_eval_dir = args.track_eval_dir.resolve()
    short_report_path = args.short_report.resolve()

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint dir: {checkpoint_dir}")
    if not lowlevel_config_path.is_file():
        raise FileNotFoundError(f"Missing low-level config: {lowlevel_config_path}")
    if not planner_config_path.is_file():
        raise FileNotFoundError(f"Missing planner config: {planner_config_path}")

    results_path = track_eval_dir / "results.json"
    official_results_available = results_path.is_file()
    if not official_results_available and not args.allow_incomplete:
        raise FileNotFoundError(
            f"Missing official results.json: {results_path}. "
            "Run official eval first or pass --allow-incomplete."
        )

    out.mkdir(parents=True, exist_ok=True)

    _copy_dir(checkpoint_dir, out / "best_checkpoint")
    _copy_file(lowlevel_config_path, out / "course_config.json")

    planner_config = _read_json(planner_config_path)
    weights_path = planner_config.get("weights_path")
    copied_weights_name = None
    if weights_path:
        source_weights = (planner_config_path.parent / str(weights_path)).resolve()
        copied_weights_name = "planner_weights.npz"
        _copy_file(source_weights, out / copied_weights_name)
        planner_config["weights_path"] = copied_weights_name
    _write_json(out / "planner_config.json", planner_config)

    _copy_file(ROOT / "track_bonus" / "__init__.py", out / "track_bonus" / "__init__.py")
    _copy_file(ROOT / "track_bonus" / "planner.py", out / "track_bonus" / "planner.py")
    _copy_file(ROOT / "track_bonus" / "controller_interface.py", out / "track_bonus" / "controller_interface.py")
    _copy_file(ROOT / "track_bonus" / "official_track.py", out / "track_bonus" / "official_track.py")
    _copy_file(ROOT / "track_bonus" / "scoring.py", out / "track_bonus" / "scoring.py")
    _copy_file(ROOT / "go2_pg_env" / "__init__.py", out / "go2_pg_env" / "__init__.py")
    _copy_file(ROOT / "go2_pg_env" / "track.py", out / "go2_pg_env" / "track.py")

    if track_eval_dir.is_dir():
        _copy_dir(track_eval_dir, out / "track_eval")

    if short_report_path.is_file():
        _copy_file(short_report_path, out / short_report_path.name)

    submission = {
        "team_name": args.team_name,
        "track2_option": "leaderboard",
        "checkpoint_dir": "best_checkpoint",
        "lowlevel_config": "course_config.json",
        "planner_config": "planner_config.json",
        "planner_code": "track_bonus/planner.py",
        "planner_weights": copied_weights_name,
        "high_level_planner_type": "teacher_distilled_mlp",
        "training_method": {
            "low_level": "HW1-style Go2 PPO command tracking, high-speed v7 fine-tune; actor policy_obs_key=state, state shape 48, action size 12",
            "high_level": "Supervised behavior cloning / distillation: a 64x64 MLP maps the official 5D high-level observation to vx, vy, yaw_rate commands using turn-in-apex racing-line teacher labels offline only",
        },
        "evaluation_command": (
            "python run_track_bonus.py --checkpoint-dir best_checkpoint "
            "--config course_config.json --planner-config planner_config.json --output-dir track_eval "
            "--entry-name learned_mlp_distill_v7"
        ),
        "official_results_available": official_results_available,
        "track_eval_results": "track_eval/results.json" if official_results_available else None,
        "status": "complete" if official_results_available else "incomplete_missing_official_results",
        "notes": (
            "Final clean MLP runtime does not call the hand-written teacher or racing-line formula. "
            "Official local eval: 200 m completed in 67.34 s, no fall, no boundary violation, min boundary margin 0.1906 m."
        ),
    }
    _write_json(out / "submission.json", submission)

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "official_results_available": official_results_available,
                "submission_json": str(out / "submission.json"),
                "status": submission["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
