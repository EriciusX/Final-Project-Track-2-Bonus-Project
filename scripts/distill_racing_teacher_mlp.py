#!/usr/bin/env python3
"""Distill a racing-line teacher planner into an MLP high-level planner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track, official_track_config
from track_bonus.planner import StarterPlannerConfig, StarterTrackPlanner, save_mlp_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=str, default="64,64")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def curvature_at_s(s_m: np.ndarray) -> np.ndarray:
    track = official_track()
    s = np.mod(np.asarray(s_m, dtype=np.float64), track.length_m)
    straight = track.straight_length_m
    turn_len = math.pi * track.turn_radius_m
    right = (s >= straight) & (s < straight + turn_len)
    left = (s >= straight + turn_len + straight) & (s < straight + turn_len + straight + turn_len)
    curv = np.zeros_like(s, dtype=np.float32)
    curv[right | left] = 1.0
    return curv


def target_to_raw(commands: np.ndarray, cfg: StarterPlannerConfig) -> np.ndarray:
    vx = 2.0 * (commands[:, 0] - float(cfg.min_speed_mps)) / (
        float(cfg.max_speed_mps) - float(cfg.min_speed_mps)
    ) - 1.0
    vy = commands[:, 1] / max(float(cfg.max_lateral_speed_mps), 1e-6)
    yaw = commands[:, 2] / max(float(cfg.max_yaw_rate_radps), 1e-6)
    return np.clip(np.stack([vx, vy, yaw], axis=1), -0.98, 0.98).astype(np.float32)


def sample_observations(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    track = official_track()
    samples = []
    n = int(args.samples)
    s = rng.uniform(0.0, track.length_m, size=n)
    curvature = curvature_at_s(s)
    lateral = rng.uniform(-0.95, 0.95, size=n)
    heading = rng.normal(0.0, 0.28, size=n)
    heading = np.clip(heading, -1.0, 1.0)
    margin = np.clip(1.0 - np.abs(lateral), 0.0, 1.0)
    samples.append(np.stack([s / track.length_m, lateral, margin, heading, curvature], axis=1))

    if args.rollout and args.rollout.exists():
        data = np.load(args.rollout)
        obs = np.asarray(data["track_observation"], dtype=np.float32)
        if obs.ndim == 3:
            obs = obs[0]
        keep = min(len(obs), n // 3)
        idx = rng.choice(len(obs), size=keep, replace=True)
        base = obs[idx].copy()
        for scale in (0.0, 0.04, 0.08):
            jitter = base.copy()
            jitter[:, 1] = np.clip(jitter[:, 1] + rng.normal(0.0, scale, size=keep), -0.98, 0.98)
            jitter[:, 2] = np.clip(1.0 - np.abs(jitter[:, 1]), 0.0, 1.0)
            jitter[:, 3] = np.clip(jitter[:, 3] + rng.normal(0.0, scale * 2.0, size=keep), -1.1, 1.1)
            samples.append(jitter)

    obs = np.concatenate(samples, axis=0).astype(np.float32)
    rng.shuffle(obs)
    return obs


class MLP(torch.nn.Module):
    def __init__(self, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers = []
        last = 5
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(last, int(size)))
            layers.append(torch.nn.Tanh())
            last = int(size)
        layers.append(torch.nn.Linear(last, 3))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


def export_weights(model: MLP) -> list[tuple[np.ndarray, np.ndarray]]:
    weights = []
    for module in model.net:
        if isinstance(module, torch.nn.Linear):
            # Planner expects row-major x @ W + b, while torch stores out x in.
            weights.append((module.weight.detach().cpu().numpy().T.astype(np.float32), module.bias.detach().cpu().numpy().astype(np.float32)))
    return weights


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    hidden_sizes = tuple(int(x.strip()) for x in str(args.hidden_sizes).split(",") if x.strip())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_cfg = StarterPlannerConfig.load(args.teacher_config)
    teacher = StarterTrackPlanner.load(args.teacher_config)
    obs = sample_observations(args, rng)
    commands = []
    for row in obs:
        ob = TrackControllerObservation(
            lap_fraction=float(row[0]),
            lateral_error_norm=float(row[1]),
            boundary_margin_norm=float(row[2]),
            heading_error_rad=float(row[3]),
            curvature_norm=float(row[4]),
        )
        commands.append(teacher.command(ob, t=99.0))
    commands_arr = np.asarray(commands, dtype=np.float32)
    target_raw = target_to_raw(commands_arr, teacher_cfg)

    obs_mean = np.asarray(teacher_cfg.obs_mean, dtype=np.float32)
    obs_scale = np.maximum(np.asarray(teacher_cfg.obs_scale, dtype=np.float32), 1e-6)
    x = torch.tensor((obs - obs_mean) / obs_scale, dtype=torch.float32)
    y = torch.tensor(target_raw, dtype=torch.float32)

    model = MLP(hidden_sizes)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    batch = int(args.batch_size)
    for epoch in range(int(args.epochs)):
        perm = torch.randperm(len(x))
        losses = []
        for start in range(0, len(x), batch):
            idx = perm[start : start + batch]
            pred = model(x[idx])
            loss = torch.mean((pred - y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if epoch % 50 == 0 or epoch == int(args.epochs) - 1:
            print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)

    weights = export_weights(model)
    save_mlp_weights(output_dir / "best_planner_weights.npz", weights)
    payload = teacher_cfg.to_dict()
    payload["planner_type"] = "mlp"
    payload["weights_path"] = "best_planner_weights.npz"
    payload["hidden_sizes"] = list(hidden_sizes)
    payload["safety_enabled"] = False
    payload.update(official_track_config())
    (output_dir / "best_planner_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with torch.no_grad():
        train_mae = torch.mean(torch.abs(model(x) - y), dim=0).cpu().numpy().tolist()
    (output_dir / "distill_metrics.json").write_text(
        json.dumps({"samples": int(len(x)), "raw_mae": train_mae}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"planner_config": str(output_dir / "best_planner_config.json"), "raw_mae": train_mae}, indent=2), flush=True)


if __name__ == "__main__":
    main()
