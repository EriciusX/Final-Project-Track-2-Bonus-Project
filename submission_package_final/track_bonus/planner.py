"""Starter high-level planner for the 200 m track bonus.

The evaluator builds the official compact 5D track observation defined in
`track_bonus/controller_interface.py`. The high-level planner maps it to the
local joystick command consumed by the HW1 Go2 locomotion policy:

    5D track observation -> [vx, vy, yaw_rate]

This file is intentionally small.  It is a weak baseline and an interface
example, not a solved full-lap controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_speed_mps: float = 0.90
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0
    weights_path: str | None = None
    hidden_sizes: tuple[int, ...] = (16, 16)
    obs_mean: tuple[float, ...] = (0.5, 0.0, 0.5, 0.0, 0.0)
    obs_scale: tuple[float, ...] = (0.5, 1.0, 0.5, math.pi, 1.0)
    safety_enabled: bool = False
    safety_boundary_margin_norm: float = 0.25
    safety_min_speed_mps: float = 0.12
    safety_speed_ramp_seconds: float = 0.0
    safety_speed_ramp_start_mps: float = 0.0
    safety_heading_error_rad: float = 1.2
    safety_curve_speed_mps: float = 0.0
    safety_edge_speed_mps: float = 0.0
    safety_edge_boundary_margin_norm: float = 0.0
    safety_centering_lateral_gain: float = 0.0
    safety_lateral_target_norm: float = 0.0
    safety_lateral_target_curve_only: bool = False
    safety_racing_line_enabled: bool = False
    safety_racing_line_style: str = "cosine"
    safety_racing_line_amplitude_norm: float = 0.0
    safety_racing_line_lateral_enabled: bool = True
    safety_racing_line_lateral_gain: float = 0.0
    safety_racing_line_heading_gain: float = 0.0
    safety_racing_line_lookahead_m: float = 0.0
    safety_racing_line_curve_lookahead_m: float = 0.0
    safety_racing_line_start_lookahead_until_m: float = 0.0
    safety_racing_line_start_lookahead_m: float = 0.0
    safety_racing_line_start_blend_until_m: float = 0.0
    safety_racing_line_start_blend_from_norm: float = 0.0
    safety_racing_line_straight_outside_extra_norm: float = 0.0
    safety_racing_line_turn_inside_extra_norm: float = 0.0
    safety_racing_line_straight_deadband_m: float = 0.0
    safety_racing_line_straight_deadband_symmetric: bool = False
    safety_racing_line_heading_bias_max_rad: float = 0.0
    safety_heading_recovery_gain: float = 0.0
    safety_curve_yaw_bias_radps: float = 0.0
    safety_speed_zone_start_m: float = -1.0
    safety_speed_zone_end_m: float = -1.0
    safety_speed_zone_mps: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "heading_slowdown": self.heading_slowdown,
            "stand_seconds": self.stand_seconds,
            "weights_path": self.weights_path,
            "hidden_sizes": list(self.hidden_sizes),
            "obs_mean": list(self.obs_mean),
            "obs_scale": list(self.obs_scale),
            "safety_enabled": self.safety_enabled,
            "safety_boundary_margin_norm": self.safety_boundary_margin_norm,
            "safety_min_speed_mps": self.safety_min_speed_mps,
            "safety_speed_ramp_seconds": self.safety_speed_ramp_seconds,
            "safety_speed_ramp_start_mps": self.safety_speed_ramp_start_mps,
            "safety_heading_error_rad": self.safety_heading_error_rad,
            "safety_curve_speed_mps": self.safety_curve_speed_mps,
            "safety_edge_speed_mps": self.safety_edge_speed_mps,
            "safety_edge_boundary_margin_norm": self.safety_edge_boundary_margin_norm,
            "safety_centering_lateral_gain": self.safety_centering_lateral_gain,
            "safety_lateral_target_norm": self.safety_lateral_target_norm,
            "safety_lateral_target_curve_only": self.safety_lateral_target_curve_only,
            "safety_racing_line_enabled": self.safety_racing_line_enabled,
            "safety_racing_line_style": self.safety_racing_line_style,
            "safety_racing_line_amplitude_norm": self.safety_racing_line_amplitude_norm,
            "safety_racing_line_lateral_enabled": self.safety_racing_line_lateral_enabled,
            "safety_racing_line_lateral_gain": self.safety_racing_line_lateral_gain,
            "safety_racing_line_heading_gain": self.safety_racing_line_heading_gain,
            "safety_racing_line_lookahead_m": self.safety_racing_line_lookahead_m,
            "safety_racing_line_curve_lookahead_m": self.safety_racing_line_curve_lookahead_m,
            "safety_racing_line_start_lookahead_until_m": self.safety_racing_line_start_lookahead_until_m,
            "safety_racing_line_start_lookahead_m": self.safety_racing_line_start_lookahead_m,
            "safety_racing_line_start_blend_until_m": self.safety_racing_line_start_blend_until_m,
            "safety_racing_line_start_blend_from_norm": self.safety_racing_line_start_blend_from_norm,
            "safety_racing_line_straight_outside_extra_norm": self.safety_racing_line_straight_outside_extra_norm,
            "safety_racing_line_turn_inside_extra_norm": self.safety_racing_line_turn_inside_extra_norm,
            "safety_racing_line_straight_deadband_m": self.safety_racing_line_straight_deadband_m,
            "safety_racing_line_straight_deadband_symmetric": self.safety_racing_line_straight_deadband_symmetric,
            "safety_racing_line_heading_bias_max_rad": self.safety_racing_line_heading_bias_max_rad,
            "safety_heading_recovery_gain": self.safety_heading_recovery_gain,
            "safety_curve_yaw_bias_radps": self.safety_curve_yaw_bias_radps,
            "safety_speed_zone_start_m": self.safety_speed_zone_start_m,
            "safety_speed_zone_end_m": self.safety_speed_zone_end_m,
            "safety_speed_zone_mps": self.safety_speed_zone_mps,
        }


class StarterTrackPlanner:
    """Conservative coordinate-to-command baseline.

    The policy is deliberately simple and conservative. Students should improve
    it by changing this controller, replacing it with an MLP, or training a
    higher-level policy that produces the same command vector.
    """

    def __init__(self, config: StarterPlannerConfig) -> None:
        if config.planner_type not in {"starter_pd", "mlp", "racing_pd"}:
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        self.config = config
        self.track: StandardOvalTrack = official_track()
        self._weights: list[tuple[np.ndarray, np.ndarray]] | None = None

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        config = StarterPlannerConfig.load(path)
        planner = cls(config)
        if config.planner_type == "mlp":
            if not config.weights_path:
                raise ValueError("MLP planner config must include weights_path.")
            weights_path = Path(config.weights_path)
            if not weights_path.is_absolute():
                weights_path = path.parent / weights_path
            planner._weights = load_mlp_weights(weights_path)
        return planner

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        if self.config.planner_type == "racing_pd":
            command = self.command_from_racing_line(obs)
            return self.apply_speed_ramp(command, t)
        if self.config.planner_type == "mlp":
            command = self.command_from_mlp(obs)
            return self.apply_speed_ramp(command, t)
        command = self.command_from_observation(obs)
        return self.apply_speed_ramp(command, t)

    def apply_speed_ramp(self, command: np.ndarray, t: float) -> np.ndarray:
        ramp_seconds = float(self.config.safety_speed_ramp_seconds)
        if ramp_seconds <= 0.0:
            return command
        elapsed = max(0.0, float(t) - float(self.config.stand_seconds))
        if elapsed >= ramp_seconds:
            return command
        ramp = np.asarray(command, dtype=np.float32).copy()
        start_speed = max(float(self.config.safety_speed_ramp_start_mps), float(self.config.min_speed_mps))
        alpha = np.clip(elapsed / max(ramp_seconds, 1e-6), 0.0, 1.0)
        speed_cap = start_speed + alpha * (float(self.config.max_speed_mps) - start_speed)
        ramp[0] = min(float(ramp[0]), float(speed_cap))
        return ramp.astype(np.float32)

    def command_from_mlp(self, obs: TrackControllerObservation) -> np.ndarray:
        if self._weights is None:
            raise ValueError("MLP weights were not loaded.")
        x = obs.as_array().astype(np.float32)
        obs_mean = np.asarray(self.config.obs_mean, dtype=np.float32)
        obs_scale = np.maximum(np.asarray(self.config.obs_scale, dtype=np.float32), 1e-6)
        x = (x - obs_mean) / obs_scale
        for idx, (weight, bias) in enumerate(self._weights):
            x = x @ weight + bias
            if idx < len(self._weights) - 1:
                x = np.tanh(x)
        raw = np.tanh(x)
        vx = float(self.config.min_speed_mps) + 0.5 * (float(raw[0]) + 1.0) * (
            float(self.config.max_speed_mps) - float(self.config.min_speed_mps)
        )
        vy = float(raw[1]) * float(self.config.max_lateral_speed_mps)
        yaw_rate = float(raw[2]) * float(self.config.max_yaw_rate_radps)
        command = np.asarray([vx, vy, yaw_rate], dtype=np.float32)
        return self.apply_safety_limits(command, obs)

    def apply_safety_limits(self, command: np.ndarray, obs: TrackControllerObservation) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32).copy()
        command[0] = np.clip(command[0], float(self.config.min_speed_mps), float(self.config.max_speed_mps))
        command[1] = np.clip(command[1], -float(self.config.max_lateral_speed_mps), float(self.config.max_lateral_speed_mps))
        command[2] = np.clip(command[2], -float(self.config.max_yaw_rate_radps), float(self.config.max_yaw_rate_radps))
        if not self.config.safety_enabled:
            return command.astype(np.float32)

        margin_scale = np.clip(
            float(obs.boundary_margin_norm) / max(float(self.config.safety_boundary_margin_norm), 1e-6),
            0.0,
            1.0,
        )
        heading_scale = np.clip(
            1.0 - abs(float(obs.heading_error_rad)) / max(float(self.config.safety_heading_error_rad), 1e-6),
            0.0,
            1.0,
        )
        speed_scale = min(float(margin_scale), float(heading_scale))
        speed_floor = float(self.config.safety_min_speed_mps)
        speed_cap = speed_floor + speed_scale * (float(self.config.max_speed_mps) - speed_floor)
        command[0] = min(float(command[0]), float(speed_cap))

        if abs(float(obs.curvature_norm)) > 0.5 and float(self.config.safety_curve_speed_mps) > 0.0:
            command[0] = min(float(command[0]), float(self.config.safety_curve_speed_mps))

        edge_margin = float(self.config.safety_edge_boundary_margin_norm)
        near_edge = edge_margin > 0.0 and float(obs.boundary_margin_norm) < edge_margin
        if near_edge:
            if float(self.config.safety_edge_speed_mps) > 0.0:
                command[0] = min(float(command[0]), float(self.config.safety_edge_speed_mps))

        lateral_target = float(self.config.safety_lateral_target_norm)
        racing_heading_bias = 0.0
        if bool(self.config.safety_racing_line_enabled):
            racing_lateral_target, racing_heading_bias = self.racing_line_target(obs)
            if bool(self.config.safety_racing_line_lateral_enabled):
                lateral_target = racing_lateral_target
        if bool(self.config.safety_lateral_target_curve_only) and abs(float(obs.curvature_norm)) <= 0.5:
            lateral_target = 0.0
        if float(self.config.safety_centering_lateral_gain) > 0.0 and (near_edge or abs(lateral_target) > 1e-6):
            lateral_error = float(obs.lateral_error_norm) - lateral_target
            centering_vy = -float(self.config.safety_centering_lateral_gain) * lateral_error
            command[1] = np.clip(
                centering_vy,
                -float(self.config.max_lateral_speed_mps),
                float(self.config.max_lateral_speed_mps),
            )
        if (
            bool(self.config.safety_racing_line_enabled)
            and bool(self.config.safety_racing_line_lateral_enabled)
            and float(self.config.safety_racing_line_lateral_gain) > 0.0
        ):
            racing_vy = float(self.config.safety_racing_line_lateral_gain) * (
                racing_lateral_target - float(obs.lateral_error_norm)
            )
            command[1] = np.clip(
                float(command[1]) + racing_vy,
                -float(self.config.max_lateral_speed_mps),
                float(self.config.max_lateral_speed_mps),
            )

        if float(self.config.safety_heading_recovery_gain) > 0.0:
            command[2] += float(self.config.safety_heading_recovery_gain) * float(obs.heading_error_rad)
        if bool(self.config.safety_racing_line_enabled) and float(self.config.safety_racing_line_heading_gain) > 0.0:
            command[2] += float(self.config.safety_racing_line_heading_gain) * racing_heading_bias
        if abs(float(obs.curvature_norm)) > 0.5 and float(self.config.safety_curve_yaw_bias_radps) > 0.0:
            command[2] += math.copysign(float(self.config.safety_curve_yaw_bias_radps), float(obs.curvature_norm))

        command[0] = np.clip(command[0], float(self.config.min_speed_mps), float(self.config.max_speed_mps))
        command[1] = np.clip(command[1], -float(self.config.max_lateral_speed_mps), float(self.config.max_lateral_speed_mps))
        command[2] = np.clip(command[2], -float(self.config.max_yaw_rate_radps), float(self.config.max_yaw_rate_radps))
        return command.astype(np.float32)

    def command_from_racing_line(self, obs: TrackControllerObservation) -> np.ndarray:
        target_lateral, heading_bias = self.racing_line_target(obs)
        if self.in_straight_target_deadband(obs, target_lateral):
            target_lateral = float(obs.lateral_error_norm)
            heading_bias = 0.0
        lateral_error_m = (float(obs.lateral_error_norm) - target_lateral) * float(self.track.half_width_m)
        heading_error = wrap_angle(float(obs.heading_error_rad) + heading_bias)

        vx = float(self.config.max_speed_mps)
        if abs(float(obs.curvature_norm)) > 0.5 and float(self.config.safety_curve_speed_mps) > 0.0:
            vx = min(vx, float(self.config.safety_curve_speed_mps))
        vx = min(vx, self.speed_zone_cap(obs, vx))
        margin_scale = np.clip(
            float(obs.boundary_margin_norm) / max(float(self.config.safety_boundary_margin_norm), 1e-6),
            0.0,
            1.0,
        )
        vx = max(float(self.config.min_speed_mps), vx * float(margin_scale))

        lateral_gain = float(self.config.safety_centering_lateral_gain)
        vy = np.clip(
            -lateral_gain * lateral_error_m,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )

        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw_rate = curvature * vx + float(self.config.k_heading) * heading_error
        if abs(float(obs.curvature_norm)) > 0.5 and float(self.config.safety_curve_yaw_bias_radps) > 0.0:
            yaw_rate += math.copysign(float(self.config.safety_curve_yaw_bias_radps), float(obs.curvature_norm))
        yaw_rate = np.clip(
            yaw_rate,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)

    def in_straight_target_deadband(self, obs: TrackControllerObservation, target_lateral: float) -> bool:
        if abs(float(obs.curvature_norm)) > 0.5:
            return False
        deadband_m = float(self.config.safety_racing_line_straight_deadband_m)
        if deadband_m <= 0.0:
            return False
        deadband_norm = deadband_m / max(float(self.track.half_width_m), 1e-6)
        lateral = float(obs.lateral_error_norm)
        if bool(self.config.safety_racing_line_straight_deadband_symmetric):
            return abs(lateral - float(target_lateral)) <= deadband_norm
        return float(target_lateral) <= lateral <= float(target_lateral) + deadband_norm

    def speed_zone_cap(self, obs: TrackControllerObservation, current_vx: float) -> float:
        zone_speed = float(self.config.safety_speed_zone_mps)
        if zone_speed <= 0.0:
            return float(current_vx)
        start = float(self.config.safety_speed_zone_start_m)
        end = float(self.config.safety_speed_zone_end_m)
        if start < 0.0 or end < 0.0:
            return float(current_vx)
        s = float(obs.lap_fraction) * float(self.track.length_m)
        if start <= end:
            in_zone = start <= s <= end
        else:
            in_zone = s >= start or s <= end
        if not in_zone:
            return float(current_vx)
        return min(float(current_vx), zone_speed)

    def racing_line_target(self, obs: TrackControllerObservation) -> tuple[float, float]:
        """Outside-apex-outside target line for the official oval.

        Positive lateral error is inside both left-hand semicircles. The target
        stays slightly outside on straights, cuts inward at each turn apex, and
        returns outside by turn exit. The derivative gives a small heading bias
        so the robot aims along the target line instead of only translating to it.
        """
        amplitude = max(float(self.config.safety_racing_line_amplitude_norm), 0.0)
        if amplitude <= 0.0:
            return 0.0, 0.0

        s = float(obs.lap_fraction) * float(self.track.length_m)
        target = self._racing_line_effective_target_at_s(s, amplitude)
        lookahead = float(self.config.safety_racing_line_lookahead_m)
        curve_lookahead = float(self.config.safety_racing_line_curve_lookahead_m)
        if abs(float(obs.curvature_norm)) > 0.5 and curve_lookahead > 0.0:
            lookahead = curve_lookahead
        start_until = float(self.config.safety_racing_line_start_lookahead_until_m)
        start_lookahead = float(self.config.safety_racing_line_start_lookahead_m)
        if start_until > 0.0 and start_lookahead > 0.0 and s <= start_until:
            lookahead = start_lookahead
        if lookahead > 0.0:
            future_target = self._racing_line_effective_target_at_s(s + lookahead, amplitude)
            lateral_delta_m = float(self.track.half_width_m) * (future_target - float(obs.lateral_error_norm))
            heading_bias = math.atan2(lateral_delta_m, max(lookahead, 1e-6))
            return float(target), self.clamp_racing_heading_bias(heading_bias)

        target_slope_per_m = self._racing_line_slope_at_s(s, amplitude)
        heading_bias = math.atan(float(self.track.half_width_m) * target_slope_per_m)
        return float(target), self.clamp_racing_heading_bias(heading_bias)

    def _racing_line_effective_target_at_s(self, s: float, amplitude: float) -> float:
        target = self._racing_line_target_at_s(s, amplitude)
        blend_until = float(self.config.safety_racing_line_start_blend_until_m)
        if blend_until <= 0.0:
            return float(target)
        s_mod = float(s) % float(self.track.length_m)
        if s_mod >= blend_until:
            return float(target)
        alpha = np.clip(s_mod / max(blend_until, 1e-6), 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        start = float(self.config.safety_racing_line_start_blend_from_norm)
        return float((1.0 - alpha) * start + alpha * target)

    def clamp_racing_heading_bias(self, heading_bias: float) -> float:
        limit = float(self.config.safety_racing_line_heading_bias_max_rad)
        if limit <= 0.0:
            return float(heading_bias)
        return float(np.clip(float(heading_bias), -limit, limit))

    def _racing_line_target_at_s(self, s: float, amplitude: float) -> float:
        s = float(s) % float(self.track.length_m)
        straight = float(self.track.straight_length_m)
        turn_len = math.pi * float(self.track.turn_radius_m)
        right_start = straight
        right_end = right_start + turn_len
        left_start = right_end + straight
        left_end = left_start + turn_len

        straight_extra = max(float(self.config.safety_racing_line_straight_outside_extra_norm), 0.0)
        target = -amplitude - straight_extra
        if str(self.config.safety_racing_line_style) == "turn_in_apex":
            turn_amp = self.turn_amplitude(amplitude)
            for turn_start in (right_start, left_start):
                maybe_target = self._racing_line_turn_in_target(
                    s=s,
                    turn_start=turn_start,
                    turn_len=turn_len,
                    outside=target,
                    inside=turn_amp,
                )
                if maybe_target is not None:
                    return float(maybe_target)
            return float(target)
        if right_start <= s < right_end:
            u = (s - right_start) / max(turn_len, 1e-6)
            target = self._racing_line_turn_target(u, self.turn_amplitude(amplitude))
        elif left_start <= s < left_end:
            u = (s - left_start) / max(turn_len, 1e-6)
            target = self._racing_line_turn_target(u, self.turn_amplitude(amplitude))
        return float(target)

    def turn_amplitude(self, amplitude: float) -> float:
        extra = max(float(self.config.safety_racing_line_turn_inside_extra_norm), 0.0)
        return float(amplitude) + extra

    def _racing_line_turn_target(self, u: float, amplitude: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        if str(self.config.safety_racing_line_style) != "late_apex":
            return float(-amplitude * math.cos(2.0 * math.pi * u))

        # Late-apex oval line: stay outside through entry, cut to a later apex,
        # then unwind toward outside before the next straight.
        points = (
            (0.00, -1.00 * amplitude),
            (0.25, -0.85 * amplitude),
            (0.58, 1.00 * amplitude),
            (1.00, -1.00 * amplitude),
        )
        for (u0, y0), (u1, y1) in zip(points[:-1], points[1:]):
            if u <= u1:
                t = (u - u0) / max(u1 - u0, 1e-6)
                t = t * t * (3.0 - 2.0 * t)
                return float((1.0 - t) * y0 + t * y1)
        return float(points[-1][1])

    def _racing_line_turn_in_target(
        self,
        *,
        s: float,
        turn_start: float,
        turn_len: float,
        outside: float,
        inside: float,
    ) -> float | None:
        # Start the diagonal turn-in on the preceding straight instead of at the
        # geometric curve boundary. This makes the target line outside-apex-outside.
        lap_len = float(self.track.length_m)
        entry_m = min(12.0, 0.45 * float(self.track.straight_length_m))
        exit_m = min(12.0, 0.45 * float(self.track.straight_length_m))
        apex_s = float(turn_start) + 0.55 * float(turn_len)
        entry_start = float(turn_start) - entry_m
        exit_end = float(turn_start) + float(turn_len) + exit_m

        entry_span = max(apex_s - entry_start, 1e-6)
        d_entry = (float(s) - entry_start) % lap_len
        if d_entry <= entry_span:
            t = np.clip(d_entry / entry_span, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)
            return float((1.0 - t) * outside + t * inside)

        exit_span = max(exit_end - apex_s, 1e-6)
        d_exit = (float(s) - apex_s) % lap_len
        if d_exit <= exit_span:
            t = np.clip(d_exit / exit_span, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)
            return float((1.0 - t) * inside + t * outside)
        return None

    def _racing_line_slope_at_s(self, s: float, amplitude: float) -> float:
        s = float(s) % float(self.track.length_m)
        straight = float(self.track.straight_length_m)
        turn_len = math.pi * float(self.track.turn_radius_m)
        right_start = straight
        right_end = right_start + turn_len
        left_start = right_end + straight
        left_end = left_start + turn_len

        if right_start <= s < right_end:
            u = (s - right_start) / max(turn_len, 1e-6)
            return float(amplitude * (2.0 * math.pi / max(turn_len, 1e-6)) * math.sin(2.0 * math.pi * u))
        if left_start <= s < left_end:
            u = (s - left_start) / max(turn_len, 1e-6)
            return float(amplitude * (2.0 * math.pi / max(turn_len, 1e-6)) * math.sin(2.0 * math.pi * u))
        return 0.0

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw_rate = np.clip(
            curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)


def mlp_shapes(input_size: int, hidden_sizes: tuple[int, ...], output_size: int) -> list[tuple[tuple[int, int], tuple[int]]]:
    layer_sizes = [int(input_size), *[int(size) for size in hidden_sizes], int(output_size)]
    return [((layer_sizes[idx], layer_sizes[idx + 1]), (layer_sizes[idx + 1],)) for idx in range(len(layer_sizes) - 1)]


def make_mlp_weights(
    *,
    hidden_sizes: tuple[int, ...] = (16, 16),
    seed: int = 0,
    scale: float = 0.10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    weights: list[tuple[np.ndarray, np.ndarray]] = []
    for weight_shape, bias_shape in mlp_shapes(5, hidden_sizes, 3):
        fan_in = max(weight_shape[0], 1)
        weight = rng.normal(0.0, scale / math.sqrt(fan_in), size=weight_shape).astype(np.float32)
        bias = np.zeros(bias_shape, dtype=np.float32)
        weights.append((weight, bias))
    return weights


def load_mlp_weights(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    payload = np.load(path)
    weights: list[tuple[np.ndarray, np.ndarray]] = []
    idx = 0
    while f"W{idx}" in payload and f"b{idx}" in payload:
        weights.append((np.asarray(payload[f"W{idx}"], dtype=np.float32), np.asarray(payload[f"b{idx}"], dtype=np.float32)))
        idx += 1
    if not weights:
        raise ValueError(f"No MLP weights found in {path}. Expected W0/b0 arrays.")
    return weights


def save_mlp_weights(path: Path, weights: list[tuple[np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for idx, (weight, bias) in enumerate(weights):
        payload[f"W{idx}"] = np.asarray(weight, dtype=np.float32)
        payload[f"b{idx}"] = np.asarray(bias, dtype=np.float32)
    np.savez_compressed(path, **payload)
