"""
Diverse Temperature Profile Families for Batch Cooling Crystallization.

Defines analytical and piecewise-differentiable cooling profiles T(t) and
associated cooling rates cr(t) = -dT/dt:
1. LinearProfile
2. CubicProfile (slow start, accelerated cooling)
3. NaturalProfile (exponential decay)
4. TwoStageProfile (two distinct cooling slopes)
5. CoolHoldCoolProfile (cool, isothermal desupersaturation hold, cool)
6. RandomMonotoneSplineProfile (piecewise linear with random positive increments)
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTemperatureProfile(ABC):
    """Abstract base class for batch temperature profiles."""

    def __init__(self, T_seed: float, T_final: float, duration: float, name: str = "BaseProfile"):
        self.T_seed = float(T_seed)
        self.T_final = float(T_final)
        self.duration = float(duration)
        self.name = name

    @abstractmethod
    def get_temperature(self, t: float) -> float:
        """Returns slurry temperature T(t) in °C."""
        pass

    @abstractmethod
    def get_cooling_rate(self, t: float) -> float:
        """Returns instantaneous cooling rate cr(t) = -dT/dt in °C/s."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_type": self.name,
            "T_seed": self.T_seed,
            "T_final": self.T_final,
            "duration": self.duration,
        }


class LinearProfile(BaseTemperatureProfile):
    """Linear cooling: T(t) = Ts - (Ts - Tf) * (t / duration)."""

    def __init__(self, T_seed: float, T_final: float, duration: float):
        super().__init__(T_seed, T_final, duration, name="Linear")
        self.cr_const = (self.T_seed - self.T_final) / self.duration

    def get_temperature(self, t: float) -> float:
        t_clamped = min(max(float(t), 0.0), self.duration)
        return self.T_seed - self.cr_const * t_clamped

    def get_cooling_rate(self, t: float) -> float:
        if 0.0 <= t <= self.duration:
            return self.cr_const
        return 0.0


class CubicProfile(BaseTemperatureProfile):
    """
    Cubic / Slow-Start cooling:
    T(t) = Ts - (Ts - Tf) * (t / duration)^3
    cr(t) = 3 * (Ts - Tf) / duration * (t / duration)^2
    Allows seeds to grow with minimal early secondary nucleation.
    """

    def __init__(self, T_seed: float, T_final: float, duration: float):
        super().__init__(T_seed, T_final, duration, name="Cubic")
        self.delta_T = self.T_seed - self.T_final

    def get_temperature(self, t: float) -> float:
        tau = min(max(float(t), 0.0), self.duration) / self.duration
        return self.T_seed - self.delta_T * (tau ** 3)

    def get_cooling_rate(self, t: float) -> float:
        if 0.0 <= t <= self.duration:
            tau = float(t) / self.duration
            return (3.0 * self.delta_T / self.duration) * (tau ** 2)
        return 0.0


class NaturalProfile(BaseTemperatureProfile):
    """
    Natural / Exponential cooling:
    Models ambient heat rejection where heat transfer slows as slurry approaches coolant temperature.
    """

    def __init__(self, T_seed: float, T_final: float, duration: float, decay_factor: float = 2.5):
        super().__init__(T_seed, T_final, duration, name="NaturalExponential")
        self.k_rate = float(decay_factor) / self.duration
        self.delta_T = self.T_seed - self.T_final
        self.norm = 1.0 - math.exp(-self.k_rate * self.duration)

    def get_temperature(self, t: float) -> float:
        t_clamped = min(max(float(t), 0.0), self.duration)
        frac = (1.0 - math.exp(-self.k_rate * t_clamped)) / self.norm
        return self.T_seed - self.delta_T * frac

    def get_cooling_rate(self, t: float) -> float:
        if 0.0 <= t <= self.duration:
            return (self.delta_T * self.k_rate / self.norm) * math.exp(-self.k_rate * float(t))
        return 0.0


class TwoStageProfile(BaseTemperatureProfile):
    """
    Two-Stage linear cooling:
    Stage 1: Ts -> T_mid over duration * split_frac
    Stage 2: T_mid -> Tf over the remaining duration.
    """

    def __init__(
        self,
        T_seed: float,
        T_final: float,
        duration: float,
        T_mid: float = 30.0,
        split_frac: float = 0.4,
    ):
        super().__init__(T_seed, T_final, duration, name="TwoStage")
        self.T_mid = float(T_mid)
        self.t1 = self.duration * float(split_frac)
        self.cr1 = (self.T_seed - self.T_mid) / max(self.t1, 1.0)
        self.cr2 = (self.T_mid - self.T_final) / max(self.duration - self.t1, 1.0)

    def get_temperature(self, t: float) -> float:
        t_val = min(max(float(t), 0.0), self.duration)
        if t_val <= self.t1:
            return self.T_seed - self.cr1 * t_val
        else:
            return self.T_mid - self.cr2 * (t_val - self.t1)

    def get_cooling_rate(self, t: float) -> float:
        if 0.0 <= t < self.t1:
            return self.cr1
        elif self.t1 <= t <= self.duration:
            return self.cr2
        return 0.0


class CoolHoldCoolProfile(BaseTemperatureProfile):
    """
    Cool - Hold - Cool profile:
    Cool down to an intermediate hold temperature, hold isothermally to let
    supersaturation decay via crystal growth, then cool to final temperature.
    """

    def __init__(
        self,
        T_seed: float,
        T_final: float,
        duration: float,
        T_hold: float = 30.5,
        hold_start_frac: float = 0.25,
        hold_end_frac: float = 0.60,
    ):
        super().__init__(T_seed, T_final, duration, name="CoolHoldCool")
        self.T_hold = float(T_hold)
        self.t_hold_start = self.duration * float(hold_start_frac)
        self.t_hold_end = self.duration * float(hold_end_frac)
        self.cr1 = (self.T_seed - self.T_hold) / max(self.t_hold_start, 1.0)
        self.cr2 = (self.T_hold - self.T_final) / max(self.duration - self.t_hold_end, 1.0)

    def get_temperature(self, t: float) -> float:
        t_val = min(max(float(t), 0.0), self.duration)
        if t_val < self.t_hold_start:
            return self.T_seed - self.cr1 * t_val
        elif t_val <= self.t_hold_end:
            return self.T_hold
        else:
            return self.T_hold - self.cr2 * (t_val - self.t_hold_end)

    def get_cooling_rate(self, t: float) -> float:
        if 0.0 <= t < self.t_hold_start:
            return self.cr1
        elif self.t_hold_start <= t <= self.t_hold_end:
            return 0.0
        elif self.t_hold_end < t <= self.duration:
            return self.cr2
        return 0.0


class RandomMonotoneSplineProfile(BaseTemperatureProfile):
    """
    Random Monotone Piecewise-Linear Profile:
    Simulates dynamic MPC-generated cooling profiles by partitioning the batch
    into M segments with random non-negative cooling rates.
    """

    def __init__(
        self,
        T_seed: float,
        T_final: float,
        duration: float,
        num_segments: int = 5,
        seed: int = None,
    ):
        super().__init__(T_seed, T_final, duration, name="RandomMonotoneSpline")
        rng = np.random.default_rng(seed)
        self.num_segments = int(num_segments)
        self.t_knots = np.linspace(0.0, self.duration, self.num_segments + 1)
        
        # Generate random positive temperature drops that sum exactly to (T_seed - T_final)
        rand_weights = rng.uniform(0.2, 1.0, size=self.num_segments)
        delta_T_total = self.T_seed - self.T_final
        drops = (rand_weights / np.sum(rand_weights)) * delta_T_total
        
        self.T_knots = np.zeros(self.num_segments + 1, dtype=np.float64)
        self.T_knots[0] = self.T_seed
        for i in range(self.num_segments):
            self.T_knots[i + 1] = self.T_knots[i] - drops[i]
        self.T_knots[-1] = self.T_final

        self.segment_durations = np.diff(self.t_knots)
        self.cr_segments = drops / self.segment_durations

    def get_temperature(self, t: float) -> float:
        t_val = min(max(float(t), 0.0), self.duration)
        idx = int(np.searchsorted(self.t_knots, t_val, side="right") - 1)
        idx = min(max(idx, 0), self.num_segments - 1)
        dt = t_val - self.t_knots[idx]
        return float(self.T_knots[idx] - self.cr_segments[idx] * dt)

    def get_cooling_rate(self, t: float) -> float:
        if 0.0 <= t <= self.duration:
            idx = int(np.searchsorted(self.t_knots, float(t), side="right") - 1)
            idx = min(max(idx, 0), self.num_segments - 1)
            return float(self.cr_segments[idx])
        return 0.0


def sample_random_profile(
    T_seed: float,
    T_final: float,
    duration: float,
    profile_type: str = "random",
    rng: np.random.Generator = None,
) -> BaseTemperatureProfile:
    """Factory function to sample or construct a cooling profile."""
    if rng is None:
        rng = np.random.default_rng()

    types = ["linear", "cubic", "natural", "two_stage", "cool_hold_cool", "random_spline"]
    p_type = rng.choice(types) if profile_type == "random" else profile_type.lower()

    if p_type == "linear":
        return LinearProfile(T_seed, T_final, duration)
    elif p_type == "cubic":
        return CubicProfile(T_seed, T_final, duration)
    elif p_type == "natural":
        decay = float(rng.uniform(1.8, 3.5))
        return NaturalProfile(T_seed, T_final, duration, decay_factor=decay)
    elif p_type == "two_stage":
        T_mid = float(rng.uniform(T_final + 0.3 * (T_seed - T_final), T_seed - 0.2 * (T_seed - T_final)))
        split = float(rng.uniform(0.3, 0.7))
        return TwoStageProfile(T_seed, T_final, duration, T_mid=T_mid, split_frac=split)
    elif p_type == "cool_hold_cool":
        T_hold = float(rng.uniform(T_final + 0.35 * (T_seed - T_final), T_seed - 0.25 * (T_seed - T_final)))
        h_start = float(rng.uniform(0.15, 0.35))
        h_end = float(rng.uniform(h_start + 0.20, min(h_start + 0.45, 0.85)))
        return CoolHoldCoolProfile(T_seed, T_final, duration, T_hold=T_hold, hold_start_frac=h_start, hold_end_frac=h_end)
    else:
        segs = int(rng.integers(4, 8))
        return RandomMonotoneSplineProfile(T_seed, T_final, duration, num_segments=segs, seed=int(rng.integers(1e6)))
