"""
Creates a common frequency grid for time-domain IFFT starting at DC
for all models to resample imported data on.

Time-domain properties (with numpy.fft.irfft):
    dt = 1 / (2 * f_max) -> temporal resolution
    t_window = 1 / df -> time length before wraparound
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class FrequencyGrid:
    f_max: float  # [Hz] highest freq (inclusive)
    n: int  # number of points

    def __post_init__(self) -> None:
        if self.f_max <= 0:
            raise ValueError(f"f_max must be > 0, got {self.f_max}")
        if self.n < 2:
            raise ValueError(f"n must be >= 2, got {self.n}")

    @classmethod
    def from_resolution(cls, f_max: float, df: float) -> "FrequencyGrid":
        # grid with spacing <= df rounded to land on discrete point
        if df <= 0 or df > f_max:
            raise ValueError("need 0 < df <= f_max")
        return cls(f_max=f_max, n=int(np.ceil(f_max / df)) + 1)

    @cached_property
    def f(self) -> np.ndarray:
        # frequency points [Hz] ranging from 0, df, 2df, ..., f_max
        return np.linspace(0.0, self.f_max, self.n)

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * self.f

    @property
    def df(self) -> float:
        return self.f_max / (self.n - 1)

    @property
    def dt(self) -> float:
        # IFFT result time step [s]
        return 1.0 / (2.0 * self.f_max)

    @property
    def t_window(self) -> float:
        # IFFT window time span [s]
        return 1.0 / self.df
