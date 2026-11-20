"""Precision handling.

Kernels are produced by factory functions parametrized on a :class:`Precision`, so a
single code path serves float32 (production on FP64-weak GPUs such as the L40S) and
float64 (verification). Factories are cached so each (kernel, precision) pair compiles once.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(frozen=True)
class Precision:
    name: str
    scalar: type
    vec3: type
    vec5: type
    np_dtype: type

    @property
    def eps(self) -> float:
        return float(np.finfo(self.np_dtype).eps)


_PRECISIONS = {
    "float32": Precision("float32", wp.float32, wp.vec3f, wp.types.vector(5, wp.float32), np.float32),
    "float64": Precision("float64", wp.float64, wp.vec3d, wp.types.vector(5, wp.float64), np.float64),
}


def get_precision(name: str) -> Precision:
    try:
        return _PRECISIONS[name]
    except KeyError:
        raise ValueError(f"unknown precision {name!r}; expected one of {sorted(_PRECISIONS)}") from None


def kernel_factory(fn):
    """Decorator for ``fn(prec: Precision, ...) -> kernels``; caches per argument tuple."""
    return functools.lru_cache(maxsize=None)(fn)
