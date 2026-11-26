"""Precision handling.

Kernels are produced by factory functions parametrized on a :class:`Precision`, so a
single code path serves float32 (production on FP64-weak GPUs such as the L40S) and
float64 (verification). Factories are cached so each (kernel, precision) pair compiles once.

The implicit RANS solver is mixed precision (docs/plan_rans.md): the residual and the state in
one precision (float64 by default) and the linear line relaxation (5x5 blocks, corrections) in
another (float32). :class:`MixedPrecision` pairs the two; factories that take it are cached the
same way.
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
    mat55: type = None  # 5x5 block (implicit Jacobians)

    @property
    def eps(self) -> float:
        return float(np.finfo(self.np_dtype).eps)


_PRECISIONS = {
    "float32": Precision("float32", wp.float32, wp.vec3f, wp.types.vector(5, wp.float32), np.float32,
                         wp.types.matrix((5, 5), wp.float32)),
    "float64": Precision("float64", wp.float64, wp.vec3d, wp.types.vector(5, wp.float64), np.float64,
                         wp.types.matrix((5, 5), wp.float64)),
}


def get_precision(name: str) -> Precision:
    try:
        return _PRECISIONS[name]
    except KeyError:
        raise ValueError(f"unknown precision {name!r}; expected one of {sorted(_PRECISIONS)}") from None


@dataclass(frozen=True)
class MixedPrecision:
    """Residual/state precision and linear-solve precision of the implicit solver."""

    residual: Precision
    solve: Precision

    @property
    def name(self) -> str:
        return f"{self.residual.name}/{self.solve.name}"


def get_mixed(residual: str = "float64", solve: str = "float32") -> MixedPrecision:
    return MixedPrecision(get_precision(residual), get_precision(solve))


def kernel_factory(fn):
    """Decorator for ``fn(prec: Precision, ...) -> kernels``; caches per argument tuple."""
    return functools.lru_cache(maxsize=None)(fn)
