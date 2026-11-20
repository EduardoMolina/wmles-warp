"""Device discovery and peer-to-peer setup for single-process multi-GPU runs."""

from __future__ import annotations

import warp as wp


def init_warp(verbose: bool = False) -> None:
    """Initialize Warp once with solver-wide settings (quiet logging, no autodiff codegen)."""
    wp.config.log_level = wp.LOG_INFO if verbose else wp.LOG_WARNING
    wp.config.enable_backward = False
    wp.init()


def resolve_devices(spec: str | list[str]) -> list[str]:
    """'auto' -> all CUDA devices (or ['cpu'] if none); otherwise validate the given list."""
    init_warp()
    if spec == "auto":
        n = wp.get_cuda_device_count()
        return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
    if isinstance(spec, str):
        spec = [spec]
    available = {str(d) for d in wp.get_devices()} | {"cpu"}
    for d in spec:
        if d not in available:
            raise ValueError(f"device {d!r} not available; have {sorted(available)}")
    return list(spec)


def enable_peer_access(devices: list[str]) -> dict[tuple[str, str], bool]:
    """Enable P2P between every CUDA device pair that supports it. Returns the support table.
    Pairs without P2P fall back to host-staged copies inside ``wp.copy``."""
    table = {}
    cuda = [d for d in devices if d.startswith("cuda")]
    for a in cuda:
        for b in cuda:
            if a == b:
                continue
            ok = bool(wp.is_peer_access_supported(a, b))
            if ok:
                wp.set_peer_access_enabled(a, b, True)
            table[(a, b)] = ok
    return table


def resolve_partitions(spec: int | str, devices: list[str]) -> int:
    if spec == "auto":
        return len(devices)
    n = int(spec)
    if n < 1:
        raise ValueError("parallel.partitions must be >= 1")
    return n
