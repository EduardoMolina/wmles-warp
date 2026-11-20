# WMLES Wrap

Compressible wall-modeled LES solver (explicit multiblock finite volume) in Python + NVIDIA Warp, single-node multi-GPU.

```bash
conda run -n pyvista python -m pytest -q
conda run -n pyvista env PYTHONPATH=src python -m wmles_wrap info cases/channel/case.yaml --write
```

Status: Phase 0 (I/O, connectivity, metrics, partitioning, halo exchange, output, restart). See CLAUDE.md.
