"""Command line interface: ``python -m wmles_wrap <command> ...``"""

from __future__ import annotations

import argparse
import logging
import sys


def _cmd_info(args) -> int:
    from .config import load_config
    from .solver import Solver

    solver = Solver(load_config(args.case))
    print(solver.summary())
    if args.write:
        print(f"wrote {solver.write_output()}")
    return 0


def _cmd_run(args) -> int:
    from .config import load_config
    from .solver import Solver

    cfg = load_config(args.case)
    solver = Solver(cfg)
    print(solver.summary())
    if cfg.run.restart_file:
        solver.read_restart(cfg.case_dir / cfg.run.restart_file)
    solver.run()
    print(f"finished at step {solver.step}, t = {solver.time:.6g}; output in {solver.out_dir}")
    return 0


def _cmd_extrude(args) -> int:
    from .io.extrude import extrude_block
    from .io.plot3d import read_plot3d, write_plot3d

    blocks = [extrude_block(b, args.nk, args.length) for b in read_plot3d(args.input)]
    write_plot3d(args.output, blocks)
    print(f"wrote {args.output}: {[b.shape[:3] for b in blocks]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wmles_wrap", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="set up the case and print grid/partition/metrics summary")
    p.add_argument("case")
    p.add_argument("--write", action="store_true", help="also write the initial field")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("run", help="run a case")
    p.add_argument("case")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("extrude", help="extrude a 2D Plot3D grid in z")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--nk", type=int, required=True)
    p.add_argument("--length", type=float, required=True)
    p.set_defaults(func=_cmd_extrude)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
