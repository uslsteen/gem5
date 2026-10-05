#!/usr/bin/env python3
"""Validate the simulated flash-attention output against the golden one"""

import argparse
import logging

import numpy as np

log = logging.getLogger("validate_flash")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True,
                        help="simulated output (O_dump.bin)")
    parser.add_argument("--ref", required=True,
                        help="golden output (O_ref.bin)")
    parser.add_argument("--shape", type=int, nargs=2, default=[16, 16],
                        metavar=("ROWS", "COLS"))
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="relative-error tolerance")
    args = parser.parse_args()

    sim = np.fromfile(args.sim, dtype=np.float32).reshape(args.shape)
    ref = np.fromfile(args.ref, dtype=np.float32).reshape(args.shape)

    mae = np.abs(sim - ref).max()
    rel = mae / (np.abs(ref).max() + 1e-9)
    ok = rel < args.tol

    log.warning("shape        : %dx%d", args.shape[0], args.shape[1])
    log.warning("max abs err  : %.6e", mae)
    log.warning("max rel err  : %.6e (tolerance %.1e)", rel, args.tol)
    log.warning("verdict      : %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    raise SystemExit(main())
