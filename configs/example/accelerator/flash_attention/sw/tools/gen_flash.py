#!/usr/bin/env python3
"""Generate the flash-attention benchmark data"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import f32_to_f16_trunc, f16_bits_to_f32 
from consts import BR, BC, D

log = logging.getLogger("gen_flash")


def golden_reference(Q, K, V, tiles, mask):
    O = np.zeros((BR, D), dtype=np.float32)
    m = np.full(BR, -np.inf, dtype=np.float32)
    l = np.zeros(BR, dtype=np.float32)
    c = np.ones(BR, dtype=np.float32)
    for j in range(tiles):
        S = Q.astype(np.float32) @ K[j].T.astype(np.float32)  # BR x BC
        if mask is not None and j == tiles - 1:
            S = np.where(mask, -np.inf, S)
        row_max = S.max(axis=1)
        m_new = np.maximum(m, row_max)
        scale = np.where(m == m_new, np.float32(1.0),
                         np.exp(m - m_new).astype(np.float32))
        P64 = np.exp(S - m_new[:, None])
        P16 = f16_bits_to_f32(f32_to_f16_trunc(P64))
        q8 = np.trunc(P16 * 127.0).astype(np.int8).reshape(BR, BC)
        l = scale * l + P64.sum(axis=1)
        c = scale
        m = m_new
        O = c[:, None] * O + q8.astype(np.float32) @ V[j].astype(np.float32)
    return O / l[:, None]


def arr(name, dtype, a):
    lines = [f"constexpr {dtype} {name}[{a.size}] = {{"]
    row = []
    for v in a.ravel():
        row.append(str(int(v)))
        if len(row) == 16:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0x5EED)
    parser.add_argument("--regime", choices=["prefill", "decode"],
                        default="prefill")
    parser.add_argument("--causal", action="store_true",
                        help="mask the strict upper triangle of the LAST "
                             "tile (causal boundary); values only")
    parser.add_argument("--gen-input", action="store_true",
                        help="generate the input tensors")
    parser.add_argument("--gen-reference", action="store_true",
                        help="generate the golden final O/l")
    parser.add_argument("--emit-header", action="store_true", default=True,
                        help="emit flash_data.hh (default: on)")
    parser.add_argument("--emit-bin", action="store_true",
                        help="emit flash_data.bin (default: off)")
    parser.add_argument(
        "--out", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "build", "flash_data.hh"))
    parser.add_argument("--input-bin", type=str, default=None,
                        help="packed Q|K|V image (default: "
                             "<out dir>/flash_data.bin)")
    parser.add_argument("--reference-bin", type=str, default=None,
                        help="golden O/l (default: <out dir>/O_ref.bin)")
    args = parser.parse_args()

    if not (args.gen_input or args.gen_reference):
        parser.error("at least one of --gen-input / --gen-reference "
                     "is required")

    out_dir = os.path.dirname(args.out)
    bin_path = args.input_bin or os.path.join(out_dir, "flash_data.bin")
    ref_path = args.reference_bin or os.path.join(out_dir, "O_ref.bin")

    rng = np.random.default_rng(args.seed)
    Q = rng.integers(-8, 9, size=(BR, D), dtype=np.int8)
    K = rng.integers(-8, 9, size=(args.tiles, BC, D), dtype=np.int8)
    V = rng.integers(-8, 9, size=(args.tiles, BC, D), dtype=np.int8)
    if args.regime == "decode":
        Q[1:, :] = 0

    mask = None
    if args.causal:
        mask = np.zeros((BR, BC), dtype=bool)
        for r in range(BR):
            mask[r, r + 1:] = True

    if args.gen_input:
        if args.emit_header:
            with open(args.out, "w") as f:
                f.write("// Generated flash-attention data (slim header вЂ” "
                        "tensor\n")
                f.write("// bytes live in flash_data.bin, loaded via "
                        "m5_read_file).\n")
                f.write("#pragma once\n#include <cstdint>\n\n")
                f.write(f"constexpr int FLASH_TILES = {args.tiles};\n")
                f.write(f"constexpr bool FLASH_MASKED = "
                        f"{str(mask is not None).lower()};\n")
                if mask is not None:
                    f.write(arr("FLASH_MASK", "uint8_t",
                                mask.astype(np.uint8)))
                    f.write("\n\n")
                else:
                    f.write("constexpr uint8_t FLASH_MASK[1] = {0};\n")
            log.warning("flash_data.hh written: tiles=%d regime=%s masked=%s",
                        args.tiles, args.regime, mask is not None)
        if args.emit_bin:
            with open(bin_path, "wb") as f:
                f.write(Q.astype(np.int8).tobytes())
                f.write(K.astype(np.int8).tobytes())
                f.write(np.transpose(V, (0, 2, 1)).astype(np.int8).tobytes())
            log.warning("flash_data.bin: %d B -> %s",
                        os.path.getsize(bin_path), bin_path)

    if args.gen_reference:
        final = golden_reference(Q, K, V, args.tiles, mask)
        with open(ref_path, "wb") as f:
            f.write(final.astype(np.float32).tobytes())
        log.warning("O_ref.bin: %d B -> %s", os.path.getsize(ref_path),
                    ref_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    main()
