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

# The host-chosen symmetric per-tensor quantization scales (zero
# points are 0): Q_real = s_q * Q_int, K_real = s_k * K_int,
# V_real = s_v * V_int. The score logit scale s_logit = s_q * s_k is
# applied to S before the softmax, and the final O/l is scaled by
# s_final = s_v / Q8_SCALE (the Q8_SCALE folds in the P -> int8
# quantization). The assignment's fixed 16-bit array output bounds
# the operand magnitudes:
#   |S_int| <= OUT_MAX  =>  |q_int|max * |k_int|max <= 255 (d = 128),
#   |O_int| <= OUT_MAX  =>  |v_int|max <= 16 (|P_q8| <= Q8_SCALE),
# which the integer ranges below satisfy.
SQ = 0.3
SK = 0.3
SV = 0.0625
Q8_SCALE = 127.0
OUT_MIN = -32768.0
OUT_MAX = 32767.0
S_LOGIT = np.float32(SQ) * np.float32(SK)  # ~ 0.09 ~ 1/sqrt(128)
S_FINAL = np.float32(SV) / np.float32(Q8_SCALE)

Q_RANGE = (-15, 15)
KV_RANGE = (-15, 15)
V_RANGE = (-16, 16)


def fp32_bits(x: np.float32) -> int:
    return int(x.view(np.uint32))


def golden_reference(Q, K, V, tiles):
    O = np.zeros((BR, D), dtype=np.float32)
    m = np.full(BR, -np.inf, dtype=np.float32)
    l = np.zeros(BR, dtype=np.float32)
    c = np.ones(BR, dtype=np.float32)
    for j in range(tiles):
        S = Q.astype(np.float32) @ K[j].T.astype(np.float32)
        S = np.clip(S, OUT_MIN, OUT_MAX)
        S = f16_bits_to_f32(f32_to_f16_trunc(S))
        S = S * S_LOGIT
        row_max = S.max(axis=1)
        m_new = np.maximum(m, row_max)
        scale = np.where(m == m_new, np.float32(1.0),
                         np.exp(m - m_new).astype(np.float32))
        P64 = np.exp(S - m_new[:, None])
        P16 = f16_bits_to_f32(f32_to_f16_trunc(P64))
        q8 = np.trunc(P16 * Q8_SCALE).astype(np.int8).reshape(BR, BC)
        l = scale * l + P64.sum(axis=1)
        c = scale
        m = m_new
        O = c[:, None] * O + q8.astype(np.float32) @ V[j].astype(np.float32)
    return (O / l[:, None]) * S_FINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0x5EED)
    parser.add_argument("--regime", choices=["prefill", "decode"],
                        default="prefill")
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
                             "<out dir>/flash_data.bin")
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
    q_lo, q_hi = Q_RANGE
    k_lo, k_hi = KV_RANGE
    v_lo, v_hi = V_RANGE
    Q = rng.integers(q_lo, q_hi + 1, size=(BR, D), dtype=np.int8)
    K = rng.integers(k_lo, k_hi + 1, size=(args.tiles, BC, D), dtype=np.int8)
    V = rng.integers(v_lo, v_hi + 1, size=(args.tiles, BC, D), dtype=np.int8)
    if args.regime == "decode":
        Q[1:, :] = 0

    s_bound = np.abs(Q.astype(np.int32) @
                     np.transpose(K, (0, 2, 1)).astype(np.int32)).max()
    assert s_bound <= OUT_MAX, f"|S| bound violated: {s_bound}"
    o_bound = BC * Q8_SCALE * max(abs(v_lo), abs(v_hi))
    assert o_bound <= OUT_MAX, f"|O| bound violated: {o_bound}"

    if args.gen_input:
        if args.emit_header:
            with open(args.out, "w") as f:
                f.write("// Generated flash-attention data (slim header --- "
                        "tensor\n")
                f.write("// bytes live in flash_data.bin, loaded via "
                        "m5_read_file).\n")
                f.write("#pragma once\n#include <cstdint>\n\n")
                f.write(f"constexpr int FLASH_TILES = {args.tiles};\n")
                f.write("// The host's symmetric per-tensor quantization "
                        "scales\n")
                f.write("// (fp32 bit patterns; the vector hart has no "
                        "DRAM route).\n")
                f.write("// s_logit = s_q * s_k, applied to the scores "
                        "before exp;\n")
                f.write("// s_final = s_v / Q8_SCALE, applied to O/l at "
                        "finalize.\n")
                f.write("constexpr std::uint32_t FLASH_LOGIT_SCALE_BITS = "
                        f"0x{fp32_bits(S_LOGIT):08X}u;\n")
                f.write("constexpr std::uint32_t FLASH_FINAL_SCALE_BITS = "
                        f"0x{fp32_bits(S_FINAL):08X}u;\n")
            log.warning("flash_data.hh written: tiles=%d regime=%s",
                        args.tiles, args.regime)
        if args.emit_bin:
            with open(bin_path, "wb") as f:
                f.write(Q.astype(np.int8).tobytes())
                f.write(K.astype(np.int8).tobytes())
                f.write(np.transpose(V, (0, 2, 1)).astype(np.int8).tobytes())
            log.warning("flash_data.bin: %d B -> %s",
                        os.path.getsize(bin_path), bin_path)

    if args.gen_reference:
        final = golden_reference(Q, K, V, args.tiles)
        with open(ref_path, "wb") as f:
            f.write(final.astype(np.float32).tobytes())
        log.warning("O_ref.bin: %d B -> %s", os.path.getsize(ref_path),
                    ref_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    main()
