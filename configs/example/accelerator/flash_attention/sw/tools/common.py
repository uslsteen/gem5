from __future__ import annotations

import math

import numpy as np

from consts import (
    F32_SIGN_SHIFT, F32_MANT_BITS, F32_EXP_MASK, F32_MANT_MASK,
    F32_IMPLICIT, F32_EXP_BIAS,
    F16_SIGN_SHIFT, F16_EXP_SHIFT, F16_EXP_MASK, F16_MANT_MASK,
    F16_INF_BITS, F16_EXP_BIAS, F16_EXP_MIN, F16_EXP_MAX, F16_UFLOW_GUARD,
    F32_TO_F16_SHIFT,
)

"""Shared fp16/fp32 bit helpers."""

def f32_to_f16_trunc(x64: np.ndarray) -> np.ndarray:
    bits = np.asarray(x64, dtype=np.float32).view(np.uint32)
    sign = (bits >> F32_SIGN_SHIFT) & 1
    exp_s = ((bits >> F32_MANT_BITS) & F32_EXP_MASK).astype(np.int32) \
        - F32_EXP_BIAS
    mant = bits & F32_MANT_MASK

    out = np.zeros(bits.shape, dtype=np.uint32)
    m_norm = (exp_s >= F16_EXP_MIN) & (exp_s <= F16_EXP_MAX)
    m_sub = (exp_s >= F16_EXP_MIN - F16_UFLOW_GUARD) & (exp_s <= F16_EXP_MIN - 1)
    m_inf = exp_s > F16_EXP_MAX
    m_zero = exp_s < F16_EXP_MIN - F16_UFLOW_GUARD

    norm = ((exp_s + F16_EXP_BIAS).astype(np.uint32) << F16_EXP_SHIFT) \
        | (mant >> F32_TO_F16_SHIFT) | (sign << F16_SIGN_SHIFT)
    shift = (-1 - exp_s).astype(np.uint32)
    sub = ((mant | F32_IMPLICIT) >> shift) | (sign << F16_SIGN_SHIFT)
    inf = (sign << F16_SIGN_SHIFT) | F16_INF_BITS
    zero = sign << F16_SIGN_SHIFT

    out[m_norm] = norm[m_norm]
    out[m_sub] = sub[m_sub]
    out[m_inf] = inf[m_inf]
    out[m_zero] = zero[m_zero]
    return out.astype(np.uint16)


def f16_bits_to_f32(h: np.ndarray) -> np.ndarray:
    bits = np.asarray(h, dtype=np.uint16).astype(np.uint32)
    sign = np.where((bits >> F16_SIGN_SHIFT) & 1, -1.0, 1.0)
    exp = (bits >> F16_EXP_SHIFT) & F16_EXP_MASK
    mant = (bits & F16_MANT_MASK).astype(np.float64)

    out = np.zeros(bits.shape, dtype=np.float64)
    m_norm = exp != 0
    m_sub = exp == 0
    m_inf = exp == F16_EXP_MASK

    out[m_norm] = np.ldexp(1.0 + mant[m_norm] / 1024.0,
                           exp[m_norm].astype(np.int64) - F16_EXP_BIAS)
    out[m_sub] = np.ldexp(mant[m_sub] / 1024.0, 1 - F16_EXP_BIAS)
    out[m_inf] = np.inf
    return (sign * out).astype(np.float32)


def fmt_float(v: float) -> str:
    v = float(v)
    if math.isnan(v):
        return '__builtin_nanf("")'
    if math.isinf(v):
        return "-__builtin_inff()" if v < 0 else "__builtin_inff()"
    return f"{v:+.9e}f"
