"""Fused Metal kernel for the self-spec canvas gated-delta step.

One launch per GDN layer replaces (kernel over clean rows) + (kernel over MASK rows) +
(repeat + einsum block-end readout):

    for t in 0..T-1:   S = g_t * S ;  S += k_t (v_t - S k_t) beta_t      (the recurrence)
                       t <  NC : y_t = S q_t      (token-causal clean rows)
                       t == NC-1: s_clean = S     (state after the clean rows, for commit)
    for t in NC..T-1:  y_t = S_end q_t            (MASK rows read the block-end state)

The per-step arithmetic of the clean rows is the same as mlx-lm's gated_delta kernel (same
thread layout, same float accumulation order), so the clean rows stay bit-compatible with
the AR step. Derived from mlx-lm's ``gated_delta.py`` kernel (MIT, Apple Inc.).
"""
from __future__ import annotations

import mlx.core as mx

_SOURCE = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_clean = s_clean + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (t < NC && thread_index_in_simdgroup == 0) {
            y_[dv_idx] = static_cast<InT>(out);
          }
          if (t == NC - 1) {
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_clean[s_idx] = static_cast<StT>(state[i]);
            }
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        // MASK rows: read out the block-end state
        auto q2 = q + b_idx * T * Hk * Dk + hk_idx * Dk + NC * Hk * Dk;
        auto y2 = y + b_idx * T * Hv * Dv + hv_idx * Dv + NC * Hv * Dv;
        for (int t = NC; t < T; ++t) {
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            out += state[i] * q2[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y2[dv_idx] = static_cast<InT>(out);
          }
          q2 += Hk * Dk;
          y2 += Hv * Dv;
        }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="trida_selfspec_canvas_gdn",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "s_clean"],
            source=_SOURCE,
        )
    return _KERNEL


def fused_available() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def canvas_gdn_fused(q, k, v, g, beta, state, n_clean: int):
    """q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv]; state: [B,Hv,Dv,Dk] (float32).
    Returns (y [B,T,Hv,Dv], s_clean [B,Hv,Dv,Dk])."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32:
        raise ValueError("fused canvas GDN kernel needs Dk % 32 == 0")
    return _kernel()(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                  ("NC", n_clean)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
