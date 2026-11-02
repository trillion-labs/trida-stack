"""CPU correctness test for the two-stream GDN decode crux — no GPU, no vLLM.

The hard, bug-prone part of the vLLM native port (DESIGN.md "The crux") is the
GDN state handling for block diffusion: the *noisy* pass must re-scan a block
starting from the *clean* block-entry recurrent state, WITHOUT corrupting the
committed clean state. This test pins that algorithm in pure PyTorch on tiny
tensors, so GPU time later goes to vLLM integration, not algorithm debugging.

It uses a minimal reference gated-delta recurrence (the delta rule: nudge state
S so that S·k ≈ v, with per-step gate/decay) — not the production Triton kernel,
but the same state-carry / snapshot-restore semantics the port must preserve.

Run (CPU):  python test_two_stream_cpu.py   (needs torch only)
"""
import torch

torch.manual_seed(0)
D = 8  # tiny head dim


def gdn_step(S, k, v, q, beta, alpha):
    """One gated-delta step. S:(D,D) recurrent state. Returns (S_new, readout)."""
    Sk = S @ k                                   # current prediction for k
    S_new = alpha * S + beta * torch.outer(v - Sk, k)   # delta-rule rank-1 correction
    return S_new, S_new @ q


def scan(S0, K, V, Q, beta, alpha, record_block_entry=None):
    """Sequential scan from state S0 over T tokens. If record_block_entry is a
    block size, also return the state at the entry of each block."""
    S = S0.clone()
    outs, entries = [], []
    T = K.shape[0]
    for t in range(T):
        if record_block_entry and t % record_block_entry == 0:
            entries.append(S.clone())
        S, o = gdn_step(S, K[t], V[t], Q[t], beta[t], alpha[t])
        outs.append(o)
    return S, torch.stack(outs), entries


def two_stream_noisy(block_entry_state, Kb, Vb, Qb, betab, alphab):
    """Noisy pass: re-scan a block from the clean block-entry state.
    MUST NOT mutate block_entry_state (snapshot/restore invariant)."""
    S_final, outs, _ = scan(block_entry_state, Kb, Vb, Qb, betab, alphab)
    return S_final, outs


def rand_seq(T):
    return (torch.randn(T, D), torch.randn(T, D), torch.randn(T, D),
            torch.rand(T), torch.rand(T) * 0.5 + 0.5)  # beta in [0,1), alpha in [0.5,1)


def test_snapshot_invariant():
    """Noisy re-scan must not corrupt the clean block-entry state."""
    S_entry = torch.randn(D, D)
    K, V, Q, b, a = rand_seq(4)
    snap = S_entry.clone()
    two_stream_noisy(S_entry, K, V, Q, b, a)
    assert torch.equal(S_entry, snap), "noisy pass mutated the clean entry state!"


def test_noisy_equals_clean_on_true_block():
    """If the noisy block equals the clean tokens, the noisy re-scan from the
    block-entry state must reproduce the clean scan's outputs for that block."""
    K, V, Q, b, a = rand_seq(8)  # 2 blocks of 4
    S_final_clean, outs_clean, entries = scan(torch.zeros(D, D), K, V, Q, b, a,
                                              record_block_entry=4)
    # re-scan block 1 (tokens 4:8) from its recorded entry state
    S_entry_b1 = entries[1]
    _, outs_noisy = two_stream_noisy(S_entry_b1, K[4:8], V[4:8], Q[4:8], b[4:8], a[4:8])
    assert torch.allclose(outs_noisy, outs_clean[4:8], atol=1e-5), \
        "noisy re-scan from block-entry state diverged from the clean scan"


def test_block_boundary_carry():
    """State carried across a block boundary must equal a single continuous scan."""
    K, V, Q, b, a = rand_seq(8)
    S_full, outs_full, _ = scan(torch.zeros(D, D), K, V, Q, b, a)
    S_b0, outs_b0, _ = scan(torch.zeros(D, D), K[:4], V[:4], Q[:4], b[:4], a[:4])
    S_b1, outs_b1, _ = scan(S_b0, K[4:], V[4:], Q[4:], b[4:], a[4:])
    assert torch.allclose(S_b1, S_full, atol=1e-5)
    assert torch.allclose(torch.cat([outs_b0, outs_b1]), outs_full, atol=1e-5)


def commit_gate(logits, threshold, mask_id):
    """Confidence-shift/threshold commit gate (mirrors the SGLang --dllm-algorithm
    LowConfidence family): commit positions with max-prob >= threshold; always
    commit at least the single highest-confidence position."""
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    commit = conf >= threshold
    if not commit.any():
        commit[conf.argmax()] = True
    pred = pred.clone()
    pred[~commit] = mask_id
    return commit, pred


def test_commit_gate():
    MASK = 99
    logits = torch.tensor([[10.0, 0, 0], [0.1, 0.0, -0.1], [8.0, 1, 0]])  # hi, ~flat, hi
    commit, pred = commit_gate(logits, threshold=0.8, mask_id=MASK)
    assert commit.tolist() == [True, False, True], commit.tolist()
    assert pred[1].item() == MASK and pred[0].item() == 0 and pred[2].item() == 0
    # all-below-threshold: still commit exactly the top-conf one
    flat = torch.zeros(3, 4)
    c2, _ = commit_gate(flat + torch.tensor([0, 0.01, 0, 0]), threshold=0.99, mask_id=MASK)
    assert c2.sum().item() == 1, "must commit at least one when all below threshold"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  ok   {t.__name__}")
        except Exception as e:
            failed += 1; print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
