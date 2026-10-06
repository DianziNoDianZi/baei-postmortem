"""REPORT 327: the interaction finder as a COMPONENT, plus its operating envelope.

WHAT THIS IS.  The one mechanism this project validated with a matched control: a read-out's own residual
error, correlated against a family of candidate JOINT blocks, ranks the interaction the task actually
needs first -- and adding that block turns a task that additive statistics provably cannot solve
(log2(K) bits, hand-computable) into an exactly solvable one.  REPORT 285/286 found it on one bench;
REPORT 325 supplied the matched control (19 wrong lags and 3 pairing-destroyed blocks all buy nothing).

WHAT THIS FILE ADDS.  Two things a usable component needs and the project never had:

  1. a CLI that takes (K, candidates, samples) and reports the whole picture at once:
     the hand-computable bar, the criterion's rank of the true lag, held-out bits with the criterion's
     own pick vs the oracle pick vs a matched WRONG block, and a permutation null for the top score;
  2. the ENVELOPE -- the same measurement across K, candidate count and sample count, so the operating
     range is documented instead of assumed.  REPORT 276 named a "cell starvation" wall for K^2-scale
     candidate families but it was never touched; this measures where the criterion stops working.

The read-out is CLOSED FORM (ridge on centred one-hot plus a temperature fitted on an internal 30%
split), so nothing here depends on how long a head was trained -- the lesson-149 objection does not
apply to this lineage, and that is stated rather than left implicit.

    python tools/diagnostics/interact_finder.py --mode demo
    python tools/diagnostics/interact_finder.py --mode envelope
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.linalg
import scipy.sparse

BITS = lambda p: float(-np.log2(np.maximum(p, 1e-12)).mean())   # noqa: E731


# --------------------------------------------------------------------------- the task
def make_stream(n: int, K: int, k: int, seed: int = 0):
    """(label, y) with y[t] = (label[t] + label[t-k]) mod K.

    Given BOTH labels the answer is a point mass; given either alone it is uniform.  So the best
    additive predictor is the marginal, i.e. exactly log2(K) bits -- the hand-computable bar.
    """
    rng = np.random.default_rng(seed)
    label = rng.integers(0, K, size=n).astype(np.int64)
    y = np.full(n, -1, dtype=np.int64)
    y[k:] = (label[k:] + label[:n - k]) % K
    return label, y


def slot_features(label, ts, W, K):
    """ONE-HOT of the label at each of the last W steps -- everything an additive model gets to see."""
    X = np.zeros((len(ts), K * W), dtype=np.float64)
    cols = np.arange(K * W).reshape(W, K)
    for c in range(W):
        X[np.arange(len(ts)), cols[c][label[ts - c]]] = 1.0
    return X


# --------------------------------------------------------------------------- the read-out (closed form)
def fit_readout(X, y, K, lam=1e-2, temp_rows=3000, temp_grid=120):
    """Ridge on centred one-hot + a temperature fitted on an internal 30% hold-back (never on eval).

    SPEED (2026-10-06, the acceleration pass): the design matrix is SPARSE by construction (each row has
    W slot ones plus one 1 per joint block), so the Gram matrix is formed with a sparse matmul and the
    system is solved by Cholesky.  On the all-cross arm (F=2960) this is the difference between a
    50-GFLOP dense product and a few million multiplies.  The temperature scan is capped in both grid
    points and rows; `_self_check_readout` asserts the fast path agrees with the dense reference.
    """
    n = X.shape[0]
    cut = int(n * 0.7)
    T = np.eye(K)[y] - 1.0 / K
    if scipy.sparse.issparse(X):
        A = scipy.sparse.hstack([X, scipy.sparse.csr_matrix(np.ones((n, 1)))]).tocsr()
        G = (A[:cut].T @ A[:cut])
        G = np.asarray(G.todense()) if scipy.sparse.issparse(G) else G
        rhs = np.asarray(A[:cut].T @ T[:cut])
    else:
        A = np.hstack([X, np.ones((n, 1))])
        G = A[:cut].T @ A[:cut]
        rhs = A[:cut].T @ T[:cut]
    G = G + lam * np.eye(A.shape[1])
    G[-1, -1] += 1e6
    Wt = scipy.linalg.solve(G, rhs, assume_a="pos")
    #: temperature on the held-back 30 percent of the fit slice, never on the eval slice
    o = A[cut:] @ Wt if scipy.sparse.issparse(A) else A[cut:] @ Wt
    o = np.asarray(o)
    yy = y[cut:]
    if len(yy) > temp_rows:                       # cap the scan's cost; the argmin is stable
        sel = np.linspace(0, len(yy) - 1, temp_rows).astype(int)
        o, yy = o[sel], yy[sel]
    best = (1.0, 1e18)
    for temp in np.exp(np.linspace(np.log(1e-3), np.log(10.0), temp_grid)):
        lo = o / temp
        lo = lo - lo.max(1, keepdims=True)
        p = np.exp(lo)
        p /= p.sum(1, keepdims=True)
        c = BITS(p[np.arange(len(yy)), yy])
        if c < best[1]:
            best = (float(temp), c)
    return Wt, best[0]


def fit_readout_dense(X, y, K, lam=1e-2):
    """The pre-acceleration reference, kept so the fast path can be checked against it (lesson 166)."""
    n = len(X)
    cut = int(n * 0.7)
    T = np.eye(K)[y] - 1.0 / K
    A = np.hstack([X, np.ones((n, 1))])
    G = A[:cut].T @ A[:cut] + lam * np.eye(A.shape[1])
    G[-1, -1] += 1e6
    Wt = np.linalg.solve(G, A[:cut].T @ T[:cut])
    o = A[cut:] @ Wt
    yy = y[cut:]
    best = (1.0, 1e18)
    for temp in np.exp(np.linspace(np.log(1e-3), np.log(10.0), 400)):
        lo = o / temp
        lo = lo - lo.max(1, keepdims=True)
        p = np.exp(lo)
        p /= p.sum(1, keepdims=True)
        c = BITS(p[np.arange(len(yy)), yy])
        if c < best[1]:
            best = (float(temp), c)
    return Wt, best[0]


def score_bits(Wt, temp, X, y, K):
    A = (_one_col(X) if scipy.sparse.issparse(X) else np.hstack([X, np.ones((len(X), 1))]))
    o = np.asarray(A @ Wt) / temp
    o = o - o.max(1, keepdims=True)
    p = np.exp(o)
    p /= p.sum(1, keepdims=True)
    return BITS(p[np.arange(len(y)), y]), float((p.argmax(1) == y).mean())


def _one_col(X):
    return scipy.sparse.hstack(
        [X, scipy.sparse.csr_matrix(np.ones((X.shape[0], 1)))]).tocsr()


def probabilities(Wt, temp, X, K):
    A = (_one_col(X) if scipy.sparse.issparse(X) else np.hstack([X, np.ones((len(X), 1))]))
    o = np.asarray(A @ Wt) / temp
    o = o - o.max(1, keepdims=True)
    p = np.exp(o)
    return p / p.sum(1, keepdims=True)


# --------------------------------------------------------------------------- the criterion
def candidate_scores(label, ts, err, K, lags, pairs=None):
    """|| J_c^T err ||^2 per candidate joint, WITHOUT materialising the (N, K^2) block.

    `J_c` is the centred one-hot of the cell (label[t], label[t-j]).  Writing mu for the column means
    and s = err.sum(axis=0) for the COLUMN sums,

        J_c^T err = J^T err - mu (x) s

    and the second term is NOT zero: every row of `err` sums to zero (p and one-hot both sum to 1), but
    the column sums need not.  The first version of this function dropped that term and the self-check
    caught it (1-9% error, and it moved the ranking).  So it is computed here explicitly.
    """
    n = len(ts)
    s = err.sum(axis=0)
    out = []
    for j in lags:
        b = label[ts - j] if pairs is None else pairs[j]
        cell = label[ts] * K + b
        R = np.zeros((K * K, K), dtype=np.float64)
        np.add.at(R, cell, err)
        counts = np.bincount(cell, minlength=K * K).astype(np.float64)
        R -= (counts / n)[:, None] * s[None, :]
        out.append(float((R ** 2).sum()))
    return np.asarray(out)


def _self_check():
    """The bincount form must equal the materialised centred form -- checked, not asserted."""
    K, W, n = 5, 3, 400
    label, y = make_stream(n + 10, K, 2, seed=7)
    ts = np.arange(W + 1, n)
    ts = ts[y[ts] >= 0]
    X = slot_features(label, ts, W, K)
    Wt, temp = fit_readout(X, y[ts], K)
    err = probabilities(Wt, temp, X, K) - np.eye(K)[y[ts]]
    fast = candidate_scores(label, ts, err, K, [1, 2, 4])
    slow = []
    for j in [1, 2, 4]:
        J = np.zeros((len(ts), K * K))
        J[np.arange(len(ts)), label[ts] * K + label[ts - j]] = 1.0
        Jc = J - J.mean(0, keepdims=True)
        slow.append(float(((Jc.T @ err) ** 2).sum()))
    rel = np.abs(fast - slow) / np.maximum(np.abs(slow), 1e-30)
    assert rel.max() < 1e-9, f"bincount criterion disagrees with the materialised one: {rel}"
    return float(rel.max())


# --------------------------------------------------------------------------- one cell
def run_cell(K: int, n_cand: int, samples: int, seed: int, W: int = 10, k: int = 4, B: int = 20,
             light: bool = False, width_budget: int = 6000):
    """One cell.  `light=True` skips the oracle/wrong/null arms -- used by the envelope sweep, where the
    question is only "does the criterion find it, and what does the pick buy".

    `width_budget` guards the closed-form solve: the joint block is K^2 wide and the ridge solve is
    O(width^3), so a cell whose width exceeds the budget is reported as SKIPPED rather than crashing.
    That limit IS the envelope's right-hand edge and is printed as such.
    """
    width = K * K + K * W + 1
    if width > width_budget:
        return {"K": K, "n_cand": n_cand, "samples": samples, "seed": seed, "skipped": True,
                "width": width, "bar": float(np.log2(K))}
    label, y = make_stream(samples, K, k, seed=seed)
    n = len(y)
    tr_n = n // 2
    lags = list(range(1, n_cand + 1))
    ts_tr = np.arange(W + 1, tr_n, 7)
    ts_tr = ts_tr[y[ts_tr] >= 0]
    ts_te = np.arange(tr_n + W + 1, n, 7)
    ts_te = ts_te[y[ts_te] >= 0]
    Xtr, Xte = slot_features(label, ts_tr, W, K), slot_features(label, ts_te, W, K)

    t0 = time.time()
    Wt, temp = fit_readout(Xtr, y[ts_tr], K)
    base_bits, base_acc = score_bits(Wt, temp, Xte, y[ts_te], K)
    err = probabilities(Wt, temp, Xtr, K) - np.eye(K)[y[ts_tr]]
    scores = candidate_scores(label, ts_tr, err, K, lags)
    order = np.argsort(-scores)
    rank = int(np.nonzero(order == (k - 1))[0][0]) + 1
    pick = lags[int(order[0])]
    t_scores = time.time() - t0

    def bits_with(j):
        Jtr = np.zeros((len(ts_tr), K * K)); Jtr[np.arange(len(ts_tr)), label[ts_tr] * K + label[ts_tr - j]] = 1.0
        Jte = np.zeros((len(ts_te), K * K)); Jte[np.arange(len(ts_te)), label[ts_te] * K + label[ts_te - j]] = 1.0
        W2, t2 = fit_readout(np.hstack([Xtr, Jtr]), y[ts_tr], K)
        return score_bits(W2, t2, np.hstack([Xte, Jte]), y[ts_te], K)

    pick_bits, pick_acc = bits_with(pick)
    out = {"K": K, "n_cand": n_cand, "samples": samples, "seed": seed, "W": W, "k": k,
           "bar": float(np.log2(K)), "base_bits": base_bits, "base_acc": base_acc,
           "rank": rank, "pick": pick, "pick_bits": pick_bits, "pick_acc": pick_acc,
           "score_seconds": t_scores, "cell_seconds": time.time() - t0, "width": width,
           "ntr": int(len(ts_tr)), "cells": int(K * K)}
    if light:
        return out

    oracle_bits, _ = bits_with(k)
    wrong = lags[int(order[1])]                     # the runner-up: a matched wrong block
    wrong_bits, _ = bits_with(wrong)
    rng = np.random.default_rng(seed + 99991)
    null = []
    for _ in range(B):
        pairs = {pick: label[ts_tr - pick][rng.permutation(len(ts_tr))]}
        null.append(candidate_scores(label, ts_tr, err, K, [pick], pairs=pairs)[0])
    null = np.asarray(null)
    out.update({"oracle_bits": oracle_bits, "wrong_lag": wrong, "wrong_bits": wrong_bits,
                "null_mean": float(null.mean()), "null_std": float(null.std()),
                "pick_z": float((scores[order[0]] - null.mean()) / (null.std() + 1e-30)),
                "cell_seconds": time.time() - t0})
    return out


# --------------------------------------------------------------------------- a strong general baseline
def _best_split(X, r):
    """Best single binary split of `r` over the columns of the 0/1 matrix X (squared-error gain).

    The features are one-hot, so every column is its own threshold: the sums and counts are a single
    sparse mat-vec, and the gain is exact.
    """
    n = len(r)
    s = X.T @ r                                   # (F,) sum of r where the column is 1
    c = np.asarray(X.sum(0)).ravel()              # (F,) how many rows are 1
    tot = float(r.sum())
    ok = (c > 0) & (c < n)
    if not ok.any():
        return None
    g = np.where(ok, s ** 2 / np.maximum(c, 1) + (tot - s) ** 2 / np.maximum(n - c, 1), -np.inf)
    f = int(np.argmax(g))
    return f, float(np.mean(r[X[:, f] > 0])), float(np.mean(r[X[:, f] <= 0]))


def _depth2(X, r):
    """A depth-2 regression tree: one root split, then one split inside each side.  Depth 2 is the
    minimum depth that can express an INTERACTION, which is why it is the fair general baseline here."""
    root = _best_split(X, r)
    if root is None:
        return None
    f1 = root[0]
    left = X[:, f1] > 0
    out = {"f1": f1}
    for side, mask in (("l", left), ("r", ~left)):
        sub = _best_split(X[mask], r[mask]) if mask.sum() > 1 else None
        out[side] = (sub[0], sub[1], sub[2]) if sub else (None, float(np.mean(r[mask])), 0.0)
        out[side + "_f"] = sub[0] if sub else None
    out["fallback"] = float(np.mean(r))
    return out


def _depth2_predict(X, t):
    if t is None:
        return np.zeros(len(X))
    f1 = t["f1"]
    left = X[:, f1] > 0
    out = np.empty(len(X))
    for side, mask in (("l", left), ("r", ~left)):
        f2, vl, vr = t[side]
        if f2 is None:
            out[mask] = float(np.mean([vl]))
        else:
            sub = X[mask]
            m2 = sub[:, f2] > 0
            out[np.flatnonzero(mask)[m2]] = vl
            out[np.flatnonzero(mask)[~m2]] = vr
    return out


def fit_gbdt(X, y, K, rounds=60, lr=0.3):
    """Multiclass gradient boosting with DEPTH-2 trees on the same inputs the criterion gets.

    Returns (prior_logits, trees) so the model can be applied to HELD-OUT rows.  The first version
    returned the training scores and the caller scored them against the test labels -- a train/test
    mix-up that does not raise, and that the additive-task instrument check caught.
    """
    n = X.shape[0]
    priors = np.bincount(y, minlength=K).astype(float)
    priors /= priors.sum()
    prior_logits = np.log(np.maximum(priors, 1e-9))
    scores = np.tile(prior_logits, (n, 1))
    trees: list[tuple[int, dict]] = []
    for _ in range(rounds):
        o = scores - scores.max(1, keepdims=True)
        p = np.exp(o)
        p /= p.sum(1, keepdims=True)
        #: the negative gradient of the multinomial log-loss is (onehot - p); fit that and ADD.
        R = np.eye(K)[y] - p
        for c in range(K):
            t = _depth2(X, R[:, c])
            trees.append((c, t))
            scores[:, c] += lr * _depth2_predict(X, t)
    return prior_logits, trees


def predict_gbdt(X, prior_logits, trees, lr):
    scores = np.tile(prior_logits, (len(X), 1))
    for c, t in trees:
        scores[:, c] += lr * _depth2_predict(X, t)
    return scores


def gbdt_scores(Xtr, ytr, Xte, K, rounds=60, lr=0.3):
    prior_logits, trees = fit_gbdt(Xtr, ytr, K, rounds=rounds, lr=lr)
    return predict_gbdt(Xte, prior_logits, trees, lr)


def gbdt_bits(Xtr, ytr, Xte, yte, K, rounds=60, lr=0.3):
    s = gbdt_scores(Xtr, ytr, Xte, K, rounds=rounds, lr=lr)
    o = s - s.max(1, keepdims=True)
    p = np.exp(o)
    p /= p.sum(1, keepdims=True)
    return BITS(p[np.arange(len(yte)), yte]), float((p.argmax(1) == yte).mean())


# --------------------------------------------------------------------------- modes
def _sanity_gbdt(K: int, samples: int, rounds: int, lr: float) -> tuple:
    """The instrument check that has to come first: on an ADDITIVE task (y = label[t], no interaction),
    the GBDT must reach ~0 bits and ~100% accuracy.  If it does not, the baseline is too weak and any
    'the GBDT fails at the interaction task' reading is an artefact of THIS implementation."""
    label, _ = make_stream(samples, K, 4, seed=0)
    y = label.copy()
    n = len(y)
    W = 10
    ts_tr = np.arange(W + 1, n // 2, 7)
    ts_te = np.arange(n // 2 + W + 1, n, 7)
    Xtr, Xte = slot_features(label, ts_tr, W, K), slot_features(label, ts_te, W, K)
    Wt, temp = fit_readout(Xtr, y[ts_tr], K)
    a_bits, a_acc = score_bits(Wt, temp, Xte, y[ts_te], K)
    g_bits, g_acc = gbdt_bits(Xtr, y[ts_tr], Xte, y[ts_te], K, rounds=rounds, lr=lr)
    return a_bits, a_acc, g_bits, g_acc


def vs_gbdt(args) -> int:
    """The one-day decisiveness test: on the SAME inputs, does a standard boosted-tree baseline already
    solve the task, or does the criterion add something a general learner does not have?"""
    print("REPORT 328: does a general booster already do this?  (same inputs, depth-2 GBDT)")
    print("  arms: additive | +criterion pick | +oracle cross | +ALL lag crosses | depth-2 GBDT")
    print("  metric: held-out bits (lower is better) and accuracy; bar = hand-computed log2(K)\n")
    print("  --- instrument check FIRST: on an additive task (y = label[t]) the GBDT must reach ~0 ---")
    for K in [int(v) for v in args.K.split(",")]:
        a_b, a_a, g_b, g_a = _sanity_gbdt(K, int(args.samples.split(",")[0]), args.rounds, args.lr)
        flag = "OK" if (g_b < 0.2 and g_a > 0.95) else "** TOO WEAK -- the comparison below is void **"
        print(f"     K={K:<4} additive {a_b:.4f} bits / acc {a_a:.4f}   "
              f"GBDT {g_b:.4f} bits / acc {g_a:.4f}   {flag}")
    print()
    print(f"  {'K':>4} {'samples':>8} {'bar':>7} {'additive':>9} {'+pick':>9} {'+oracle':>9} "
          f"{'+allX':>9} {'GBDT':>9} {'GBDTacc':>8} {'rank':>5} {'sec':>6}")
    for K in [int(v) for v in args.K.split(",")]:
        for samples in [int(v) for v in args.samples.split(",")]:
            k, W = 4, 10
            label, y = make_stream(samples, K, k, seed=0)
            n = len(y)
            tr_n = n // 2
            lags = list(range(1, int(args.cands.split(",")[0]) + 1))
            ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
            ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
            Xtr, Xte = slot_features(label, ts_tr, W, K), slot_features(label, ts_te, W, K)
            t0 = time.time()
            Wt, temp = fit_readout(Xtr, y[ts_tr], K)
            add_bits, _ = score_bits(Wt, temp, Xte, y[ts_te], K)
            err = probabilities(Wt, temp, Xtr, K) - np.eye(K)[y[ts_tr]]
            sc = candidate_scores(label, ts_tr, err, K, lags)
            order = np.argsort(-sc)
            rank = int(np.nonzero(order == (k - 1))[0][0]) + 1
            pick = lags[int(order[0])]

            def blk(j, ts):
                J = np.zeros((len(ts), K * K)); J[np.arange(len(ts)), label[ts] * K + label[ts - j]] = 1.0
                return J

            def bits_add(Jtr, Jte):
                W2, t2 = fit_readout(np.hstack([Xtr, Jtr]), y[ts_tr], K)
                return score_bits(W2, t2, np.hstack([Xte, Jte]), y[ts_te], K)[0]

            pick_bits = bits_add(blk(pick, ts_tr), blk(pick, ts_te))
            orac_bits = bits_add(blk(k, ts_tr), blk(k, ts_te))
            allx_width = K * K * len(lags) + K * W + 1
            if allx_width <= 6000:
                allx_bits = bits_add(np.hstack([blk(j, ts_tr) for j in lags]),
                                     np.hstack([blk(j, ts_te) for j in lags]))
                allx_s = f"{allx_bits:>9.4f}"
            else:
                allx_s = f"{'SKIP':>9}"          # the exhaustive cross is O(W*K^2) wide: that IS the cost
            g_bits, g_acc = gbdt_bits(Xtr, y[ts_tr], Xte, y[ts_te], K,
                                      rounds=args.rounds, lr=args.lr)
            print(f"  {K:>4} {samples:>8} {np.log2(K):>7.4f} {add_bits:>9.4f} {pick_bits:>9.4f} "
                  f"{orac_bits:>9.4f} {allx_s} {g_bits:>9.4f} {g_acc:>8.4f} {rank:>5} "
                  f"{time.time() - t0:>6.1f}")
    print("\n  reading: if the GBDT column already sits near 0, a standard general learner finds this")
    print("  interaction on its own, and the criterion's value is COST (one candidate instead of all of")
    print("  them, no labels needed for the search), not capability.")
    return 0


def fit_fm(X, y, K, dim=8, epochs=300, lr=0.05, seed=0):
    """Factorisation machine, one-vs-rest over K classes: linear + LOW-RANK pairwise interactions.

    SPEED (the acceleration pass): every class keeps its OWN embedding (V is (F, K*dim) and is reshaped
    to (n, K, dim) in one BLAS call) -- so the model class is IDENTICAL to K independent one-vs-rest FMs,
    which is what the first version ran in a Python loop at K times the cost.  Measured 5.07 s -> below
    a second per cell at K=16.  Weakening the opponent to a shared V would have been the easy way, and
    that is exactly what lesson 168 forbids.
    """
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)      # one thread per worker process: 8 workers x N threads would thrash
    n, F = X.shape
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    w = torch.zeros(F, K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    V = (torch.randn(F, K * dim) * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([w, b, V], lr=lr)
    X2 = Xt ** 2
    for _ in range(epochs):
        xv = (Xt @ V).view(n, K, dim)
        xv2 = (X2 @ (V ** 2)).view(n, K, dim)
        pair = 0.5 * (xv ** 2 - xv2).sum(-1)              # (n, K)
        logits = Xt @ w + b + pair
        loss = torch.nn.functional.cross_entropy(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return (w.detach(), b.detach(), V.detach(), dim)


def predict_fm(X, models, K):
    import torch
    w, b, V, dim = models
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    n = Xt.shape[0]
    xv = (Xt @ V).view(n, K, dim)
    xv2 = ((Xt ** 2) @ (V ** 2)).view(n, K, dim)
    pair = 0.5 * (xv ** 2 - xv2).sum(-1)
    return (Xt @ w + b + pair).numpy()


def fm_bits(Xtr, ytr, Xte, yte, K, dim=8, epochs=300, lr=0.05):
    models = fit_fm(Xtr, ytr, K, dim=dim, epochs=epochs, lr=lr)
    s = predict_fm(Xte, models, K)
    o = s - s.max(1, keepdims=True)
    p = np.exp(o)
    p /= p.sum(1, keepdims=True)
    return BITS(p[np.arange(len(yte)), yte]), float((p.argmax(1) == yte).mean())


def vs_all(args) -> int:
    """The three-way decider: our criterion vs a low-rank FM vs a GBDT with real capacity.

    Reports, per arm, held-out bits and accuracy AND the feature width -- because "who needs how many
    columns to reach the same number" is the question that decides whether the underlying interaction
    PARAMETERISATION has to change.
    """
    print("REPORT 329: criterion vs FM (low-rank pairwise) vs a GBDT with capacity")
    print("  same inputs for every arm; widths are reported because they are the cost axis\n")
    print("  --- instrument check: on an ADDITIVE task every arm must reach ~0 ---")
    K0 = int(args.K.split(",")[0])
    a_b, a_a, g_b, g_a = _sanity_gbdt(K0, int(args.samples.split(",")[0]), args.rounds, args.lr)
    f_b, f_a = fm_bits(*_sanity_additive(K0, int(args.samples.split(",")[0])), K=K0,
                       dim=args.dim, epochs=args.epochs)
    print(f"     K={K0}: additive {a_b:.4f}/{a_a:.4f}   GBDT {g_b:.4f}/{g_a:.4f}   "
          f"FM {f_b:.4f}/{f_a:.4f}")
    print()
    print(f"  {'K':>4} {'samples':>8} {'bar':>7} {'additive':>9} {'+pick':>9} {'+allX':>9} "
          f"{'GBDT':>8} {'FM':>8} {'FMacc':>7} {'rank':>5} {'widths pick/allX/FM':>22} {'sec':>6}")
    for K in [int(v) for v in args.K.split(",")]:
        for samples in [int(v) for v in args.samples.split(",")]:
            k, W = 4, 10
            label, y = make_stream(samples, K, k, seed=0)
            n = len(y); tr_n = n // 2
            lags = list(range(1, int(args.cands.split(",")[0]) + 1))
            ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
            ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
            Xtr, Xte = slot_features(label, ts_tr, W, K), slot_features(label, ts_te, W, K)
            t0 = time.time()
            Wt, temp = fit_readout(Xtr, y[ts_tr], K)
            add_bits, _ = score_bits(Wt, temp, Xte, y[ts_te], K)
            err = probabilities(Wt, temp, Xtr, K) - np.eye(K)[y[ts_tr]]
            sc = candidate_scores(label, ts_tr, err, K, lags)
            order = np.argsort(-sc)
            rank = int(np.nonzero(order == (k - 1))[0][0]) + 1
            pick = lags[int(order[0])]

            def blk(j, ts):
                J = np.zeros((len(ts), K * K)); J[np.arange(len(ts)), label[ts] * K + label[ts - j]] = 1.0
                return J

            def bits_add(Jtr, Jte):
                W2, t2 = fit_readout(np.hstack([Xtr, Jtr]), y[ts_tr], K)
                return score_bits(W2, t2, np.hstack([Xte, Jte]), y[ts_te], K)[0]

            pick_bits = bits_add(blk(pick, ts_tr), blk(pick, ts_te))
            allx_w = K * K * len(lags) + K * W + 1
            allx_s = (f"{bits_add(np.hstack([blk(j, ts_tr) for j in lags]), np.hstack([blk(j, ts_te) for j in lags])):.4f}"
                      if allx_w <= 8000 else "SKIP")
            g_bits, g_acc = gbdt_bits(Xtr, y[ts_tr], Xte, y[ts_te], K, rounds=args.rounds, lr=args.lr)
            fm_b, fm_a = fm_bits(Xtr, y[ts_tr], Xte, y[ts_te], K, dim=args.dim, epochs=args.epochs)
            widths = f"{K*K+K*W}/{allx_w}/{K*W+K*args.dim}"
            print(f"  {K:>4} {samples:>8} {np.log2(K):>7.4f} {add_bits:>9.4f} {pick_bits:>9.4f} "
                  f"{allx_s:>9} {g_bits:>8.4f} {fm_b:>8.4f} {fm_a:>7.4f} {rank:>5} {widths:>22} "
                  f"{time.time() - t0:>6.1f}")
    print("\n  what decides 'change the underlying parameterisation': if FM matches +pick at a")
    print("  narrower width, the K^2 table is the thing to replace (low-rank); if +pick wins at equal")
    print("  width, the selection criterion is the asset and the parameterisation can stay.")
    return 0


def _sanity_additive(K: int, samples: int):
    label, _ = make_stream(samples, K, 4, seed=0)
    y = label.copy()
    n = len(y); W = 10
    ts_tr = np.arange(W + 1, n // 2, 7)
    ts_te = np.arange(n // 2 + W + 1, n, 7)
    return slot_features(label, ts_tr, W, K), y[ts_tr], slot_features(label, ts_te, W, K), y[ts_te]


def make_multi_stream(n: int, K: int, pairs, seed: int = 0):
    """y[t] = sum over the given lag pairs of (label[t-a] + label[t-b]), mod K.

    The 2M lags are required to be distinct: if one lag appeared twice the sum would fold into
    "2 * label[t-a]", which is visible from a single variable and would break the additive floor.
    Given any single label value the answer is uniform, so the additive floor is exactly log2(K).
    """
    used = [l for p in pairs for l in p]
    if len(set(used)) != len(used):
        raise SystemExit(f"lag pairs must have distinct lags, got {pairs}")
    rng = np.random.default_rng(seed)
    label = rng.integers(0, K, size=n).astype(np.int64)
    y = np.full(n, -1, dtype=np.int64)
    k = max(used)
    acc = np.zeros(n - k, dtype=np.int64)
    for a, b in pairs:
        acc = (acc + label[k - a:n - a] + label[k - b:n - b]) % K
    y[k:] = acc
    return label, y


def pair_block(label, ts, i, j, K):
    X = np.zeros((len(ts), K * K))
    X[np.arange(len(ts)), label[ts - i] * K + label[ts - j]] = 1.0
    return X


def pair_block_sparse(label, ts, i, j, K):
    """The same block as CSR: exactly one 1 per row.  This is what makes the big-F fits affordable."""
    n = len(ts)
    cell = label[ts - i] * K + label[ts - j]
    return scipy.sparse.csr_matrix((np.ones(n), (np.arange(n), cell)), shape=(n, K * K))


def design(X, blocks):
    """Column-stack the slot features with any number of sparse joint blocks (still sparse)."""
    mats = [X if scipy.sparse.issparse(X) else scipy.sparse.csr_matrix(X)]
    mats += list(blocks)
    return scipy.sparse.hstack(mats).tocsr()


def _self_check_readout():
    """The accelerated read-out must agree with the pre-acceleration reference (lesson 166).

    Two things are asserted separately, because they are two different claims:
      (i)  the LINEAR ALGEBRA is unchanged -- with the same temperature grid, the weights and the bits
           must match the dense reference to 1e-9.  This is what "sparse + Cholesky" claims.
      (ii) the PRODUCTION setting only coarsens the temperature SCAN (400 -> 120 grid points, rows
           capped at 3000).  That does move bits, so it is MEASURED and reported rather than asserted
           away -- the first version of this check asserted the two settings were identical and the
           difference (0.3910 vs 0.3957) showed up immediately.
    """
    K, W, n = 4, 6, 900
    label, y, span = make_components_stream(n, 8, [(1, 2), (3, 5)], 2, seed=3)
    ts = np.arange(W + 1, len(y), 2)
    ts = ts[y[ts] >= 0]
    Xd = slot_features(label, ts, W, 8)
    blocks_d = [pair_block(label, ts, 1, 2, 8), pair_block(label, ts, 3, 5, 8)]
    blocks_s = [pair_block_sparse(label, ts, 1, 2, 8), pair_block_sparse(label, ts, 3, 5, 8)]
    Ad, As = np.hstack([Xd] + blocks_d), design(Xd, blocks_s)
    Wd, td = fit_readout_dense(Ad, y[ts], span)
    Ws, tst = fit_readout(As, y[ts], span, temp_rows=10 ** 9, temp_grid=400)
    rel = float(np.abs(Wd - Ws).max() / max(1.0, np.abs(Wd).max()))
    assert rel < 1e-9, f"fast read-out changed the linear algebra: relative {rel}"
    bd, _ = score_bits(Wd, td, Ad, y[ts], span)
    bs, _ = score_bits(Ws, tst, As, y[ts], span)
    assert abs(bd - bs) < 1e-9, f"same settings disagree: {bd} vs {bs}"
    Wp, tp = fit_readout(As, y[ts], span)
    bp, _ = score_bits(Wp, tp, As, y[ts], span)
    return bd, bp


def pair_scores(label, ts, err, K_lab, n_cls, pairs, second=None):
    """The centred-bincount criterion, generalised to any lag pair.

    `K_lab` is the alphabet of the LABELS (which index the cell) and `n_cls` is the number of target
    classes (the width of `err`).  In the single-interaction benches the two coincide; in the
    multi-interaction task the labels keep an 8-letter alphabet while the packed target has more
    classes, and conflating them is exactly the shape error that stopped the first run.
    `second` optionally overrides the second lag's values (used to build the pairing-destroyed null of
    REPORT 338: each candidate gets its OWN independent shuffle, so the null-max over N candidates is
    measured rather than assumed).
    """
    n = len(ts)
    s = err.sum(axis=0)
    out = []
    for (i, j) in pairs:
        b = label[ts - j] if second is None else second[(i, j)]
        cell = label[ts - i] * K_lab + b
        R = np.zeros((K_lab * K_lab, n_cls), dtype=np.float64)
        np.add.at(R, cell, err)
        counts = np.bincount(cell, minlength=K_lab * K_lab).astype(np.float64)
        R -= (counts / n)[:, None] * s[None, :]
        out.append(float((R ** 2).sum()))
    return np.asarray(out)


def make_components_stream(n: int, K_lab: int, pairs, comps: int, seed: int = 0):
    """A MULTI-INTERACTION task whose additive floor is PROVABLE, not assumed.

    Two designs were rejected before this one, and both rejections are the point:

    ① y = sum over pairs of (label[t-a] + label[t-b]) mod K.  The answer then depends on the JOINT of
       two pair-blocks -- a 4-way term -- which no additively-composed block family can express.  The
       pilot showed the exhaustive-cross UPPER BOUND sitting on the additive bar, i.e. the task was out
       of the method's CLASS rather than the method failing.
    ② a random table g(cell) per interaction.  Measured: the additive arm sat BELOW the hand-computed
       floor (0.8439 against 1.0000), i.e. the table leaked into single-variable statistics.

    Here g_m(a, b) = (a + b) mod comps with K_lab a multiple of comps.  Then for a fixed a the value
    runs uniformly as b varies (and vice versa), so NO single label carries information and the
    additive floor is exactly M * log2(comps) -- a property that can be checked, not hoped for.
    """
    if K_lab % comps != 0:
        raise SystemExit("K_lab must be a multiple of comps for the floor to be exact")
    rng = np.random.default_rng(seed)
    label = rng.integers(0, K_lab, size=n).astype(np.int64)
    kmax = max(max(p) for p in pairs)
    acc = np.zeros(n - kmax, dtype=np.int64)
    mult, span = 1, 1
    for (a, b) in pairs:
        g = (label[kmax - a:n - a] + label[kmax - b:n - b]) % comps
        acc = acc + g * mult
        mult *= comps
        span *= comps
    y = np.full(n, -1, dtype=np.int64)
    y[kmax:] = acc
    return label, y, span


def boosting(args) -> int:
    """REPORT 330: multi-round boosting over DISCOVERED pairs -- the task-kind axis.

    Round r: fit the read-out on the additive slots plus the blocks chosen so far, take its residual,
    score every remaining candidate pair against that residual, add the best one, refit.  Each true
    interaction owns one component of the label, so the EXHAUSTIVE-CROSS arm is a computable upper
    bound: if that arm cannot reach ~0, the task is out of class and the run is void.
    """
    K_lab, W, comps = 8, 10, 2
    lags = list(range(W))
    all_pairs = [(i, j) for i in lags for j in lags if i < j]
    print("REPORT 330: multi-round boosting on M-interaction tasks (each interaction = one label bit)")
    print(f"  K_lab={K_lab}, W={W}, candidates = {len(all_pairs)} lag pairs, "
          f"upper bound = add ALL pair blocks")
    print(f"  arms: additive | boosting (M rounds, then +4) | FM(dim {args.dim}) | GBDT({args.rounds} rounds) | allCross\n")
    print(f"  {'M':>3} {'K':>4} {'bar':>6} {'add':>7} {'allCross':>9} {'boost@M':>8} {'boost@M+4':>10} "
          f"{'FM':>8} {'GBDT':>8} {'hits':>6} {'sec':>6}")
    for M in [int(v) for v in args.M.split(",")]:
        samples = int(args.samples.split(",")[0])
        base = [(1, 2), (3, 5), (4, 7), (6, 9), (0, 8), (2, 6), (5, 8), (1, 9)][:M]
        true_pairs = set(base)
        label, y, span = make_components_stream(samples + 20, K_lab, base, comps, seed=0)
        K, bar = span, float(np.log2(span))
        n = len(y); tr_n = n // 2
        ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
        ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
        Xtr, Xte = slot_features(label, ts_tr, W, K_lab), slot_features(label, ts_te, W, K_lab)
        t0 = time.time()
        Wt, temp = fit_readout(Xtr, y[ts_tr], K)
        add_bits, _ = score_bits(Wt, temp, Xte, y[ts_te], K)

        #: the computable upper bound FIRST -- if it cannot solve the task, nothing below is readable
        Atr = np.hstack([Xtr] + [pair_block(label, ts_tr, i, j, K_lab) for (i, j) in all_pairs])
        Ate = np.hstack([Xte] + [pair_block(label, ts_te, i, j, K_lab) for (i, j) in all_pairs])
        Wa, ta = fit_readout(Atr, y[ts_tr], K)
        ac_bits, _ = score_bits(Wa, ta, Ate, y[ts_te], K)

        chosen, bits_by_round = [], []
        Cur_tr, Cur_te = Xtr, Xte
        Wc, tc = Wt, temp
        for rnd in range(M + 4):
            err = probabilities(Wc, tc, Cur_tr, K) - np.eye(K)[y[ts_tr]]
            cand = [p for p in all_pairs if p not in chosen]
            sc = pair_scores(label, ts_tr, err, K_lab, K, cand)
            pick = cand[int(np.argsort(-sc)[0])]
            chosen.append(pick)
            Cur_tr = np.hstack([Cur_tr, pair_block(label, ts_tr, pick[0], pick[1], K_lab)])
            Cur_te = np.hstack([Cur_te, pair_block(label, ts_te, pick[0], pick[1], K_lab)])
            Wc, tc = fit_readout(Cur_tr, y[ts_tr], K)
            bits_by_round.append(score_bits(Wc, tc, Cur_te, y[ts_te], K)[0])
        hits = sum(1 for p in chosen[:M] if p in true_pairs)

        fm_b, _ = fm_bits(Xtr, y[ts_tr], Xte, y[ts_te], K, dim=args.dim, epochs=args.epochs)
        g_b, _ = gbdt_bits(Xtr, y[ts_tr], Xte, y[ts_te], K, rounds=args.rounds, lr=args.lr)
        print(f"  {M:>3} {K:>4} {bar:>6.3f} {add_bits:>7.4f} {ac_bits:>9.4f} {bits_by_round[M-1]:>8.4f} "
              f"{bits_by_round[M+3]:>10.4f} {fm_b:>8.4f} {g_b:>8.4f} {hits:>3}/{M:<2} "
              f"{time.time() - t0:>6.1f}")
    print("\n  every arm must be read against the allCross column: if allCross is not ~0, the task is")
    print("  out of class and the row is void (that is how the first task design was rejected).")
    print("  B1: boost@M <= 0.05 at M=8?   B2: boost@M < FM and < GBDT at M>=4?")
    return 0


_M_BASE = [(1, 2), (3, 5), (4, 7), (6, 9), (0, 8), (2, 6), (5, 8), (1, 9)]
#: DISJOINT lags: with _M_BASE the 8 pairs reuse lags (16 slots into 10 lags), and a candidate that
#: shares ONE lag with an unmodelled true interaction correlates with the residual through that lag
#: alone -- a spurious top pick.  This set separates "my pair choice" from "the mechanism".
_M_DISJOINT = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]


def _pairs_for(M, pmode):
    base = _M_DISJOINT if pmode else _M_BASE
    if M > len(base):
        raise SystemExit(f"M={M} exceeds the available true pairs ({len(base)})")
    return base[:M]


def _boost_cell(job):
    """One (M, samples, seed) cell of the multi-interaction sweep -- a top-level function so it can be
    farmed out to worker processes (the acceleration pass)."""
    M, samples, seed, W, K_lab, comps, dim, epochs, rounds, lam = job
    base = _M_BASE[:M]
    true_pairs = set(base)
    label, y, span = make_components_stream(samples + 20, K_lab, base, comps, seed=seed)
    K = span
    n = len(y)
    tr_n = n // 2
    ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
    ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
    dense_tr = slot_features(label, ts_tr, W, K_lab)
    dense_te = slot_features(label, ts_te, W, K_lab)
    xs_tr = scipy.sparse.csr_matrix(dense_tr)
    xs_te = scipy.sparse.csr_matrix(dense_te)
    all_pairs = [(i, j) for i in range(W) for j in range(W) if i < j]

    def blk(i, j, tr):
        return pair_block_sparse(label, ts_tr if tr else ts_te, i, j, K_lab)

    t0 = time.time()
    Wt, temp = fit_readout(xs_tr, y[ts_tr], K, lam=lam)
    add_bits, _ = score_bits(Wt, temp, xs_te, y[ts_te], K)

    #: THE CEILING THAT MUST REACH 0: the M true blocks, and nothing else.  The first design used "add
    #: all 45 blocks" as its upper bound; with n_train < F that arm is not identifiable at all, so its
    #: failure said nothing about the task (lesson 169).  The oracle is the right ceiling.
    Otr = design(xs_tr, [blk(i, j, True) for (i, j) in base])
    Ote = design(xs_te, [blk(i, j, False) for (i, j) in base])
    Wo, to = fit_readout(Otr, y[ts_tr], K, lam=lam)
    orc_bits, _ = score_bits(Wo, to, Ote, y[ts_te], K)

    chosen, bits_by_round = [], []
    Cur_tr, Cur_te = xs_tr, xs_te
    Wc, tc = Wt, temp
    for _ in range(M + 4):
        err = probabilities(Wc, tc, Cur_tr, K) - np.eye(K)[y[ts_tr]]
        cand = [p for p in all_pairs if p not in chosen]
        pick = cand[int(np.argmax(pair_scores(label, ts_tr, err, K_lab, K, cand)))]
        chosen.append(pick)
        Cur_tr = design(Cur_tr, [blk(pick[0], pick[1], True)])
        Cur_te = design(Cur_te, [blk(pick[0], pick[1], False)])
        Wc, tc = fit_readout(Cur_tr, y[ts_tr], K, lam=lam)
        bits_by_round.append(score_bits(Wc, tc, Cur_te, y[ts_te], K)[0])
    hits = sum(1 for p in chosen[:M] if p in true_pairs)

    fm_b, _ = fm_bits(dense_tr, y[ts_tr], dense_te, y[ts_te], K, dim=dim, epochs=epochs)
    g_b, _ = gbdt_bits(dense_tr, y[ts_tr], dense_te, y[ts_te], K, rounds=rounds, lr=0.3)
    wide_w = K_lab * K_lab * len(all_pairs) + K_lab * W + 1
    wide = float("nan")
    if wide_w <= 4096 and len(ts_tr) > wide_w:          # only where the wide family is identifiable
        Atr = design(xs_tr, [blk(i, j, True) for (i, j) in all_pairs])
        Ate = design(xs_te, [blk(i, j, False) for (i, j) in all_pairs])
        Wa, ta = fit_readout(Atr, y[ts_tr], K, lam=lam)
        wide = score_bits(Wa, ta, Ate, y[ts_te], K)[0]
    return {"M": M, "samples": samples, "seed": seed, "K": K, "bar": float(np.log2(K)),
            "add": add_bits, "oracle": orc_bits, "boostM": bits_by_round[M - 1],
            "boostM4": bits_by_round[M + 3], "fm": fm_b, "gbdt": g_b, "wide": wide,
            "hits": hits, "chosen": chosen[:M], "sec": time.time() - t0}


def boosting2(args) -> int:
    """REPORT 331: the fixed bench + the accelerated sweep.

    Five things changed from REPORT 330: (1) the interaction value is (a+b) mod comps so the additive
    floor is PROVABLE and the additive arm is an instrument check rather than an assumption; (2) the
    ceiling is the ORACLE (the M true blocks), not "all 45 blocks", because with n_train < F the wide
    family is not identifiable and its failure was uninformative; (3) the read-out is sparse + Cholesky;
    (4) FM is batched over classes with per-class embeddings (same model class, K times cheaper);
    (5) cells run in parallel worker processes.
    """
    rel_a, rel_b = _self_check_readout()
    print("REPORT 331: multi-interaction bench (fixed) + accelerated sweep")
    print(f"  read-out self-check: identical linear algebra (asserted); the production temperature scan")
    print(f"  is coarser, measured cost {rel_b - rel_a:+.6f} bits ({rel_a:.6f} -> {rel_b:.6f})\n")
    K_lab, W, comps = 8, 10, 2
    Ks = [int(v) for v in args.K.split(",")]
    W = Ks[0] if len(Ks) > 1 else W
    Ms = [int(v) for v in args.M.split(",")]
    samps = [int(v) for v in args.samples.split(",")]
    seeds = list(range(args.seeds))
    jobs = [(M, s, sd, W, K_lab, comps, args.dim, args.epochs, args.rounds, args.lam)
            for M in Ms for s in samps for sd in seeds]
    print(f"  {len(jobs)} cells, workers={args.workers}, K_lab={K_lab}, W={W}, comps={comps}, "
          f"dim={args.dim}, epochs={args.epochs}, rounds={args.rounds}, lam={args.lam}\n")
    t0 = time.time()
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_boost_cell, jobs))
    else:
        rows = [_boost_cell(j) for j in jobs]
    print(f"  {'M':>3} {'K':>5} {'sam':>7} {'bar':>6} {'add':>7} {'ORACLE':>7} {'boost@M':>8} "
          f"{'boost@M+4':>10} {'FM':>7} {'GBDT':>7} {'wide':>7} {'hit':>5} {'s':>5}")
    for r in sorted(rows, key=lambda r: (r["M"], r["samples"], r["seed"])):
        wd = f"{r['wide']:.4f}" if r["wide"] == r["wide"] else "n/a"
        print(f"  {r['M']:>3} {r['K']:>5} {r['samples']:>7} {r['bar']:>6.3f} {r['add']:>7.4f} "
              f"{r['oracle']:>7.4f} {r['boostM']:>8.4f} {r['boostM4']:>10.4f} {r['fm']:>7.4f} "
              f"{r['gbdt']:>7.4f} {wd:>7} {r['hits']:>2}/{r['M']:<2} {r['sec']:>5.1f}")
    #: every criterion derived from the numbers, never hard-coded (the project's rule ⑧)
    print("\n  --- derived verdict ---")
    bad_add = [r for r in rows if abs(r["add"] - r["bar"]) > 0.05]
    bad_orc = [r for r in rows if r["oracle"] > 0.05]
    print(f"    instrument: additive arm within 0.05 of the bar in "
          f"{len(rows) - len(bad_add)}/{len(rows)} cells  "
          f"({'PASS' if not bad_add else 'FAIL -- bench invalid'})")
    print(f"    instrument: ORACLE reaches <=0.05 bits in {len(rows) - len(bad_orc)}/{len(rows)} "
          f"({'PASS' if not bad_orc else 'FAIL -- bench out of class or fit broken'})")
    ok = [r for r in rows if not bad_add and not bad_orc]
    if ok:
        allhit = sum(1 for r in ok if r["hits"] == r["M"])
        win = sum(1 for r in ok if r["boostM"] < min(r["fm"], r["gbdt"]))
        print(f"    on the {len(ok)} valid cells: all M pairs found in {allhit}/{len(ok)}; "
              f"boosting beats both opponents in {win}/{len(ok)}")
    print(f"  total {time.time() - t0:.1f} s")
    return 0


def speed(args) -> int:
    """Report the acceleration: the two hot paths, before and after (the numbers go in the REPORT)."""
    K_lab, W, n = 8, 10, 20000
    print("REPORT 331 SPEED: the two hot paths")
    ba, bb = _self_check_readout()
    print(f"  read-out equivalence: linear algebra asserted identical; coarsened temperature scan costs "
          f"{bb - ba:+.6f} bits ({ba:.6f} -> {bb:.6f})")
    label, y, span = make_components_stream(n, K_lab, _M_BASE[:4], 2, seed=0)
    ts = np.arange(W + 1, len(y) // 2, 7); ts = ts[y[ts] >= 0]
    dense = slot_features(label, ts, W, K_lab)
    sp = scipy.sparse.csr_matrix(dense)
    pairs = [(i, j) for i in range(W) for j in range(W) if i < j]
    blocks = [pair_block_sparse(label, ts, i, j, K_lab) for (i, j) in pairs]
    A = design(sp, blocks)
    Ad = np.hstack([dense] + [pair_block(label, ts, i, j, K_lab) for (i, j) in pairs])
    t0 = time.time(); fit_readout(A, y[ts], span); fast = time.time() - t0
    t0 = time.time(); fit_readout_dense(Ad, y[ts], span); ref = time.time() - t0
    t0 = time.time(); fm_bits(dense, y[ts], dense, y[ts], span, dim=args.dim, epochs=100); fm = time.time() - t0
    t0 = time.time(); gbdt_bits(dense, y[ts], dense, y[ts], span, rounds=60); gb = time.time() - t0
    print(f"  fit_readout  all-cross F={A.shape[1]}:  fast {fast * 1000:.0f} ms   dense {ref * 1000:.0f} ms"
          f"   speedup {ref / max(fast, 1e-9):.1f}x")
    print(f"  FM (K={span}, dim={args.dim}, 100 epochs, 1 config):      {fm * 1000:.0f} ms"
          f"   (was 5071 ms as K one-vs-rest models)")
    print(f"  GBDT (K={span}, 60 rounds):                            {gb * 1000:.0f} ms")
    return 0


def _component_targets(label, ts, pairs, comps):
    """One target per interaction (the same g(a,b) = (a+b) mod comps the floor is proved from)."""
    return [((label[ts - a] + label[ts - b]) % comps) for (a, b) in pairs]


def _multi_bits(design_tr, design_te, ys_tr, ys_te, comps, lam):
    """M independent heads on ONE shared design: parameters grow like M*comps*F, not 2^M*F."""
    tot, accs = 0.0, []
    for ytr, yte in zip(ys_tr, ys_te):
        Wt, temp = fit_readout(design_tr, ytr, comps, lam=lam)
        b, a = score_bits(Wt, temp, design_te, yte, comps)
        tot += b
        accs.append(a)
    return tot, float(np.mean(accs))


def _boost_cell_multi(job):
    """REPORT 332: the multi-head interface.  REPORT 331 showed that packing M interactions into one
    K = 2^M-class index makes the read-out's parameter count exponential, and that at M=4 even the
    ORACLE fails -- so the limit was the interface, not the selection.  Here each interaction owns its
    own head, the criterion scores a candidate by the SUM of the per-head residual correlations, and the
    additive floor is M*log2(comps) as before."""
    M, samples, seed, W, K_lab, comps, dim, epochs, rounds, lam, pmode, no_opp = job
    base = _pairs_for(M, pmode)
    true_pairs = set(base)
    label, y, span = make_components_stream(samples + 20, K_lab, base, comps, seed=seed)
    n = len(y)
    tr_n = n // 2
    ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
    ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
    ys_tr = _component_targets(label, ts_tr, base, comps)
    ys_te = _component_targets(label, ts_te, base, comps)
    xs_tr = scipy.sparse.csr_matrix(slot_features(label, ts_tr, W, K_lab))
    xs_te = scipy.sparse.csr_matrix(slot_features(label, ts_te, W, K_lab))
    dense_tr = np.asarray(xs_tr.todense()) if scipy.sparse.issparse(xs_tr) else xs_tr
    dense_te = np.asarray(xs_te.todense()) if scipy.sparse.issparse(xs_te) else xs_te
    all_pairs = [(i, j) for i in range(W) for j in range(W) if i < j]

    def blk(i, j, tr):
        return pair_block_sparse(label, ts_tr if tr else ts_te, i, j, K_lab)

    t0 = time.time()
    add_bits, _ = _multi_bits(xs_tr, xs_te, ys_tr, ys_te, comps, lam)
    Otr = design(xs_tr, [blk(i, j, True) for (i, j) in base])
    Ote = design(xs_te, [blk(i, j, False) for (i, j) in base])
    orc_bits, _ = _multi_bits(Otr, Ote, ys_tr, ys_te, comps, lam)

    #: THE OPPONENTS, on the SAME inputs and given M heads as well, so "one head per interaction" is not
    #: a privilege of our method: a multi-head FM (low-rank pairwise) and a multi-head depth-2 GBDT.
    fm_tot = gb_tot = 0.0
    if not no_opp:
        for ytr, yte in zip(ys_tr, ys_te):
            fm_b, _ = fm_bits(dense_tr, ytr, dense_te, yte, comps, dim=dim, epochs=epochs)
            g_b, _ = gbdt_bits(dense_tr, ytr, dense_te, yte, comps, rounds=rounds, lr=0.3)
            fm_tot += fm_b
            gb_tot += g_b

    chosen, bits_by_round = [], []
    Cur_tr, Cur_te = xs_tr, xs_te
    for _ in range(M + 4):
        cand = [p for p in all_pairs if p not in chosen]
        sc = np.zeros(len(cand))
        for ytr in ys_tr:                      # the criterion is the SUM over heads
            Wt, temp = fit_readout(Cur_tr, ytr, comps, lam=lam)
            err = probabilities(Wt, temp, Cur_tr, comps) - np.eye(comps)[ytr]
            sc += pair_scores(label, ts_tr, err, K_lab, comps, cand)
        pick = cand[int(np.argmax(sc))]
        chosen.append(pick)
        Cur_tr = design(Cur_tr, [blk(pick[0], pick[1], True)])
        Cur_te = design(Cur_te, [blk(pick[0], pick[1], False)])
        b, _ = _multi_bits(Cur_tr, Cur_te, ys_tr, ys_te, comps, lam)
        bits_by_round.append(b)
    hits = sum(1 for p in chosen[:M] if p in true_pairs)
    return {"M": M, "samples": samples, "seed": seed, "K": comps ** M, "W": W,
            "n_cand": len(all_pairs), "bar": float(M * np.log2(comps)), "add": add_bits,
            "oracle": orc_bits, "boostM": bits_by_round[M - 1], "boostM4": bits_by_round[M + 3],
            "fm": fm_tot, "gbdt": gb_tot, "wide": float("nan"),
            "hits": hits, "chosen": chosen[:M], "sec": time.time() - t0}


def multi(args) -> int:
    """REPORT 332: the "how many interactions" axis with the interface fixed (one head per interaction)."""
    rel_a, rel_b = _self_check_readout()
    print("REPORT 332: MANY interactions, one head each (params linear in M instead of 2^M)")
    print(f"  read-out equivalence asserted; coarsened temperature scan costs {rel_b - rel_a:+.6f} bits\n")
    K_lab, W, comps = int(args.K_lab.split(",")[0]), args.W, 2
    Ms = [int(v) for v in args.M.split(",")]
    samps = [int(v) for v in args.samples.split(",")]
    seeds = list(range(args.seeds))
    jobs = [(M, s, sd, W, K_lab, comps, args.dim, args.epochs, args.rounds, args.lam, args.pairs,
             int(args.no_opponents))
            for M in Ms for s in samps for sd in seeds]
    print(f"  {len(jobs)} cells, workers={args.workers}, K_lab={K_lab}, W={W}, comps={comps}, "
          f"heads = M, lam={args.lam}, pairs={'disjoint' if args.pairs else 'shared'}\n")
    t0 = time.time()
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_boost_cell_multi, jobs))
    else:
        rows = [_boost_cell_multi(j) for j in jobs]
    print(f"  {'M':>3} {'cand':>5} {'sam':>7} {'bar':>6} {'add':>7} {'ORACLE':>7} {'boost@M':>8} "
          f"{'FM(M heads)':>12} {'GBDT(M heads)':>14} {'hit':>5} {'s':>5}")
    for r in sorted(rows, key=lambda r: (r["W"], r["M"], r["samples"], r["seed"])):
        print(f"  {r['M']:>3} {r['n_cand']:>5} {r['samples']:>7} {r['bar']:>6.3f} {r['add']:>7.4f} "
              f"{r['oracle']:>7.4f} {r['boostM']:>8.4f} {r['fm']:>12.4f} {r['gbdt']:>14.4f} "
              f"{r['hits']:>2}/{r['M']:<2} {r['sec']:>5.1f}")
    print("\n  --- derived verdict ---")
    bad_add = [r for r in rows if abs(r["add"] - r["bar"]) > 0.05]
    bad_orc = [r for r in rows if r["oracle"] > 0.05]
    print(f"    instrument: additive arm within 0.05 of the bar in {len(rows) - len(bad_add)}/{len(rows)}"
          f"  ({'PASS' if not bad_add else 'FAIL'})")
    print(f"    instrument: ORACLE within 0.05 bits in {len(rows) - len(bad_orc)}/{len(rows)}"
          f"  ({'PASS' if not bad_orc else 'FAIL'})")
    ok = [r for r in rows if not bad_add and not bad_orc]
    if ok:
        allhit = sum(1 for r in ok if r["hits"] == r["M"])
        reached = sum(1 for r in ok if r["boostM"] - r["oracle"] < 0.05)
        print(f"    on the {len(ok)} valid cells: all M pairs found in {allhit}/{len(ok)}; "
              f"boosting within 0.05 of the oracle in {reached}/{len(ok)}")
        for M in sorted({r['M'] for r in ok}):
            g = [r for r in ok if r['M'] == M]
            print(f"      M={M}: oracle {np.mean([r['oracle'] for r in g]):.4f}  "
                  f"boost {np.mean([r['boostM'] for r in g]):.4f}  "
                  f"FM {np.mean([r['fm'] for r in g]):.4f}  "
                  f"GBDT {np.mean([r['gbdt'] for r in g]):.4f}  "
                  f"hits {sum(r['hits'] for r in g)}/{M * len(g)}")
        if not args.no_opponents:
            wb_fm = sum(1 for r in ok if r["boostM"] < r["fm"] - 0.05)
            wb_gb = sum(1 for r in ok if r["boostM"] < r["gbdt"] - 0.05)
            print(f"    boosting beats the multi-head FM in {wb_fm}/{len(ok)} cells; "
                  f"beats the multi-head GBDT in {wb_gb}/{len(ok)}")
    print(f"  total {time.time() - t0:.1f} s")
    return 0


# --------------------------------------------------------------- REPORT 334: low-rank, gradient-free
def _ridge(D, T, lam):
    F = D.shape[1]
    G = D.T @ D + lam * np.eye(F)
    return np.linalg.solve(G, D.T @ T)


def slot_features_sparse(label, ts, W, K):
    """The same W-step one-hot slot features as CSR.  Each row has EXACTLY W ones, so the dense form is
    wasteful by a factor of K (345 MB -> 8 MB at K=64, W=30, n=22k) and the Gram matrix becomes the
    bottleneck for no reason.  This is the acceleration the discovery test needed."""
    n = len(ts)
    rows = np.repeat(np.arange(n), W)
    cols = (np.arange(W)[None, :] * K + label[ts[:, None] - np.arange(W)[None, :]]).ravel()
    return scipy.sparse.csr_matrix((np.ones(n * W), (rows, cols)), shape=(n, K * W))


def _stack(Xs, pi):
    n = Xs.shape[0]
    if scipy.sparse.issparse(Xs):
        return scipy.sparse.hstack([Xs, scipy.sparse.csr_matrix(pi),
                                    scipy.sparse.csr_matrix(np.ones((n, 1)))]).tocsr()
    return np.hstack([Xs, pi, np.ones((n, 1))])


def _ridge_any(D, T, lam):
    F = D.shape[1]
    G = D.T @ D
    G = np.asarray(G.todense()) if scipy.sparse.issparse(G) else G
    G = G + lam * np.eye(F)
    rhs = np.asarray(D.T @ T) if scipy.sparse.issparse(D) else D.T @ T
    return np.linalg.solve(G, rhs)


def fit_als(Xs, cell, T, K_lab, dim=8, sweeps=6, lam=1e-3, seed=0):
    """Low-rank interaction read-out fitted by ALTERNATING LEAST SQUARES -- no gradients anywhere.

    Model (same least-squares objective as `fit_readout`, i.e. centred one-hot + temperature):

        L[t,c] = x_t . W[:,c] + b[c] + sum_r P[a_t, r] Q[b_t, r] R[r, c]

    where a_t, b_t are the two lag LABELS of the pair.  Parameter count is
    K_lab*dim*2 + dim*n_cls + slots*W, i.e. LINEAR in the alphabet, against the exact joint table's
    K_lab^2 * n_cls.  Each ALS step fixes two factors and solves the third by ridge, so the whole fit is
    closed form -- which is the point: REPORT 329 showed that the gradient-trained low-rank FM collapses
    at K=32, and the question here is whether the STRUCTURE or the FIT METHOD was to blame.
    """
    rng = np.random.default_rng(seed)
    n, n_cls = T.shape[0], T.shape[1]
    a, b = cell // K_lab, cell % K_lab
    P = rng.normal(0, 0.1, (K_lab, dim))
    Q = rng.normal(0, 0.1, (K_lab, dim))
    R = rng.normal(0, 0.1, (dim, n_cls))
    W = np.zeros((Xs.shape[1], n_cls))
    bb = np.zeros(n_cls)
    for _ in range(sweeps):
        pi = P[a] * Q[b]
        D = _stack(Xs, pi)
        Coef = _ridge_any(D, T, lam)
        W = Coef[:Xs.shape[1]]
        R = Coef[Xs.shape[1]:Xs.shape[1] + dim]
        bb = Coef[-1]
        T_res = T - Xs @ W - bb
        #: P given (Q, R): one shared dim-vector per label value, all (t, c) pairs stacked
        for av in np.unique(a):
            m = a == av
            Dz = np.einsum('nr,rc->ncr', Q[b[m]], R).reshape(-1, dim)
            if Dz.shape[0] < dim + 2:
                continue
            P[av] = _ridge(Dz, T_res[m].reshape(-1)[:, None], lam).ravel()
        for bv in np.unique(b):
            m = b == bv
            Dz = np.einsum('nr,rc->ncr', P[a[m]], R).reshape(-1, dim)
            if Dz.shape[0] < dim + 2:
                continue
            Q[bv] = _ridge(Dz, T_res[m].reshape(-1)[:, None], lam).ravel()
    return {"W": W, "b": bb, "P": P, "Q": Q, "R": R}


def predict_als(m, Xs, cell, K_lab):
    a, b = cell // K_lab, cell % K_lab
    return Xs @ m["W"] + m["b"] + (m["P"][a] * m["Q"][b]) @ m["R"]


def fit_adam_lowrank(Xs, cell, T, K_lab, dim=8, steps=400, lr=0.05, seed=0):
    """The SAME model class and the SAME objective, fitted by Adam -- so that "structure" and "fit
    method" can be separated (this is the controlled twin of `fit_als`)."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    n, n_cls = T.shape
    a = torch.tensor(cell // K_lab)
    b = torch.tensor(cell % K_lab)
    Xt = torch.tensor(Xs, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)
    P = (torch.randn(K_lab, dim) * 0.1).requires_grad_(True)
    Q = (torch.randn(K_lab, dim) * 0.1).requires_grad_(True)
    R = (torch.randn(dim, n_cls) * 0.1).requires_grad_(True)
    W = torch.zeros(Xs.shape[1], n_cls, requires_grad=True)
    bb = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.Adam([P, Q, R, W, bb], lr=lr)
    for _ in range(steps):
        pi = P[a] * Q[b]
        L = Xt @ W + bb + pi @ R
        loss = ((L - Tt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return {"W": W.detach().numpy(), "b": bb.detach().numpy(), "P": P.detach().numpy(),
            "Q": Q.detach().numpy(), "R": R.detach().numpy()}


def _temp_and_bits(pred_fn, T, y, n_cls):
    """Temperature on an internal hold-back, then bits -- the same convention as `fit_readout`."""
    n = len(y)
    cut = int(n * 0.7)
    o = pred_fn(np.arange(cut, n))
    yy = y[cut:]
    best = (1.0, 1e18)
    for temp in np.exp(np.linspace(np.log(1e-3), np.log(10.0), 120)):
        lo = o / temp
        lo = lo - lo.max(1, keepdims=True)
        p = np.exp(lo)
        p /= p.sum(1, keepdims=True)
        c = BITS(p[np.arange(len(yy)), yy])
        if c < best[1]:
            best = (float(temp), c)
    return best[0]


def _als_cell(job):
    """REPORT 334: one (K_lab, samples, seed) cell -- 'given the correct pair, WHAT CAN CASH IT?'"""
    K_lab, samples, seed, W, comps, dim, sweeps, lam, steps = job
    pair = (1, 2)
    label, y, span = make_components_stream(samples + 20, K_lab, [pair], comps, seed=seed)
    n = len(y)
    tr_n = n // 2
    ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
    ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
    Xtr = slot_features(label, ts_tr, W, K_lab)
    Xte = slot_features(label, ts_te, W, K_lab)
    ctr = label[ts_tr - pair[0]] * K_lab + label[ts_tr - pair[1]]
    cte = label[ts_te - pair[0]] * K_lab + label[ts_te - pair[1]]
    n_cls = comps
    Ttr = np.eye(n_cls)[y[ts_tr]] - 1.0 / n_cls
    Tte = np.eye(n_cls)[y[ts_te]] - 1.0 / n_cls
    t0 = time.time()

    def bits_of(pred_tr_fn, pred_te_fn, name):
        temp = _temp_and_bits(pred_tr_fn, Ttr, y[ts_tr], n_cls)
        o = pred_te_fn() / temp
        o = o - o.max(1, keepdims=True)
        p = np.exp(o)
        p /= p.sum(1, keepdims=True)
        return BITS(p[np.arange(len(y[ts_te])), y[ts_te]]), float((p.argmax(1) == y[ts_te]).mean())

    #: additive floor
    Wt, temp0 = fit_readout(scipy.sparse.csr_matrix(Xtr), y[ts_tr], n_cls, lam=lam)
    add_bits, _ = score_bits(Wt, temp0, scipy.sparse.csr_matrix(Xte), y[ts_te], n_cls)

    #: exact joint table (the incumbent): K_lab^2 columns per class
    Jtr = pair_block(label, ts_tr, pair[0], pair[1], K_lab)
    Jte = pair_block(label, ts_te, pair[0], pair[1], K_lab)
    exact_params = K_lab * K_lab * n_cls
    exact = float("nan")
    if Xtr.shape[0] > exact_params:            # only where it is identifiable at all
        Wx, tx = fit_readout(scipy.sparse.csr_matrix(np.hstack([Xtr, Jtr])), y[ts_tr], n_cls, lam=lam)
        exact, _ = score_bits(Wx, tx, scipy.sparse.csr_matrix(np.hstack([Xte, Jte])), y[ts_te], n_cls)

    #: low-rank, fitted two ways on the IDENTICAL model class and objective
    m_als = fit_als(Xtr, ctr, Ttr, K_lab, dim=dim, sweeps=sweeps, lam=lam, seed=seed)
    als_bits, als_acc = bits_of(
        lambda rows: predict_als(m_als, Xtr[rows], ctr[rows], K_lab),
        lambda: predict_als(m_als, Xte, cte, K_lab), "als")
    m_ad = fit_adam_lowrank(Xtr, ctr, Ttr, K_lab, dim=dim, steps=steps, seed=seed)
    ad_bits, ad_acc = bits_of(
        lambda rows: predict_als(m_ad, Xtr[rows], ctr[rows], K_lab),
        lambda: predict_als(m_ad, Xte, cte, K_lab), "adam")
    lr_params = 2 * K_lab * dim + dim * n_cls
    return {"K_lab": K_lab, "samples": samples, "seed": seed, "bar": float(np.log2(n_cls)),
            "add": add_bits, "exact": exact, "als": als_bits, "als_acc": als_acc,
            "adam": ad_bits, "adam_acc": ad_acc, "exact_params": exact_params,
            "lr_params": lr_params, "ntr": int(Xtr.shape[0]), "sec": time.time() - t0}


def als(args) -> int:
    """REPORT 334: can a LOW-RANK, GRADIENT-FREE read-out cash the interaction where the exact K^2 table
    becomes unfittable and where the gradient-trained low-rank model collapses?"""
    print("REPORT 334: low-rank + gradient-free (ALS) vs the exact K^2 table vs the same model on Adam")
    print("  one correct pair, one component (bar = log2(comps) = 1 bit); F and n_train are reported\n")
    print(f"  {'K_lab':>5} {'sam':>7} {'n_train':>8} {'exactP':>8} {'lrP':>6} {'add':>7} {'EXACT':>8} "
          f"{'ALS':>8} {'ALS acc':>8} {'ADAM':>8} {'ADAM acc':>9} {'s':>5}")
    jobs = [(K, s, sd, args.W, 2, args.dim, args.sweeps, args.lam, args.steps)
            for K in [int(v) for v in args.K_lab.split(",")]
            for s in [int(v) for v in args.samples.split(",")]
            for sd in range(args.seeds)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_als_cell, jobs))
    else:
        rows = [_als_cell(j) for j in jobs]
    for r in sorted(rows, key=lambda r: (r["K_lab"], r["samples"], r["seed"])):
        ex = f"{r['exact']:.4f}" if r["exact"] == r["exact"] else "n/a"
        print(f"  {r['K_lab']:>5} {r['samples']:>7} {r['ntr']:>8} {r['exact_params']:>8} "
              f"{r['lr_params']:>6} {r['add']:>7.4f} {ex:>8} {r['als']:>8.4f} {r['als_acc']:>8.4f} "
              f"{r['adam']:>8.4f} {r['adam_acc']:>9.4f} {r['sec']:>5.1f}")
    print("\n  --- derived verdict ---")
    ok = [r for r in rows if r["exact"] == r["exact"]]
    if ok:
        als_win = sum(1 for r in ok if r["als"] < r["adam"] - 0.05)
        als_near = sum(1 for r in ok if r["als"] - r["exact"] < 0.05)
        print(f"    where the exact table is identifiable ({len(ok)} cells): ALS beats Adam in "
              f"{als_win}/{len(ok)}; ALS within 0.05 of the exact table in {als_near}/{len(ok)}")
        print(f"    mean bits -- exact {np.mean([r['exact'] for r in ok]):.4f}  "
              f"ALS {np.mean([r['als'] for r in ok]):.4f}  Adam {np.mean([r['adam'] for r in ok]):.4f}")
    no_exact = [r for r in rows if r["exact"] != r["exact"]]
    if no_exact:
        print(f"    where the exact table is NOT identifiable ({len(no_exact)} cells: n_train < "
              f"K_lab^2*n_cls): ALS {np.mean([r['als'] for r in no_exact]):.4f}  "
              f"Adam {np.mean([r['adam'] for r in no_exact]):.4f}")
    return 0


def fit_als_multi(Xs, cells, T, K_lab, dim=8, sweeps=6, lam=1e-3, seed=0):
    """ALS with SEVERAL low-rank interaction blocks at once (one per discovered pair).

        L[t,c] = x_t.W[:,c] + b[c] + sum_k sum_r P_k[a_t^k, r] Q_k[b_t^k, r] R_k[r, c]
    """
    rng = np.random.default_rng(seed)
    n, n_cls = T.shape[0], T.shape[1]
    K = len(cells)
    if K == 0:
        D = np.hstack([Xs, np.ones((n, 1))])
        Coef = _ridge(D, T, lam)
        return {"W": Coef[:-1], "b": Coef[-1], "blocks": []}
    A = [c // K_lab for c in cells]
    B = [c % K_lab for c in cells]
    P = [rng.normal(0, 0.1, (K_lab, dim)) for _ in range(K)]
    Q = [rng.normal(0, 0.1, (K_lab, dim)) for _ in range(K)]
    R = [rng.normal(0, 0.1, (dim, n_cls)) for _ in range(K)]
    W = np.zeros((Xs.shape[1], n_cls))
    bb = np.zeros(n_cls)
    for _ in range(sweeps):
        pis = [P[k][A[k]] * Q[k][B[k]] for k in range(K)]
        D = np.hstack([Xs] + pis + [np.ones((n, 1))])
        Coef = _ridge(D, T, lam)
        W = Coef[:Xs.shape[1]]
        for k in range(K):
            R[k] = Coef[Xs.shape[1] + k * dim: Xs.shape[1] + (k + 1) * dim]
        bb = Coef[-1]
        base = Xs @ W + bb
        for k in range(K):
            others = sum((pis[j] @ R[j] for j in range(K) if j != k), np.zeros_like(T))
            T_res = T - base - others
            for av in np.unique(A[k]):
                m = A[k] == av
                Dz = np.einsum('nr,rc->ncr', Q[k][B[k][m]], R[k]).reshape(-1, dim)
                if Dz.shape[0] < dim + 2:
                    continue
                P[k][av] = _ridge(Dz, T_res[m].reshape(-1)[:, None], lam).ravel()
            pis[k] = P[k][A[k]] * Q[k][B[k]]
            others = sum((pis[j] @ R[j] for j in range(K) if j != k), np.zeros_like(T))
            T_res = T - base - others
            for bv in np.unique(B[k]):
                m = B[k] == bv
                Dz = np.einsum('nr,rc->ncr', P[k][A[k][m]], R[k]).reshape(-1, dim)
                if Dz.shape[0] < dim + 2:
                    continue
                Q[k][bv] = _ridge(Dz, T_res[m].reshape(-1)[:, None], lam).ravel()
    return {"W": W, "b": bb, "blocks": list(zip(P, Q, R)), "A": A, "B": B}


def predict_als_multi(m, Xs, cells, K_lab):
    out = Xs @ m["W"] + m["b"]
    for (P, Q, R), c in zip(m["blocks"], cells):
        a, b = c // K_lab, c % K_lab
        out = out + (P[a] * Q[b]) @ R
    return out


def _als_multi_cell(job):
    """REPORT 335/337: THE COMPOSITION -- the binned-residual criterion selecting pairs, with a LOW-RANK
    gradient-free read-out doing the cashing, one head per interaction, alphabet up to 128.

    The criterion's cost is O(n) per candidate REGARDLESS of the alphabet (it is a binned residual
    average, not a fitted model), so the K^2 sample wall that REPORT 331-334 hit applies only to the
    read-out -- which is exactly what the low-rank form removes.

    REPORT 337 adds `fm_on`: an INDEPENDENT low-rank opponent (a factorisation machine over the same
    slot inputs, fitted by gradient descent with cross-entropy) on the same heads, so the two methods can
    be compared at MATCHED parameter counts instead of only against exact tables and Adam twins.
    """
    M, samples, seed, W, K_lab, comps, dim, sweeps, lam, fm_on, epochs = job
    base = _pairs_for(M, 0)
    label, y, span = make_components_stream(samples + 20, K_lab, base, comps, seed=seed)
    n = len(y)
    tr_n = n // 2
    ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
    ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
    ys_tr = _component_targets(label, ts_tr, base, comps)
    ys_te = _component_targets(label, ts_te, base, comps)
    Xtr = slot_features(label, ts_tr, W, K_lab)
    Xte = slot_features(label, ts_te, W, K_lab)
    all_pairs = [(i, j) for i in range(W) for j in range(W) if i < j]

    def cells_of(pairs, ts):
        return [label[ts - i] * K_lab + label[ts - j] for (i, j) in pairs]

    t0 = time.time()
    #: additive floor, per head
    add_bits = 0.0
    for ytr, yte in zip(ys_tr, ys_te):
        Wt, temp = fit_readout(scipy.sparse.csr_matrix(Xtr), ytr, comps, lam=lam)
        b, _ = score_bits(Wt, temp, scipy.sparse.csr_matrix(Xte), yte, comps)
        add_bits += b

    total_bits, hits, picks = 0.0, 0, []
    fm_bits_tot = 0.0
    for head, (ytr, yte, true_pair) in enumerate(zip(ys_tr, ys_te, base)):
        if fm_on:
            fb, _ = fm_bits(Xtr, ytr, Xte, yte, comps, dim=dim, epochs=epochs)
            fm_bits_tot += fb
        Ttr = np.eye(comps)[ytr] - 1.0 / comps
        #: round 0: no block -- the residual the criterion reads
        m0 = fit_als_multi(Xtr, [], Ttr, K_lab, dim=dim, sweeps=1, lam=lam, seed=seed)
        o0 = predict_als_multi(m0, Xtr, [], K_lab)
        cut0 = int(len(ytr) * 0.7)
        t0best = (1.0, 1e18)
        for t in np.exp(np.linspace(np.log(1e-3), np.log(10.0), 120)):
            lo = o0[cut0:] / t
            lo = lo - lo.max(1, keepdims=True)
            pp = np.exp(lo)
            pp /= pp.sum(1, keepdims=True)
            cc = BITS(pp[np.arange(len(ytr) - cut0), ytr[cut0:]])
            if cc < t0best[1]:
                t0best = (float(t), cc)
        lo = o0 / t0best[0]
        lo = lo - lo.max(1, keepdims=True)
        p0 = np.exp(lo)
        p0 /= p0.sum(1, keepdims=True)
        err = p0 - np.eye(comps)[ytr]
        sc = pair_scores(label, ts_tr, err, K_lab, comps, all_pairs)
        pick = all_pairs[int(np.argmax(sc))]
        picks.append(pick)
        if pick == true_pair:
            hits += 1
        m1 = fit_als_multi(Xtr, cells_of([pick], ts_tr), Ttr, K_lab, dim=dim, sweeps=sweeps,
                           lam=lam, seed=seed)
        o_tr = predict_als_multi(m1, Xtr, cells_of([pick], ts_tr), K_lab)
        #: temperature on the internal hold-back, then bits on the eval slice
        cut = int(len(ytr) * 0.7)
        best = (1.0, 1e18)
        for t in np.exp(np.linspace(np.log(1e-3), np.log(10.0), 120)):
            lo = o_tr[cut:] / t
            lo = lo - lo.max(1, keepdims=True)
            pp = np.exp(lo)
            pp /= pp.sum(1, keepdims=True)
            cc = BITS(pp[np.arange(len(ytr) - cut), ytr[cut:]])
            if cc < best[1]:
                best = (float(t), cc)
        o_te = predict_als_multi(m1, Xte, cells_of([pick], ts_te), K_lab) / best[0]
        o_te = o_te - o_te.max(1, keepdims=True)
        pt = np.exp(o_te)
        pt /= pt.sum(1, keepdims=True)
        total_bits += BITS(pt[np.arange(len(yte)), yte])
    return {"M": M, "K_lab": K_lab, "samples": samples, "seed": seed, "bar": float(M * np.log2(comps)),
            "add": add_bits, "bits": total_bits, "hits": hits, "picks": picks, "fm": fm_bits_tot,
            "our_params": M * (2 * K_lab * dim + dim * comps + K_lab * W * comps),
            "fm_params": M * (K_lab * W * comps + K_lab * W * dim * comps),
            "n_cand": len(all_pairs), "sec": time.time() - t0}


def alsmulti(args) -> int:
    """REPORT 335: the composition, at an alphabet the exact K^2 table can no longer handle."""
    print("REPORT 335: COMPOSITION -- binned-residual criterion + low-rank gradient-free read-out")
    print("  one head per interaction; the criterion scores all lag pairs against that head's residual\n")
    print(f"  {'K_lab':>5} {'M':>3} {'cand':>5} {'sam':>7} {'bar':>6} {'add':>7} {'ours':>8} "
          f"{'ourP':>7} {'FM(1indep)':>10} {'fmP':>8} {'hits':>6} {'s':>6}")
    jobs = [(M, s, sd, args.W, K, 2, args.dim, args.sweeps, args.lam, int(args.fm), args.epochs)
            for K in [int(v) for v in args.K_lab.split(",")]
            for M in [int(v) for v in args.M.split(",")]
            for s in [int(v) for v in args.samples.split(",")]
            for sd in range(args.seeds)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_als_multi_cell, jobs))
    else:
        rows = [_als_multi_cell(j) for j in jobs]
    for r in sorted(rows, key=lambda r: (r["K_lab"], r["M"], r["samples"], r["seed"])):
        print(f"  {r['K_lab']:>5} {r['M']:>3} {r['n_cand']:>5} {r['samples']:>7} {r['bar']:>6.3f} "
              f"{r['add']:>7.4f} {r['bits']:>8.4f} {r['our_params']:>7} {r['fm']:>10.4f} "
              f"{r['fm_params']:>8} {r['hits']:>3}/{r['M']:<2} {r['sec']:>6.1f}")
    print("\n  --- derived verdict ---")
    bad_add = [r for r in rows if abs(r["add"] - r["bar"]) > 0.05]
    print(f"    instrument: additive arm within 0.05 of the bar in {len(rows) - len(bad_add)}/{len(rows)}"
          f"  ({'PASS' if not bad_add else 'FAIL'})")
    solved = [r for r in rows if r["bits"] < 0.05 and r["hits"] == r["M"]]
    print(f"    all M pairs found AND bits ~0 in {len(solved)}/{len(rows)} cells")
    for K in sorted({r["K_lab"] for r in rows}):
        g = [r for r in rows if r["K_lab"] == K]
        w = sum(1 for r in g if r["bits"] < r["fm"] - 0.05)
        print(f"      K_lab={K}: ours {np.mean([r['bits'] for r in g]):.4f} bits @ "
              f"{g[0]['our_params']} params | FM {np.mean([r['fm'] for r in g]):.4f} bits @ "
              f"{g[0]['fm_params']} params | ours wins {w}/{len(g)}")
    return 0


def _pool_cell(job):
    """REPORT 338: one (K_lab, N, seed) cell of the candidate-pool sweep.

    The question is what happens when the pool is mostly junk: does the true pair still rank first, and
    does its score clear the null-max of N pairing-destroyed candidates?
    """
    K_lab, N, samples, seed, comps, dim, sweeps, lam, B = job
    W = int(np.ceil((1.0 + np.sqrt(1.0 + 8.0 * N)) / 2.0))
    pair = (1, 2)
    label, y, span = make_components_stream(samples + 20, K_lab, [pair], comps, seed=seed)
    n = len(y)
    tr_n = n // 2
    ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
    ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
    Xtr = slot_features(label, ts_tr, W, K_lab)
    Xte = slot_features(label, ts_te, W, K_lab)
    ys_tr = _component_targets(label, ts_tr, [pair], comps)[0]
    ys_te = _component_targets(label, ts_te, [pair], comps)[0]
    all_pairs = [(i, j) for i in range(W) for j in range(W) if i < j][:N]
    if pair not in all_pairs:
        return {"skip": True}
    t0 = time.time()
    Wt, temp = fit_readout(scipy.sparse.csr_matrix(Xtr), ys_tr, comps, lam=lam)
    add_bits, _ = score_bits(Wt, temp, scipy.sparse.csr_matrix(Xte), ys_te, comps)
    err = probabilities(Wt, temp, scipy.sparse.csr_matrix(Xtr), comps) - np.eye(comps)[ys_tr]
    scores = pair_scores(label, ts_tr, err, K_lab, comps, all_pairs)
    order = np.argsort(-scores)
    rank_true = int(np.nonzero(order == all_pairs.index(pair))[0][0]) + 1
    top = all_pairs[int(order[0])]
    #: the null of "the best of N junk candidates": each candidate gets its OWN pairing shuffle
    rng = np.random.default_rng(seed + 4242)
    null_max = []
    for _ in range(B):
        second = {p: label[ts_tr - p[1]][rng.permutation(len(ts_tr))] for p in all_pairs}
        null_max.append(float(pair_scores(label, ts_tr, err, K_lab, comps, all_pairs,
                                          second=second).max()))
    thr = float(np.quantile(null_max, 0.95))
    true_score = float(scores[all_pairs.index(pair)])
    top_score = float(scores[int(order[0])])
    #: does the top pick actually cash it?  (low-rank ALS on the picked pair)
    ctr = label[ts_tr - top[0]] * K_lab + label[ts_tr - top[1]]
    cte = label[ts_te - top[0]] * K_lab + label[ts_te - top[1]]
    Ttr = np.eye(comps)[ys_tr] - 1.0 / comps
    m1 = fit_als(Xtr, ctr, Ttr, K_lab, dim=dim, sweeps=sweeps, lam=lam, seed=seed)
    o_tr = predict_als(m1, Xtr, ctr, K_lab)
    cut = int(len(ys_tr) * 0.7)
    best = (1.0, 1e18)
    for t in np.exp(np.linspace(np.log(1e-3), np.log(10.0), 120)):
        lo = o_tr[cut:] / t
        lo = lo - lo.max(1, keepdims=True)
        pp = np.exp(lo)
        pp /= pp.sum(1, keepdims=True)
        cc = BITS(pp[np.arange(len(ys_tr) - cut), ys_tr[cut:]])
        if cc < best[1]:
            best = (float(t), cc)
    o_te = predict_als(m1, Xte, cte, K_lab) / best[0]
    o_te = o_te - o_te.max(1, keepdims=True)
    pt = np.exp(o_te)
    pt /= pt.sum(1, keepdims=True)
    pick_bits = BITS(pt[np.arange(len(ys_te)), ys_te])
    return {"K_lab": K_lab, "N": N, "samples": samples, "seed": seed, "W": W, "n_pool": len(all_pairs),
            "bar": float(np.log2(comps)), "add": add_bits, "rank_true": rank_true,
            "top": top, "top_is_true": int(top == pair), "true_score": true_score,
            "top_score": top_score, "thr": thr, "null_max_mean": float(np.mean(null_max)),
            "clears": int(true_score > thr), "top_clears": int(top_score > thr),
            "pick_bits": pick_bits, "sec": time.time() - t0}


def pool(args) -> int:
    """REPORT 338: the SEARCH axis -- how big can the candidate pool get before the criterion breaks?"""
    print("REPORT 338: candidate-pool sweep -- the search axis, with a measured null-max per N")
    print("  one interaction (1,2); the null is 'the best of N pairing-destroyed candidates'\n")
    print(f"  {'K_lab':>5} {'N':>5} {'W':>4} {'pool':>5} {'add':>7} {'rank':>5} {'top':>9} "
          f"{'true':>6} {'thr95':>10} {'nullMax':>10} {'clear':>6} {'bits':>8} {'s':>5}")
    jobs = [(K, N, int(args.samples.split(",")[0]), sd, 2, args.dim, args.sweeps, args.lam, 5)
            for K in [int(v) for v in args.K_lab.split(",")]
            for N in [int(v) for v in args.cands.split(",")]
            for sd in range(args.seeds)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = [r for r in ex.map(_pool_cell, jobs) if not r.get("skip")]
    else:
        rows = [r for r in (_pool_cell(j) for j in jobs) if not r.get("skip")]
    for r in sorted(rows, key=lambda r: (r["K_lab"], r["N"], r["seed"])):
        print(f"  {r['K_lab']:>5} {r['N']:>5} {r['W']:>4} {r['n_pool']:>5} {r['add']:>7.4f} "
              f"{r['rank_true']:>5} {str(r['top']):>9} {r['true_score']:>6.3f} {r['thr']:>10.3f} "
              f"{r['null_max_mean']:>10.3f} {r['clears']:>6} {r['pick_bits']:>8.4f} {r['sec']:>5.1f}")
    print("\n  --- derived verdict (docs/PREREG_338.md) ---")
    bad_add = [r for r in rows if abs(r["add"] - r["bar"]) > 0.05]
    print(f"    instrument: additive within 0.05 of the bar in {len(rows) - len(bad_add)}/{len(rows)}"
          f"  ({'PASS' if not bad_add else 'FAIL'})")
    for K in sorted({r["K_lab"] for r in rows}):
        g = [r for r in rows if r["K_lab"] == K]
        for N in sorted({r["N"] for r in g}):
            h = [r for r in g if r["N"] == N]
            r1 = sum(1 for r in h if r["rank_true"] == 1)
            cl = sum(1 for r in h if r["clears"])
            print(f"      K_lab={K} N={N:<5} rank-1 {r1}/{len(h)}   true>thr {cl}/{len(h)}   "
                  f"null-max {np.mean([r['null_max_mean'] for r in h]):.3f}")
    print("    C2 (rank-1 at N=200/1000) and C3 (true > null-max 95th) are read off the rows above.")
    return 0


def pair_scores(label, ts, err, K_lab, n_cls, pairs, second=None, weighted=False):
    """The centred-bincount criterion, generalised to any lag pair.

    `weighted=True` is the INPUT ADAPTER of REPORT 340.  The raw score `sum_c ||R_c||^2` sums over
    cells with EQUAL weight, and on a real stream the biggest pair-cell can hold 33-45% of all rows
    (REPORT 339), so the score is dominated by whichever candidate happens to cover that cell -- not by
    whether it explains the residual.  Dividing each cell's contribution by its count makes the
    expected contribution of every cell equal under the null (E||R_c||^2 ~ n_c for i.i.d. residuals),
    which is the minimal shape fix.  On a uniform synthetic stream it must change NOTHING (asserted in
    `_self_check_weighting`).
    """
    n = len(ts)
    s = err.sum(axis=0)
    out = []
    for (i, j) in pairs:
        b = label[ts - j] if second is None else second[(i, j)]
        cell = label[ts - i] * K_lab + b
        R = np.zeros((K_lab * K_lab, n_cls), dtype=np.float64)
        np.add.at(R, cell, err)
        counts = np.bincount(cell, minlength=K_lab * K_lab).astype(np.float64)
        R -= (counts / n)[:, None] * s[None, :]
        sq = (R ** 2).sum(1)
        if weighted:
            nz = counts > 0
            out.append(float((sq[nz] / counts[nz]).sum()))
        else:
            out.append(float(sq.sum()))
    return np.asarray(out)


def _self_check_weighting():
    """On a stream with (near) equal cell counts the adapter must not change the RANKING at all."""
    K_lab, W, n = 8, 10, 20000
    label, y, span = make_components_stream(n, K_lab, [(1, 2)], 2, seed=5)
    ts = np.arange(W + 1, len(y), 7)
    ts = ts[y[ts] >= 0]
    X = slot_features(label, ts, W, K_lab)
    Wt, temp = fit_readout(scipy.sparse.csr_matrix(X), y[ts], span)
    err = probabilities(Wt, temp, scipy.sparse.csr_matrix(X), span) - np.eye(span)[y[ts]]
    pairs = [(i, j) for i in range(W) for j in range(W) if i < j]
    a = pair_scores(label, ts, err, K_lab, span, pairs)
    b = pair_scores(label, ts, err, K_lab, span, pairs, weighted=True)
    ra, rb = int(np.argsort(-a)[0]), int(np.argsort(-b)[0])
    return ra == rb, float(np.corrcoef(a, b)[0, 1])


def als_table(model, K_lab, comps):
    """THE OUTPUT ADAPTER: (P, Q, R) -> the interaction table g(a, b), i.e. one human-readable thing.

        L_ab(c) = sum_r P[a,r] Q[b,r] R[r,c]      ->      g(a,b) = argmax_c L_ab(c)

    This is the analogue of a vocoder: the model's internal factors turned back into something a person
    can read, check, and use.  REPORT 340 prints it and scores it against the known truth.
    """
    L = np.einsum('ar,br,rc->abc', model["P"], model["Q"], model["R"])
    return L.argmax(-1)


def _emit_cell(job):
    """REPORT 340: end-to-end -- stream in, criterion picks the pair, low-rank read-out fitted, and the
    interaction table EMITTED and checked against the known g(a,b) = (a+b) mod comps."""
    K_lab, samples, seed, W, comps, dim, sweeps, lam, N = job
    pair = (1, 2)
    label, y, span = make_components_stream(samples + 20, K_lab, [pair], comps, seed=seed)
    n = len(y)
    tr_n = n // 2
    ts_tr = np.arange(W + 1, tr_n, 7); ts_tr = ts_tr[y[ts_tr] >= 0]
    ts_te = np.arange(tr_n + W + 1, n, 7); ts_te = ts_te[y[ts_te] >= 0]
    Xtr = slot_features(label, ts_tr, W, K_lab)
    all_pairs = [(i, j) for i in range(W) for j in range(W) if i < j][:N]
    Wt, temp = fit_readout(scipy.sparse.csr_matrix(Xtr), y[ts_tr], span)
    err = probabilities(Wt, temp, scipy.sparse.csr_matrix(Xtr), span) - np.eye(span)[y[ts_tr]]
    sc = pair_scores(label, ts_tr, err, K_lab, span, all_pairs)
    pick = all_pairs[int(np.argmax(sc))]
    ctr = label[ts_tr - pick[0]] * K_lab + label[ts_tr - pick[1]]
    Ttr = np.eye(span)[y[ts_tr]] - 1.0 / span
    m = fit_als(Xtr, ctr, Ttr, K_lab, dim=dim, sweeps=sweeps, lam=lam, seed=seed)
    tbl = als_table(m, K_lab, comps)
    A, B = np.meshgrid(np.arange(K_lab), np.arange(K_lab), indexing="ij")
    truth = (A + B) % comps
    exact = float((tbl == truth).mean())
    #: the count-weighted accuracy is the one that matters: right where the DATA lives
    counts = np.bincount(ctr, minlength=K_lab * K_lab).reshape(K_lab, K_lab).astype(float)
    wacc = float((counts * (tbl == truth)).sum() / counts.sum())
    return {"K_lab": K_lab, "N": N, "pick": pick, "hit": int(pick == pair), "exact": exact,
            "wacc": wacc, "corner": tbl[:6, :6].tolist()}


def emit(args) -> int:
    """REPORT 340: the OUTPUT adapter -- emit the interaction as a table a person can read."""
    Ks, W = [int(v) for v in args.K_lab.split(",")], args.W
    print("REPORT 340: emit the interaction table (stream -> criterion -> low-rank -> g(a,b))")
    print("  truth is known here: g(a,b) = (a+b) mod comps, so the emitted table can be SCORED\n")
    print(f"  {'K_lab':>5} {'cand':>5} {'pick':>9} {'hit':>4} {'table exact':>12} {'count-weighted':>15}")
    jobs = [(K, int(args.samples.split(",")[0]), sd, W, 2, args.dim, args.sweeps, args.lam,
             int(args.cands.split(",")[0]))
            for K in Ks for sd in range(args.seeds)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_emit_cell, jobs))
    else:
        rows = [_emit_cell(j) for j in jobs]
    for r in sorted(rows, key=lambda r: r["K_lab"]):
        print(f"  {r['K_lab']:>5} {r['N']:>5} {str(r['pick']):>9} {r['hit']:>4} "
              f"{r['exact']:>12.4f} {r['wacc']:>15.4f}")
    print("\n  emitted table corner (rows = label at lag 1, cols = label at lag 2):")
    for r in sorted(rows, key=lambda r: r["K_lab"])[:1]:
        for row in r["corner"]:
            print("    " + " ".join(str(v) for v in row))
    print("\n  exact = share of ALL cells where the emitted component equals the truth;")
    print("  count-weighted = the same share weighted by how often each cell actually occurs.")
    return 0


def _real_cell(job):
    """REPORT 340: the INPUT adapter, tested on the REAL stream (corpus_193.txt, pinned).

    A known interaction is INJECTED into the real byte stream -- y = (tok[t-a] + tok[t-b]) mod comps --
    so the truth is known while the marginals, the repeats and the local structure are all real.  With
    `noise > 0` the label is replaced by a uniform draw with that probability, which WEAKENS the
    injected signal: that is the regime where a statistic dominated by one big cell should fail first.
    """
    import real_seq_data as R
    K, pair, W, stride, comps, lam, seed, shuffle, noise, rare = job
    raw = R.load_corpus()
    ids, names = R.raw_tokens(raw, "byte")
    n = len(ids)
    bounds = R.block_bounds(n)
    train_end = bounds[R.TRAIN_BLOCKS - 1][1]
    tok, _ = R.discretise(ids, names, train_end, K)
    rng = np.random.default_rng(seed)
    if shuffle:
        tok = tok[rng.permutation(len(tok))]
    a, b = pair
    ts = np.arange(W + max(pair), train_end - 20, stride)
    y = (tok[ts - a] + tok[ts - b]) % comps
    if noise > 0:
        flip = rng.random(len(ts)) < noise
        y = np.where(flip, rng.integers(0, comps, size=len(ts)), y)
    #: THE RARE-CELL REGIME.  With `rare` > 0 the signal exists ONLY where x_{t-a} is one of the rarest
    #: symbols, so the informative cells are the SMALL ones while the big cells are pure noise.  That is
    #: the one regime where dividing each cell by its count can help (it amplifies the small cells),
    #: and REPORT 340 tested only the opposite regime.
    informative = 1.0
    if rare > 0:
        sym_counts = np.bincount(tok[:train_end], minlength=K)
        keep = np.argsort(sym_counts)[:rare]          # the `rare` least frequent symbol ids
        mask = np.isin(tok[ts - a], keep)
        informative = float(mask.mean())
        y = np.where(mask, y, rng.integers(0, comps, size=len(ts)))
    X = slot_features(tok, ts, W, K)
    Wt, temp = fit_readout(scipy.sparse.csr_matrix(X), y, comps, lam=lam)
    add_bits, add_acc = score_bits(Wt, temp, scipy.sparse.csr_matrix(X), y, comps)
    cnt_y = np.bincount(y, minlength=comps).astype(float)
    py = cnt_y / cnt_y.sum()
    floor = float(-(py[py > 0] * np.log2(py[py > 0])).sum())
    err = probabilities(Wt, temp, scipy.sparse.csr_matrix(X), comps) - np.eye(comps)[y]
    pairs = [(i, j) for i in range(W) for j in range(W) if i < j]
    cell = tok[ts - a] * K + tok[ts - b]
    counts = np.bincount(cell, minlength=K * K).astype(float)
    su = pair_scores(tok, ts, err, K, comps, pairs)
    sw = pair_scores(tok, ts, err, K, comps, pairs, weighted=True)
    ru = int(np.nonzero(np.argsort(-su) == pairs.index(pair))[0][0]) + 1
    rw = int(np.nonzero(np.argsort(-sw) == pairs.index(pair))[0][0]) + 1
    return {"K": K, "n": len(ts), "pair": pair, "shuffle": int(shuffle), "n_cand": len(pairs),
            "noise": noise, "rare": rare, "informative": informative,
            "floor": floor, "add_bits": add_bits, "add_acc": add_acc,
            "rank_unw": ru, "rank_w": rw,
            "top_unw": pairs[int(np.argsort(-su)[0])], "top_w": pairs[int(np.argsort(-sw)[0])],
            "max_cell_share": float(counts.max() / counts.sum()),
            "cells_occupied": float((counts > 0).mean())}


def real(args) -> int:
    """REPORT 340/341: the input adapter on the real stream, with the truth injected by construction."""
    print("REPORT 341: REAL stream + injected interaction, incl. the RARE-CELL regime")
    print("  y = (tok[t-1] + tok[t-8]) mod 2; `rare`>0 keeps the signal ONLY on the rarest symbols\n")
    print(f"  {'K':>4} {'cand':>5} {'noise':>5} {'rare':>5} {'infor':>6} {'rows':>7} {'floor':>7} "
          f"{'additive':>9} {'maxCell':>8} {'rank等权':>8} {'rank加权':>8} {'top等权':>9} {'top加权':>9}")
    jobs = [(K, (1, 8), W, args.real_stride, 2, args.lam, sd, 0, nz, rr)
            for K in [int(v) for v in args.K_lab.split(",")]
            for W in [int(v) for v in args.Ws.split(",")]
            for nz in [float(v) for v in args.noise.split(",")]
            for rr in [int(v) for v in args.rare.split(",")]
            for sd in range(args.seeds)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_real_cell, jobs))
    else:
        rows = [_real_cell(j) for j in jobs]
    for r in sorted(rows, key=lambda r: (r["K"], r["n_cand"], r["noise"], r["rare"])):
        print(f"  {r['K']:>4} {r['n_cand']:>5} {r['noise']:>5.2f} {r['rare']:>5} "
              f"{r['informative']:>6.3f} {r['n']:>7} {r['floor']:>7.4f} {r['add_bits']:>9.4f} "
              f"{r['max_cell_share']:>8.3f} {r['rank_unw']:>8} {r['rank_w']:>8} "
              f"{str(r['top_unw']):>9} {str(r['top_w']):>9}")
    print("\n  --- derived verdict ---")
    for K in sorted({r["K"] for r in rows}):
        for rr in sorted({r["rare"] for r in rows if r["K"] == K}):
            g = [r for r in rows if r["K"] == K and r["rare"] == rr]
            unw = sum(1 for r in g if r["rank_unw"] == 1)
            wtd = sum(1 for r in g if r["rank_w"] == 1)
            inf = g[0]["informative"]
            print(f"    K={K:<4} rare={rr:<2} (signal on {inf:.1%} of rows, maxCell "
                  f"{g[0]['max_cell_share']:.3f}):  raw #1 in {unw}/{len(g)}   weighted #1 in "
                  f"{wtd}/{len(g)}")
    print("    the RARE-CELL rows are where the adapter could matter; everything else is noise.")
    return 0


def _natural_cell(job):
    """REPORT 342: NATURAL discovery on the real stream -- no injection anywhere.

    Target = the real next symbol.  The additive read-out sees the window; the criterion then picks the
    lag pair that best explains its residual.  Arm A adds THAT pair; arm B adds a RANDOM pair of the
    same width.  The two arms differ in exactly one thing (which pair), so the gap is the criterion's
    contribution -- and if there is nothing to find, the gap is zero.  This is the first test in the
    whole line whose "truth" is not constructed by us.
    """
    import real_seq_data as R
    K, W, stride, lam, seed, dim, sweeps, n_rand = job
    raw = R.load_corpus()
    ids, names = R.raw_tokens(raw, "byte")
    n = len(ids)
    bounds = R.block_bounds(n)
    train_end = bounds[R.TRAIN_BLOCKS - 1][1]
    tok, _ = R.discretise(ids, names, train_end, K)
    ts = np.arange(W + 1, train_end - 20, stride)
    y = tok[ts]                                   #: the REAL next symbol
    X = slot_features_sparse(tok, ts - 1, W, K)    #: lags 1..W (lag 0 is excluded: it is the target's own past)
    t0 = time.time()
    #: additive (no interaction block)
    T = np.eye(K)[y] - 1.0 / K
    m0 = fit_als(X, np.zeros(len(ts), dtype=np.int64), T, K, dim=dim, sweeps=1, lam=lam, seed=seed)
    o0 = predict_als(m0, X, np.zeros(len(ts), dtype=np.int64), K)
    cut = int(len(ts) * 0.7)
    temp0 = _best_temp(o0, y, K, cut)
    add_bits, _ = _bits_with_temp(o0, y, K, temp0, cut)
    #: the criterion's pick
    err = _softmax(o0 / temp0) - np.eye(K)[y]
    pairs = [(i, j) for i in range(W) for j in range(W) if i < j]
    sc = pair_scores(tok, ts - 1, err, K, K, pairs, weighted=True)
    pick = pairs[int(np.argmax(sc))]

    def bits_with_pair(p):
        c = tok[ts - 1 - p[0]] * K + tok[ts - 1 - p[1]]
        m = fit_als(X, c, T, K, dim=dim, sweeps=sweeps, lam=lam, seed=seed)
        o = predict_als(m, X, c, K)
        t = _best_temp(o, y, K, cut)
        return _bits_with_temp(o, y, K, t, cut)[0]

    bits_A = bits_with_pair(pick)
    rng = np.random.default_rng(seed)
    others = [p for p in pairs if p != pick]
    idx = rng.choice(len(others), size=min(n_rand, len(others)), replace=False)
    rand_bits = np.array([bits_with_pair(others[int(i)]) for i in idx])
    #: NULL = 0.5: a pick unrelated to real predictive value beats half of the random draws.
    frac_beaten = float((rand_bits > bits_A + 1e-9).mean())
    rank_bits = None
    if len(pairs) <= 80:                      #: exact: where does the pick land when ranked by bits?
        allbits = np.array([bits_A if p == pick else bits_with_pair(p) for p in pairs])
        rank_bits = int(np.nonzero(np.argsort(allbits) == pairs.index(pick))[0][0]) + 1
    return {"K": K, "W": W, "n": len(ts), "n_cand": len(pairs), "add": add_bits, "pick": pick,
            "bitsA": bits_A, "rand_mean": float(rand_bits.mean()), "rand_min": float(rand_bits.min()),
            "frac_beaten": frac_beaten, "rank_bits": rank_bits, "sec": time.time() - t0}


def _softmax(o):
    o = o - o.max(1, keepdims=True)
    p = np.exp(o)
    return p / p.sum(1, keepdims=True)


def _best_temp(o, y, K, cut):
    best = (1.0, 1e18)
    for t in np.exp(np.linspace(np.log(1e-3), np.log(10.0), 80)):
        b, _ = _bits_with_temp(o, y, K, t, cut)
        if b < best[1]:
            best = (float(t), b)
    return best[0]


def _bits_with_temp(o, y, K, temp, cut):
    p = _softmax(o / temp)
    b = BITS(p[np.arange(cut, len(y)), y[cut:]])
    return b, float((p[cut:].argmax(1) == y[cut:]).mean())


def natural(args) -> int:
    """REPORT 342: does the criterion find anything REAL in real text?  Arm A vs a random pair."""
    print("REPORT 342: NATURAL discovery -- target is the real next symbol, nothing is injected")
    print("  arm A = additive + the criterion's pick;  arm B = additive + a random pair (same width)\n")
    print(f"  {'K':>4} {'W':>3} {'cand':>5} {'rows':>7} {'additive':>9} {'pick':>9} {'A':>8} "
          f"{'B mean':>8} {'frac>rand':>10} {'rankbybits':>11} {'s':>5}")
    jobs = [(K, W, args.real_stride, args.lam, sd, args.dim, args.sweeps, args.n_rand)
            for K in [int(v) for v in args.K_lab.split(",")]
            for W in [int(v) for v in args.Ws.split(",")]
            for sd in range(args.seeds)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_natural_cell, jobs))
    else:
        rows = [_natural_cell(j) for j in jobs]
    for r in sorted(rows, key=lambda r: (r["K"], r["W"], r["n"])):
        rb = "-" if r["rank_bits"] is None else f"{r['rank_bits']}/{r['n_cand']}"
        print(f"  {r['K']:>4} {r['W']:>3} {r['n_cand']:>5} {r['n']:>7} {r['add']:>9.4f} "
              f"{str(r['pick']):>9} {r['bitsA']:>8.4f} {r['rand_mean']:>8.4f} "
              f"{r['frac_beaten']:>10.3f} {rb:>11} {r['sec']:>5.1f}")
    print("\n  --- derived verdict ---")
    fb = np.array([r["frac_beaten"] for r in rows])
    print(f"    mean share of random pairs the pick beats: {fb.mean():.3f} "
          f"(NULL = 0.500; min {fb.min():.3f}, max {fb.max():.3f})")
    print(f"    cells where it beat EVERY random draw: {int((fb > 0.999).sum())}/{len(rows)}")
    rk = [r["rank_bits"] for r in rows if r["rank_bits"] is not None]
    if rk:
        print(f"    where it lands when ALL candidate pairs are ranked by real held-out bits: "
              f"{sorted(rk)}  (of {rows[0]['n_cand']}; rank 1 would be perfect alignment)")
    print("    nothing was injected: the target is the real next symbol, so a value at the null means")
    print("    the criterion found nothing real here, and a value above it means it found something.")
    return 0


def _fields_cell(job):
    """REPORT 343: THE FEATURE-CROSS SETTING, on real records.

    The corpus's LINES are the records; byte positions inside a line are the FIELDS.  The target is a
    real property of each line:

        y = 1  iff  line[0] == line[-1]

    which is true of real formatted text (table rows ``|...|``, quoted lines, bracketed lines, rules).
    The additive read-out sees each field on its own and CANNOT decide equality of two fields; the pair
    (first, last) can -- and its truth table is known in closed form, g(a,b) = [a == b], so the EMITTED
    table (REPORT 340's output adapter) can be scored cell-for-cell against it.

    This is the setting where real applications live (user x item, field A x field B), and the one where
    FM / DeepFM / GBDT are the standard opponents -- so both are run on exactly the same fields.
    """
    import real_seq_data as R
    K, F, n_lines, lam, seed, dim, sweeps, epochs, rounds, n_rand = job
    raw = R.load_corpus()
    #: RECORDS = fixed-size blocks of WORDS, FIELDS = positions inside the block.
    #: The line-based version failed for a measurable reason: on this mostly-CHINESE corpus the last
    #: byte of a line is almost always a UTF-8 continuation byte, which is NOT in the top-K byte
    #: vocabulary, so the "last byte" field was CONSTANT (measured: `uniq = [0]`) and the parity target
    #: degenerated into a function of one field -- which is why the additive arm kept reaching -0.0000.
    #: Word tokens have no such problem: the last word of a block is a real word.
    ids, names = R.raw_tokens(raw, "word")
    B = F + 2
    nb = len(ids) // B
    recs = ids[:nb * B].reshape(nb, B)
    rng = np.random.default_rng(seed)
    rng.shuffle(recs)
    n = int(min(nb, n_lines))
    recs = recs[:n]
    cut = int(n * 0.7)
    counts = np.bincount(recs[:cut].ravel(), minlength=int(ids.max()) + 1)
    keep = np.lexsort((np.arange(counts.size), -counts))[:K - 1]
    lut = np.zeros(counts.size, dtype=np.int64)
    lut[keep] = np.arange(1, K)
    pos = list(range(F - 1)) + [B - 1]
    cols = lut[recs[:, pos]]
    #: THE TARGET, final form.  Four designs failed on the way here, and each failure is a fact:
    #:   1) [line[0] == line[-1]]: after mapping, both ends concentrate on the same classes, so equality
    #:      is largely predictable from ONE field (additive reached -0.0000 bits).
    #:   2) [first is a digit] AND [last is a letter]: digits are not in the top-K bytes AND is
    #:      LINEARLY SEPARABLE anyway (a threshold on the sum of two log-odds implements AND).
    #:   3) [first is a LETTER] XOR [last is a LETTER]: this corpus is mostly CHINESE in UTF-8, so NO
    #:      ASCII letter is in the top-15 byte vocabulary (3 of 63 at K=64).
    #:   4) the same XOR over "is ASCII", BALANCED BY RESAMPLING: balancing changed the conditional
    #:      distribution -- the balanced subset is exactly the rows where the two ends disagree, and
    #:      inside it one field predicts the label, so the additive arm hit -0.0000 bits again.
    #: Lesson: never balance by resampling when the label is a relation between features; it can turn a
    #: non-separable relation into a separable one.  So the label is left alone and its entropy is
    #: REPORTED as an instrument check, and the target is a parity over two REAL fields:
    #:
    #:     y = (column 0 + last column) mod 2          (non-separable, balanced-ish, known table)
    raw_keep = np.concatenate([[0], keep])               #: mapped id -> raw byte (0 = <unk>)
    y_all = ((cols[:, 0] + cols[:, F - 1]) % 2).astype(np.int64)
    p1 = float(y_all.mean())
    H_y = float(-(p1 * np.log2(max(p1, 1e-12)) + (1 - p1) * np.log2(max(1 - p1, 1e-12))))
    if H_y < 0.2:
        return {"K": K, "F": F, "degenerate": 4, "p1": p1, "H": H_y, "n": 0}
    H_y = float(-(p1 * np.log2(max(p1, 1e-12)) + (1 - p1) * np.log2(max(1 - p1, 1e-12))))
    #: NO RESAMPLING.  The label is a relation between two fields, so balancing it would change the
    #: conditional distribution and can make a non-separable relation separable (attempt 4 above).
    cols, y_all, n = cols, y_all, len(y_all)
    cut = int(n * 0.7)
    #: a CONSTANT field makes any target a function of the remaining fields -- assert none is constant
    const_fields = [k for k in range(F) if len(np.unique(cols[:, k])) < 3]
    if const_fields:
        return {"K": K, "F": F, "degenerate": 5, "p1": p1, "H": H_y, "n": 0,
                "const_fields": const_fields}
    blocks = [scipy.sparse.csr_matrix((np.ones(n), (np.arange(n), cols[:, k])), shape=(n, K))
              for k in range(F)]
    Xall = scipy.sparse.hstack(blocks).tocsr()
    Xtr, Xte = Xall[:cut].tocsr(), Xall[cut:].tocsr()
    ytr, yte = y_all[:cut], y_all[cut:]
    tr, te = slice(0, cut), slice(cut, n)
    comps = 2
    Ttr = np.eye(comps)[ytr] - 1.0 / comps
    t0 = time.time()
    pairs = [(i, j) for i in range(F) for j in range(F) if i < j]
    zero = np.zeros(Xtr.shape[0], dtype=np.int64)

    def bits_with(p):
        ctr = (cols[tr, p[0]] * K + cols[tr, p[1]]) if p else zero
        m = fit_als(Xtr, ctr, Ttr, K, dim=dim, sweeps=sweeps if p else 1, lam=lam, seed=seed)
        o = predict_als(m, Xtr, ctr, K)
        temp = _best_temp(o, ytr, comps, int(0.7 * len(ytr)))
        ote = (predict_als(m, Xte, (cols[te, p[0]] * K + cols[te, p[1]]) if p else np.zeros(Xte.shape[0], np.int64), K)
               if p else predict_als(m, Xte, np.zeros(Xte.shape[0], np.int64), K))
        return _bits_with_temp(ote, yte, comps, temp, 0)[0], m

    add_bits, _ = bits_with(None)
    err_o = None
    m0 = fit_als(Xtr, zero, Ttr, K, dim=dim, sweeps=1, lam=lam, seed=seed)
    o0 = predict_als(m0, Xtr, zero, K)
    t0v = _best_temp(o0, ytr, comps, int(0.7 * len(ytr)))
    err = _softmax(o0 / t0v) - np.eye(comps)[ytr]
    sc = pair_scores(cols[tr], np.arange(len(ytr)), err, K, comps, pairs, weighted=True) \
        if False else None
    #: the criterion needs (field_i, field_j) cells -- build them directly
    def score_pair(p):
        cell = cols[tr, p[0]] * K + cols[tr, p[1]]
        R_ = np.zeros((K * K, comps))
        np.add.at(R_, cell, err)
        cnt = np.bincount(cell, minlength=K * K).astype(float)
        R_ -= (cnt / len(cell))[:, None] * err.sum(0)[None, :]
        nz = cnt > 0
        return float(((R_ ** 2).sum(1)[nz] / cnt[nz]).sum())
    scv = np.array([score_pair(p) for p in pairs])
    pick = pairs[int(np.argmax(scv))]
    truth_pair = (0, F - 1)
    pick_bits, m_pick = bits_with(pick)
    #: with many candidates the exact all-pairs ranking is too expensive, so a random-pair control is
    #: used instead -- the SAME device as REPORT 342, and the null is again 0.5.
    rank_pick = rank_true = None
    frac_beaten = float("nan")
    if len(pairs) <= 200:
        allbits = np.array([pick_bits if p == pick else bits_with(p)[0] for p in pairs])
        rank_pick = int(np.nonzero(np.argsort(allbits) == pairs.index(pick))[0][0]) + 1
        rank_true = int(np.nonzero(np.argsort(allbits) == pairs.index(truth_pair))[0][0]) + 1
    else:
        rng2 = np.random.default_rng(seed + 99)
        others = [p for p in pairs if p != pick]
        idx = rng2.choice(len(others), size=min(n_rand, len(others)), replace=False)
        rb = np.array([bits_with(others[int(i)])[0] for i in idx])
        frac_beaten = float((rb > pick_bits + 1e-9).mean())
    #: score the EMITTED table of the pick against the known truth when the pick is the truth pair
    tbl_acc = float("nan")
    if pick == truth_pair:
        tbl = als_table(m_pick, K, comps)
        A, B = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
        truth_tbl = ((A + B) % 2).astype(int)
        tbl_acc = float((tbl == truth_tbl).mean())
    Xd_tr, Xd_te = Xtr.toarray(), Xte.toarray()
    fm_b, _ = fm_bits(Xd_tr, ytr, Xd_te, yte, comps, dim=dim, epochs=epochs)
    gb_b, _ = gbdt_bits(Xd_tr, ytr, Xd_te, yte, comps, rounds=rounds, lr=0.3)
    return {"K": K, "F": F, "n": n, "n_cand": len(pairs), "add": add_bits, "pick": pick,
            "pick_bits": pick_bits, "rank_pick": rank_pick, "rank_true": rank_true,
            "frac_beaten": frac_beaten,
            "truth_bits": (float(allbits[pairs.index(truth_pair)]) if rank_pick is not None else
                           float("nan")), "tbl_acc": tbl_acc,
            "fm": fm_b, "gbdt": gb_b, "p1_raw": p1, "H_raw": H_y, "sec": time.time() - t0}


def fields(args) -> int:
    """REPORT 343: the feature-cross setting -- real records, real fields, a real property."""
    print("REPORT 343: FEATURE CROSS on real records (corpus words; fields = positions in a block)")
    print("  target: y = (field0 + field_last) mod 2   -- non-separable, so only the pair can do it")
    print("  the additive read-out sees fields singly; only the pair (first,last) can decide it\n")
    print(f"  {'K':>4} {'F':>3} {'lines':>7} {'cand':>5} {'additive':>9} {'pick':>9} {'pick bits':>10} "
          f"{'rank/frac':>11} {'beaten':>7} {'table acc':>10} {'FM':>8} {'GBDT':>8} {'s':>5}")
    jobs = [(K, F, args.n_lines, args.lam, sd, args.dim, args.sweeps, args.epochs, args.rounds,
             args.n_rand)
            for K in [int(v) for v in args.K_lab.split(",")]
            for F in [int(v) for v in args.Fs.split(",")]
            for sd in range(args.seeds)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_fields_cell, jobs))
    else:
        rows = [_fields_cell(j) for j in jobs]
    for r in sorted(rows, key=lambda r: (r["K"], r.get("n", 0))):
        if r.get("degenerate"):
            extra = ""
            if r.get("const_fields"):
                extra = f", constant fields {r['const_fields']}"
            print(f"  {r['K']:>4} {'-':>3} {r['n']:>7}   DEGENERATE({r['degenerate']}): "
                  f"p1={r['p1']:.5f}, H(y)={r['H']:.5f} bits{extra}")
            continue
        ta = "n/a" if r["tbl_acc"] != r["tbl_acc"] else f"{r['tbl_acc']:.4f}"
        rp = "-" if r["rank_pick"] is None else f"{r['rank_pick']}/{r['n_cand']}"
        fb = "-" if r["frac_beaten"] != r["frac_beaten"] else f"{r['frac_beaten']:.3f}"
        print(f"  {r['K']:>4} {r['F']:>3} {r['n']:>7} {r['n_cand']:>5} {r['add']:>9.4f} "
              f"{str(r['pick']):>9} {r['pick_bits']:>10.4f} {rp:>11} {fb:>7} "
              f"{ta:>10} {r['fm']:>8.4f} {r['gbdt']:>8.4f} {r['sec']:>5.1f}")
    print("\n  --- derived verdict ---")
    rows = [r for r in rows if not r.get("degenerate")]
    if not rows:
        print("    every cell was degenerate -- no reading is possible")
        return 0
    hit = sum(1 for r in rows if tuple(r["pick"]) == (0, r["F"] - 1))
    w_fm = sum(1 for r in rows if r["pick_bits"] < r["fm"] - 0.01)
    w_gb = sum(1 for r in rows if r["pick_bits"] < r["gbdt"] - 0.01)
    print(f"    the criterion picked the TRUE pair (first,last) in {hit}/{len(rows)} cells")
    print(f"    its pair beats the multi-class FM in {w_fm}/{len(rows)} and the GBDT in {w_gb}/{len(rows)}")
    print(f"    mean bits: additive {np.mean([r['add'] for r in rows]):.4f} | pick "
          f"{np.mean([r['pick_bits'] for r in rows]):.4f} | FM {np.mean([r['fm'] for r in rows]):.4f} "
          f"| GBDT {np.mean([r['gbdt'] for r in rows]):.4f}")
    print("    'table acc' scores the EMITTED interaction table against the known g(a,b)=[a==b].")
    return 0


def _tune_cell(job):
    """REPORT 345: give the OPPONENTS a real budget before believing the gap.

    REPORT 344's headline (100-370x at 1035 candidates) was measured with FM dim=8 / 60 epochs and
    GBDT 40 rounds -- defaults.  A gap that survives only at the opponent's default budget is not a
    result.  So this cell runs the same data and OUR method once, then the FM and the GBDT at several
    budgets, and reports the opponent's BEST.
    """
    import real_seq_data as R
    K, F, n_lines, lam, seed, sweeps, fm_grid, gb_rounds = job
    raw = R.load_corpus()
    ids, _ = R.raw_tokens(raw, "word")
    B = F + 2
    nb = len(ids) // B
    recs = ids[:nb * B].reshape(nb, B)
    rng = np.random.default_rng(seed)
    rng.shuffle(recs)
    n = int(min(nb, n_lines))
    recs = recs[:n]
    cut = int(n * 0.7)
    counts = np.bincount(recs[:cut].ravel(), minlength=int(ids.max()) + 1)
    keep = np.lexsort((np.arange(counts.size), -counts))[:K - 1]
    lut = np.zeros(counts.size, dtype=np.int64)
    lut[keep] = np.arange(1, K)
    pos = list(range(F - 1)) + [B - 1]
    cols = lut[recs[:, pos]]
    y = ((cols[:, 0] + cols[:, F - 1]) % 2).astype(np.int64)
    ytr, yte = y[:cut], y[cut:]
    X = scipy.sparse.hstack([scipy.sparse.csr_matrix((np.ones(n), (np.arange(n), cols[:, k])),
                                                     shape=(n, K)) for k in range(F)],
                            format="csr")
    Xtr, Xte = X[:cut], X[cut:]
    pairs = [(i, j) for i in range(F) for j in range(F) if i < j]
    Ttr = np.eye(2)[ytr] - 0.5
    t0 = time.time()
    #: our method: additive -> criterion picks the pair -> add it
    m0 = fit_als(Xtr, np.zeros(cut, np.int64), Ttr, K, dim=8, sweeps=1, lam=lam, seed=seed)
    o0 = predict_als(m0, Xtr, np.zeros(cut, np.int64), K)
    t = _best_temp(o0, ytr, 2, int(0.7 * cut))
    err = _softmax(o0 / t) - np.eye(2)[ytr]
    sc = np.array([_pair_score_cols(cols[:cut], err, p, K) for p in pairs])
    pick = pairs[int(np.argmax(sc))]
    ctr, cte = cols[:cut, pick[0]] * K + cols[:cut, pick[1]], cols[cut:, pick[0]] * K + cols[cut:, pick[1]]
    m1 = fit_als(Xtr, ctr, Ttr, K, dim=8, sweeps=sweeps, lam=lam, seed=seed)
    ot = predict_als(m1, Xte, cte, K)
    ours, _ = _bits_with_temp(ot, yte, 2, _best_temp(predict_als(m1, Xtr, ctr, K), ytr, 2,
                                                     int(0.7 * cut)), 0)
    Xd_tr, Xd_te = Xtr.toarray(), Xte.toarray()
    fm = {}
    for (dim, epochs, lr) in fm_grid:
        b, _ = fm_bits(Xd_tr, ytr, Xd_te, yte, 2, dim=dim, epochs=epochs, lr=lr)
        fm[(dim, epochs, lr)] = b
    gb = {}
    for rr in gb_rounds:
        b, _ = gbdt_bits(Xd_tr, ytr, Xd_te, yte, 2, rounds=rr, lr=0.3)
        gb[rr] = b
    return {"K": K, "F": F, "n_cand": len(pairs), "n": n, "pick": pick, "ours": ours,
            "fm": fm, "gbdt": gb, "sec": time.time() - t0}


def _pair_score_cols(cols_tr, err, p, K):
    cell = cols_tr[:, p[0]] * K + cols_tr[:, p[1]]
    R_ = np.zeros((K * K, err.shape[1]))
    np.add.at(R_, cell, err)
    cnt = np.bincount(cell, minlength=K * K).astype(float)
    R_ -= (cnt / len(cell))[:, None] * err.sum(0)[None, :]
    nz = cnt > 0
    return float(((R_ ** 2).sum(1)[nz] / cnt[nz]).sum())


def tune(args) -> int:
    """REPORT 345: the opponents with a real budget."""
    print("REPORT 345: opponents tuned -- does the 344 gap survive a real FM/GBDT budget?")
    print("  same data, same fields, same target; our method is run once per cell\n")
    fm_grid = [(8, 60, 0.05), (8, 400, 0.05), (16, 120, 0.02), (4, 400, 0.10)]
    gb_rounds = [40, 400]
    jobs = [(int(args.K_lab.split(",")[0]), F, args.n_lines, args.lam, sd, args.sweeps,
             fm_grid, gb_rounds)
            for F in [int(v) for v in args.Fs.split(",")] for sd in range(args.seeds)]
    print(f"  FM grid (dim, epochs, lr) = {fm_grid};  GBDT rounds = {gb_rounds}")
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_tune_cell, jobs))
    else:
        rows = [_tune_cell(j) for j in jobs]
    for r in sorted(rows, key=lambda r: r["F"]):
        best_fm = min(r["fm"].values())
        best_fm_cfg = min(r["fm"], key=r["fm"].get)
        best_gb = min(r["gbdt"].values())
        print(f"\n  F={r['F']} ({r['n_cand']} candidates, n={r['n']})   "
              f"OURS {r['ours']:.4f} bits   pick {r['pick']}")
        for cfg, b in sorted(r["fm"].items(), key=lambda kv: kv[1]):
            print(f"      FM  dim={cfg[0]:<3} epochs={cfg[1]:<4} lr={cfg[2]:<5} -> {b:.4f}")
        for rr, b in sorted(r["gbdt"].items()):
            print(f"      GBDT rounds={rr:<4} -> {b:.4f}")
        print(f"      BEST opponent: FM {best_fm:.4f} (dim {best_fm_cfg[0]}, {best_fm_cfg[1]} ep, "
              f"lr {best_fm_cfg[2]})  vs ours {r['ours']:.4f}  ->  ours is "
              f"{'better by %.4f' % (best_fm - r['ours']) if best_fm > r['ours'] else 'NOT better'}"
              f";  GBDT best {best_gb:.4f}")
    print("\n  the claim 'the gap is real' requires the gap to survive the opponent's BEST budget.")
    return 0


def _gbm_cell(job):
    """REPORT 346: the REAL competitor.  REPORT 344/345 measured 'FM/GBDT' against MY hand-rolled
    implementations; a gradient-boosted tree that I wrote myself is not the thing practitioners use.

    LightGBM is installed (4.7.0), so this runs the same data and the same fields through it with a
    budget sweep, alongside our method.  If a real GBDT closes the gap, the REPORT 344 advantage was an
    artifact of a weak baseline and the honest conclusion is that the component has no accuracy edge.
    """
    import real_seq_data as R
    import lightgbm as lgb
    K, F, n_lines, lam, seed, sweeps, gbm_grid = job
    raw = R.load_corpus()
    ids, _ = R.raw_tokens(raw, "word")
    B = F + 2
    nb = len(ids) // B
    recs = ids[:nb * B].reshape(nb, B)
    rng = np.random.default_rng(seed)
    rng.shuffle(recs)
    n = int(min(nb, n_lines))
    recs = recs[:n]
    cut = int(n * 0.7)
    counts = np.bincount(recs[:cut].ravel(), minlength=int(ids.max()) + 1)
    keep = np.lexsort((np.arange(counts.size), -counts))[:K - 1]
    lut = np.zeros(counts.size, dtype=np.int64)
    lut[keep] = np.arange(1, K)
    pos = list(range(F - 1)) + [B - 1]
    cols = lut[recs[:, pos]]
    y = ((cols[:, 0] + cols[:, F - 1]) % 2).astype(np.int64)
    ytr, yte = y[:cut], y[cut:]
    Xtr, Xte = cols[:cut], cols[cut:]
    pairs = [(i, j) for i in range(F) for j in range(F) if i < j]
    t0 = time.time()
    #: ours: additive -> criterion picks the pair -> low-rank block (same as REPORT 343/344)
    Xs = scipy.sparse.hstack([scipy.sparse.csr_matrix((np.ones(n), (np.arange(n), cols[:, k])),
                                                      shape=(n, K)) for k in range(F)], format="csr")
    Ttr = np.eye(2)[ytr] - 0.5
    m0 = fit_als(Xs[:cut], np.zeros(cut, np.int64), Ttr, K, dim=8, sweeps=1, lam=lam, seed=seed)
    o0 = predict_als(m0, Xs[:cut], np.zeros(cut, np.int64), K)
    t = _best_temp(o0, ytr, 2, int(0.7 * cut))
    err = _softmax(o0 / t) - np.eye(2)[ytr]
    sc = np.array([_pair_score_cols(cols[:cut], err, p, K) for p in pairs])
    pick = pairs[int(np.argmax(sc))]
    ctr, cte = cols[:cut, pick[0]] * K + cols[:cut, pick[1]], cols[cut:, pick[0]] * K + cols[cut:, pick[1]]
    m1 = fit_als(Xs[:cut], ctr, Ttr, K, dim=8, sweeps=sweeps, lam=lam, seed=seed)
    ours, _ = _bits_with_temp(predict_als(m1, Xs[cut:], cte, K), yte, 2,
                              _best_temp(predict_als(m1, Xs[:cut], ctr, K), ytr, 2, int(0.7 * cut)), 0)
    gbm = {}
    for cfg in gbm_grid:
        clf = lgb.LGBMClassifier(**cfg, verbose=-1, n_jobs=1)
        clf.fit(Xtr, ytr, categorical_feature=list(range(F)))
        p = np.clip(clf.predict_proba(Xte)[:, 1], 1e-9, 1 - 1e-9)
        gbm[tuple(sorted(cfg.items()))] = float(-np.log2(np.where(yte == 1, p, 1 - p)).mean())
    return {"K": K, "F": F, "n": n, "n_cand": len(pairs), "pick": pick, "pick_true": int(pick == (0, F - 1)),
            "ours": ours, "gbm": gbm, "sec": time.time() - t0}


def gbm(args) -> int:
    """REPORT 346: LightGBM, a real gradient-boosted tree, on the same fields."""
    print("REPORT 346: the REAL competitor -- LightGBM 4.7.0 on the same fields and target")
    print("  (REPORT 344/345 used a hand-rolled GBDT; this is the thing practitioners actually run)\n")
    grid = [{"num_leaves": 31, "n_estimators": 300, "learning_rate": 0.1},
            {"num_leaves": 127, "n_estimators": 1500, "learning_rate": 0.05},
            {"num_leaves": 255, "n_estimators": 2000, "learning_rate": 0.03},
            {"num_leaves": 1023, "n_estimators": 2000, "learning_rate": 0.03}]
    jobs = [(int(args.K_lab.split(",")[0]), F, args.n_lines, args.lam, sd, args.sweeps, grid)
            for F in [int(v) for v in args.Fs.split(",")] for sd in range(args.seeds)]
    print(f"  LightGBM grid: {[ (c['num_leaves'], c['n_estimators'], c['learning_rate']) for c in grid ]}")
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_gbm_cell, jobs))
    else:
        rows = [_gbm_cell(j) for j in jobs]
    for r in sorted(rows, key=lambda r: r["F"]):
        best = min(r["gbm"].values())
        best_cfg = min(r["gbm"], key=r["gbm"].get)
        print(f"\n  F={r['F']} ({r['n_cand']} candidates, n={r['n']})  "
              f"OURS {r['ours']:.4f}  pick {r['pick']} ({'TRUE' if r['pick_true'] else 'wrong'})")
        for cfg, b in sorted(r["gbm"].items(), key=lambda kv: kv[1]):
            tag = "  <- best" if b == best else ""
            print(f"      LightGBM leaves={cfg[0][1]:<5} trees={cfg[1][1]:<5} lr={cfg[2][1]:<5} -> "
                  f"{b:.4f}{tag}")
        print(f"      BEST LightGBM {best:.4f} vs OURS {r['ours']:.4f}  ->  "
              f"{'ours better by %.4f' % (best - r['ours']) if best > r['ours'] else 'LIGHTGBM WINS'}")
    print("\n  if LightGBM closes the gap, REPORT 344's advantage was a weak-baseline artifact.")
    return 0


def demo(args) -> int:
    rel = _self_check()
    print("REPORT 327: the interaction finder as a component")
    print(f"  self-check: bincount criterion == materialised centred form, max rel err {rel:.2e}\n")
    print(f"  {'K':>4} {'cand':>5} {'samples':>8} {'bar':>7} {'base':>7} {'rank':>5} {'pick':>5} "
          f"{'+pick':>8} {'+oracle':>8} {'wrong':>8} {'pick_z':>7} {'sec':>6}")
    for K in [int(v) for v in args.K.split(",")]:
        for n_cand in [int(v) for v in args.cands.split(",")]:
            for samples in [int(v) for v in args.samples.split(",")]:
                r = run_cell(K, n_cand, samples, seed=0, k=4)
                if r.get("skipped"):
                    print(f"  {K:>4} {n_cand:>5} {samples:>8}   SKIPPED (joint width {r['width']} "
                          f"> closed-form budget)")
                    continue
                print(f"  {r['K']:>4} {r['n_cand']:>5} {r['samples']:>8} {r['bar']:>7.4f} "
                      f"{r['base_bits']:>7.4f} {r['rank']:>5} {r['pick']:>5} {r['pick_bits']:>8.4f} "
                      f"{r['oracle_bits']:>8.4f} {r['wrong_bits']:>8.4f} {r['pick_z']:>7.1f} "
                      f"{r['cell_seconds']:>6.1f}")
    return 0


def envelope(args) -> int:
    print("REPORT 327 ENVELOPE: where does the criterion stop finding the interaction?")
    print("  grid: K x candidates x samples, 3 seeds each; k=4; W=10; read-out closed form")
    print("  a cell is a SUCCESS when the criterion's rank of the true lag is 1 in all seeds\n")
    Ks = [int(v) for v in args.K.split(",")]
    cands = [int(v) for v in args.cands.split(",")]
    samps = [int(v) for v in args.samples.split(",")]
    seeds = list(range(args.seeds))
    print(f"  {'K':>4} {'cand':>6} {'samples':>9} {'cells':>7} {'train':>7} {'cells/tr':>9} "
          f"{'rank-1':>7} {'base':>8} {'+pick':>8} {'bar':>7} {'sec':>6}")
    rows = []
    for K in Ks:
        for c in cands:
            for s in samps:
                rs = [run_cell(K, c, s, seed=sd, k=4, light=True) for sd in seeds]
                if rs[0].get("skipped"):
                    print(f"  {K:>4} {c:>6} {s:>9} {K*K:>7} {'-':>7} {'-':>9} {'SKIPPED':>7} "
                          f"{'-':>8} {'-':>8} {rs[0]['bar']:>7.4f} {'-':>6}   (joint width "
                          f"{rs[0]['width']} > budget)")
                    continue
                ok = sum(1 for r in rs if r["rank"] == 1)
                mb = float(np.mean([r["base_bits"] for r in rs]))
                mp = float(np.mean([r["pick_bits"] for r in rs]))
                sec = float(np.mean([r["cell_seconds"] for r in rs]))
                ntr = int(np.mean([r["ntr"] for r in rs]))
                rows.append({"K": K, "cand": c, "samples": s, "rank1": ok, "n": len(rs),
                             "base": mb, "pick": mp, "bar": rs[0]["bar"], "sec": sec,
                             "cells": K * K, "ntr": ntr})
                print(f"  {K:>4} {c:>6} {s:>9} {K*K:>7} {ntr:>7} {K*K/max(ntr,1):>9.3f} "
                      f"{ok:>3}/{len(rs):<3} {mb:>8.4f} {mp:>8.4f} {rs[0]['bar']:>7.4f} {sec:>6.1f}")
    if rows:
        print(f"\n  rank-1 in every seed: {sum(1 for r in rows if r['rank1'] == r['n'])}/{len(rows)} cells")
        print(f"  the pick reaches the bar ({rows[0]['bar']:.2f} bits at K={rows[0]['K']}) or better "
              f"in {sum(1 for r in rows if r['pick'] < r['bar'] * 0.1)}/{len(rows)} cells")
    print("  cost model: the closed-form read-out costs O((K^2 + K*W)^3) -- the K^2 joint column block,")
    print("  not the sample count, is the wall; REPORT 276 named it and this is where it bites.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("demo", "envelope", "vs-gbdt", "vs-all", "boosting",
                                       "boosting2", "multi", "speed", "als", "alsmulti", "pool",
                                       "emit", "real", "natural", "fields", "tune", "gbm"),
                    default="demo")
    ap.add_argument("--F", type=int, default=7, help="number of fields per record (fields mode)")
    ap.add_argument("--Fs", default="7,20,46",
                    help="field counts to sweep in fields mode (candidates = C(F,2))")
    ap.add_argument("--n_lines", type=int, default=20000, help="records to use (fields mode)")
    ap.add_argument("--n_rand", type=int, default=20, help="random-pair draws per cell (natural mode)")
    ap.add_argument("--real_stride", type=int, default=40, help="position stride on the real stream")
    ap.add_argument("--noise", default="0.0", help="probability the real-stream label is noise")
    ap.add_argument("--rare", default="0,1,2,4",
                    help="keep the signal only where x_{t-1} is one of the R rarest symbols")
    ap.add_argument("--Ws", default="10,30", help="window lengths for the real-stream pool sweep")
    ap.add_argument("--sweeps", type=int, default=6, help="ALS sweeps (als mode)")
    ap.add_argument("--steps", type=int, default=400, help="Adam steps for the low-rank twin (als mode)")
    ap.add_argument("--fm", action="store_true",
                    help="alsmulti: also run an INDEPENDENT low-rank FM opponent on the same heads")
    ap.add_argument("--M", default="1,2,4,8", help="number of true interactions (boosting mode)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--lam", type=float, default=1e-2, help="ridge lambda for the closed-form read-out")
    ap.add_argument("--W", type=int, default=10, help="window length (multi mode)")
    ap.add_argument("--pairs", type=int, default=0,
                    help="0 = true pairs may share lags (default), 1 = disjoint lags")
    ap.add_argument("--no-opponents", action="store_true",
                    help="skip the multi-head FM/GBDT arms (they dominate the runtime)")
    ap.add_argument("--K_lab", default="8",
                    help="label alphabet size (multi: first value; als: comma list)")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--dim", type=int, default=8, help="FM embedding dimension")
    ap.add_argument("--epochs", type=int, default=300, help="FM epochs per class")
    ap.add_argument("--K", default="8,32,128")
    ap.add_argument("--cands", default="20,100")
    ap.add_argument("--samples", default="20000,80000")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    if args.mode == "demo":
        return demo(args)
    if args.mode == "envelope":
        return envelope(args)
    if args.mode == "vs-gbdt":
        return vs_gbdt(args)
    if args.mode == "vs-all":
        return vs_all(args)
    if args.mode == "speed":
        return speed(args)
    if args.mode == "multi":
        return multi(args)
    if args.mode == "als":
        return als(args)
    if args.mode == "alsmulti":
        return alsmulti(args)
    if args.mode == "pool":
        return pool(args)
    if args.mode == "emit":
        return emit(args)
    if args.mode == "real":
        return real(args)
    if args.mode == "natural":
        return natural(args)
    if args.mode == "fields":
        return fields(args)
    if args.mode == "tune":
        return tune(args)
    if args.mode == "gbm":
        return gbm(args)
    return boosting(args) if args.mode == "boosting" else boosting2(args)


if __name__ == "__main__":
    raise SystemExit(main())
