"""REPORT 193 step 0/1: the REAL-sequence data pipeline and the n-gram baseline family.

WHAT THIS FILE IS FOR.  Before any substrate is built (`real_seq_substrate.py`), two things must be
true or the whole round is unreadable:

    S0  the harness does not leak.  If the stream is shuffled -- which destroys every temporal
        dependency and leaves only the symbol frequencies -- EVERY model must fall back to the
        unigram entropy.  A model that scores better than unigram on a shuffled stream is reading
        something it was not supposed to see.
    S1  the discretisation did not destroy the local structure (Expert 1's P0).  If the best
        n-gram cannot beat unigram, no substrate can be asked a meaningful question on this stream.

AND it produces the numbers the substrate will be compared against, on a position set computed
once, so that both sides are scored on the same samples.

REPORT 197 CORRECTION.  The sentence above was false as first written: `eval_positions` is called
by THIS file's own `run_config` (all 5353 test windows, offsets 3..23, 112 413 positions) and was
NEVER called by the substrate script, which builds its own set through `dense_positions` (600
windows, offsets 5..23, 11 400 positions).  Inside the substrate script the arms and their n-gram
reference do share `pos_all`, so REPORT 194's S2 comparison is sound; but the S1 verdict and the
0.20-bit bar that authorises the round were measured on a different, larger sample.  Stated here
because the docstring claimed otherwise.

THE TWO THINGS THE EXPERTS CHANGED, and which are therefore done here and nowhere else:

    E2-a  the vocabulary is counted on the TRAINING blocks only and then applied to the test
          blocks.  Counting it on the whole file would put the test block's frequency statistics
          into every arm's input representation, raising the substrate and the baselines by
          different amounts and poisoning the one number the round exists to measure.
    E2-b  the metric is BPS (bits per symbol = cross-entropy in bits), not accuracy.  Accuracy
          thresholds written in percentage points are not comparable across K: five points is a
          ~15% relative gain at K=16 and a ~40% one at K=64.

THE METRIC.  For a model that assigns probability p to the true symbol at each evaluation
position, BPS = mean(-log2 p).  Unigram BPS is the natural zero point: a model that knows only the
symbol frequencies cannot go below it on held-out text.  Accuracy is reported as a secondary
number, split by whether the TRUE symbol is <unk>, because Top-K truncation makes <unk> a very
frequent class and a model can look good by predicting it (Expert 1's trap A).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

#: REPORT 193.3: the corpus is REPORT.md truncated at the byte length it had when section 193 was
#: written.  The truncation is not cosmetic -- the last 12.5% of the file is the TEST block, and if
#: it were allowed to grow it would contain the description of this very experiment.
#: REPORT 244: the corpus used to be REPORT.md[:CORPUS_BYTES], i.e. the lab notebook itself.
#: Editing the notebook therefore edited the experiment -- which happened, via a line-ending
#: change.  The bytes are now a standalone file, verified against the same pinned sha256.
CORPUS = ROOT / "corpus_193.txt"
CORPUS_BYTES = 1_027_937
CORPUS_SHA256 = "616554b3c615dd91e45cc07a04ff8a65b1458d82b579e102e1748df685196c7a"

UNK = 0                      #: the truncation class; always id 0
BLOCKS = 8                   #: contiguous, equal-length blocks
TRAIN_BLOCKS = 7             #: blocks 0..6 train, block 7 test
#: Smoothing grid.  It reaches 10.0 because the first run of the S0 control (a fully shuffled
#: stream) measured the 4-gram at 1.77 bits WORSE than unigram at K=64: with alpha capped at 1.0 a
#: context seen once is still predicted with probability ~0.5, which is confident nonsense.  A
#: baseline family that cannot fall back to unigram when the data has no structure is not a
#: baseline, it is a handicap the substrate would be measured against unfairly -- in the substrate's
#: FAVOUR, which is the direction that produces false positives.
ALPHAS = (0.01, 0.05, 0.2, 1.0, 10.0)

_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


# ---------------------------------------------------------------------------------------------
# corpus and discretisation
# ---------------------------------------------------------------------------------------------

def load_corpus() -> bytes:
    raw = CORPUS.read_bytes()[:CORPUS_BYTES]
    got = hashlib.sha256(raw).hexdigest()
    if got != CORPUS_SHA256:
        raise SystemExit(
            f"corpus snapshot changed: sha256 {got} != {CORPUS_SHA256}.  The frozen spec (REPORT\n"
            f"193.3) pins the first {CORPUS_BYTES} bytes of {CORPUS.name}; if the file was edited\n"
            f"before that offset the experiment is no longer the frozen one.  Aborting rather than\n"
            f"running an experiment whose corpus is not the one the predictions were written for.")
    return raw


def raw_tokens(raw: bytes, gran: str) -> tuple[np.ndarray, list[str]]:
    """Symbol ids before the vocabulary is applied, plus the human-readable name of each id."""
    if gran == "byte":
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        names = [f"byte:{b:02x}" for b in range(256)]
        return arr, names
    if gran == "word":
        text = raw.decode("utf-8", errors="replace")
        toks = _WORD_RE.findall(text)
        index: dict[str, int] = {}
        names = []
        out = np.empty(len(toks), dtype=np.int64)
        for i, tk in enumerate(toks):
            j = index.get(tk)
            if j is None:
                j = len(names)
                index[tk] = j
                names.append(tk)
            out[i] = j
        return out, names
    raise SystemExit(f"unknown granularity {gran!r}")


def block_bounds(n: int) -> list[tuple[int, int]]:
    return [(b * n // BLOCKS, (b + 1) * n // BLOCKS) for b in range(BLOCKS)]


def build_vocab(raw_ids: np.ndarray, train_end: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top ``k - 1`` raw ids by frequency on the TRAIN side only.  Returns (keep_ids, map_lut).

    ``keep_ids`` is ordered by descending training frequency (ties broken by raw id, so the result
    is deterministic); ``map_lut`` maps a raw id to ``0`` (<unk>) or ``1..k-1``.
    """
    counts = np.bincount(raw_ids[:train_end], minlength=int(raw_ids.max()) + 1)
    order = np.lexsort((np.arange(counts.size), -counts))
    keep = order[:k - 1]
    lut = np.zeros(counts.size, dtype=np.int64)
    lut[keep] = np.arange(1, len(keep) + 1)
    return keep, lut


def discretise(raw_ids: np.ndarray, raw_names: list[str], train_end: int, k: int):
    keep, lut = build_vocab(raw_ids, train_end, k)
    tok = lut[raw_ids]
    names = ["<unk>"] + [raw_names[int(i)] for i in keep]
    return tok, names


# ---------------------------------------------------------------------------------------------
# evaluation positions
# ---------------------------------------------------------------------------------------------

def eval_positions(n: int, train_end: int, lag: int, ctx: int, steps: int) -> np.ndarray:
    """The ONE position set every model is scored on (REPORT 193.6).

    Test block only, and a position qualifies when

      * ``t + lag`` is still inside the test block  -- otherwise the target is train-side text;
      * ``t`` has at least ``ctx`` symbols of test-side context before it, so the n-gram models
        have a full context at every position;
      * ``t`` is at least ``ctx`` steps into its 24-step substrate window.  Without this the first
        positions of every window would be scored from a substrate state that had seen fewer
        symbols than the n-gram context, and the comparison would flatter the baseline for a reason
        that has nothing to do with the substrate.
    """
    start, end = train_end, n
    #: only positions the substrate can actually visit: it consumes whole `steps`-long windows
    #: from `start`, so the trailing partial window has no substrate state at all.  Scoring the
    #: baselines on positions the substrate never sees would compare two different sample sets.
    covered_end = start + steps * ((end - start) // steps)
    cand = []
    for t in range(start, min(end, covered_end)):
        if t + lag >= end:
            break
        if t - start < ctx:
            continue
        if (t - start) % steps < ctx:
            continue
        cand.append(t)
    return np.asarray(cand, dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# n-gram family: vectorised interpolated backoff
# ---------------------------------------------------------------------------------------------

class Table:
    """Pair counts for one order, stored as sorted keys so a lookup is a ``searchsorted``."""

    def __init__(self, codes: np.ndarray, targets: np.ndarray, v: int) -> None:
        pair = codes * v + targets
        self.pair_key, self.pair_cnt = np.unique(pair, return_counts=True)
        self.code_key, self.code_tot = np.unique(codes, return_counts=True)
        #: how many (context, target) observations this order saw at all -- reported so a reader
        #: can tell an order that is informative from one that is empty.
        self.n_obs = int(codes.size)

    def counts(self, codes: np.ndarray, v: int) -> tuple[np.ndarray, np.ndarray]:
        if self.code_key.size == 0:
            # REPORT 197: an order with no observations at all (possible for a short fit range at
            # a high order) used to raise IndexError from the clip-then-index below.  An empty
            # table means "this order knows nothing", i.e. total count 0 for every context, which
            # makes the interpolation fall back to the lower order -- the correct answer.
            return np.zeros((codes.size, v)), np.zeros(codes.size)
        idx = np.searchsorted(self.code_key, codes)
        idx = np.clip(idx, 0, max(0, self.code_key.size - 1))
        tot = np.where(self.code_key[idx] == codes, self.code_tot[idx], 0).astype(np.float64)
        q = codes[:, None] * v + np.arange(v, dtype=np.int64)[None, :]
        j = np.searchsorted(self.pair_key, q)
        j = np.clip(j, 0, max(0, self.pair_key.size - 1))
        cnt = np.where(self.pair_key[j] == q, self.pair_cnt[j], 0).astype(np.float64)
        return cnt, tot


def context_codes(tok: np.ndarray, pos: np.ndarray, order: int, v: int) -> np.ndarray:
    """Base-``v`` code of ``(x(t), x(t-1), ..., x(t-order+2))``.  Order 1 has an empty context."""
    code = np.zeros(pos.size, dtype=np.int64)
    for i in range(order - 1):
        code = code * v + tok[pos - i]
    return code


class Ngram:
    """Interpolated backoff up to ``order``, fitted on one contiguous range of the token stream.

    ``p_n(w) = (count_n(ctx_n, w) + alpha * p_{n-1}(w)) / (total_n(ctx_n) + alpha)``

    with ``p_0`` uniform, which is the standard formulation in which a missing context falls
    exactly back to the lower order.  The whole thing is evaluated as a ``(positions, V)`` matrix,
    so BPS, accuracy and the <unk> split all come off the same probabilities.
    """

    def __init__(self, tok: np.ndarray, fit_start: int, fit_end: int, v: int,
                 order: int, lag: int) -> None:
        self.v, self.order = v, order
        t = np.arange(fit_start, fit_end, dtype=np.int64)
        t = t[t + lag < fit_end]
        if order > 1:
            t = t[(t - order + 1) >= fit_start]
        self.tables = [None]
        for n in range(1, order + 1):
            codes = context_codes(tok, t, n, v)
            self.tables.append(Table(codes, tok[t + lag], v))

    def probs(self, tok: np.ndarray, pos: np.ndarray, alpha: float,
              upto: int | None = None) -> np.ndarray:
        """The interpolated distribution at each position, using orders ``1..upto``.

        ``upto`` exists so that one fitted table set serves every order in the family: building a
        separate model per order re-derives the lower-order counts once per order (ten table
        builds instead of four on a 900k-token stream), which is pure waste and was measured as
        such before this parameter existed.
        """
        v = self.v
        last = self.order if upto is None else min(int(upto), self.order)
        p = np.full((pos.size, v), 1.0 / v)
        for n in range(1, last + 1):
            cnt, tot = self.tables[n].counts(context_codes(tok, pos, n, v), v)
            p = (cnt + alpha * p) / (tot + alpha)[:, None]
        return p


def score(p: np.ndarray, targets: np.ndarray) -> dict:
    """BPS / accuracy, overall and restricted to positions whose TRUE symbol is not <unk>."""
    hit = -np.log2(np.clip(p[np.arange(targets.size), targets], 1e-12, None))
    pred = p.argmax(axis=1)
    real = targets != UNK
    out = {
        "bps": float(hit.mean()),
        "acc": float((pred == targets).mean()),
        "n": int(targets.size),
        "unk_true_rate": float((~real).mean()),
        "unk_pred_rate": float((pred == UNK).mean()),
    }
    if bool(real.any()):
        out["bps_real"] = float(hit[real].mean())
        out["acc_real"] = float((pred[real] == targets[real]).mean())
        out["n_real"] = int(real.sum())
    else:
        out["bps_real"] = float("nan")
        out["acc_real"] = float("nan")
        out["n_real"] = 0
    return out


# ---------------------------------------------------------------------------------------------
# one full configuration
# ---------------------------------------------------------------------------------------------

def prepare(gran: str, k: int) -> dict:
    """Corpus -> ids, vocabulary, block split, and everything the substrate script needs."""
    raw = load_corpus()
    raw_ids, raw_names = raw_tokens(raw, gran)
    n = int(raw_ids.size)
    bounds = block_bounds(n)
    train_end = bounds[TRAIN_BLOCKS][0]
    tok, names = discretise(raw_ids, raw_names, train_end, k)

    def hist(a: np.ndarray) -> np.ndarray:
        return np.bincount(a, minlength=k).astype(np.float64) / max(1, a.size)

    pt, pe = hist(tok[:train_end]), hist(tok[train_end:])
    kl = float(np.sum(np.where(pe > 0, pe * np.log2(np.clip(pe, 1e-12, None)
                                                     / np.clip(pt, 1e-12, None)), 0.0)))
    return {
        "gran": gran, "k": k, "tok": tok, "names": names, "n": n,
        "train_end": int(train_end), "test_start": int(train_end),
        "test_unk_rate": float((tok[train_end:] == UNK).mean()),
        "train_unk_rate": float((tok[:train_end] == UNK).mean()),
        "top1_share": float(pt.max()), "freq_kl": kl,
    }


def fit_models(d: dict, lag: int, max_order: int, tok: np.ndarray | None = None):
    """Fit the whole family once: alpha per order (chosen on a TRAIN-side slice), one test model.

    Returns ``(model, alphas)`` where ``model`` is fitted on the full training side and
    ``model.probs(..., upto=n)`` is order ``n``'s distribution.  Alpha is never chosen on the test
    block; the validation slice is the last eighth of the training side.
    """
    tok = d["tok"] if tok is None else tok
    v, te = d["k"], d["train_end"]
    vstart = te - max(256, te // 8)
    ctx = max(max_order - 1, 1)
    alphas = {n: ALPHAS[0] for n in range(1, max_order + 1)}
    vpos = np.arange(vstart + ctx, te, dtype=np.int64)
    vpos = vpos[vpos + lag < te]
    if vpos.size:
        val = Ngram(tok, 0, vstart, v, max_order, lag)
        vtgt = tok[vpos + lag]
        for n in range(1, max_order + 1):
            best, best_bps = ALPHAS[0], float("inf")
            for a in ALPHAS:
                b = score(val.probs(tok, vpos, a, upto=n), vtgt)["bps"]
                if b < best_bps:
                    best, best_bps = a, b
            alphas[n] = best
    return Ngram(tok, 0, te, v, max_order, lag), alphas


def run_config(gran: str, k: int, lag: int, max_order: int, d: dict | None = None,
               pos: np.ndarray | None = None) -> dict:
    d = prepare(gran, k) if d is None else d
    tok, v, te, n = d["tok"], d["k"], d["train_end"], d["n"]
    ctx = max(max_order - 1, 1)
    if pos is None:
        pos = eval_positions(n, te, lag, ctx, steps=24)
    tgt = tok[pos + lag]

    model, alphas = fit_models(d, lag, max_order)
    rows = {}
    for order in range(1, max_order + 1):
        p = model.probs(tok, pos, alphas[order], upto=order)
        rows[order] = score(p, tgt) | {"alpha": alphas[order]}
    return {"gran": gran, "k": k, "lag": lag, "n_pos": int(pos.size),
            "pos": pos, "rows": rows, "meta": {
                "test_unk_rate": d["test_unk_rate"], "train_unk_rate": d["train_unk_rate"],
                "top1_share": d["top1_share"], "freq_kl": d["freq_kl"], "n": n,
                "train_end": te, "vocab": [d["names"][i] for i in range(min(8, v))],
            }}


def shuffled_control(gran: str, k: int, lag: int, max_order: int, seed: int = 0) -> dict:
    """S0: destroy every temporal dependency (shuffle the WHOLE stream) and re-measure.

    Everything else is held fixed -- same vocabulary size, same block split, same position set --
    so a model whose BPS does not collapse to the unigram value on this stream is reading
    information that is not in the data, and the harness is broken.
    """
    d = prepare(gran, k)
    rng = np.random.default_rng(seed)
    d = dict(d)
    d["tok"] = d["tok"][rng.permutation(d["tok"].size)]
    return run_config(gran, k, lag, max_order, d=d)


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------

def print_config(res: dict, max_order: int) -> None:
    m = res["meta"]
    print(f"  {res['gran']:<5} K={res['k']:<3} lag={res['lag']:<3} "
          f"positions={res['n_pos']:<7} test<unk>={m['test_unk_rate']:.3f} "
          f"train<unk>={m['train_unk_rate']:.3f} top1={m['top1_share']:.3f} KL={m['freq_kl']:.4f}")
    base = res["rows"][1]["bps"]
    cells = []
    for order in range(1, max_order + 1):
        r = res["rows"][order]
        cells.append(f"{order}:{r['bps']:.3f}({base - r['bps']:+.3f})")
    print(f"        BPS  " + "  ".join(cells))
    best = min(range(2, max_order + 1), key=lambda o: res["rows"][o]["bps"],
               default=1)
    rb, ru = res["rows"][best], res["rows"][1]
    print(f"        best order {best}: BPS {rb['bps']:.4f} vs unigram {ru['bps']:.4f}  "
          f"delta={ru['bps'] - rb['bps']:+.4f} bits   acc {rb['acc']:.4f} "
          f"(unigram acc {ru['acc']:.4f}, chance {1.0 / res['k']:.4f})")
    print(f"        <unk>: true {rb['unk_true_rate']:.3f} of positions, "
          f"predicted {rb['unk_pred_rate']:.3f};  BPS|true!=unk {rb['bps_real']:.4f} "
          f"(unigram {ru['bps_real']:.4f})   acc|true!=unk {rb['acc_real']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gran", default="byte,word")
    ap.add_argument("--k", default="16,64")
    ap.add_argument("--lag", default="1,2,4,8,16")
    ap.add_argument("--orders", type=int, default=4)
    ap.add_argument("--sanity", action="store_true", help="also run S0 (shuffled-stream control)")
    args = ap.parse_args()

    grans = [g for g in args.gran.split(",") if g]
    ks = [int(x) for x in args.k.split(",")]
    lags = [int(x) for x in args.lag.split(",")]

    print("REPORT 193 step 0/1: real-sequence pipeline + n-gram baseline family")
    print(f"  corpus {CORPUS.name}[:{CORPUS_BYTES}]  sha256 {CORPUS_SHA256[:16]}...")
    print(f"  blocks {BLOCKS} ({TRAIN_BLOCKS} train / {BLOCKS - TRAIN_BLOCKS} test), "
          f"vocabulary counted on the train blocks ONLY, metric BPS (bits/symbol)")
    print()

    if args.sanity:
        print("=" * 100)
        print("S0  SHUFFLED-STREAM CONTROL -- no model may be BETTER than unigram on a stream with")
        print("    no temporal structure.  (Being WORSE is the correct behaviour, not a failure:")
        print("    a model that conditions on contexts which carry no information is confidently")
        print("    wrong, and the criterion is one-sided for that reason -- see REPORT 193.7 S0,")
        print("    which was corrected here after the first run, before any substrate was built.)")
        print("=" * 100)
        worst = 0.0
        for gran in grans:
            for k in ks:
                for lag in lags[:1] + lags[-1:]:
                    r = shuffled_control(gran, k, lag, args.orders)
                    u = r["rows"][1]["bps"]
                    dev = min(r["rows"][o]["bps"] - u
                              for o in range(1, args.orders + 1))
                    worst = min(worst, dev)
                    print(f"  {gran:<5} K={k:<3} lag={lag:<3} unigram {u:.4f}  "
                          f"best (most negative) deviation over orders 1..{args.orders} = "
                          f"{dev:+.4f}")
        print()
        print(f"  -> S0 {'PASS' if worst > -0.02 else 'FAIL'}  (best improvement over unigram "
              f"{worst:+.4f} bits; must be > -0.02)")
        print()

    print("=" * 100)
    print("S1  N-GRAM FAMILY -- BPS, and the delta against unigram in bits")
    print("=" * 100)
    verdicts = []
    for gran in grans:
        for k in ks:
            dd = prepare(gran, k)
            for lag in lags:
                if lag >= 24:
                    continue
                res = run_config(gran, k, lag, args.orders, d=dd)
                print_config(res, args.orders)
                best = min(range(2, args.orders + 1),
                           key=lambda o: res["rows"][o]["bps"], default=1)
                delta = res["rows"][1]["bps"] - res["rows"][best]["bps"]
                verdicts.append((gran, k, lag, delta))
            print()
    print("=" * 100)
    worst = min(v[3] for v in verdicts) if verdicts else 0.0
    print(f"  S1 (P0: best n-gram beats unigram by >= 0.20 bits) -> "
          f"{'PASS' if worst >= 0.20 else 'FAIL'}   worst cell {worst:+.3f} bits")
    print(f"  these deltas are the bar the substrate must clear in REPORT 193.7 S2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
