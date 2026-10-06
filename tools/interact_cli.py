"""interact_cli.py -- "is there a PAIRWISE interaction here, or am I chasing noise?"

WHO THIS IS FOR.  Not someone who wants the best predictive model (use LightGBM).  It is for someone
who has a table, a target, and a nagging suspicion that "these two columns together" matter -- and who
needs to know, quickly and honestly, whether that suspicion survives being looked at.

WHY THAT IS WORTH A TOOL.  Most pairwise-interaction hypotheses are false.  Measured on two real
datasets (REPORT 347): the single best field pair buys 0.002-0.012 bits over a plain additive logistic
regression, while a tuned GBDT buys 0.12 bits by combining MANY weak effects.  An analyst who "adds the
cross and watches the AUC" therefore reads noise, and with N candidate pairs the noise gets worse by
selection.  This tool answers the question with a measured null instead:

    * it scores EVERY candidate pair against the additive model's residual (no tuning, closed form);
    * it ranks them by real held-out bits, so the answer is not the criterion's own opinion;
    * it reports the criterion's permutation null over N candidates (how good the best of N junk pairs
      looks), so "the best pair" can be compared with "the best of N coin flips";
    * it prints the cell table for whatever it picks, so a human can look at the actual cells.

    python tools/diagnostics/interact_cli.py --csv runs/adult.csv --target __target__
    python tools/diagnostics/interact_cli.py --csv runs/credit_g.csv --target __target__ --min_cell 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def bits_of(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.log2(np.where(y == 1, p, 1 - p)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--fields", default="", help="comma list; default = all non-numeric columns")
    ap.add_argument("--min_cell", type=int, default=20)
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null_draws", type=int, default=200,
                    help="shuffled candidates used for the criterion's null")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(args.csv)
    y = df[args.target].to_numpy().astype(int)
    if args.fields:
        cat = args.fields.split(",")
    else:
        cat = [c for c in df.columns if c != args.target
               and not pd.api.types.is_numeric_dtype(df[c])]
    codes_cols = []
    for c in cat:
        cc = pd.Categorical(df[c]).codes.astype(np.int64)
        if (cc < 0).any():
            cc = np.where(cc < 0, cc.max() + 1, cc)
        codes_cols.append(cc)
    codes = np.stack(codes_cols, axis=1)
    K = [int(codes[:, k].max()) + 1 for k in range(len(cat))]
    n = len(df)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    cut = int(n * 0.7)
    tr, te = idx[:cut], idx[cut:]
    print(f"interact_cli: {n} rows, {len(cat)} fields, target rate {y.mean():.3f} "
          f"(train {len(tr)} / held-out {len(te)})")
    print(f"  fields: {', '.join(f'{c}(K={K[k]})' for k, c in enumerate(cat))}")

    Xtr = np.hstack([np.eye(K[k])[codes[tr, k]] for k in range(len(cat))])
    Xte = np.hstack([np.eye(K[k])[codes[te, k]] for k in range(len(cat))])
    ytr, yte = y[tr], y[te]
    lr = LogisticRegression(max_iter=args.max_iter, C=1.0).fit(Xtr, ytr)
    add_bits = bits_of(lr.predict_proba(Xte)[:, 1], yte)
    pairs = [(i, j) for i in range(len(cat)) for j in range(i + 1, len(cat))]
    print(f"\n  additive model (logistic regression, {Xtr.shape[1]} one-hot columns): "
          f"{add_bits:.4f} bits held-out")
    print(f"  candidate pairs: {len(pairs)}")

    err = lr.predict_proba(Xtr)[:, 1] - ytr

    def cell_of(p, rows):
        return codes[rows, p[0]] * K[p[1]] + codes[rows, p[1]]

    def score(p):
        cell = cell_of(p, tr)
        m = K[p[0]] * K[p[1]]
        R = np.zeros(m)
        np.add.at(R, cell, err)
        cnt = np.bincount(cell, minlength=m).astype(float)
        R = R - (cnt / len(cell)) * err.sum()
        nz = cnt > 0
        return float(((R ** 2)[nz] / cnt[nz]).sum())

    sc = np.array([score(p) for p in pairs])
    pick = pairs[int(np.argmax(sc))]
    #: the criterion's permutation null: the best score obtainable from N candidates with no signal
    null_max = []
    for _ in range(20):
        e = err[rng.permutation(len(err))]
        vals = []
        for p in pairs:
            cell = cell_of(p, tr)
            m = K[p[0]] * K[p[1]]
            R = np.zeros(m)
            np.add.at(R, cell, e)
            cnt = np.bincount(cell, minlength=m).astype(float)
            R = R - (cnt / len(cell)) * e.sum()
            nz = cnt > 0
            vals.append(float(((R ** 2)[nz] / cnt[nz]).sum()))
        null_max.append(max(vals))
    thr = float(np.quantile(null_max, 0.95))

    def bits_with(p):
        cell_tr, cell_te = cell_of(p, tr), cell_of(p, te)
        cnt = np.bincount(cell_tr, minlength=K[p[0]] * K[p[1]])
        keep = np.flatnonzero(cnt >= args.min_cell)
        if len(keep) < 2:
            return float("nan")
        lut = {int(c): i for i, c in enumerate(keep)}
        Btr = np.zeros((len(tr), len(keep)))
        Bte = np.zeros((len(te), len(keep)))
        for r, c in enumerate(cell_tr):
            j = lut.get(int(c))
            if j is not None:
                Btr[r, j] = 1.0
        for r, c in enumerate(cell_te):
            j = lut.get(int(c))
            if j is not None:
                Bte[r, j] = 1.0
        m = LogisticRegression(max_iter=args.max_iter, C=1.0).fit(np.hstack([Xtr, Btr]), ytr)
        return bits_of(m.predict_proba(np.hstack([Xte, Bte]))[:, 1], yte)

    allbits = np.array([bits_with(p) for p in pairs])
    order = np.argsort(allbits)
    rank_pick = int(np.nonzero(order == pairs.index(pick))[0][0]) + 1
    best_pair = pairs[int(order[0])]
    ours = float(allbits[pairs.index(pick)])
    draws = rng.choice(len(pairs), size=min(25, len(pairs)), replace=False)
    rand_bits = allbits[draws]
    print(f"\n  criterion's pick      {cat[pick[0]]} x {cat[pick[1]]}   ->  {ours:.4f} bits "
          f"(gain {add_bits - ours:+.4f}; rank {rank_pick} of {len(pairs)} by held-out bits)")
    print(f"  best of all pairs     {cat[best_pair[0]]} x {cat[best_pair[1]]}   ->  "
          f"{allbits.min():.4f} bits (gain {add_bits - allbits.min():+.4f})")
    print(f"  25 random pairs: mean {rand_bits.mean():.4f}, best {rand_bits.min():.4f}")
    print(f"  criterion's null over {len(pairs)} candidates: 95th pct = {thr:.1f}, "
          f"pick's score = {sc.max():.1f}  ->  {'CLEARS the null' if sc.max() > thr else 'INSIDE the null'}")

    print("\n  --- VERDICT ---")
    gain = add_bits - allbits.min()
    if allbits.min() >= add_bits - 0.005:
        print("    NO pair helps: the best of all candidate pairs is no better than the additive model.")
        print("    => the interaction hypothesis is dead; stop here (this is the common answer).")
    elif sc.max() <= thr:
        print("    The criterion's best candidate is INSIDE its own permutation null, i.e. the best of")
        print(f"    {len(pairs)} candidates looks like the best of {len(pairs)} coin flips.")
        print("    => no evidence; a raw 'add the cross and watch the AUC' reading here is noise.")
    else:
        print(f"    The criterion's pick CLEARS the null (score {sc.max():.1f} vs {thr:.1f}) and the best")
        print(f"    pair is worth {gain:+.4f} bits.  Look at the table below before using it.")

    #: the artifact a human actually reads
    p = best_pair
    cell_tr = cell_of(p, tr)
    cell_te = cell_of(p, te)
    base = ytr.mean()
    rows_ = []
    for c in np.unique(cell_tr):
        m = cell_tr == c
        if m.sum() < args.min_cell:
            continue
        a, b = int(c) // K[p[1]], int(c) % K[p[1]]
        mte = cell_te == c
        rows_.append((float(ytr[m].mean()), int(m.sum()),
                      float(yte[mte].mean()) if mte.sum() else float("nan"), int(mte.sum()),
                      str(df[cat[p[0]]].astype("category").cat.categories[a]) if False else a,
                      str(df[cat[p[1]]].astype("category").cat.categories[b]) if False else b))
    rows_.sort(reverse=True)
    print(f"\n  --- cell table for {cat[p[0]]} x {cat[p[1]]} (target rate by cell; train base {base:.3f}) ---")
    print(f"  {'train':>7} {'n_tr':>6}  {'held-out':>9} {'n_te':>6}   cell")
    for rb, ntr_, rte, nte_, a, b in rows_[:8]:
        print(f"  {rb:>7.3f} {ntr_:>6}  {rte:>9.3f} {nte_:>6}   {cat[p[0]]}={a} & {cat[p[1]]}={b}")
    print("\n  caution: a high cell rate can come from the MARGINALS (either field alone) rather than from")
    print("  the interaction -- that is why the bits gain, not the table, is the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
