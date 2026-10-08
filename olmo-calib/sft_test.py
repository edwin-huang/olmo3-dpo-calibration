"""Does SFT copy a more knowledgeable teacher's confidence? (teacher-imitation test)

For a pair of checkpoints A -> B on the same questions:
  1. Fit the best uniform sharpening of A, Q = normalize(P_A ** s), by minimizing mean KL(P_B || Q).
     Q is the "nothing question-specific happened" baseline: B = A with one temperature change.
  2. Group questions by how much A knew (A's probability on the correct answer) and compare B's
     confidence with Q's. Teacher imitation predicts B > Q mainly where A did NOT know the answer;
     a pure temperature change predicts B ≈ Q in every group.
  3. Per subject: does the residual (B conf - Q conf) grow as A's subject accuracy falls?

Usage: python sft_test.py results_robust/mmlu5_full [results_robust/arc5_full]
"""
import json
import sys
from collections import defaultdict

import numpy as np

PAIRS = [
    ("Olmo-3-1025-7B", "Olmo-3-7B-Think-SFT", "SFT: base -> Think SFT"),
    ("Olmo-3-7B-Think-SFT", "Olmo-3-7B-Instruct-SFT", "SFT: Think SFT -> Instruct SFT"),
    ("Olmo-3-7B-Instruct-SFT", "Olmo-3-7B-Instruct-DPO", "DPO: Instruct"),
    ("Olmo-3-7B-Think-SFT", "Olmo-3-7B-Think-DPO", "DPO: Think"),
    ("Olmo-3-1025-7B", "Olmo-3-7B-RL-Zero-Mix", "RL: base -> RL-Zero (control)"),
]
BINS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001]


def load(d, name):
    rows = [json.loads(l) for l in open(f"{d}/{name}.jsonl")]
    p = np.array([r["probs"] for r in rows], dtype=np.float64)
    return p, np.array([r["answer"] for r in rows]), [r["subject"] for r in rows]


def sharpen(p, s):
    q = np.power(np.clip(p, 1e-12, 1), s)
    return q / q.sum(1, keepdims=True)


def fit_s(pa, pb):
    grid = np.exp(np.linspace(np.log(0.3), np.log(5), 400))
    lb = np.log(np.clip(pb, 1e-12, 1))
    kl = [np.mean(np.sum(pb * (lb - np.log(sharpen(pa, s))), 1)) for s in grid]
    k = int(np.argmin(kl))
    return grid[k], kl[k], np.mean(np.sum(pb * (lb - np.log(pa)), 1))


def auroc(conf, correct):
    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty(len(conf))
    ranks[order] = np.arange(1, len(conf) + 1)
    pos = correct.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def run(d):
    print(f"\n################ {d}")
    for a, b, label in PAIRS:
        pa, ans, subj = load(d, a)
        pb, ans_b, _ = load(d, b)
        assert (ans == ans_b).all(), "question order differs"
        s, kl_fit, kl_raw = fit_s(pa, pb)
        q = sharpen(pa, s)
        idx = np.arange(len(ans))
        conf_a, conf_b, conf_q = pa.max(1), pb.max(1), q.max(1)
        corr_a = pa.argmax(1) == ans
        corr_b = pb.argmax(1) == ans
        know = pa[idx, ans]  # how much A knew: its probability on the correct answer
        resid = conf_b - conf_q

        print(f"\n=== {label}   ({a} -> {b})")
        print(f"best uniform sharpening s = {s:.2f}  (KL(B||A) {kl_raw:.4f} -> KL(B||A^s) {kl_fit:.4f}, "
              f"explains {1 - kl_fit / kl_raw:.0%})")
        print(f"AUROC  A {auroc(conf_a, corr_a):.3f}   B {auroc(conf_b, corr_b):.3f}   "
              f"sharpened A {auroc(conf_q, corr_a):.3f}")
        print(f"{'A p(correct)':<14}{'n':>6}{'acc A':>7}{'acc B':>7}{'conf A':>8}{'conf B':>8}"
              f"{'sharp':>8}{'B-sharp':>9}{'95% CI':>18}{'B p(corr)':>10}{'sharp p':>9}{'changed':>8}")
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (know >= lo) & (know < hi)
            if m.sum() < 30:
                continue
            ci = boot_ci(resid[m])
            changed = (pa[m].argmax(1) != pb[m].argmax(1)).mean()
            print(f"[{lo:.2f},{min(hi, 1):.2f}){'':<1}{m.sum():>6}{corr_a[m].mean():>7.3f}{corr_b[m].mean():>7.3f}"
                  f"{conf_a[m].mean():>8.3f}{conf_b[m].mean():>8.3f}{conf_q[m].mean():>8.3f}"
                  f"{resid[m].mean():>+9.3f}   [{ci[0]:+.3f},{ci[1]:+.3f}]"
                  f"{pb[m][np.arange(m.sum()), ans[m]].mean():>10.3f}{q[m][np.arange(m.sum()), ans[m]].mean():>9.3f}"
                  f"{changed:>8.1%}")
        for name, m in [("A wrong", ~corr_a), ("A right", corr_a)]:
            ci = boot_ci(resid[m])
            print(f"  {name:<9} n={m.sum():>5}  B-sharp {resid[m].mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")

        # Subject level: does the residual grow where A knows the subject less?
        by = defaultdict(list)
        for i, sj in enumerate(subj):
            by[sj].append(i)
        acc_s = np.array([corr_a[v].mean() for v in by.values()])
        res_s = np.array([resid[v].mean() for v in by.values()])
        if len(by) > 5:
            r = np.corrcoef(acc_s, res_s)[0, 1]
            print(f"  subjects: corr(A's subject accuracy, mean B-sharp) = {r:+.3f} over {len(by)} subjects")


for d in sys.argv[1:]:
    run(d)
