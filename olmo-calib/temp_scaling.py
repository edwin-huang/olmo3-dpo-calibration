"""Is post-training overconfidence just a global sharpening?

For each checkpoint's per-question probabilities, fit one temperature T on half the
questions (p_i ∝ p_i^(1/T)) and report ECE on the other half before/after. If one
scalar removes most of the ECE, the miscalibration is a uniform "temperature" change.
AUROC (does confidence rank right answers above wrong ones?) is unaffected by T, so
it measures whether the model's ability to tell what it knows changed separately.

Usage: python temp_scaling.py results_robust/mmlu5_full [more dirs...]
"""
import glob
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from plot import ORDER  # noqa: E402


def scaled(probs, t):
    logs = [math.log(max(p, 1e-12)) / t for p in probs]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    z = sum(e)
    return [x / z for x in e]


def ece(confs, correct, n_bins=15):
    bins = [[0, 0.0, 0] for _ in range(n_bins)]
    for c, y in zip(confs, correct):
        b = bins[min(int(c * n_bins), n_bins - 1)]
        b[0] += 1
        b[1] += c
        b[2] += y
    return sum(abs(s - k) for n, s, k in bins if n) / len(confs)


def nll(rows, t):
    return -sum(math.log(max(scaled(r["probs"], t)[r["answer"]], 1e-12)) for r in rows) / len(rows)


def fit_t(rows):
    lo, hi = math.log(0.2), math.log(10.0)  # golden-section search on log T
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(rows, math.exp(a)), nll(rows, math.exp(b))
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(rows, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(rows, math.exp(b))
    return math.exp((lo + hi) / 2)


def auroc(confs, correct):
    pos = [c for c, y in zip(confs, correct) if y]
    neg = [c for c, y in zip(confs, correct) if not y]
    ranked = sorted([(c, 1) for c in pos] + [(c, 0) for c in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):  # average ranks over ties
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(rows, t):
    probs = [scaled(r["probs"], t) for r in rows]
    confs = [max(p) for p in probs]
    correct = [int(p.index(max(p)) == r["answer"]) for p, r in zip(probs, rows)]
    return ece(confs, correct), auroc(confs, correct), sum(confs) / len(confs)


def main():
    for d in sys.argv[1:]:
        print(f"\n##### {d}")
        print(f"{'checkpoint':<36}{'T*':>6}{'ECE':>8}{'ECE@T*':>8}{'removed':>9}{'AUROC':>8}{'conf':>7}{'conf@T*':>8}")
        paths = sorted(glob.glob(f"{d}/*.jsonl"),
                       key=lambda p: ORDER.index(os.path.basename(p)[:-6]) if os.path.basename(p)[:-6] in ORDER else 99)
        for p in paths:
            rows = [json.loads(line) for line in open(p)]
            idx = list(range(len(rows)))
            random.Random(0).shuffle(idx)
            fit = [rows[i] for i in idx[: len(rows) // 2]]
            held = [rows[i] for i in idx[len(rows) // 2 :]]
            t = fit_t(fit)
            e0, auc, c0 = evaluate(held, 1.0)
            e1, _, c1 = evaluate(held, t)
            print(f"{os.path.basename(p)[:-6]:<36}{t:>6.2f}{e0:>8.3f}{e1:>8.3f}{(1 - e1 / e0):>9.0%}{auc:>8.3f}{c0:>7.3f}{c1:>8.3f}")


if __name__ == "__main__":
    main()
