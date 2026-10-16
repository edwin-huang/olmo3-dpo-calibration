"""Put immediate answers on the self-consistency scale (no GPU).

Self-consistency samples 8 reasoning chains at temperature 0.7 and uses agreement with the majority answer as
confidence (cot_test.py sample(), analyze_perq.py). Here the same is done for immediate answers: an immediate answer
is one draw from the saved chat-format A-D probabilities, so "8 immediate samples at T 0.7" is simulated exactly
from p ** (1/0.7), renormalized. Ties go to the earliest letter, as in cot_test.py. Expected values over 4,000
simulated sets of 8 per question. The 95% intervals on DPO - SFT come from a paired bootstrap over questions
(2,000 resamples).

Usage: python equal_terms.py results_cot   (writes nothing; output kept in results_cot/equal_terms.out)
"""
import json
import sys

import numpy as np

d = sys.argv[1]
L = "ABCD"
K, TEMP, REPS, BOOT = 8, 0.7, 4000, 2000
rng = np.random.default_rng(0)


def load(name):
    modes = {"direct": {}, "cot": {}}
    for line in open(f"{d}/{name}.questions.jsonl"):
        r = json.loads(line)
        modes[r["mode"]][r["qi"]] = r
    sc = {}
    for line in open(f"{d}/{name}.sc8.questions.jsonl"):
        r = json.loads(line)
        sc[r["qi"]] = r
    return modes, sc


def sampled_agreement(p, answer):
    """Expected (majority share, majority correct) for K draws at TEMP from letter probabilities p."""
    q = np.asarray(p, dtype=float) ** (1 / TEMP)
    q /= q.sum()
    counts = rng.multinomial(K, q, size=REPS)  # [REPS, 4]
    best = counts.argmax(1)  # argmax returns the first maximum: earliest letter wins ties
    conf = counts.max(1) / K
    return conf.mean(), (best == L.index(answer)).mean()


rows = {}
for name in ("Olmo-3-7B-Instruct-SFT", "Olmo-3-7B-Instruct-DPO"):
    modes, sc = load(name)
    per = []
    for qi, s in sc.items():
        dr, cr = modes["direct"][qi], modes["cot"][qi]
        assert dr["answer"] == s["answer"] == cr["answer"] and dr["src"] == s["src"], qi
        p = dr["probs"]
        votes = s["votes"]
        counts = {x: votes.count(x) for x in L}
        maj = max(L, key=lambda x: counts[x])
        valid = [v for v in votes if v]
        ds_conf, ds_ok = sampled_agreement(p, s["answer"])
        per.append({
            "src": s["src"],
            "d_conf": max(p), "d_ok": L[int(np.argmax(p))] == s["answer"],
            "ds_conf": ds_conf, "ds_ok": ds_ok,
            "c_conf": max(cr["probs"]), "c_ok": L[int(np.argmax(cr["probs"]))] == s["answer"],
            "sc_conf": counts[maj] / len(votes), "sc_ok": maj == s["answer"],
            "scv_conf": counts[maj] / len(valid) if valid else 0.0,
        })
    rows[name] = per

measures = [
    ("immediate, letter probability (write-up)", "d_conf", "d_ok"),
    ("immediate, agreement of 8 samples at T 0.7 (new)", "ds_conf", "ds_ok"),
    ("after one greedy chain, letter probability", "c_conf", "c_ok"),
    ("after reasoning, agreement of 8 chains (write-up)", "sc_conf", "sc_ok"),
    ("after reasoning, 8 chains, missing answers dropped", "scv_conf", "sc_ok"),
]
S, D = rows["Olmo-3-7B-Instruct-SFT"], rows["Olmo-3-7B-Instruct-DPO"]
for src in ("mmlu", "arc"):
    s_ = [r for r in S if r["src"] == src]
    d_ = [r for r in D if r["src"] == src]
    print(f"\n== {src.upper()} ({len(s_)} questions): conf, acc, excess (conf - acc) ==")
    print(f"{'measure':<52}{'SFT conf':>9}{'acc':>6}{'excess':>8}{'DPO conf':>10}{'acc':>6}{'excess':>8}{'DPO-SFT':>9}"
          f"{'95% CI':>18}")
    n = len(s_)
    boot = rng.integers(0, n, size=(BOOT, n))  # paired bootstrap over questions, same resamples for every measure
    for label, kc, ko in measures:
        sc_, sa = np.mean([r[kc] for r in s_]), np.mean([r[ko] for r in s_])
        dc, da = np.mean([r[kc] for r in d_]), np.mean([r[ko] for r in d_])
        diff_q = np.array([(dr[kc] - dr[ko]) - (sr[kc] - sr[ko]) for sr, dr in zip(s_, d_)], dtype=float)
        lo, hi = np.percentile(diff_q[boot].mean(1), [2.5, 97.5])
        print(f"{label:<52}{sc_:>9.3f}{sa:>6.3f}{sc_ - sa:>+8.3f}{dc:>10.3f}{da:>6.3f}{dc - da:>+8.3f}"
              f"{(dc - da) - (sc_ - sa):>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>18}")
