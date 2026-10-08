"""Summarize attribution.py output: which DPO pairs push toward listed-option overconfidence?

push_conf > 0 means the pair's DPO update (at the start of training) sharpens the A-D answer
distribution on MMLU. Because the sample has 2000 option-listing and 2000 other pairs, shares
of the total push are reweighted to the real dataset (7,017 vs 252,905 pairs).

Usage: python analyze_attr.py results_attr
"""
import json
import random
import re
import sys
from collections import defaultdict

N_MC, N_ALL = 7017, 259922
out = sys.argv[1] if len(sys.argv) > 1 else "results_attr"
rows = [json.loads(l) for l in open(f"{out}/instruct_pairs.jsonl")]
info = json.load(open(f"{out}/instruct_direction.json"))

# Remove the part of push_conf that just tracks push_acc (regression across pairs), leaving
# the "more confident, not more correct" component.
xs, ys = [r["push_acc"] for r in rows], [r["push_conf"] for r in rows]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
for r in rows:
    r["push_conf_resid"] = r["push_conf"] - slope * r["push_acc"]
    r["source"] = re.sub(r"-request-.*$", "", r["prompt_id"] or "?")


def mean_ci(v, n_boot=2000, seed=0):
    rng = random.Random(seed)
    m = sum(v) / len(v)
    boots = sorted(sum(rng.choices(v, k=len(v))) / len(v) for _ in range(n_boot))
    return m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def show(title, groups, key, min_n=1):
    print(f"\n{title}  [{key}]")
    print(f"{'group':<58}{'n':>6}{'mean':>10}{'95% CI':>22}{'>0':>7}")
    for g, rs in sorted(groups.items(), key=lambda kv: -sum(r[key] for r in kv[1]) / len(kv[1])):
        if len(rs) < min_n:
            continue
        m, lo, hi = mean_ci([r[key] for r in rs])
        pos = sum(r[key] > 0 for r in rs) / len(rs)
        print(f"{str(g)[:57]:<58}{len(rs):>6}{m:>10.3f}   [{lo:>8.3f}, {hi:>8.3f}]{pos:>7.0%}")


print("sanity check (first-order prediction vs measured, MMLU):")
print(f"  conf change: measured {info['conf_change']:+.4f}, predicted {info['conf_change_first_order']:+.4f}")
print(f"  acc  change: measured {info['acc_change']:+.4f}, predicted {info['acc_change_first_order']:+.4f}")
print(f"  cos(∇conf, ∇acc) = {info['cos_gconf_gacc']:+.3f};  push_conf ~ {slope:+.3f} × push_acc across pairs")

by_mc = defaultdict(list)
for r in rows:
    by_mc["option-listing" if r["is_mc"] else "other"].append(r)
for key in ("push_conf", "push_conf_resid", "push_acc"):
    show("option-listing vs other", by_mc, key)

# Share of the whole dataset's push that comes from option-listing pairs.
for key in ("push_conf", "push_conf_resid"):
    rng = random.Random(0)
    mc, ot = [r[key] for r in by_mc["option-listing"]], [r[key] for r in by_mc["other"]]
    def share(a, b):
        tot_mc, tot_ot = N_MC * sum(a) / len(a), (N_ALL - N_MC) * sum(b) / len(b)
        return tot_mc / (tot_mc + tot_ot)
    boots = sorted(share(rng.choices(mc, k=len(mc)), rng.choices(ot, k=len(ot))) for _ in range(2000))
    print(f"\nshare of total {key} from option-listing pairs (2.7% of data): "
          f"{share(mc, ot):.1%}  [95% CI {boots[50]:.1%}, {boots[1949]:.1%}]")

by_type = defaultdict(list)
for r in rows:
    by_type[(r["preference_type"], "MC" if r["is_mc"] else "other")].append(r)
show("by preference type x option-listing", by_type, "push_conf_resid", min_n=20)

by_src = defaultdict(list)
for r in rows:
    by_src[r["source"]].append(r)
show("by prompt source (n >= 30)", by_src, "push_conf_resid", min_n=30)

# Is it just length? DPO length bias: chosen is usually longer.
d = [r["len_chosen"] - r["len_rejected"] for r in rows]
p = [r["push_conf_resid"] for r in rows]
md, mp = sum(d) / len(d), sum(p) / len(p)
corr = sum((a - md) * (b - mp) for a, b in zip(d, p)) / (
    (sum((a - md) ** 2 for a in d) * sum((b - mp) ** 2 for b in p)) ** 0.5)
print(f"\ncorr(len_chosen - len_rejected, push_conf_resid) = {corr:+.3f}")
