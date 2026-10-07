"""Do the DPO preference pairs reward sounding certain?

For every pair we compare the final assistant turn of `chosen` vs `rejected`:
  - hedge rate      (might, possibly, likely, ... per 100 words)
  - booster rate    (definitely, clearly, always, ... per 100 words)
  - admits          (explicit uncertainty: "I'm not sure", "I can't verify", ...)
The key number is admit_chosen: among pairs where exactly one side admits
uncertainty, how often is that side the chosen one? 50% = neutral; well below
50% = the data teaches the model to stop admitting uncertainty.

Groups: Instruct DPO split by preference_type, and Think DPO as a whole.
Lexicon counts are crude; read the dumped examples before trusting the numbers.

Usage: python analyze_dpo_data.py [data_dir] [out_dir]
"""
import glob
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

DATA = sys.argv[1] if len(sys.argv) > 1 else "../data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"

HEDGES = r"might|may|could|possibly|perhaps|probably|likely|unlikely|seems?|appears?|suggests?|approximately|roughly|generally|typically|usually|often|i think|i believe|it depends|depending on"
BOOSTERS = r"definitely|certainly|clearly|undoubtedly|absolutely|without a doubt|always|never|guaranteed|of course|obviously|surely|indeed|precisely|exactly"
ADMITS = (
    r"i'?m not (?:entirely |completely |100% )?(?:sure|certain)|i am not (?:entirely |completely )?(?:sure|certain)"
    r"|i don'?t know|i do not know|i'?m unsure|i cannot be (?:sure|certain)|i can'?t be (?:sure|certain)"
    r"|it'?s unclear|it is unclear|i may be wrong|i might be wrong|i could be wrong"
    r"|i (?:can'?t|cannot|am unable to|'m unable to) (?:verify|confirm)|i don'?t have (?:access|reliable|enough) "
    r"|as of my (?:last )?(?:knowledge|training) (?:cutoff|update)|to the best of my knowledge"
    r"|i (?:would )?recommend (?:verifying|double-checking|checking with)|please (?:verify|double-check)"
)
RE_H = re.compile(rf"\b(?:{HEDGES})\b")
RE_B = re.compile(rf"\b(?:{BOOSTERS})\b")
RE_A = re.compile(ADMITS)
RE_THINK = re.compile(r"<think>.*?</think>", re.S)


def final_answer(messages):
    """Last assistant turn, with any <think>...</think> reasoning stripped."""
    if isinstance(messages, str):
        text = messages
    else:
        turns = [m for m in messages if m.get("role") == "assistant"]
        text = (turns[-1] if turns else messages[-1]).get("content") or ""
    text = RE_THINK.sub("", text)
    if "</think>" in text:  # opening tag lived in the prompt/template
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def feats(text):
    low = text.lower()
    words = max(len(low.split()), 1)
    return {
        "words": words,
        "hedge": 100 * len(RE_H.findall(low)) / words,
        "boost": 100 * len(RE_B.findall(low)) / words,
        "admits": bool(RE_A.search(low)),
    }


def iter_pairs(paths, group_fn, columns):
    for path in sorted(paths):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=columns):
            for row in batch.to_pylist():
                yield group_fn(row), row


class Stats:
    def __init__(self):
        self.n = 0
        self.sum = defaultdict(float)
        self.cmp = Counter()  # chosen-vs-rejected comparisons
        self.examples = []  # pairs where only the rejected side admits uncertainty
        self.models = Counter()

    def add(self, row, c, r):
        self.n += 1
        for k in ("words", "hedge", "boost"):
            self.sum[f"d_{k}"] += c[k] - r[k]
            self.sum[f"c_{k}"] += c[k]
            self.sum[f"r_{k}"] += r[k]
        self.cmp["c_longer"] += c["words"] > r["words"]
        if c["hedge"] != r["hedge"]:
            self.cmp["hedge_diff"] += 1
            self.cmp["c_less_hedged"] += c["hedge"] < r["hedge"]
        self.cmp["c_admits"] += c["admits"]
        self.cmp["r_admits"] += r["admits"]
        if c["admits"] != r["admits"]:
            self.cmp["one_admits"] += 1
            self.cmp["chosen_is_admitter"] += c["admits"]
            if r["admits"] and len(self.examples) < 2000:
                self.examples.append(row)
        self.models[(row.get("chosen_model"), row.get("rejected_model"))] += 1

    def summary(self):
        n = self.n
        def prop(k, d):
            den = self.cmp[d] if isinstance(d, str) else d
            p = self.cmp[k] / den if den else float("nan")
            se = math.sqrt(p * (1 - p) / den) if den else float("nan")
            return {"p": p, "se": se, "n": den}
        return {
            "pairs": n,
            "mean_words_chosen": self.sum["c_words"] / n,
            "mean_words_rejected": self.sum["r_words"] / n,
            "hedge_per100_chosen": self.sum["c_hedge"] / n,
            "hedge_per100_rejected": self.sum["r_hedge"] / n,
            "boost_per100_chosen": self.sum["c_boost"] / n,
            "boost_per100_rejected": self.sum["r_boost"] / n,
            "chosen_longer": prop("c_longer", n),
            "chosen_less_hedged": prop("c_less_hedged", "hedge_diff"),
            "admit_rate_chosen": prop("c_admits", n),
            "admit_rate_rejected": prop("r_admits", n),
            "admit_chosen": prop("chosen_is_admitter", "one_admits"),
            "top_model_pairs": [f"{c} > {r}: {k}" for (c, r), k in self.models.most_common(5)],
        }


def main():
    groups = defaultdict(Stats)
    sources = defaultdict(lambda: defaultdict(Stats))  # preference_type -> source -> Stats

    inst = glob.glob(f"{DATA}/instruct_dpo/**/*.parquet", recursive=True)
    think = glob.glob(f"{DATA}/think_dpo/**/*.parquet", recursive=True)
    print(f"instruct files: {len(inst)}  think files: {len(think)}", flush=True)

    cols_i = ["chosen", "rejected", "chosen_model", "rejected_model", "prompt_id", "preference_type"]
    for g, row in iter_pairs(inst, lambda r: f"instruct/{r['preference_type']}", cols_i):
        c, r = feats(final_answer(row["chosen"])), feats(final_answer(row["rejected"]))
        groups[g].add(row, c, r)
        src = re.sub(r"[-_]\d+(?:-\d+)*$", "", row["prompt_id"] or "?")[:70]
        sources[row["preference_type"]][src].add(row, c, r)
        if groups[g].n % 50000 == 0:
            print(g, groups[g].n, flush=True)

    if think:
        cols_t = [c for c in pq.ParquetFile(think[0]).schema_arrow.names
                  if c in ("chosen", "rejected", "chosen_model", "rejected_model", "dataset_source", "id")]
        for g, row in iter_pairs(think, lambda r: "think/delta_learning", cols_t):
            c, r = feats(final_answer(row["chosen"])), feats(final_answer(row["rejected"]))
            groups[g].add(row, c, r)
            sources["think"][row.get("dataset_source") or "?"].add(row, c, r)

    summary = {g: s.summary() for g, s in sorted(groups.items())}
    by_source = {
        t: {src: s.summary() for src, s in sorted(d.items(), key=lambda kv: -kv[1].n)[:12]}
        for t, d in sources.items()
    }
    with open(f"{OUT}/dpo_data_analysis.json", "w") as f:
        json.dump({"groups": summary, "by_source": by_source}, f, indent=2)

    # Pairs where the rejected answer admits uncertainty and the chosen one doesn't:
    # read these to check whether the lexicon is catching real hedging.
    rng = random.Random(0)
    with open(f"{OUT}/dpo_data_examples.jsonl", "w", encoding="utf-8") as f:
        for g, s in sorted(groups.items()):
            for row in rng.sample(s.examples, min(15, len(s.examples))):
                prompt = row["chosen"][0]["content"] if isinstance(row["chosen"], list) else ""
                f.write(json.dumps({
                    "group": g,
                    "prompt": prompt[:1500],
                    "chosen": final_answer(row["chosen"])[:1500],
                    "rejected": final_answer(row["rejected"])[:1500],
                    "chosen_model": row.get("chosen_model"),
                    "rejected_model": row.get("rejected_model"),
                }, ensure_ascii=False) + "\n")

    def fmt(p):
        return f"{p['p']:.3f}±{p['se']:.3f} (n={p['n']})"
    print(f"\n{'group':<38}{'pairs':>8}{'words c/r':>14}{'hedge c/r':>13}{'boost c/r':>13}")
    for g, s in summary.items():
        print(f"{g:<38}{s['pairs']:>8}{s['mean_words_chosen']:>7.0f}/{s['mean_words_rejected']:<6.0f}"
              f"{s['hedge_per100_chosen']:>6.2f}/{s['hedge_per100_rejected']:<6.2f}"
              f"{s['boost_per100_chosen']:>6.2f}/{s['boost_per100_rejected']:<6.2f}")
    print()
    for g, s in summary.items():
        print(f"{g:<38} chosen_less_hedged={fmt(s['chosen_less_hedged'])}")
        print(f"{'':<38} admit_chosen={fmt(s['admit_chosen'])}  "
              f"admit rate c/r={s['admit_rate_chosen']['p']:.3f}/{s['admit_rate_rejected']['p']:.3f}")
    print(f"\nwrote {OUT}/dpo_data_analysis.json and {OUT}/dpo_data_examples.jsonl")


if __name__ == "__main__":
    main()
