"""Build the DPO training sets for the 8xH100 runs (D0-D4) as single parquet files open-instruct can read.

Every set keeps only role/content in the chosen/rejected message lists. Sets written to --out:
  instruct_full      all 260K Instruct DPO pairs (D0 reproduce, D1 same data at Think's LR)
  think_full         all 150K Think DPO pairs (D2 Think data at Instruct's LR)
  d4_full            Instruct pairs minus flagged pairs (D4 at full scale)
  q25_all            65K random Instruct pairs (control for every 25% run)
  q25_delta          65K delta-learning pairs only (D3)
  q25_judged         65K GPT-judged pairs only (D3)
  q25_d4             65K unflagged pairs with the same preference-type mix as q25_all (D4)
  instruct_meta      one row per Instruct pair: source, rejected opening class, token count, flags

A pair is flagged (dropped for D4) when its rejected response is
  answer-first : <= 5 tokens, or opens with an option label / bare number / yes-no / "The answer is" (see LETTER, ANSWER)
  degenerate   : empty, no letters or digits at all ("******", "[]"), or a long generation loop
Token attribution put ~25% of DPO's net push in answer-first openings and found degenerate rejected outputs heavily
over-weighted, so D4 asks whether removing these pairs removes most of the MCQ overconfidence.

Usage: python make_dpo_data.py --out /workspace/dpo_data [--instruct DIR_OR_HF_ID] [--think DIR_OR_HF_ID]
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import zlib

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

INSTRUCT = "allenai/dolci-3-instruct-dpo-with-metadata"
THINK = "allenai/Dolci-Think-DPO-7B"
TOKENIZER = "allenai/Olmo-3-7B-Instruct-SFT"
Q25 = 65000

# A tightened version of openings.py's ANSWER pattern. On 20K rejected responses, 28% of that pattern's matches were
# false positives: the article "A" ("A family of bears..."), numbered lists ("1. To calculate..."), and lowercase part
# labels ("(a) **..."). It also missed "The correct answer is: **(B)...". Option letters are now uppercase and must be
# followed by . ) : or end the line; a bare number must end its line.
LETTER = re.compile(r"^\s*\(?[A-E]\)?([\.\):]\s|[\.\):]?\s*$|\)\s)")
ANSWER = re.compile(r"^\s*((yes|no|true|false|positive|negative|neutral|correct|incorrect)\b"
                    r"|[-+]?\$?\d[\d,]*(\.\d+)?%?\s*\.?\s*(\n|$)"
                    r"|\**(the |my )?(final |correct |best )?answer\**\s*(is|:)(?!\s*let'?s think)|\**(option|choice)\s+[A-E1-5]\b)",
                    re.I)
# Checked with judge_answer_first.py (gpt-6-luna on 650 stratified rejected responses): 94% of flagged openings are
# answer-first (the old pattern: 75%). The judge's broader notion also counts first-sentence answers in prose ("A spin
# class typically lasts 45 to 60 minutes"); the rule deliberately targets only the bare-label form.
PREAMBLE = re.compile(r"^\s*(#+\s|\*\*[^*]{0,60}\*\*\s*\n|sure|certainly|of course|absolutely|great|okay|ok\b|alright|"
                      r"here('s| is| are)|let'?s|to (solve|answer|determine|find|calculate|address|create|write)|"
                      r"i('m| am| will| would| can|'d| have)?\b|thank|in (this|order)|as an ai|what a|"
                      r"(this|that) (is|sounds)|good question|hello|hi\b)", re.I)


def parquet_files(src):
    if not os.path.exists(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, repo_type="dataset", allow_patterns=["*.parquet"])
    files = sorted(glob.glob(f"{src}/**/*.parquet", recursive=True))
    assert files, f"no parquet files under {src}"
    return files


def read(src, cols):
    rows = []
    for f in parquet_files(src):
        rows += pq.read_table(f, columns=cols).to_pylist()
    return rows


def clean(msgs):
    return [{"role": m["role"], "content": m["content"] or ""} for m in msgs]


def classify(tok, text):
    """Opening class of a response and its token count (exact up to 64 tokens, else '>64')."""
    head = tok.encode(text[:400], add_special_tokens=False).ids
    n = len(head) if len(text) <= 400 else 65
    opening = tok.decode(head[:12])
    if n <= 5 or LETTER.match(opening) or ANSWER.match(opening):
        return "answer-first", n
    if PREAMBLE.match(opening):
        return "preamble", n
    return "content", n


def degenerate(text):
    t = text.strip()
    if not t or not any(c.isalnum() for c in t):
        return True
    b = t.encode()
    return len(b) > 1000 and len(zlib.compress(b)) / len(b) < 0.05


def write(out, name, rows):
    table = pa.Table.from_pylist([{"chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    pq.write_table(table, f"{out}/{name}.parquet")
    types = collections.Counter(r.get("preference_type", "think") for r in rows)
    print(f"  {name:<15} {len(rows):>7} pairs  {dict(types)}")
    return {"n": len(rows), "types": dict(types)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--instruct", default=INSTRUCT)
    ap.add_argument("--think", default=THINK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_think", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(args.seed)

    rows = read(args.instruct, ["chosen", "rejected", "preference_type", "prompt_id", "chosen_model", "rejected_model"])
    meta, examples = [], collections.defaultdict(list)
    for i, r in enumerate(rows):
        r["chosen"], r["rejected"] = clean(r["chosen"]), clean(r["rejected"])
        rej = r["rejected"][-1]["content"]
        cls, n = classify(tok, rej)
        degen = degenerate(rej)
        r["flag"] = cls == "answer-first" or degen
        meta.append({"idx": i, "preference_type": r["preference_type"], "source": r["prompt_id"].rsplit("-", 1)[0],
                     "rejected_model": r["rejected_model"], "rejected_class": cls, "rejected_ntok": n,
                     "chosen_class": classify(tok, r["chosen"][-1]["content"])[0], "degenerate": degen, "flag": r["flag"]})
        key = "degenerate" if degen else cls
        if r["flag"] and len(examples[key]) < 8:
            examples[key].append(rej[:100])
    pq.write_table(pa.Table.from_pylist(meta), f"{args.out}/instruct_meta.parquet")

    n = len(rows)
    flagged = [r for r in rows if r["flag"]]
    by_type = collections.Counter((m["preference_type"], m["rejected_class"], m["degenerate"]) for m in meta)
    print(f"Instruct: {n} pairs; flagged {len(flagged)} ({len(flagged) / n:.1%})")
    for k, v in sorted(by_type.items()):
        print(f"  {k[0]:<28} rejected {k[1]:<13} degenerate={k[2]!s:<5} {v:>7}")
    for k, v in examples.items():
        print(f"examples, {k}:")
        for s in v:
            print("   ", json.dumps(s))

    summary = {}
    print("writing:")
    summary["instruct_full"] = write(args.out, "instruct_full", rows)
    summary["d4_full"] = write(args.out, "d4_full", [r for r in rows if not r["flag"]])
    shuffled = rows[:]
    rng.shuffle(shuffled)
    q25 = shuffled[:Q25]
    summary["q25_all"] = write(args.out, "q25_all", q25)
    for name, ptype in (("q25_delta", "delta_learning"), ("q25_judged", "llm_judged")):
        summary[name] = write(args.out, name, [r for r in shuffled if r["preference_type"] == ptype][:Q25])
    # D4 at 25%: unflagged pairs, matching q25_all's preference-type counts so only the filter differs
    want = collections.Counter(r["preference_type"] for r in q25)
    d4 = []
    for ptype, k in want.items():
        pool = [r for r in shuffled if r["preference_type"] == ptype and not r["flag"]]
        assert len(pool) >= k, (ptype, len(pool), k)
        d4 += pool[:k]
    rng.shuffle(d4)
    summary["q25_d4"] = write(args.out, "q25_d4", d4)
    summary["q25_all"]["flagged"] = sum(r["flag"] for r in q25)

    if not args.skip_think:
        think = read(args.think, ["chosen", "rejected", "preference_type"])
        for r in think:
            r["chosen"], r["rejected"] = clean(r["chosen"]), clean(r["rejected"])
        summary["think_full"] = write(args.out, "think_full", think)
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
