"""What do the openings of DPO responses look like, and which openings carry the push?

Classifies the first ~12 tokens of each chosen / rejected response:
  answer-first : a bare label/letter/yes-no/number, "The answer is...", or a whole response of <= 5 tokens
  preamble     : "Sure", "Here's", "To solve...", "Let's", "I ...", a markdown heading, restating the task
  content      : anything else (starts straight into prose, code, a list, ...)
Then splits the opening push (first 5 tokens) by class, finds where chosen and rejected first
diverge, and prints the prompts and openings of the pairs with the largest push.

Usage: python openings.py results_tokens/instruct_tokens.jsonl
"""
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from transformers import AutoTokenizer

from attribution import load_pairs

tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
recs = [json.loads(l) for l in open(sys.argv[1])]
rows = {r["idx"]: r for r in load_pairs(10 ** 6, 10 ** 6) if r["idx"] in {x["idx"] for x in recs}}

ANSWER = re.compile(r"^\s*(\(?[A-Ea-e]\)?[\.\):]?(\s|$)|(yes|no|true|false|positive|negative|neutral|correct|incorrect)\b"
                    r"|[-+]?\d[\d,\.]*\s*$|[-+]?\d[\d,\.]*\s*[\.\n]|\**(the )?(final )?answer\**\s*(is|:)|\**option\s+[A-E1-5]\b|\**choice\s+[A-E1-5]\b)",
                    re.I)
PREAMBLE = re.compile(r"^\s*(#+\s|\*\*[^*]{0,60}\*\*\s*\n|sure|certainly|of course|absolutely|great|okay|ok\b|alright|"
                      r"here('s| is| are)|let'?s|to (solve|answer|determine|find|calculate|address|create|write)|"
                      r"i('m| am| will| would| can|'d| have)?\b|thank|in (this|order)|as an ai|what a|"
                      r"(this|that) (is|sounds)|good question|hello|hi\b)", re.I)


def classify(text, n_tokens):
    if n_tokens <= 5 or ANSWER.match(text):
        return "answer-first"
    if PREAMBLE.match(text):
        return "preamble"
    return "content"


def opening(seq, k=12):
    return tok.decode(seq["tokens"][:k])


cls_count = {"chosen": Counter(), "rejected": Counter()}
open_push = defaultdict(float)  # (side, class) -> push from that side's first 5 tokens
rest_push = defaultdict(float)
pair_cls = Counter()
diverge = []
for r in recs:
    for side, sign in (("chosen", 1), ("rejected", -1)):
        s = r[side]
        L = len(s["d"])
        c = classify(opening(s), L)
        cls_count[side][c] += 1
        open_push[(side, c)] += sign * sum(s["d"][:5]) / L
        rest_push[(side, c)] += sign * sum(s["d"][5:]) / L
        r[f"cls_{side}"] = c
    pair_cls[(r["cls_chosen"], r["cls_rejected"])] += 1
    a, b = r["chosen"]["tokens"], r["rejected"]["tokens"]
    diverge.append(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))

total = sum(r["push"] for r in recs)
print(f"{len(recs)} pairs, total push {total:+.2f}")
print("\nhow responses open (share of responses):")
for side in ("chosen", "rejected"):
    n = sum(cls_count[side].values())
    print(f"  {side:<9}" + "  ".join(f"{c} {cls_count[side][c] / n:.0%}" for c in ("answer-first", "preamble", "content")))
print("\npair types (chosen opening / rejected opening):")
for (c, rj), n in pair_cls.most_common():
    print(f"  {c:<13} / {rj:<13} {n:>5} ({n / len(recs):.0%})")
d = np.array(diverge)
print(f"\nfirst token where chosen and rejected differ: token 0 in {np.mean(d == 0):.0%} of pairs, "
      f"within first 5 in {np.mean(d < 5):.0%}, median {np.median(d):.0f}")
print("\nshare of total push, by side and opening class (first 5 tokens / rest of response):")
for side in ("chosen", "rejected"):
    for c in ("answer-first", "preamble", "content"):
        print(f"  {side:<9}{c:<13} first 5: {open_push[(side, c)] / total:>+7.1%}   rest: {rest_push[(side, c)] / total:>+7.1%}"
              f"   (n = {cls_count[side][c]})")

print("\n================ pairs with the largest push ================")
for r in sorted(recs, key=lambda x: -x["push"])[:8]:
    row = rows[r["idx"]]
    prompt = next((m["content"] or "" for m in reversed(row["chosen"]) if m["role"] == "user"), "")
    print(f"\n--- idx {r['idx']}  push {r['push']:+.3f}  {r['preference_type']}  {r['prompt_id'][:50]}")
    print(f"PROMPT: {prompt[:300]!r}")
    for side in ("chosen", "rejected"):
        s = r[side]
        print(f"{side.upper()} [{r['cls_' + side]}, {len(s['d'])} tok]: {tok.decode(s['tokens'][:40])!r}")
