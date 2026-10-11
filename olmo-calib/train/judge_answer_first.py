"""Check make_dpo_data.py's answer-first rule against an LLM judge on a stratified sample of rejected responses.

Strata (from one shard of the Instruct DPO data): responses the rule flags, responses only the old openings.py
pattern flagged, and responses neither flags. The judge sees the prompt and the opening of the response and says
whether the response commits to its answer before any reasoning or explanation. Precision and recall of the rule
are reweighted to the full population by stratum sizes.

Usage: python judge_answer_first.py SHARD.parquet OUT.jsonl  (reads OPENAI_API_KEY from env or ../../.env)
"""
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from make_dpo_data import TOKENIZER, classify

MODEL = "gpt-6-luna"
N = {"rule": 200, "old_only": 150, "neither": 300}
OLD = re.compile(r"^\s*(\(?[A-Ea-e]\)?[\.\):]?(\s|$)|(yes|no|true|false|positive|negative|neutral|correct|incorrect)\b"
                 r"|[-+]?\d[\d,\.]*\s*$|[-+]?\d[\d,\.]*\s*[\.\n]|\**(the )?(final )?answer\**\s*(is|:)|\**option\s+[A-E1-5]\b|\**choice\s+[A-E1-5]\b)",
                 re.I)
INSTR = """You label the opening of an AI assistant's response.

answer_first = true if the response commits to its final answer at the very start, before any reasoning, explanation,
restating, or preamble. Examples: a bare option letter ("B", "(C) water vapor"), a number ("42."), yes/no ("Yes, ..."),
a short answer word or phrase ("Negative", "earth"), "The answer is X". Also true if the entire response is just a
short answer.
answer_first = false if it opens with reasoning, steps, a restatement, a greeting/preamble ("Sure! Here's..."), a
heading, a numbered list of steps or items, or prose/code that isn't a direct answer to the question.

Reply with JSON only: {"answer_first": true|false}"""


def key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")):
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no OPENAI_API_KEY")


def judge(api_key, prompt, opening):
    body = {"model": MODEL, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": INSTR},
                         {"role": "user", "content": f"PROMPT (truncated):\n{prompt[:600]}\n\nRESPONSE OPENING:\n{opening}"}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=120))
            return bool(json.loads(out["choices"][0]["message"]["content"])["answer_first"])
        except Exception as e:  # rate limits / transient errors
            print(f"retry {attempt}: {e}", flush=True)
            time.sleep(2 ** attempt * 5)
    return None


def main():
    shard, out = sys.argv[1], sys.argv[2]
    tok = Tokenizer.from_pretrained(TOKENIZER)
    rows = pq.read_table(shard, columns=["rejected"]).to_pylist()
    pools = {"rule": [], "old_only": [], "neither": []}
    for r in rows:
        msgs = r["rejected"]
        text = msgs[-1]["content"] or ""
        prompt = next((m["content"] or "" for m in reversed(msgs[:-1]) if m["role"] == "user"), "")
        cls, n = classify(tok, text)
        ids = tok.encode(text[:400], add_special_tokens=False).ids
        old = n <= 5 or bool(OLD.match(tok.decode(ids[:12])))
        s = "rule" if cls == "answer-first" else "old_only" if old else "neither"
        pools[s].append({"prompt": prompt, "opening": text[:300], "ntok": n, "old": old})
    sizes = {k: len(v) for k, v in pools.items()}
    print("population:", sizes, flush=True)
    api_key = key()
    sample = []
    for s, k in N.items():
        for x in random.Random(0).sample(pools[s], min(k, len(pools[s]))):
            sample.append(dict(x, stratum=s))
    with ThreadPoolExecutor(16) as ex:
        labels = list(ex.map(lambda x: judge(api_key, x["prompt"], x["opening"]), sample))
    with open(out, "w") as f:
        for x, y in zip(sample, labels):
            f.write(json.dumps(dict(x, judge=y)) + "\n")
    rate = {}
    for s in N:
        ys = [y for x, y in zip(sample, labels) if x["stratum"] == s and y is not None]
        rate[s] = sum(ys) / max(len(ys), 1)
        print(f"{s:<9} judged {len(ys):>4}  judge says answer-first: {rate[s]:.0%}")
    print(f"failed calls: {sum(y is None for y in labels)}")
    # population-weighted: true answer-first = sum over strata of size * rate
    tp = sizes["rule"] * rate["rule"]
    total_pos = tp + sizes["old_only"] * rate["old_only"] + sizes["neither"] * rate["neither"]
    rule = [(x, y) for x, y in zip(sample, labels) if x["stratum"] == "rule" and y is not None]
    old_in_rule = sum(x["old"] for x, _ in rule) / len(rule)  # share of rule-flagged that the old pattern also flags
    old_tp_in_rule = sum(x["old"] and y for x, y in rule) / len(rule)
    old_flagged = sizes["rule"] * old_in_rule + sizes["old_only"]
    old_tp = sizes["rule"] * old_tp_in_rule + sizes["old_only"] * rate["old_only"]
    print(f"new rule: precision {rate['rule']:.0%}, recall {tp / total_pos:.0%}, flags {sizes['rule']}")
    print(f"old rule: precision {old_tp / old_flagged:.0%}, recall {old_tp / total_pos:.0%}, flags {old_flagged:.0f}")


if __name__ == "__main__":
    main()
