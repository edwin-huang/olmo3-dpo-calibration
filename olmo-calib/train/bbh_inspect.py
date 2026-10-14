"""Inspect BBH outputs: accuracy overall and on non-truncated responses, response lengths, and what the truncated
responses of a model look like (repetition in their last 200 words). Usage: python bbh_inspect.py /workspace/results_ckpt"""
import json
import sys

R0 = sys.argv[1]
names = ["sft", "dpo_released", "q25_plain", "q25_weighted", "full_weighted"]


def rep(t):
    w = t.split()[-200:]
    return 1 - len(set(w)) / max(1, len(w))


for n in names:
    rows = [json.loads(l) for l in open(f"{R0}/bbh/{n}/bbh.responses.jsonl")]
    nt = [r for r in rows if not r["truncated"]]
    tr = [r for r in rows if r["truncated"]]
    lens = sorted(len(r["response"]) for r in rows)
    print(f"{n:<14} acc {sum(r['correct'] for r in rows) / len(rows):.3f} | truncated {len(tr):>4} | non-truncated acc "
          f"{sum(r['correct'] for r in nt) / len(nt):.3f} | median chars {lens[len(lens) // 2]} p90 {lens[int(.9 * len(lens))]}"
          f" | repetition in truncated {sum(rep(r['response']) for r in tr) / max(1, len(tr)):.2f}")
rows = [json.loads(l) for l in open(f"{R0}/bbh/full_weighted/bbh.responses.jsonl")]
for r in [r for r in rows if r["truncated"]][:2]:
    print("--- task", r["task"], "| target", r["target"])
    print("START:", r["response"][:400].replace("\n", " "))
    print("END:  ", r["response"][-300:].replace("\n", " "))
g = [json.loads(l) for l in open(f"{R0}/full_w5/final_full/gen.responses.jsonl")]
print("full_weighted chat generations truncated at 2048 tokens:", sum(x["truncated"] for x in g), "of", len(g))
