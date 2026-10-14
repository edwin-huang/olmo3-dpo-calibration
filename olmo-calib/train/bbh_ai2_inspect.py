"""Break down Ai2-style BBH scores (eval_ckpt part bbh_ai2): why are answers marked wrong?

Per model: Ai2's score (olmes extraction, bug included), the score with the bare-letter fallback fixed (raw string),
and each response's category:
  correct                     scored correct by Ai2's extraction
  bare letter, correct        multiple-choice answer stated as a bare letter (e.g. "the answer is B", "**B**") that is
                              right but scored wrong (needs the fallback that the non-raw string disables)
  no answer extracted         nothing matched (rambling, truncated, or an answer in another form)
  wrong                       an answer was extracted and it is wrong
Also: share of responses with an explicit answer phrase ("answer is" / "Answer:"), truncation, mean tokens.

Usage: python bbh_ai2_inspect.py /workspace/results_ckpt   (run from olmo-calib/ so eval_ckpt imports)
"""
import json
import re
import sys

sys.path.insert(0, "/workspace/olmo-calib")
from eval_ckpt import BBH_ANSWER_REGEX, OLMO_3_REGEXES, olmes_bbh_extract, olmes_exact_match  # noqa: E402

R0 = sys.argv[1]
names = ["sft", "dpo_released", "q25_plain", "q25_weighted", "full_weighted"]


def fixed_extract(text, task):
    """olmes extraction with the intended raw-string fallback r'\\b([A-Z])\\b' for multiple-choice tasks."""
    mc = task not in BBH_ANSWER_REGEX
    ans = "\\([A-Z]\\)" if mc else BBH_ANSWER_REGEX[task]
    regexes = list(OLMO_3_REGEXES) + ([r"\b([A-Z])\b"] if mc else []) + ["(?i)($ANS$)"]
    out = ""
    for rx in regexes:
        found = re.findall(rx.replace("$ANS$", ans), text)
        if found:
            out = found[-1]
            break
    for left, right in (("$", "$"), ("\\(", "\\)"), ("(", ")"), ("**", "**"), ("***", "***"), ("\\[", "\\]"),
                        ("'", "'"), ("`", "`"), ('"', '"')):
        if re.match(ans, left):
            continue
        out = re.sub(f"^{re.escape(left)}(.*){re.escape(right)}$", "\\1", out).strip()
    if mc and len(out) == 1:
        out = f"({out})"
    return out


BARE = re.compile(r"(?i)(?:answer\s*(?:is|:)\s*(?:option\s*)?\**\s*|\*\*\s*)([A-R])\b(?!\))")
PHRASE = re.compile(r"(?i)answer\s*(?:is|:)")
print(f"{'model':<14}{'Ai2 score':>10}{'fixed':>8}{'correct':>9}{'bare-letter right':>19}{'no answer':>11}{'wrong':>8}"
      f"{'answer phrase':>15}{'truncated':>11}{'mean tok':>10}")
for n in names:
    rows = [json.loads(l) for l in open(f"{R0}/bbh/{n}/bbh_ai2.responses.jsonl")]
    cat = {"correct": 0, "bare": 0, "none": 0, "wrong": 0}
    fixed = 0
    for r in rows:
        task, text, gold = r["task"], r["response"], r["target"]
        # the fix only changes multiple-choice tasks (the fallback is MC-only); elsewhere reuse the stored score,
        # which also avoids re-running the slow word_sorting fallback regex on long responses
        fixed += olmes_exact_match(fixed_extract(text, task), gold, task) if task not in BBH_ANSWER_REGEX else r["correct"]
        if r["correct"]:
            cat["correct"] += 1
            continue
        mc = task not in BBH_ANSWER_REGEX
        m = BARE.findall(text[-400:]) if mc else []
        if mc and m and f"({m[-1].upper()})" == gold:
            cat["bare"] += 1
        elif not r["pred"]:
            cat["none"] += 1
        else:
            cat["wrong"] += 1
    N = len(rows)
    phrase = sum(bool(PHRASE.search(r["response"][-600:])) for r in rows) / N
    trunc = sum(r["truncated"] for r in rows) / N
    tok = sum(r["n_tokens"] for r in rows) / N
    print(f"{n:<14}{sum(r['correct'] for r in rows) / N:>10.3f}{fixed / N:>8.3f}{cat['correct'] / N:>9.3f}{cat['bare'] / N:>19.3f}"
          f"{cat['none'] / N:>11.3f}{cat['wrong'] / N:>8.3f}{phrase:>15.3f}{trunc:>11.3f}{tok:>10.0f}")
