"""LLM-judge win rate of DPO checkpoints against Instruct-SFT on held-out chat prompts (run locally; the API key
stays on this machine).

Reads <results>/<tag>/gen.responses.jsonl (eval_ckpt.py's "gen" part) for each tag and for the reference tag, asks
gpt-6-luna which of the two responses is better, with the order alternating by prompt id so position bias cancels
across prompts. Win = 1, tie = 0.5. Judgments are cached in <results>/<tag>/judge_vs_<ref>.jsonl, so reruns only judge
what is new. Also reports a length-adjusted win rate (logistic fit of the outcome on the log length ratio, read off at
equal length), since DPO tends to lengthen responses and judges favour length.

Usage: python judge_winrate.py results_ckpt small/sft small/base8a small/simpo8a ...
"""
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = "gpt-6-luna"
INSTRS = {  # JUDGE_STYLE env: "strict" (default; warns against length) or "neutral" (AlpacaEval-like, no length warning)
    "strict": """You compare two AI assistant responses to the same user prompt. Decide which response a careful expert
would prefer overall: correct, helpful, follows the instructions, clear, and appropriately concise. Do not prefer a
response for being longer unless the extra content adds real value. Output JSON only: {"winner": "1"} or
{"winner": "2"} or {"winner": "tie"}.""",
    "neutral": """You compare two AI assistant responses to the same user prompt. Which response is better? Output JSON
only: {"winner": "1"} or {"winner": "2"} or {"winner": "tie"}.""",
}
STYLE = os.environ.get("JUDGE_STYLE", "strict")
INSTR = INSTRS[STYLE]


def key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")):
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no OPENAI_API_KEY")


def judge(api_key, prompt, r1, r2):
    body = {"model": MODEL, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": INSTR},
                         {"role": "user", "content": f"USER PROMPT:\n{prompt[:6000]}\n\n=== RESPONSE 1 ===\n{r1[:8000]}"
                                                     f"\n\n=== RESPONSE 2 ===\n{r2[:8000]}"}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=180))
            w = str(json.loads(out["choices"][0]["message"]["content"])["winner"]).strip().lower()
            if w in ("1", "2", "tie"):
                return w
        except Exception as e:  # rate limits / transient errors
            print(f"retry {attempt}: {e}", flush=True)
            time.sleep(2 ** attempt * 5)
    return None


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def length_adjusted(rows):
    """Logistic fit score ~ a + b * log(len_model / len_ref) on win/loss (ties as half-weight both); returns sigmoid(a)."""
    xs = [math.log((r["n_model"] + 1) / (r["n_ref"] + 1)) for r in rows]
    ys = [r["score"] for r in rows]
    a = b = 0.0
    for _ in range(50):  # Newton steps on the 2-parameter logistic log-likelihood (fractional targets are fine)
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-(a + b * x)))
            ga += y - p; gb += (y - p) * x
            w = p * (1 - p); haa += w; hab += w * x; hbb += w * x * x
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            break
        a += (hbb * ga - hab * gb) / det
        b += (haa * gb - hab * ga) / det
    return 1 / (1 + math.exp(-a)), b


def main():
    root, ref_tag, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    api_key = key()
    ref = load(os.path.join(root, ref_tag, "gen.responses.jsonl"))
    ref_name = ref_tag.replace("/", "_") + ("" if STYLE == "strict" else f"_{STYLE}")
    print(f"{'tag':<28}{'n':>5}{'win':>8}{'±se':>7}{'len-adj':>9}{'len ratio':>11}{'trunc':>7}")
    for tag in tags:
        gen = load(os.path.join(root, tag, "gen.responses.jsonl"))
        cache_path = os.path.join(root, tag, f"judge_vs_{ref_name}.jsonl")
        done = load(cache_path) if os.path.exists(cache_path) else {}
        todo = [i for i in gen if i in ref and i not in done]

        def one(i):
            g, r = gen[i], ref[i]
            model_first = i % 2 == 0
            w = judge(api_key, g["prompt"], *((g["response"], r["response"]) if model_first else (r["response"], g["response"])))
            if w is None:
                return None
            score = 0.5 if w == "tie" else float((w == "1") == model_first)
            return {"id": i, "winner": w, "model_first": model_first, "score": score,
                    "n_model": g["n_tokens"], "n_ref": r["n_tokens"]}

        with ThreadPoolExecutor(16) as ex, open(cache_path, "a", encoding="utf-8") as f:
            for res in ex.map(one, todo):
                if res:
                    f.write(json.dumps(res) + "\n")
                    done[res["id"]] = res
        rows = [done[i] for i in sorted(done)]
        win = sum(r["score"] for r in rows) / len(rows)
        se = (sum((r["score"] - win) ** 2 for r in rows) / len(rows)) ** 0.5 / len(rows) ** 0.5
        lc, _ = length_adjusted(rows)
        ratio = math.exp(sum(math.log((r["n_model"] + 1) / (r["n_ref"] + 1)) for r in rows) / len(rows))
        trunc = sum(g["truncated"] for g in gen.values()) / len(gen)
        print(f"{tag:<28}{len(rows):>5}{win:>8.3f}{se:>7.3f}{lc:>9.3f}{ratio:>11.2f}{trunc:>7.1%}")
        json.dump({"ref": ref_tag, "n": len(rows), "win": win, "se": se, "win_length_adjusted": lc,
                   "length_ratio": ratio, "truncated": trunc},
                  open(os.path.join(root, tag, f"judge_vs_{ref_name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
