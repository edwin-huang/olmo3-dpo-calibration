"""Do static agreement weights go stale during training? (inference only)

Agreement-weighted DPO fixes each pair's weight w0 = sigmoid(-5 m0) from the SFT margin m0 (mean per-token log-prob,
chosen - rejected) for the whole run. The actual gradient weight at a later state is w0 * sigmoid(-beta h), beta 5,
h = margin now - m0 (DPO's own sigmoid, which already shrinks pairs whose margin has grown). Stale-weights hypothesis
for the full-scale run (T* 1.61 -> 1.82 while the 25% run stayed at 1.50): as training goes on, more of this weight
lands on pairs the model ALREADY agrees with now, i.e. static weighting drifts back toward plain DPO's pattern.

For 4,000 single-turn q25_all pairs (in both q25_w5's and full_w5's training data; responses <= 2048 tokens, prompt
<= 1024, so nothing is truncated) score per-token mean log-probs under each model, then at each model state report
the share of gradient weight on currently-agreed pairs under: plain DPO sigmoid(-5h); static w0 sigmoid(-5h);
dynamic sigmoid(-5 m_now) sigmoid(-5h) (what current-model weights would assign at the same state).

  python stale_weights.py score NAME MODEL_PATH      (one model per GPU; writes /workspace/results_why/stale/NAME.json)
  python stale_weights.py analyze NAME1 NAME2 ...    (first name = SFT)
"""
import json
import os
import random
import sys

OUT = "/workspace/results_why/stale"
N, MAX_RESP = 4000, 2048


def pairs():
    import pyarrow.parquet as pq
    meta_n = pq.read_metadata("/workspace/dpo_data/instruct_meta.parquet").num_rows
    perm = list(range(meta_n))
    random.Random(0).shuffle(perm)  # make_dpo_data.py's shuffle: q25_all row i = instruct_full row perm[i]
    q25 = pq.read_table("/workspace/dpo_data/q25_all.parquet").slice(0, 30000).to_pylist()
    full = pq.read_table("/workspace/dpo_data/instruct_full.parquet", columns=["chosen"])
    for i in range(0, 30000, 3000):  # mapping check
        assert q25[i]["chosen"][-1]["content"] == full.column("chosen")[perm[i]].as_py()[-1]["content"], i
    rows = [(i, r) for i, r in enumerate(q25) if len(r["chosen"]) == 2]
    random.Random(11).shuffle(rows)
    return [(perm[i], r) for i, r in rows], perm


def cmd_score(name, path):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")

    def encode(msgs):
        prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        if not full.startswith(prefix):
            return None
        p = tok(prefix, add_special_tokens=False).input_ids
        r = tok(full[len(prefix):], add_special_tokens=False).input_ids
        return (torch.tensor([p + r]), len(p)) if 1 < len(r) <= MAX_RESP and len(p) <= 1024 else None

    sel = []
    for idx, r in pairs()[0]:
        c = encode([{"role": m["role"], "content": m["content"]} for m in r["chosen"]])
        j = encode([{"role": m["role"], "content": m["content"]} for m in r["rejected"]])
        if c and j:
            sel.append((idx, c, j))
        if len(sel) == N:
            break
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def lp(ids, start):
        ids = ids.cuda()
        z = model(ids).logits[0, start - 1:-1].float()
        return F.log_softmax(z, -1).gather(1, ids[0, start:, None])[:, 0].mean().item()

    res = [{"idx": idx, "lc": lp(*c), "lr": lp(*j)} for idx, c, j in sel]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"name": name, "path": path, "pairs": res}, open(f"{OUT}/{name}.json", "w"))
    print(f"{name}: scored {len(res)} pairs")


def cmd_analyze(names):
    import torch
    cache = torch.load("/workspace/ref_cache/89203f41147b0c40.pt", map_location="cpu")
    R = {n: json.load(open(f"{OUT}/{n}.json"))["pairs"] for n in names}
    idx = [p["idx"] for p in R[names[0]]]
    assert all([p["idx"] for p in R[n]] == idx for n in names)
    t = lambda xs: torch.tensor(xs, dtype=torch.float64)
    m0c = (cache["chosen_logps"].double() - cache["rejected_logps"].double())[idx]   # what training used
    m0 = t([p["lc"] - p["lr"] for p in R[names[0]]])                                  # same, re-scored here
    print(f"{len(idx)} pairs; corr(SFT margin re-scored, training cache) {torch.corrcoef(torch.stack([m0, m0c]))[0, 1]:.4f}, "
          f"mean abs diff {(m0 - m0c).abs().mean():.4f}; SFT agrees {(m0 > 0).double().mean():.1%}")
    w0 = torch.sigmoid(-5 * m0c)
    sg = torch.sigmoid
    print(f"\n{'state':<22}{'agrees':>8}{'margin':>8}{'agreed slope':>13}"
          f"{'| weight share on pairs agreed NOW:':>36}{'plain':>7}{'static':>8}{'dynamic':>9}"
          f"{'| static push left':>19}{'| static on crossed':>20}")
    for n in names:
        m = t([p["lc"] - p["lr"] for p in R[n]])
        h = m - m0
        now = m > 0
        ag = m0 > 0
        X = torch.stack([m0[ag], torch.ones(int(ag.sum()), dtype=torch.float64)], 1)
        slope = torch.linalg.lstsq(X, m[ag][:, None]).solution[0, 0].item()
        eff = {"plain": sg(-5 * h), "static": w0 * sg(-5 * h), "dynamic": sg(-5 * m) * sg(-5 * h)}
        share = {k: (v[now].sum() / v.sum()).item() for k, v in eff.items()}
        left = (eff["static"].sum() / (w0 * 0.5).sum()).item()   # total static push relative to step 0
        crossed = (~ag) & now                                      # SFT disagreed, model now agrees
        on_crossed = (eff["static"][crossed].sum() / eff["static"].sum()).item()
        print(f"{n:<22}{now.double().mean():>8.1%}{m.mean():>+8.3f}{slope:>13.2f}{'':>36}{share['plain']:>7.3f}"
              f"{share['static']:>8.3f}{share['dynamic']:>9.3f}{left:>19.2f}{on_crossed:>20.3f}")
    print("\nweight share = sum of that method's per-pair gradient weight on pairs with margin > 0 under this state, / total."
          "\nstatic push left = total static gradient weight vs step 0. crossed = SFT disagreed, this state agrees.")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        cmd_score(sys.argv[2], sys.argv[3])
    else:
        cmd_analyze(sys.argv[2:])
