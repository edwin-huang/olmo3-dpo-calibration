"""Which tokens inside a DPO pair carry its push toward listed-option overconfidence?

Direction v: at Instruct-SFT, ∇conf on 1000 MMLU questions with its ∇acc part removed, unit norm
(same as direction_edit.py's MMLU direction). For every response token y_t, its log-prob's
directional derivative along v, by central finite differences with the model in fp32:
    d_t = [log π_{θ+εv}(y_t) − log π_{θ−εv}(y_t)] / 2ε
A pair's push = mean_t d_t(chosen) − mean_t d_t(rejected), the length-normalized DPO update's
push along v (attribution.py's push_conf, up to scale). Each token is stored with its SFT entropy,
SFT probability and position, so the push can be split by where it comes from.

Usage: python token_attribution.py --pairs-from results_attr/instruct_pairs.jsonl --n 2000
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution import SFT, encode_side, load_pairs
from calib_eval import letter_token_ids
from direction_edit import sources, unit_direction


@torch.no_grad()
def token_stats(model, ids, start):
    ids = ids.cuda()
    logits = model(ids, logits_to_keep=ids.shape[1] - start + 1).logits[0, :-1].float()
    lp = F.log_softmax(logits, -1)
    tgt = ids[0, start:]
    logp = lp.gather(1, tgt[:, None])[:, 0]
    ent = -(lp.exp() * lp).sum(-1)
    return logp.cpu(), ent.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-from", default="results_attr/instruct_pairs.jsonl")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.02, help="DPO moved ~0.021 along v")
    ap.add_argument("--out", default="results_tokens")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "instruct_tokens.jsonl")

    tok = AutoTokenizer.from_pretrained(SFT)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1. The direction, computed in bf16 (as in direction_edit.py), then kept on CPU.
    v_path = os.path.join(args.out, "v_mmlu.pt")
    if os.path.exists(v_path):
        v = torch.load(v_path)
    else:
        m = AutoModelForCausalLM.from_pretrained(SFT, dtype=torch.bfloat16, device_map="cuda")
        m.gradient_checkpointing_enable()
        m.config.use_cache = False
        m.train()
        srcs, _ = sources()
        rows, shots = srcs["mmlu"]
        v = [t.cpu() for t in unit_direction(m, list(m.parameters()), tok, rows, shots, letter_token_ids(tok))]
        torch.save(v, v_path)
        del m
        torch.cuda.empty_cache()

    # 2. fp32 model so that a perturbation of size eps along a unit direction is resolvable.
    model = AutoModelForCausalLM.from_pretrained(SFT, dtype=torch.float32, device_map="cuda").eval()
    params = list(model.parameters())
    v = [t.cuda().float() for t in v]

    def shift(scale):
        with torch.no_grad():
            for p, d in zip(params, v):
                p.add_(d, alpha=scale)

    # Same pairs as the gradient attribution, so the two estimates can be compared pair by pair.
    wanted = [json.loads(l)["idx"] for l in open(args.pairs_from)][: args.n]
    wanted_set = set(wanted)
    by_idx = {r["idx"]: r for r in load_pairs(10 ** 6, 10 ** 6) if r["idx"] in wanted_set}
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["idx"] for l in open(out_path)}
    todo = [i for i in wanted if i in by_idx and i not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    # Encode once; three passes over all sequences (θ, θ+εv, θ−εv) so weights move only twice.
    seqs = []
    for i in todo:
        r = by_idx[i]
        c, rj = encode_side(tok, r["chosen"]), encode_side(tok, r["rejected"])
        if c is not None and rj is not None:
            seqs.append((r, c, rj))
    for chunk in range(0, len(seqs), 250):
        part = seqs[chunk : chunk + 250]
        base = [(token_stats(model, *c), token_stats(model, *rj)) for _, c, rj in part]
        shift(args.eps)
        plus = [(token_stats(model, *c)[0], token_stats(model, *rj)[0]) for _, c, rj in part]
        shift(-2 * args.eps)
        minus = [(token_stats(model, *c)[0], token_stats(model, *rj)[0]) for _, c, rj in part]
        shift(args.eps)
        with open(out_path, "a") as f:
            for (r, c, rj), b, pl, mi in zip(part, base, plus, minus):
                rec = {"idx": r["idx"], "is_mc": r["is_mc"], "preference_type": r["preference_type"],
                       "prompt_id": r["prompt_id"]}
                for k, (seq, (logp, ent), lp_p, lp_m) in enumerate(
                        [(c, b[0], pl[0], mi[0]), (rj, b[1], pl[1], mi[1])]):
                    side = "chosen" if k == 0 else "rejected"
                    ids, start = seq
                    d = ((lp_p - lp_m) / (2 * args.eps)).tolist()
                    rec[side] = {"tokens": ids[0, start:].tolist(), "d": d,
                                 "logp": logp.tolist(), "ent": ent.tolist()}
                rec["push"] = (sum(rec["chosen"]["d"]) / len(rec["chosen"]["d"])
                               - sum(rec["rejected"]["d"]) / len(rec["rejected"]["d"]))
                f.write(json.dumps(rec) + "\n")
        print(f"{chunk + len(part)}/{len(seqs)}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
