"""Inference-only probes of HOW and WHY DPO sharpens listed-option answers (no training).

Models (A = before, B = after): Instruct-SFT vs Instruct-DPO by default; any two checkpoints via --a/--b.
All MMLU work uses the 5-shot letters prompt of calib_eval.py; the answer position is the last token, and the letter
logits are those of " A".." D".

  decomp   A1 + A2. Is B's change a uniform rescaling of the (centered) letter logits? How much of the final normalized
           residual lies in the letter-unembedding subspace? Direct logit attribution of the predicted-letter margin
           (letter logit minus the mean of the other three) to the embedding, every attention head and every MLP:
           margin = sum of contributions exactly, since Olmo 3 normalizes each block's output before adding it and the
           final RMSNorm's scale is fixed per example. Reports where B adds margin vs A.
  patch    A3. Activation patching, all on A's prompts: (i) residual stream at every position after layer l taken from
           B, A's layers continue; (ii) the reverse (A's residual into B); (iii) one attention or MLP block's output
           replaced by B's, all positions; (iv) the same, answer position only. Metric: letter-logit scale c relative
           to A (least squares on centered logits); share of the A->B effect = log c / log c_B.
  norm     A4. Centered letter logits = (U_g h) / rms(h) with U_g the letter readout in residual space. Splits B's
           letter-logit scale into a raw-signal factor (U_g h) and a normalization factor (final RMS), with the two
           counterfactuals; the part of g*h in W_U's effective null space (bottom --k-null singular directions); each
           block's share of ||h_L||^2; last-MLP neurons pointing into the null space ("entropy neurons", Stolfo et
           al. 2024) and their activation change.
  squeeze  C1. Teacher-forced on DPO training pairs (chosen and rejected responses) and on wikitext: per token, the
           change in p(target), p(A's top-1), entropy, and where the probability mass that leaves the target goes
           (A's top-1 vs rest). The squeezing effect of a negative gradient predicts mass flowing to A's top-1 on
           rejected responses beyond what plain sharpening of the whole distribution would give.
  firsttok C2. First response token for DPO training prompts (chat template): entropy and top-1 probability under A
           and B, by prompt type (option-listing / short chosen answer / long chosen answer), plus the best uniform
           sharpening s* (min KL(p_B || softmax(s z_A))) per type.

Usage: python why.py decomp --n 1000 ; python why.py patch --n 400 ; python why.py squeeze --n 300 ; python why.py firsttok
Writes results_why/<test>__<A>__<B>.json
"""
import argparse
import itertools
import json
import math
import os
import random
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from calib_eval import build_prompts, load_task

LETTERS = "ABCD"
OPT = re.compile(r"(?m)^\s*(?:\(?[A-Ea-e][\).:]|\(?[1-5][\).])\s+\S")


def load(name):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    return m


def short(name):
    return name.rstrip("/").split("/")[-1]


def mmlu(tok, n, seed=0):
    test, shots = load_task("mmlu")
    idx = list(range(len(test)))
    random.Random(seed).shuffle(idx)
    test = [test[i] for i in sorted(idx[:n])]
    return test, build_prompts(test, shots, 5, tok, chat=False)


def letter_ids(tok):
    ids = [tok.encode(f" {l}", add_special_tokens=False) for l in LETTERS]
    assert all(len(i) == 1 for i in ids)
    return torch.tensor([i[0] for i in ids], device="cuda")


def batches(tok, prompts, bs):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        ix = order[s : s + bs]
        yield ix, tok([prompts[i] for i in ix], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")


def centered(z):
    return z - z.mean(-1, keepdim=True)


def scale_fit(z, z_ref):
    """c minimizing ||centered(z) - c centered(z_ref)||^2 over all questions, and R^2."""
    a, b = centered(z).flatten(), centered(z_ref).flatten()
    c = (a @ b / (b @ b)).item()
    r2 = 1 - ((a - c * b) ** 2).sum().item() / ((a - a.mean()) ** 2).sum().item()
    return c, r2


# ---------------------------------------------------------------- A1 + A2


@torch.no_grad()
def run_decomp(model, tok, prompts, lid, pred=None, bs=8):
    """Letter logits, letter-subspace share of the final normalized residual, and per-component margin attribution."""
    L = len(model.model.layers)
    W_U = model.lm_head.weight[lid].float()  # [4, d]
    U = W_U - W_U.mean(0, keepdim=True)  # centered letter directions (rank 3)
    Q = torch.linalg.svd(U.T, full_matrices=False).U[:, :3]  # orthonormal basis of the letter subspace [d, 3]
    g_f = model.model.norm.weight.float()
    cap = {}
    hooks = []
    for l, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, a, l=l: cap.__setitem__(("heads_in", l), a[0][:, -1].float())))
        def attn_hook(m, a, o, l=l):  # must return None, or PyTorch replaces the module's output
            cap[("attn_pre", l)] = a[0][:, -1].float()
            cap[("attn", l)] = o[:, -1].float()
        hooks.append(layer.post_attention_layernorm.register_forward_hook(attn_hook))
        hooks.append(layer.post_feedforward_layernorm.register_forward_hook(
            lambda m, a, o, l=l: cap.__setitem__(("mlp", l), o[:, -1].float())))
    n_heads = model.config.num_attention_heads
    hd = model.config.hidden_size // n_heads
    W_O = [layer.self_attn.o_proj.weight.float() for layer in model.model.layers]  # [d, n_heads*hd]
    g_a = [layer.post_attention_layernorm.weight.float() for layer in model.model.layers]
    N = len(prompts)
    Z = torch.zeros(N, 4, device="cuda")
    share = torch.zeros(N, device="cuda")
    dla_emb = torch.zeros(N, device="cuda")
    dla_attn = torch.zeros(N, L, device="cuda")
    dla_mlp = torch.zeros(N, L, device="cuda")
    dla_head = torch.zeros(L, n_heads, device="cuda")  # summed over questions
    for ix, enc in batches(tok, prompts, bs):
        out = model(**enc, output_hidden_states=True)
        z = out.logits[:, -1].float()[:, lid]
        Z[ix] = z
        # exact pre-norm final residual: embedding + every block's (normalized) output
        h_L = (out.hidden_states[0][:, -1].float()
               + sum(cap[("attn", l)] + cap[("mlp", l)] for l in range(L)))
        rms = h_L.pow(2).mean(-1, keepdim=True).add(model.model.norm.variance_epsilon).sqrt()
        n_vec = g_f * h_L / rms
        share[ix] = (n_vec @ Q).norm(dim=-1) / n_vec.norm(dim=-1)  # fraction of the readout vector in the letter subspace
        p = (pred[ix] if pred is not None else z.argmax(-1))
        u = U[p]  # [b, d] centered direction of the predicted letter (= margin direction x 4/3)
        u = u * (4 / 3)  # u . logits = z_pred - mean(z_others)
        v = u * g_f / rms  # readout vector for a residual contribution
        dla_emb[ix] = (v * out.hidden_states[0][:, -1].float()).sum(-1)
        for l in range(L):
            dla_attn[ix, l] = (v * cap[("attn", l)]).sum(-1)
            dla_mlp[ix, l] = (v * cap[("mlp", l)]).sum(-1)
            x = cap[("attn_pre", l)]
            w = v * g_a[l] / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()  # through post-attention norm
            per = (w @ W_O[l]) * cap[("heads_in", l)]  # [b, n_heads*hd]
            dla_head[l] += per.view(len(ix), n_heads, hd).sum(-1).sum(0)
        cap.clear()
    for h in hooks:
        h.remove()
    pz = Z.gather(1, (pred if pred is not None else Z.argmax(-1))[:, None])[:, 0]
    margin = pz - (Z.sum(-1) - pz) / 3
    total = dla_emb + dla_attn.sum(1) + dla_mlp.sum(1)
    return {"Z": Z, "share": share, "margin": margin, "dla_total": total, "dla_emb": dla_emb,
            "dla_attn": dla_attn, "dla_mlp": dla_mlp, "dla_head": dla_head / len(prompts)}


def cmd_decomp(args):
    tok = AutoTokenizer.from_pretrained(args.a)
    tok.padding_side = "left"
    test, prompts = mmlu(tok, args.n)
    lid = letter_ids(tok)
    res = {}
    pred = None
    for tag, name in (("A", args.a), ("B", args.b)):
        m = load(name)
        r = run_decomp(m, tok, prompts, lid, pred=pred)
        if pred is None:
            pred = r["Z"].argmax(-1)  # A's prediction defines the margin direction for both models
        res[tag] = r
        del m
        torch.cuda.empty_cache()
    A, B = res["A"], res["B"]
    c, r2 = scale_fit(B["Z"], A["Z"])
    same = (A["Z"].argmax(-1) == B["Z"].argmax(-1)).float().mean().item()
    L = A["dla_attn"].shape[1]
    d_attn = (B["dla_attn"] - A["dla_attn"]).mean(0)
    d_mlp = (B["dla_mlp"] - A["dla_mlp"]).mean(0)
    d_emb = (B["dla_emb"] - A["dla_emb"]).mean().item()
    d_margin = (B["margin"] - A["margin"]).mean().item()
    out = {
        "a": args.a, "b": args.b, "n": len(prompts), "same_pred": same,
        "letter_logit_scale_c": c, "scale_fit_r2": r2,
        "margin_A": A["margin"].mean().item(), "margin_B": B["margin"].mean().item(),
        "dla_check_max_abs_err": max((r["dla_total"] - r["margin"]).abs().max().item() for r in (A, B)),
        "letter_share_A": A["share"].mean().item(), "letter_share_B": B["share"].mean().item(),
        "d_margin": d_margin, "d_emb": d_emb,
        "d_attn_by_layer": d_attn.tolist(), "d_mlp_by_layer": d_mlp.tolist(),
        "attn_by_layer_A": A["dla_attn"].mean(0).tolist(), "mlp_by_layer_A": A["dla_mlp"].mean(0).tolist(),
    }
    dh = (B["dla_head"] - A["dla_head"])
    top = torch.topk(dh.flatten().abs(), 20).indices.tolist()
    out["top_heads_d"] = [(i // dh.shape[1], i % dh.shape[1], dh.flatten()[i].item(),
                           A["dla_head"].flatten()[i].item()) for i in top]
    save(args, "decomp", out)
    print(f"same prediction {same:.1%}; letter-logit scale B/A c = {c:.3f} (R2 {r2:.3f}); margin {out['margin_A']:.2f} -> "
          f"{out['margin_B']:.2f}; DLA check err {out['dla_check_max_abs_err']:.2e}")
    print(f"share of final normalized residual in the letter subspace: A {out['letter_share_A']:.4f}  B {out['letter_share_B']:.4f}")
    print(f"margin change {d_margin:+.3f} = embedding {d_emb:+.3f} + attention {d_attn.sum().item():+.3f} + MLP {d_mlp.sum().item():+.3f}")
    print("by layer (B - A):  " + "  ".join(f"L{l}:{d_attn[l].item() + d_mlp[l].item():+.2f}" for l in range(L)))
    print("by layer attn:     " + "  ".join(f"{d_attn[l].item():+.2f}" for l in range(L)))
    print("by layer mlp:      " + "  ".join(f"{d_mlp[l].item():+.2f}" for l in range(L)))
    print("top heads by |change| (layer, head, change, A's value): " +
          ", ".join(f"L{l}H{h} {d:+.2f} ({a:+.2f})" for l, h, d, a in out["top_heads_d"][:12]))


# ---------------------------------------------------------------- A3


@torch.no_grad()
def letter_logits(model, tok, prompts, lid, bs=8, hook=None):
    Z = torch.zeros(len(prompts), 4, device="cuda")
    for ix, enc in batches(tok, prompts, bs):
        if hook is not None:
            hook["ix"] = ix
        Z[ix] = model(**enc).logits[:, -1].float()[:, lid]
    return Z


def capture(model, enc):
    """Forward pass returning letter-position logits and every layer's residual and block outputs (all positions)."""
    cur, hooks = {}, []
    for l, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, l=l: cur.__setitem__(("res", l), o.detach())))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            lambda m, a, o, l=l: cur.__setitem__(("attn", l), o.detach())))
        hooks.append(layer.post_feedforward_layernorm.register_forward_hook(
            lambda m, a, o, l=l: cur.__setitem__(("mlp", l), o.detach())))
    logits = model(**enc).logits[:, -1].float()
    for h in hooks:
        h.remove()
    return logits, cur


def patched_logits(model, enc, kind, l, src, last_only=False):
    mod = {"res": model.model.layers[l], "attn": model.model.layers[l].post_attention_layernorm,
           "mlp": model.model.layers[l].post_feedforward_layernorm}[kind]
    def repl(m, a, o):
        if last_only:
            o = o.clone()
            o[:, -1] = src[:, -1]
            return o
        return src
    h = mod.register_forward_hook(repl)
    try:
        return model(**enc).logits[:, -1].float()
    finally:
        h.remove()


@torch.no_grad()
def cmd_patch(args):
    tok = AutoTokenizer.from_pretrained(args.a)
    tok.padding_side = "left"
    test, prompts = mmlu(tok, args.n)
    lid = letter_ids(tok)
    ma, mb = load(args.a), load(args.b)
    L = len(ma.model.layers)
    N = len(prompts)
    variants = [(v, l) for l in range(L) for v in ("res_B_into_A", "res_A_into_B", "attn_all", "mlp_all", "attn_last", "mlp_last")]
    Zv = {k: torch.zeros(N, 4, device="cuda") for k in variants}
    ZA, ZB = torch.zeros(N, 4, device="cuda"), torch.zeros(N, 4, device="cuda")
    for bi, (ix, enc) in enumerate(batches(tok, prompts, 8)):  # batch-major: B's and A's activations stay on the GPU
        za, capA = capture(ma, enc)
        zb, capB = capture(mb, enc)
        ZA[ix], ZB[ix] = za[:, lid], zb[:, lid]
        for (v, l) in variants:
            if v == "res_B_into_A":
                z = patched_logits(ma, enc, "res", l, capB[("res", l)])
            elif v == "res_A_into_B":
                z = patched_logits(mb, enc, "res", l, capA[("res", l)])
            else:
                kind, where = v.split("_")
                z = patched_logits(ma, enc, kind, l, capB[(kind, l)], last_only=(where == "last"))
            Zv[(v, l)][ix] = z[:, lid]
        del capA, capB
        if bi % 10 == 0:
            print(f"batch {bi}", flush=True)
    cB, _ = scale_fit(ZB, ZA)
    lB = math.log(cB)
    share = {k: math.log(max(scale_fit(Z, ZA)[0], 1e-6)) / lB for k, Z in Zv.items()}
    out = {"a": args.a, "b": args.b, "n": N, "scale_B": cB}
    for v in ("res_B_into_A", "res_A_into_B", "attn_all", "mlp_all", "attn_last", "mlp_last"):
        out[v] = [share[(v, l)] for l in range(L)]
    print(f"B's letter-logit scale vs A = {cB:.3f}. Share of the A->B log-scale change, by layer l:")
    print("  res B->A: B's residual after layer l, A's layers above.  res A->B: A's residual into B (1 - share = B's upper layers' part).")
    for l in range(L):
        print(f"L{l:>2}: res B->A {out['res_B_into_A'][l]:+.2f}  res A->B {out['res_A_into_B'][l]:+.2f} | attn all "
              f"{out['attn_all'][l]:+.2f} last {out['attn_last'][l]:+.2f} | mlp all {out['mlp_all'][l]:+.2f} last {out['mlp_last'][l]:+.2f}")
    save(args, "patch", out)


# ---------------------------------------------------------------- A4: signal vs normalization


@torch.no_grad()
def norm_stats(model, tok, prompts, lid, k_null, bs=8):
    """At the answer position: unnormalized letter signal U_g h (centered), final RMS, the part of g*h in W_U's
    effective null space, each block's share of ||h_L||^2 (h_L . c / ||h_L||^2), and the last MLP's neuron activations."""
    L = len(model.model.layers)
    g_f = model.model.norm.weight.float()
    W_U = model.lm_head.weight.float()
    U = W_U[lid] - W_U[lid].mean(0, keepdim=True)
    Ug = U * g_f  # letter readout in residual space: centered letter logits = (Ug @ h) / rms(h)
    Vt = torch.linalg.svd(W_U, full_matrices=False).Vh  # right singular vectors, descending singular values
    null = Vt[-k_null:]  # [k, d]: directions the unembedding barely reads
    cap = {}
    hooks = []
    for l, layer in enumerate(model.model.layers):
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            lambda m, a, o, l=l: cap.__setitem__(("attn", l), o[:, -1].float())))
        hooks.append(layer.post_feedforward_layernorm.register_forward_hook(
            lambda m, a, o, l=l: cap.__setitem__(("mlp", l), o[:, -1].float())))
    hooks.append(model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(
        lambda m, a: cap.__setitem__("neurons", a[0][:, -1].float())))
    N = len(prompts)
    raw = torch.zeros(N, 4, device="cuda"); rms = torch.zeros(N, device="cuda"); nullfrac = torch.zeros(N, device="cuda")
    share_attn = torch.zeros(N, L, device="cuda"); share_mlp = torch.zeros(N, L, device="cuda"); share_emb = torch.zeros(N, device="cuda")
    neurons = None
    for ix, enc in batches(tok, prompts, bs):
        out = model(**enc, output_hidden_states=True)
        emb = out.hidden_states[0][:, -1].float()
        h = emb + sum(cap[("attn", l)] + cap[("mlp", l)] for l in range(L))
        raw[ix] = h @ Ug.T
        rms[ix] = h.pow(2).mean(-1).add(model.model.norm.variance_epsilon).sqrt()
        gh = g_f * h
        nullfrac[ix] = (gh @ null.T).norm(dim=-1) / gh.norm(dim=-1)
        hh = (h * h).sum(-1)
        share_emb[ix] = (h * emb).sum(-1) / hh
        for l in range(L):
            share_attn[ix, l] = (h * cap[("attn", l)]).sum(-1) / hh
            share_mlp[ix, l] = (h * cap[("mlp", l)]).sum(-1) / hh
        if neurons is None:
            neurons = torch.zeros(N, cap["neurons"].shape[-1], device="cuda")
        neurons[ix] = cap["neurons"]
        cap.clear()
    for h_ in hooks:
        h_.remove()
    return {"raw": raw, "rms": rms, "nullfrac": nullfrac, "share_emb": share_emb, "share_attn": share_attn,
            "share_mlp": share_mlp, "neurons": neurons}


def cmd_norm(args):
    tok = AutoTokenizer.from_pretrained(args.a)
    tok.padding_side = "left"
    test, prompts = mmlu(tok, args.n)
    lid = letter_ids(tok)
    res = {}
    for tag, name in (("A", args.a), ("B", args.b)):
        m = load(name)
        res[tag] = norm_stats(m, tok, prompts, lid, args.k_null)
        if tag == "A":  # neuron output directions of the last MLP, in the final readout space
            W_down = m.model.layers[-1].mlp.down_proj.weight.float()  # [d, n_neurons]
            g_ff = m.model.layers[-1].post_feedforward_layernorm.weight.float()
            g_f = m.model.norm.weight.float()
            Vt = torch.linalg.svd(m.lm_head.weight.float(), full_matrices=False).Vh
            out_dirs = (g_f * g_ff)[:, None] * W_down  # how each neuron's output reaches the readout (up to norms)
            nf = (Vt[-args.k_null:] @ out_dirs).norm(dim=0) / out_dirs.norm(dim=0)
            wnorm = W_down.norm(dim=0)
        del m
        torch.cuda.empty_cache()
    A, B = res["A"], res["B"]
    c_total, _ = scale_fit(B["raw"] / B["rms"][:, None], A["raw"] / A["rms"][:, None])
    c_raw, r2_raw = scale_fit(B["raw"], A["raw"])
    c_norm = (A["rms"] / B["rms"]).mean().item()
    c_cf_norm, _ = scale_fit(A["raw"] / B["rms"][:, None], A["raw"] / A["rms"][:, None])  # SFT signal, DPO's RMS
    c_cf_sig, _ = scale_fit(B["raw"] / A["rms"][:, None], A["raw"] / A["rms"][:, None])   # DPO signal, SFT's RMS
    d_share_attn = (B["share_attn"] - A["share_attn"]).mean(0)
    d_share_mlp = (B["share_mlp"] - A["share_mlp"]).mean(0)
    dn = (B["neurons"] - A["neurons"]).mean(0) * wnorm
    top_nf = torch.topk(nf, 50).indices  # most null-space-pointing neurons
    rows = sorted(top_nf.tolist(), key=lambda j: -abs(dn[j].item()))[:12]
    out = {"a": args.a, "b": args.b, "n": len(prompts), "k_null": args.k_null,
           "letter_scale_total": c_total, "raw_signal_scale": c_raw, "raw_signal_r2": r2_raw,
           "rms_ratio_A_over_B": c_norm, "counterfactual_sft_signal_dpo_rms": c_cf_norm,
           "counterfactual_dpo_signal_sft_rms": c_cf_sig,
           "rms_A": A["rms"].mean().item(), "rms_B": B["rms"].mean().item(),
           "nullfrac_A": A["nullfrac"].mean().item(), "nullfrac_B": B["nullfrac"].mean().item(),
           "d_norm_share_attn": d_share_attn.tolist(), "d_norm_share_mlp": d_share_mlp.tolist(),
           "norm_share_mlp_A": A["share_mlp"].mean(0).tolist(),
           "null_neurons": [(j, nf[j].item(), A["neurons"][:, j].mean().item(), B["neurons"][:, j].mean().item(), dn[j].item())
                            for j in rows]}
    save(args, "norm", out)
    print(f"letter-logit scale B/A {c_total:.3f} = raw signal x{c_raw:.3f} (R2 {r2_raw:.3f}) and RMS ratio x{c_norm:.3f}")
    print(f"counterfactuals: SFT's signal with DPO's RMS -> x{c_cf_norm:.3f}; DPO's signal with SFT's RMS -> x{c_cf_sig:.3f}")
    print(f"final RMS {out['rms_A']:.2f} -> {out['rms_B']:.2f}; null-space fraction (bottom {args.k_null} of W_U) "
          f"{out['nullfrac_A']:.3f} -> {out['nullfrac_B']:.3f}")
    print("change in each block's share of ||h_L||^2 (B - A), layers 16-31:")
    print("  attn: " + " ".join(f"L{l}:{d_share_attn[l].item():+.3f}" for l in range(16, 32)))
    print("  mlp:  " + " ".join(f"L{l}:{d_share_mlp[l].item():+.3f}" for l in range(16, 32)))
    print(f"  (A's mlp share at L31 {A['share_mlp'][:, -1].mean().item():+.3f}, L30 {A['share_mlp'][:, -2].mean().item():+.3f})")
    print("last-MLP neurons pointing most into W_U's null space, by |activation change x ||w_out|||: " +
          ", ".join(f"n{j} nf {f:.2f} act {a:+.3f}->{b:+.3f}" for j, f, a, b, d in out["null_neurons"][:8]))


# ---------------------------------------------------------------- A5: causal test of the last-MLP null-space neurons


@torch.no_grad()
def last_mlp_acts(model, tok, prompts, bs):
    acts = {}
    def grab(m, a):
        acts["x"] = a[0][:, -1].float()
    h = model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(grab)
    out = None
    for ix, enc in batches(tok, prompts, bs):
        model(**enc, logits_to_keep=1)
        if out is None:
            out = torch.zeros(len(prompts), acts["x"].shape[-1], device="cuda")
        out[ix] = acts["x"]
    h.remove()
    return out


@torch.no_grad()
def logits_with_neurons(model, tok, prompts, lid, neurons, values, bs):
    """Letter logits with the given last-MLP neurons' answer-position activations set to `values` [N, k]."""
    state = {}
    idx = torch.tensor(neurons, device="cuda")
    def setv(m, a):
        x = a[0].clone()
        x[:, -1, idx] = values[state["ix"]].to(x.dtype)
        return (x,)
    h = model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(setv)
    Z = torch.zeros(len(prompts), 4, device="cuda")
    for ix, enc in batches(tok, prompts, bs):
        state["ix"] = torch.tensor(ix, device="cuda")
        Z[ix] = model(**enc, logits_to_keep=1).logits[:, -1].float()[:, lid]
    h.remove()
    return Z


def calib_from_logits(Z, test):
    from temp_scaling import ece, fit_t
    p = F.softmax(Z, -1)
    rows = [{"probs": r.tolist(), "answer": t["answer"]} for r, t in zip(p, test)]
    conf = [max(r["probs"]) for r in rows]
    corr = [int(r["probs"].index(max(r["probs"])) == r["answer"]) for r in rows]
    return {"acc": sum(corr) / len(corr), "conf": sum(conf) / len(conf), "ece": ece(conf, corr), "t_star": fit_t(rows)}


def cmd_neurons(args):
    """Swap the answer-position activations of selected last-MLP neurons between A and B. Selections: the k neurons
    pointing most into W_U's null space ranked by |activation change x ||w_out|||, for several k, and the same number
    of neurons with the largest change but LOW null-space fraction (control)."""
    if args.mem_frac:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    tok = AutoTokenizer.from_pretrained(args.a)
    tok.padding_side = "left"
    test, prompts = mmlu(tok, args.n)
    lid = letter_ids(tok)
    m = load(args.b)
    actB = last_mlp_acts(m, tok, prompts, args.bs)
    ZB = letter_logits(m, tok, prompts, lid, bs=args.bs)
    del m; torch.cuda.empty_cache()
    m = load(args.a)
    actA = last_mlp_acts(m, tok, prompts, args.bs)
    ZA = letter_logits(m, tok, prompts, lid, bs=args.bs)
    W_down = m.model.layers[-1].mlp.down_proj.weight.float()
    g = m.model.norm.weight.float() * m.model.layers[-1].post_feedforward_layernorm.weight.float()
    Vt = torch.linalg.svd(m.lm_head.weight.float(), full_matrices=False).Vh
    out_dirs = g[:, None] * W_down
    nf = (Vt[-args.k_null:] @ out_dirs).norm(dim=0) / out_dirs.norm(dim=0)
    dn = ((actB - actA).mean(0).abs() * W_down.norm(dim=0))
    hi = [j for j in torch.argsort(dn, descending=True).tolist() if nf[j] >= 0.6]
    lo = [j for j in torch.argsort(dn, descending=True).tolist() if nf[j] < 0.3]
    cB, _ = scale_fit(ZB, ZA)
    res = {"a": args.a, "b": args.b, "n": len(prompts), "scale_B": cB, "A": calib_from_logits(ZA, test),
           "B": calib_from_logits(ZB, test), "runs": []}
    print(f"A: {res['A']}\nB: {res['B']}  (letter scale B/A {cB:.3f})")
    for label, pool in (("null", hi), ("low-null control", lo)):
        for k in args.ks:
            sel = pool[:k]
            Z = logits_with_neurons(m, tok, prompts, lid, sel, actB[:, sel], args.bs)  # A with B's values
            c, _ = scale_fit(Z, ZA)
            r = {"set": label, "k": k, "neurons": sel, "A_with_B_values_share": math.log(c) / math.log(cB),
                 "A_with_B_values": calib_from_logits(Z, test)}
            res["runs"].append(r)
            print(f"{label:<17} k={k:<3} A with B's values: share of log-scale {r['A_with_B_values_share']:+.2f}  "
                  f"T* {r['A_with_B_values']['t_star']:.2f}  ECE {r['A_with_B_values']['ece']:.3f}", flush=True)
    del m; torch.cuda.empty_cache()
    m = load(args.b)
    for r in res["runs"]:
        sel = r["neurons"]
        Z = logits_with_neurons(m, tok, prompts, lid, sel, actA[:, sel], args.bs)  # B with A's values
        c, _ = scale_fit(Z, ZA)
        r["B_with_A_values_share"] = math.log(c) / math.log(cB)
        r["B_with_A_values"] = calib_from_logits(Z, test)
        print(f"{r['set']:<17} k={r['k']:<3} B with A's values: share left {r['B_with_A_values_share']:+.2f}  T* "
              f"{r['B_with_A_values']['t_star']:.2f}  ECE {r['B_with_A_values']['ece']:.3f}  acc {r['B_with_A_values']['acc']:.3f}", flush=True)
    save(args, "neurons", res)


# ---------------------------------------------------------------- C3: DPO pairs under A and B


@torch.no_grad()
def response_logps(model, ids, start):
    """Mean per-token log-prob of the response, plus logits at up to 8 random response positions (for s*)."""
    ids = ids.cuda()
    z = model(ids).logits[0, start - 1 : -1].float()
    lp = F.log_softmax(z, -1).gather(1, ids[0, start:, None])[:, 0]
    keep = torch.randperm(z.shape[0], generator=torch.Generator().manual_seed(0))[:8].to(z.device)
    return lp.mean().item(), z[keep].cpu()


def cmd_pairs(args):
    """For DPO training pairs: SFT-style margin (mean log-prob chosen - rejected, as in dpo_norm) under A and B.
    Is B's margin a rescaled copy of A's (gain) or a shift? How often does A already agree (margin > 0)? And the
    token-level sharpening s* of B vs A on chosen responses, split by whether A agrees with the pair."""
    import pyarrow.parquet as pq
    src = {"instruct": "/workspace/dpo_data/q25_all.parquet", "think": "/workspace/dpo_data/think_full.parquet"}[args.data]
    rows = pq.read_table(src).slice(0, 30000).to_pylist()
    rows = [r for r in rows if len(r["chosen"]) == 2]
    random.Random(7).shuffle(rows)
    rows = rows[: args.n]
    tok = AutoTokenizer.from_pretrained(args.a)
    enc = []
    for r in rows:
        c, j = encode(tok, r["chosen"], args.max_resp), encode(tok, r["rejected"], args.max_resp)
        if c and j:
            enc.append((c, j))
    out = {}
    for tag, name in (("A", args.a), ("B", args.b)):
        m = load(name)
        lc, lr, zc = [], [], []
        for (c, j) in enc:
            a, z = response_logps(m, *c)
            b, _ = response_logps(m, *j)
            lc.append(a); lr.append(b); zc.append(z)
        out[tag] = {"lc": lc, "lr": lr, "zc": zc}
        del m
        torch.cuda.empty_cache()
    A, B = out["A"], out["B"]
    mA = torch.tensor([c - r for c, r in zip(A["lc"], A["lr"])])
    mB = torch.tensor([c - r for c, r in zip(B["lc"], B["lr"])])
    dc = torch.tensor(B["lc"]) - torch.tensor(A["lc"])
    dr = torch.tensor(B["lr"]) - torch.tensor(A["lr"])

    def fit(x, y):
        X = torch.stack([x, torch.ones_like(x)], 1)
        sol = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
        r2 = 1 - ((y - X @ sol) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return sol[0].item(), sol[1].item(), r2.item()

    res = {"a": args.a, "b": args.b, "data": args.data, "n": len(enc), "agree_A": (mA > 0).float().mean().item(),
           "agree_B": (mB > 0).float().mean().item(), "margin_A_mean": mA.mean().item(), "margin_B_mean": mB.mean().item()}
    res["fit_all"] = fit(mA, mB)
    print(f"{args.data}: {len(enc)} pairs; A agrees with {res['agree_A']:.0%} (mean margin {res['margin_A_mean']:+.3f}), "
          f"B agrees with {res['agree_B']:.0%} (mean margin {res['margin_B_mean']:+.3f})")
    print(f"margin_B = {res['fit_all'][0]:.2f} x margin_A {res['fit_all'][1]:+.3f}  (R2 {res['fit_all'][2]:.2f})")
    for label, sel in (("A agrees (margin>0)", mA > 0), ("A disagrees (margin<0)", mA <= 0)):
        f = fit(mA[sel], mB[sel])
        za = torch.cat([z for z, s in zip(A["zc"], sel.tolist()) if s]).cuda()
        zb = torch.cat([z for z, s in zip(B["zc"], sel.tolist()) if s]).cuda()
        pb = F.softmax(zb, -1)
        s_star = min((((pb * (F.log_softmax(zb, -1) - F.log_softmax(sc * za, -1))).sum(-1).mean().item()), sc)
                     for sc in [0.7 + 0.025 * i for i in range(41)])[1]
        g = {"n": int(sel.sum()), "fit": f, "d_logp_chosen": dc[sel].mean().item(), "d_logp_rejected": dr[sel].mean().item(),
             "margin_change": (mB - mA)[sel].mean().item(), "s_star_chosen_tokens": s_star}
        res[label] = g
        print(f"{label:<24} n {g['n']:>5}: margin_B = {f[0]:.2f} x margin_A {f[1]:+.3f} (R2 {f[2]:.2f}) | d logp chosen "
              f"{g['d_logp_chosen']:+.3f} rejected {g['d_logp_rejected']:+.3f} | margin change {g['margin_change']:+.3f} | "
              f"token s* on chosen {s_star:.3f}")
    edges = torch.quantile(mA, torch.linspace(0, 1, 11))
    res["by_decile"] = []
    for k in range(10):
        sel = (mA >= edges[k]) & (mA <= edges[k + 1])
        res["by_decile"].append((edges[k].item(), edges[k + 1].item(), mA[sel].mean().item(), (mB - mA)[sel].mean().item(),
                                 dc[sel].mean().item(), dr[sel].mean().item()))
    print("by A-margin decile (A margin -> change in margin; d chosen; d rejected): " +
          "  ".join(f"{a:+.2f}->{d:+.2f} ({c:+.2f};{r:+.2f})" for _, _, a, d, c, r in res["by_decile"]))
    # per-pair values, for follow-up analyses (e.g. response length x agreement)
    res["per_pair"] = [{"lcA": A["lc"][i], "lrA": A["lr"][i], "lcB": B["lc"][i], "lrB": B["lr"][i],
                        "len_c": enc[i][0][0].shape[1] - enc[i][0][1], "len_r": enc[i][1][0].shape[1] - enc[i][1][1]}
                       for i in range(len(enc))]
    save(args, f"pairs_{args.data}", res)


# ---------------------------------------------------------------- A6: the same circuit at commit points in training text?


@torch.no_grad()
def run_dla(model, tok, prompts, ab, bs=8):
    """Direct logit attribution of margin z[a] - z[b] at the last position of each prompt (left-padded batches):
    embedding, each attention block, each MLP block, each attention head. Same exact decomposition as run_decomp."""
    L = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    hd = model.config.hidden_size // n_heads
    W_U = model.lm_head.weight.float()
    g_f = model.model.norm.weight.float()
    W_O = [layer.self_attn.o_proj.weight.float() for layer in model.model.layers]
    g_a = [layer.post_attention_layernorm.weight.float() for layer in model.model.layers]
    cap, hooks = {}, []
    for l, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, a, l=l: cap.__setitem__(("heads_in", l), a[0][:, -1].float())))
        def attn_hook(m, a, o, l=l):
            cap[("attn_pre", l)] = a[0][:, -1].float()
            cap[("attn", l)] = o[:, -1].float()
        hooks.append(layer.post_attention_layernorm.register_forward_hook(attn_hook))
        hooks.append(layer.post_feedforward_layernorm.register_forward_hook(
            lambda m, a, o, l=l: cap.__setitem__(("mlp", l), o[:, -1].float())))
    N = len(prompts)
    margin = torch.zeros(N, device="cuda")
    attn = torch.zeros(N, L, device="cuda"); mlp = torch.zeros(N, L, device="cuda")
    heads = torch.zeros(N, L, n_heads, device="cuda")
    abt = torch.tensor(ab, device="cuda")
    for ix, enc in batches(tok, prompts, bs):
        out = model(**enc, output_hidden_states=True)
        z = out.logits[:, -1].float()
        a, b = abt[ix, 0], abt[ix, 1]
        margin[ix] = z.gather(1, a[:, None])[:, 0] - z.gather(1, b[:, None])[:, 0]
        h_L = out.hidden_states[0][:, -1].float() + sum(cap[("attn", l)] + cap[("mlp", l)] for l in range(L))
        rms = h_L.pow(2).mean(-1, keepdim=True).add(model.model.norm.variance_epsilon).sqrt()
        v = (W_U[a] - W_U[b]) * g_f / rms
        for l in range(L):
            attn[ix, l] = (v * cap[("attn", l)]).sum(-1)
            mlp[ix, l] = (v * cap[("mlp", l)]).sum(-1)
            w = v * g_a[l] / cap[("attn_pre", l)].pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            heads[ix, l] = ((w @ W_O[l]) * cap[("heads_in", l)]).view(len(ix), n_heads, hd).sum(-1)
        cap.clear()
    for h in hooks:
        h.remove()
    return {"margin": margin, "attn": attn, "mlp": mlp, "heads": heads}


def cmd_bridge(args):
    """Per-head / per-block change (B - A) in the direct contribution to A's top-1-vs-top-2 margin, in three contexts:
    MCQ letters (5-shot MMLU, top-2 among A-D), commit points in DPO training data (first response token of chosen
    responses <= 20 tokens, chat template), and ordinary text (wikitext, random cut points). Correlates the head-change
    profiles across contexts: if commit points share MCQ's profile and wikitext does not, the circuit DPO amplifies is a
    general 'commit to an answer' readout."""
    import pyarrow.parquet as pq
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.a)
    tok.padding_side = "left"
    n = args.n
    _, mmlu_prompts = mmlu(tok, n)
    lid = letter_ids(tok)
    rows = pq.read_table("/workspace/dpo_data/q25_all.parquet").slice(30000, 30000).to_pylist()
    commit = []
    for r in rows:
        if len(r["chosen"]) != 2:
            continue
        ans = r["chosen"][-1]["content"] or ""
        if 0 < len(tok(ans, add_special_tokens=False).input_ids) <= 20:
            p = tok.apply_chat_template([{"role": "user", "content": r["chosen"][0]["content"]}], tokenize=False,
                                        add_generation_prompt=True)
            if len(tok(p, add_special_tokens=False).input_ids) <= 1500:
                commit.append(p)
        if len(commit) >= n:
            break
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ids = tok("\n".join(t for t in wiki["text"] if t.strip()), add_special_tokens=False).input_ids
    rng = random.Random(3)
    text = []
    for i in range(n):
        start = rng.randrange(0, len(ids) - 600)
        text.append(tok.decode(ids[start : start + rng.randrange(100, 500)]))
    contexts = {"mcq_letters": mmlu_prompts, "commit_points": commit, "wikitext": text}
    ma = load(args.a)
    ab = {}
    with torch.no_grad():  # A's top-1 and top-2 (MCQ: among the letters)
        for name, prompts in contexts.items():
            Z = torch.zeros(len(prompts), 2, dtype=torch.long)
            for ix, enc in batches(tok, prompts, 8):
                z = ma(**enc, logits_to_keep=1).logits[:, -1].float()
                if name == "mcq_letters":
                    top = lid[z[:, lid].topk(2, -1).indices]
                else:
                    top = z.topk(2, -1).indices
                Z[ix] = top.cpu()
            ab[name] = Z.tolist()
    resA = {name: run_dla(ma, tok, prompts, ab[name]) for name, prompts in contexts.items()}
    del ma; torch.cuda.empty_cache()
    mb = load(args.b)
    resB = {name: run_dla(mb, tok, prompts, ab[name]) for name, prompts in contexts.items()}
    out = {"a": args.a, "b": args.b, "n": {k: len(v) for k, v in contexts.items()}}
    dheads = {}
    print(f"{'context':<14}{'n':>5}{'margin A':>10}{'margin B':>10}{'ratio':>7}   attn d / mlp d by layers 16-31")
    for name in contexts:
        A, B = resA[name], resB[name]
        mA, mB = A["margin"].mean().item(), B["margin"].mean().item()
        da, dm = (B["attn"] - A["attn"]).mean(0), (B["mlp"] - A["mlp"]).mean(0)
        dheads[name] = (B["heads"] - A["heads"]).mean(0).flatten()
        out[name] = {"margin_A": mA, "margin_B": mB, "d_attn": da.tolist(), "d_mlp": dm.tolist(),
                     "heads_A": A["heads"].mean(0).flatten().tolist(), "d_heads": dheads[name].tolist()}
        top = torch.topk(dheads[name].abs(), 6).indices.tolist()
        print(f"{name:<14}{len(contexts[name]):>5}{mA:>10.3f}{mB:>10.3f}{mB / mA:>7.2f}   attn: " +
              " ".join(f"{da[l].item():+.2f}" for l in range(16, 32)))
        print(f"{'':<46}mlp:  " + " ".join(f"{dm[l].item():+.2f}" for l in range(16, 32)))
        print(f"{'':<46}top heads: " + ", ".join(f"L{i // 32}H{i % 32} {dheads[name][i].item():+.3f}" for i in top))
    names = list(contexts)
    out["head_change_corr"] = {}
    for i in range(3):
        for j in range(i + 1, 3):
            r = torch.corrcoef(torch.stack([dheads[names[i]], dheads[names[j]]]))[0, 1].item()
            out["head_change_corr"][f"{names[i]}~{names[j]}"] = r
            print(f"corr of per-head change: {names[i]} ~ {names[j]} = {r:+.2f}")
    save(args, "bridge", out)


# ---------------------------------------------------------------- C1


def dpo_pairs(n, seed=0):
    import pyarrow.parquet as pq
    rows = pq.read_table("/workspace/dpo_data/q25_all.parquet").slice(0, 20000).to_pylist()
    rows = [r for r in rows if len(r["chosen"]) == 2]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def encode(tok, msgs, max_resp):
    msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
    prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    if not full.startswith(prefix):
        return None
    p = tok(prefix, add_special_tokens=False).input_ids[-1024:]
    r = tok(full[len(prefix):], add_special_tokens=False).input_ids[:max_resp]
    return (torch.tensor([p + r]), len(p)) if len(r) > 1 else None


@torch.no_grad()
def token_stats(ma, mb, ids, start):
    ids = ids.cuda()
    za = ma(ids).logits[0, start - 1 : -1].float()
    zb = mb(ids).logits[0, start - 1 : -1].float()
    y = ids[0, start:]
    la, lb = F.log_softmax(za, -1), F.log_softmax(zb, -1)
    pa, pb = la.exp(), lb.exp()
    top = pa.argmax(-1)
    py_a, py_b = pa.gather(1, y[:, None])[:, 0], pb.gather(1, y[:, None])[:, 0]
    pt_a, pt_b = pa.gather(1, top[:, None])[:, 0], pb.gather(1, top[:, None])[:, 0]
    ha, hb = -(pa * la).sum(-1), -(pb * lb).sum(-1)
    keep = torch.randperm(za.shape[0], generator=torch.Generator().manual_seed(0))[:8].to(za.device)  # for the s* fit
    return {"y_is_top": (y == top), "dlogp_y": lb.gather(1, y[:, None])[:, 0] - la.gather(1, y[:, None])[:, 0],
            "dp_y": py_b - py_a, "dp_top": pt_b - pt_a, "p_top_a": pt_a, "dH": hb - ha, "Ha": ha,
            "za": za[keep], "zb": zb[keep]}


def summarize_tokens(stats_list, label):
    keys = ["dlogp_y", "dp_y", "dp_top", "dH", "Ha", "p_top_a"]
    cat = {k: torch.cat([s[k] for s in stats_list]) for k in keys + ["y_is_top"]}
    out = {"label": label, "tokens": int(cat["dH"].numel())}
    for sel_name, sel in (("all", torch.ones_like(cat["y_is_top"])), ("target_is_A_top1", cat["y_is_top"]),
                          ("target_not_top1", ~cat["y_is_top"])):
        if sel.sum() == 0:
            continue
        d = {k: cat[k][sel].float().mean().item() for k in keys}
        d["n"] = int(sel.sum())
        if sel_name == "target_not_top1":
            # of the probability mass the target loses (when it loses), what share does A's top-1 token gain?
            lost = (-cat["dp_y"][sel]).clamp(min=0)
            gain = cat["dp_top"][sel].clamp(min=0)
            d["top1_gain_per_target_loss"] = (gain.sum() / lost.sum().clamp(min=1e-9)).item()
        out[sel_name] = d
    # best single temperature-like scale for B vs A on these tokens (KL(p_B || softmax(s z_A)))
    za = torch.cat([s["za"] for s in stats_list]).cuda()
    zb = torch.cat([s["zb"] for s in stats_list]).cuda()
    pb = F.softmax(zb, -1)
    best = min((((pb * (F.log_softmax(zb, -1) - F.log_softmax(sc * za, -1))).sum(-1).mean().item()), sc)
               for sc in [0.7 + 0.025 * i for i in range(41)])
    out["s_star"] = best[1]
    return out


def cmd_squeeze(args):
    tok = AutoTokenizer.from_pretrained(args.a)
    ma, mb = load(args.a), load(args.b)
    res = {}
    groups = {"chosen": [], "rejected": []}
    for r in dpo_pairs(args.n):
        for side in ("chosen", "rejected"):
            e = encode(tok, r[side], 256)
            if e:
                s = token_stats(ma, mb, *e)
                groups[side].append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in s.items()})
    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ids = tok("\n".join(t for t in wiki["text"] if t.strip()), add_special_tokens=False).input_ids
    groups["wikitext"] = []
    for i in range(args.n // 4):
        s = token_stats(ma, mb, torch.tensor([ids[i * 512 : (i + 1) * 512]]), 32)
        groups["wikitext"].append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in s.items()})
    for g, lst in groups.items():
        res[g] = summarize_tokens(lst, g)
        a, nt = res[g].get("all", {}), res[g].get("target_not_top1", {})
        print(f"{g:<9} tokens {res[g]['tokens']:>7}  mean dlogp(target) {a.get('dlogp_y', 0):+.3f}  dH {a.get('dH', 0):+.3f}  "
              f"d p(A top1) {a.get('dp_top', 0):+.4f}  s* {res[g]['s_star']:.3f} | target!=top1: d p(target) "
              f"{nt.get('dp_y', 0):+.4f}  d p(top1) {nt.get('dp_top', 0):+.4f}  top1 gain / target loss "
              f"{nt.get('top1_gain_per_target_loss', 0):.2f}", flush=True)
    save(args, "squeeze", res)


# ---------------------------------------------------------------- C2


def cmd_firsttok(args):
    tok = AutoTokenizer.from_pretrained(args.a)
    ma, mb = load(args.a), load(args.b)
    import pyarrow.parquet as pq
    rows = pq.read_table("/workspace/dpo_data/q25_all.parquet").slice(20000, 20000).to_pylist()
    rows = [r for r in rows if len(r["chosen"]) == 2]
    random.Random(1).shuffle(rows)
    groups = {"option_listing": [], "short_answer": [], "long_answer": []}
    for r in rows:
        user = r["chosen"][0]["content"] or ""
        n_ans = len(tok(r["chosen"][-1]["content"] or "", add_special_tokens=False).input_ids)
        g = "option_listing" if len(OPT.findall(user)) >= 3 else "short_answer" if n_ans <= 20 else "long_answer"
        if len(groups[g]) >= args.n:
            continue
        p = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        groups[g].append(tok(p, add_special_tokens=False).input_ids[-2048:])
        if all(len(v) >= args.n for v in groups.values()):
            break
    res = {}
    with torch.no_grad():
        for g, lst in groups.items():
            za_all, zb_all = [], []
            for ids in lst:
                x = torch.tensor([ids], device="cuda")
                za_all.append(ma(x, logits_to_keep=1).logits[0, -1].float())
                zb_all.append(mb(x, logits_to_keep=1).logits[0, -1].float())
            za, zb = torch.stack(za_all), torch.stack(zb_all)
            la, lb = F.log_softmax(za, -1), F.log_softmax(zb, -1)
            pa, pb = la.exp(), lb.exp()
            best = min((((pb * (lb - F.log_softmax(sc * za, -1))).sum(-1).mean().item()), sc)
                       for sc in [0.6 + 0.025 * i for i in range(57)])
            res[g] = {"n": len(lst), "H_A": (-(pa * la).sum(-1)).mean().item(), "H_B": (-(pb * lb).sum(-1)).mean().item(),
                      "top1_A": pa.max(-1).values.mean().item(), "top1_B": pb.max(-1).values.mean().item(),
                      "same_top1": (pa.argmax(-1) == pb.argmax(-1)).float().mean().item(), "s_star": best[1]}
            r = res[g]
            print(f"{g:<15} n {r['n']:>4}  entropy {r['H_A']:.3f} -> {r['H_B']:.3f}  top-1 p {r['top1_A']:.3f} -> "
                  f"{r['top1_B']:.3f}  same top-1 {r['same_top1']:.0%}  s* {r['s_star']:.3f}", flush=True)
    save(args, "firsttok", res)


# ---------------------------------------------------------------- A7: causal test of the gain account


def gain_hooks(model, heads, mlps, s):
    """Scale the selected heads' and MLP blocks' contributions to the residual stream by s, at every position.
    heads = {layer: [head, ...]}: each head's share of the post-attention-norm output, with that norm's scale held
    fixed (the same split decomp uses), gets multiplied by s. mlps = [layer, ...]: the whole normalized MLP output."""
    n_heads = model.config.num_attention_heads
    hd = model.config.hidden_size // n_heads
    hooks, z = [], {}
    for l, hs in heads.items():
        layer = model.model.layers[l]
        cols = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in hs]).cuda()
        W = layer.self_attn.o_proj.weight[:, cols].float()  # [d, k*hd]
        g = layer.post_attention_layernorm.weight.float()
        eps = layer.post_attention_layernorm.variance_epsilon

        def grab(m, a, l=l, cols=cols):
            z[l] = a[0][..., cols]

        def scale(m, a, o, l=l, W=W, g=g, eps=eps):  # returns the new output on purpose
            x = a[0].float()
            rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
            contrib = (z.pop(l).float() @ W.T) * g / rms
            return (o.float() + (s - 1) * contrib).to(o.dtype)
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(grab))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(scale))
    for l in mlps:
        hooks.append(model.model.layers[l].post_feedforward_layernorm.register_forward_hook(lambda m, a, o: o * s))
    return hooks


@torch.no_grad()
def wiki_stats(model, chunks):
    """Mean next-token NLL and mean top-two log-prob gap on wikitext chunks (positions 32+)."""
    nll = gap = n = 0.0
    for ids in chunks:
        ids = ids.cuda()
        lp = F.log_softmax(model(ids).logits[0, 31:-1].float(), -1)
        y = ids[0, 32:]
        nll += -lp.gather(1, y[:, None]).sum().item()
        top2 = lp.topk(2, -1).values
        gap += (top2[:, 0] - top2[:, 1]).sum().item()
        n += len(y)
    return nll / n, gap / n


@torch.no_grad()
def cmd_gain(args):
    """Is the overconfidence CAUSED by a gain on SFT's existing answer components? In SFT, scale the top-k heads by
    their SFT letter-gap contribution (and/or late MLPs) by s and see whether T* rises to DPO's level with answers,
    accuracy and AUROC unchanged (sufficiency); in DPO, scale the same components by 1/s and see whether T* falls back
    (necessity). Controls: k random heads; side effects on wikitext NLL and top-two gap."""
    from mechanism import text_batches
    from temp_scaling import auroc
    tok = AutoTokenizer.from_pretrained(args.a)
    tok.padding_side = "left"
    test, prompts = mmlu(tok, args.n)
    lid = letter_ids(tok)
    chunks = [ids for src, ids, _ in itertools.islice(text_batches(tok, 32), 32) if src == "wikitext"]
    A = load(args.a)
    dla = run_decomp(A, tok, prompts, lid)["dla_head"]  # [L, H] mean contribution to SFT's letter gap
    L, H = dla.shape
    order = torch.argsort(dla.flatten(), descending=True).tolist()
    top = lambda k: order[:k]
    rnd = random.Random(0).sample(order[100:], 30)
    as_map = lambda flat: {l: [i % H for i in flat if i // H == l] for l in sorted({i // H for i in flat})}
    late_mlps = list(range(19, 31))
    late_all = {l: list(range(H)) for l in range(15, L)}
    print("top-10 SFT answer heads:", ", ".join(f"L{i // H}H{i % H} {dla.flatten()[i].item():+.3f}" for i in top(10)))
    rows, base = [], {}

    def run(model, tag, name, heads, mlps, s):
        hooks = gain_hooks(model, heads, mlps, s) if (heads or mlps) and s != 1 else []
        try:
            Z = letter_logits(model, tok, prompts, lid)
            wnll, wgap = wiki_stats(model, chunks)
        finally:
            for h in hooks:
                h.remove()
        cal = calib_from_logits(Z, test)
        p = F.softmax(Z, -1)
        conf = p.max(-1).values.tolist()
        corr = [int(a == t["answer"]) for a, t in zip(p.argmax(-1).tolist(), test)]
        if name == "baseline":
            base[tag] = (Z, wnll, wgap)
        c, r2 = scale_fit(Z, base["SFT"][0])
        r = {"model": tag, "condition": name, "s": s, "n_heads": sum(len(v) for v in heads.values()), "n_mlps": len(mlps),
             "letter_scale_vs_sft": c, "scale_r2": r2, "same_pred_as_own_baseline": (Z.argmax(-1) == base[tag][0].argmax(-1)).float().mean().item(),
             "acc": cal["acc"], "ece": cal["ece"], "t_star": cal["t_star"], "auroc": auroc(conf, corr),
             "wiki_nll": wnll, "wiki_dnll": wnll - base[tag][1], "wiki_gap_ratio": wgap / base[tag][2]}
        rows.append(r)
        print(f"{tag:<4} {name:<34} s={s:<5.3g} scale {c:5.3f} (R2 {r2:.3f}) same {r['same_pred_as_own_baseline']:.3f} "
              f"acc {r['acc']:.3f} ECE {r['ece']:.3f} T* {r['t_star']:.2f} AUROC {r['auroc']:.3f} | wiki dNLL {r['wiki_dnll']:+.4f} "
              f"gap x{r['wiki_gap_ratio']:.3f}", flush=True)

    run(A, "SFT", "baseline", {}, [], 1)
    for s in (1.3, 1.5):
        for k in (10, 30, 100):
            run(A, "SFT", f"top-{k} answer heads", as_map(top(k)), [], s)
        run(A, "SFT", "30 random heads (control)", as_map(rnd), [], s)
        run(A, "SFT", "late MLPs (L19-30)", {}, late_mlps, s)
        run(A, "SFT", "top-30 heads + late MLPs", as_map(top(30)), late_mlps, s)
    run(A, "SFT", "all heads + MLPs, L15-31", late_all, list(range(15, L)), 1.3)
    del A
    torch.cuda.empty_cache()
    B = load(args.b)
    run(B, "DPO", "baseline", {}, [], 1)
    for s in (1.3, 1.5):
        run(B, "DPO", "top-30 heads + late MLPs", as_map(top(30)), late_mlps, 1 / s)
        run(B, "DPO", "top-30 answer heads", as_map(top(30)), [], 1 / s)
        run(B, "DPO", "late MLPs (L19-30)", {}, late_mlps, 1 / s)
    run(B, "DPO", "all heads + MLPs, L15-31", late_all, list(range(15, L)), 1 / 1.3)
    save(args, "gain", {"a": args.a, "b": args.b, "n": len(prompts), "top_heads": [(i // H, i % H, dla.flatten()[i].item()) for i in top(30)],
                        "random_heads": [(i // H, i % H) for i in rnd], "rows": rows})


@torch.no_grad()
def cmd_transplant(args):
    """Follow-up to gain: does DPO's change live in what the attributed components WRITE? At the answer position,
    replace selected heads' contributions (post-attention-norm, each block's norm scale held fixed as in decomp),
    whole attention blocks, or MLP blocks in A (SFT) with B's (DPO) values on the same prompt; later layers then run
    as A. Metric: letter scale vs A and its share of B's (log c / log c_B)."""
    tok = AutoTokenizer.from_pretrained(args.a)
    tok.padding_side = "left"
    test, prompts = mmlu(tok, args.n)
    lid = letter_ids(tok)
    A, B = load(args.a), load(args.b)
    dla = run_decomp(A, tok, prompts, lid)["dla_head"]
    L, H = dla.shape
    hd = A.config.hidden_size // H
    order = torch.argsort(dla.flatten(), descending=True).tolist()
    dpath = os.path.join(args.out, f"decomp__{short(args.a)}__{short(args.b)}.json")
    changed = [int(l) * H + int(h) for l, h, *_ in json.load(open(dpath))["top_heads_d"][:20]]
    rnd = random.Random(0).sample(order[100:], 30)
    union = sorted(set(order[:100]) | set(changed) | set(rnd))
    by_layer = lambda flat: {l: [i % H for i in flat if i // H == l] for l in sorted({i // H for i in flat})}
    late = list(range(15, L))
    conds = [("top-10 answer heads", order[:10], [], []), ("top-30 answer heads", order[:30], [], []),
             ("top-100 answer heads", order[:100], [], []), ("20 most-changed heads", changed, [], []),
             ("30 random heads (control)", rnd, [], []), ("late MLPs L19-30", [], list(range(19, 31)), []),
             ("top-30 heads + late MLPs", order[:30], list(range(19, 31)), []),
             ("all attention L15-31", [], [], late), ("all attention + MLPs L15-31", [], late, late)]
    if args.controls:  # size-matched random controls for the MLP conditions (12 MLP blocks, like L19-30)
        rmlp = [sorted(random.Random(s).sample(range(L), 12)) for s in range(3)]
        for s, m in enumerate(rmlp):
            print(f"random MLP set {s}: layers {m}")
            conds.append((f"12 random MLPs s{s} (control)", [], m, []))
        conds.append(("30 random heads + 12 random MLPs s0 (control)", rnd, rmlp[0], []))
    W = {(m, l): mod.model.layers[l].self_attn.o_proj.weight.float() for m, mod in (("A", A), ("B", B)) for l in range(L)}

    def head_contrib(m, l, z, x, heads):
        """[b, k, d] contributions of `heads` at the last position: g * (W_O[:, h] z_h) / rms(x)."""
        norm = (A if m == "A" else B).model.layers[l].post_attention_layernorm
        rms = x.pow(2).mean(-1, keepdim=True).add(norm.variance_epsilon).sqrt()  # [b, 1]
        out = []
        for h in heads:
            sl = slice(h * hd, (h + 1) * hd)
            out.append((z[:, sl] @ W[(m, l)][:, sl].T) * norm.weight.float() / rms)
        return torch.stack(out, 1)

    Z = {name: torch.zeros(len(prompts), 4, device="cuda") for name, *_ in conds}
    ZA, ZB = torch.zeros(len(prompts), 4, device="cuda"), torch.zeros(len(prompts), 4, device="cuda")
    ul = by_layer(union)
    for ix, enc in batches(tok, prompts, args.bs):
        # 1. B's last-position values
        cap, hooks = {}, []
        for l, layer in enumerate(B.model.layers):
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(("z", l), a[0][:, -1].float())))
            def attn_b(m, a, o, l=l):
                cap[("attn", l)] = o[:, -1].clone()
                if l in ul:
                    cap[("heads", l)] = dict(zip(ul[l], head_contrib("B", l, cap[("z", l)], a[0][:, -1].float(), ul[l]).unbind(1)))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(attn_b))
            hooks.append(layer.post_feedforward_layernorm.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("mlp", l), o[:, -1].clone())))
        ZB[ix] = B(**enc, logits_to_keep=1).logits[:, -1].float()[:, lid]
        for h_ in hooks:
            h_.remove()
        ZA[ix] = A(**enc, logits_to_keep=1).logits[:, -1].float()[:, lid]
        # 2. A with B's values transplanted at the last position
        for name, heads, mlps, attn_layers in conds:
            hl, hooks, za = by_layer(heads), [], {}
            for l, layer in enumerate(A.model.layers):
                if l in hl or l in attn_layers:
                    hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: za.__setitem__(l, a[0][:, -1].float())))
                    def attn_a(m, a, o, l=l, hs=hl.get(l, []), whole=l in attn_layers):
                        o = o.clone()
                        if whole:
                            o[:, -1] = cap[("attn", l)]
                        else:
                            own = head_contrib("A", l, za[l], a[0][:, -1].float(), hs)
                            new = torch.stack([cap[("heads", l)][h] for h in hs], 1)
                            o[:, -1] = (o[:, -1].float() + (new - own).sum(1)).to(o.dtype)
                        return o
                    hooks.append(layer.post_attention_layernorm.register_forward_hook(attn_a))
                if l in mlps:
                    def mlp_a(m, a, o, l=l):
                        o = o.clone()
                        o[:, -1] = cap[("mlp", l)]
                        return o
                    hooks.append(layer.post_feedforward_layernorm.register_forward_hook(mlp_a))
            Z[name][ix] = A(**enc, logits_to_keep=1).logits[:, -1].float()[:, lid]
            for h_ in hooks:
                h_.remove()
    cB, _ = scale_fit(ZB, ZA)
    rows = []
    for name, heads, mlps, attn_layers in conds:
        c, r2 = scale_fit(Z[name], ZA)
        cal = calib_from_logits(Z[name], test)
        r = {"condition": name, "letter_scale": c, "r2": r2, "share_of_dpo": math.log(c) / math.log(cB), "t_star": cal["t_star"],
             "acc": cal["acc"], "ece": cal["ece"], "same_pred_as_sft": (Z[name].argmax(-1) == ZA.argmax(-1)).float().mean().item()}
        rows.append(r)
        print(f"{name:<30} scale {c:.3f} (R2 {r2:.3f}) share {r['share_of_dpo']:+.2f}  T* {r['t_star']:.2f}  acc {r['acc']:.3f}  same {r['same_pred_as_sft']:.3f}", flush=True)
    calA, calB = calib_from_logits(ZA, test), calib_from_logits(ZB, test)
    print(f"SFT T* {calA['t_star']:.2f}  DPO T* {calB['t_star']:.2f}  DPO letter scale {cB:.3f}")
    save(args, "transplant", {"a": args.a, "b": args.b, "n": len(prompts), "dpo_scale": cB, "sft_t": calA["t_star"],
                              "dpo_t": calB["t_star"], "changed_heads": [(i // H, i % H) for i in changed], "rows": rows})


def save(args, test, obj):
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{test}__{short(args.a)}__{short(args.b)}.json")
    json.dump(obj, open(path, "w"), indent=1, default=float)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["decomp", "patch", "squeeze", "firsttok", "norm", "neurons", "pairs", "bridge", "gain", "transplant"])
    ap.add_argument("--data", default="instruct", choices=["instruct", "think"], help="pairs: which DPO data")
    ap.add_argument("--max-resp", type=int, default=1024, help="pairs: response tokens scored")
    ap.add_argument("--k-null", type=int, default=64, help="norm/neurons: bottom singular directions of W_U counted as null space")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 10, 30, 100], help="neurons: set sizes")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--controls", action="store_true", help="transplant: add size-matched random-MLP controls")
    ap.add_argument("--mem-frac", type=float, default=0.0, help="cap this process's GPU memory (shared GPU)")
    ap.add_argument("--a", default="allenai/Olmo-3-7B-Instruct-SFT")
    ap.add_argument("--b", default="allenai/Olmo-3-7B-Instruct-DPO")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results_why")
    args = ap.parse_args()
    {"decomp": cmd_decomp, "patch": cmd_patch, "squeeze": cmd_squeeze, "firsttok": cmd_firsttok,
     "norm": cmd_norm, "neurons": cmd_neurons, "pairs": cmd_pairs, "bridge": cmd_bridge, "gain": cmd_gain, "transplant": cmd_transplant}[args.test](args)


if __name__ == "__main__":
    main()
