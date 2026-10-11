"""Do the fast attention kernels compute the same Olmo 3 as the reference implementation on long sequences?

Training auto-selects FlashAttention-4 on Blackwell (FA3 on Hopper); evals use sdpa. Olmo 3 mixes 4096-token
sliding-window layers with full-attention layers, and DPO training sees up to 16K tokens, so short-prompt checks
don't cover the sliding window. Per-token log-probs of a 12K-token wikitext sequence under each implementation are
compared with eager attention, split into positions < 4096 and >= 4096.

Usage: python attn_check.py [model]   (one GPU)
"""
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1] if len(sys.argv) > 1 else "allenai/Olmo-3-7B-Instruct-SFT"
L = 12288
tok = AutoTokenizer.from_pretrained(model_id)
wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
ids = tok("\n".join(t for t in wiki["text"] if t.strip()), add_special_tokens=False).input_ids[:L]
x = torch.tensor([ids], device="cuda")


@torch.no_grad()
def logprobs(impl):
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation=impl).eval()
    torch.cuda.synchronize()
    t = time.time()
    lp = F.log_softmax(m(x).logits[0, :-1].float(), -1).gather(1, x[0, 1:, None])[:, 0].cpu()
    torch.cuda.synchronize()
    dt = time.time() - t
    del m
    torch.cuda.empty_cache()
    return lp, dt


ref, dt = logprobs("eager")
print(f"eager: mean logprob {ref.mean():.4f}, forward {dt:.1f}s", flush=True)
for impl in ("sdpa", "flash_attention_4", "flash_attention_3", "flash_attention_2"):
    try:
        lp, dt = logprobs(impl)
    except Exception as e:  # not installed / unsupported on this GPU
        print(f"{impl:<18} unavailable: {type(e).__name__}: {str(e)[:120]}", flush=True)
        continue
    d = (lp - ref).abs()
    print(f"{impl:<18} max|dlp| <4096 {d[:4095].max():.3f}  >=4096 {d[4095:].max():.3f}   "
          f"mean|dlp| <4096 {d[:4095].mean():.4f}  >=4096 {d[4095:].mean():.4f}   forward {dt:.1f}s", flush=True)
