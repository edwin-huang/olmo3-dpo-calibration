"""Are the SFT-test LoRA models overconfident everywhere, or only on multiple choice?

Compares the base model (A) with base + a LoRA from sft_known.py (B) on held-out text: wikitext
test (the prose LoRA trained on wikitext train) and chat responses. Same measures as
decision_points.py: per uncertainty bin, the sharpening s* that best maps A onto B
(s* > 1 = B sharper). If prose fine-tuning raises MMLU T* to ~1.7 but leaves text at s* ~ 1,
multiple-choice calibration is a fragile, separate property; if text sharpens too, it's
general overfitting.

Usage: python lora_text_probe.py --loras prose_s0_ep4,known_s0_ep4,unknown_s0_ep4
"""
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision_points import probe
from label_trivia import MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loras", default="prose_s0_ep4,known_s0_ep4,unknown_s0_ep4")
    ap.add_argument("--dir", default="results_sftk")
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--out", default="results_lora_text")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")  # its chat template; same vocab
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    for name in args.loras.split(","):
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            print(f"skip {name}")
            continue
        print(f"\n######## base -> base + LoRA {name}", flush=True)
        b = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
        b = PeftModel.from_pretrained(b, os.path.join(args.dir, f"lora_{name}")).merge_and_unload().eval()
        json.dump(probe(base, b, tok, args.n_seq), open(path, "w"), indent=1)
        del b
        torch.cuda.empty_cache()
    print("ALL DONE")


if __name__ == "__main__":
    main()
