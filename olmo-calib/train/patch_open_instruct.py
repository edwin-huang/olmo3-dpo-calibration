"""Patch open-instruct (commit 11826255) for the olmo-calib DPO reruns. Idempotent; each patch asserts its anchor.

1. dpo_tune_cache.py: model-only snapshots. Upstream only saves full accelerate state (~100 GB for a 7B with
   optimizer) at --checkpointing_steps. With SNAPSHOT_EVERY=N set, this also saves the bf16 HF model (~15 GB) to
   <output_dir>/snap_step_<k> every N optimizer steps, so calibration can be measured along the run. Snapshot folders
   don't match is_checkpoint_folder(), so keep_last_n_checkpoints never deletes them.
2. model_utils.py: saving crashed with "GenerationConfig is invalid" under transformers 5.4. save_with_accelerate sets
   a clean generation config on the DeepSpeed wrapper but saves the unwrapped model, which still carries
   Olmo-3-7B-Instruct-SFT's temperature=0.6 / top_p=0.95 without do_sample. Setting do_sample=True keeps those
   sampling defaults and makes the config valid.
3. dpo_tune_cache.py: with LOG_BATCH_INDEX=1, log every micro-batch's example indices to
   <output_dir>/batch_index_rank<r>.txt, to check that a resumed run continues the same data order.
4. dpo_tune_cache.py: DPO_DETACH=chosen|rejected trains only the other half of the DPO gradient (root-cause tests:
   does the sharpening come from pushing rejected responses down or chosen responses up?).
5. dpo_tune_cache.py, fix tests: DPO_AGREE_K=k multiplies each pair's loss by sigmoid(-k * reference margin) /
   DPO_AGREE_NORM (reference margin = mean per-token log-prob chosen minus rejected under the SFT model, from the
   reference cache), so pairs the SFT model already agrees with count less. DPO_NLL=alpha adds alpha x the chosen
   response's mean per-token negative log-likelihood (RPO-style).
6. dpo_tune_cache.py, Brief 2 tests (use with --loss_type dpo_norm, so the reference cache holds per-token-averaged
   log-probs): DPO_GAMMA=g gives SimPO-with-reference, loss -log sigmoid(beta * (h - g)); DPO_IPO_TAU=t gives IPO on
   the same margin, loss (h - 1/(2t))^2; h = (pi(y_c) - ref(y_c)) - (pi(y_r) - ref(y_r)), the reference-relative margin.

Usage: python patch_open_instruct.py /workspace/open-instruct
"""
import sys

MARK = "# olmo-calib:"
PATCHES = [
    ("open_instruct/dpo_tune_cache.py",
     "                if completed_steps >= args.max_train_steps:\n                    break\n",
     "before",
     """                # olmo-calib: model-only snapshots
                snapshot_every = int(os.environ.get("SNAPSHOT_EVERY", "0"))
                if snapshot_every and completed_steps % snapshot_every == 0 and completed_steps < args.max_train_steps:
                    model_utils.save_with_accelerate(
                        accelerator, model, tokenizer, os.path.join(args.output_dir, f"snap_step_{completed_steps}"),
                        args.use_lora, chat_template_name=tc.chat_template_name,
                    )
                    accelerator.wait_for_everyone()

"""),
    ("open_instruct/model_utils.py",
     "    unwrapped_model: transformers.PreTrainedModel = accelerator.unwrap_model(model)\n",
     "after",
     """    # olmo-calib: the unwrapped model keeps the SFT checkpoint's sampling defaults without do_sample (invalid in transformers 5.4)
    gen_config = getattr(unwrapped_model, "generation_config", None)
    if gen_config is not None and not gen_config.do_sample and (
        gen_config.temperature not in (None, 1.0) or gen_config.top_p not in (None, 1.0)
    ):
        gen_config.do_sample = True
"""),
    ("open_instruct/dpo_tune_cache.py",
     "            episode += len(batch[\"chosen_input_ids\"]) * accelerator.num_processes\n",
     "after",
     """            # olmo-calib: LOG_BATCH_INDEX=1 appends each micro-batch's example indices (one line per micro-batch, per rank)
            if os.environ.get("LOG_BATCH_INDEX"):
                with open(os.path.join(args.output_dir, f"batch_index_rank{accelerator.process_index}.txt"), "a") as idx_f:
                    idx_f.write(" ".join(str(int(i)) for i in batch["index"]) + "\\n")
"""),
    ("open_instruct/dpo_tune_cache.py",
     "                )  # `aux_loss` is only used when `args.load_balancing_loss = True`\n",
     "after",
     """                # olmo-calib: DPO_DETACH=chosen|rejected stops the gradient through that side's log-probs, so only
                # the other half of DPO's update trains (the sigmoid pair weights still use both sides)
                detach_side = os.environ.get("DPO_DETACH")
                if detach_side == "chosen":
                    policy_chosen_logps = policy_chosen_logps.detach()
                elif detach_side == "rejected":
                    policy_rejected_logps = policy_rejected_logps.detach()
"""),
    ("open_instruct/dpo_tune_cache.py",
     "                loss = losses.mean()\n",
     "before",
     """                # olmo-calib: DPO_AGREE_K / DPO_AGREE_NORM agreement weighting, DPO_NLL chosen-likelihood term (fix tests)
                agree_k = float(os.environ.get("DPO_AGREE_K", "0"))
                if agree_k:
                    ref_lp = reference_cache[batch["index"]]
                    ref_margin = (ref_lp["chosen_logps"] - ref_lp["rejected_logps"]).float()
                    pair_w = torch.sigmoid(-agree_k * ref_margin) / float(os.environ.get("DPO_AGREE_NORM", "1"))
                    losses = losses * pair_w.to(losses.dtype)
                nll_alpha = float(os.environ.get("DPO_NLL", "0"))
                if nll_alpha:
                    losses = losses - nll_alpha * policy_chosen_logps
"""),
    ("open_instruct/dpo_tune_cache.py",
     "                # olmo-calib: DPO_AGREE_K / DPO_AGREE_NORM agreement weighting, DPO_NLL chosen-likelihood term (fix tests)\n",
     "before",
     """                # olmo-calib: DPO_GAMMA (SimPO-with-reference) / DPO_IPO_TAU (IPO) on the reference-relative margin
                ref_gamma = os.environ.get("DPO_GAMMA")
                ipo_tau = os.environ.get("DPO_IPO_TAU")
                if ref_gamma or ipo_tau:
                    ref_lp6 = reference_cache[batch["index"]]
                    h6 = (policy_chosen_logps - ref_lp6["chosen_logps"]) - (policy_rejected_logps - ref_lp6["rejected_logps"])
                    if ipo_tau:
                        losses = (h6 - 1 / (2 * float(ipo_tau))) ** 2
                    else:
                        losses = -torch.nn.functional.logsigmoid(args.beta * (h6 - float(ref_gamma)))
"""),
]

root = sys.argv[1]
for rel, anchor, where, snippet in PATCHES:
    path = f"{root}/{rel}"
    src = open(path).read()
    if snippet.strip().splitlines()[0].strip() in src:
        print("already patched", rel)
        continue
    assert src.count(anchor) == 1, f"anchor not found in {rel}; open-instruct changed, re-check"
    src = src.replace(anchor, snippet + anchor if where == "before" else anchor + snippet)
    open(path, "w").write(src)
    print("patched", rel)
