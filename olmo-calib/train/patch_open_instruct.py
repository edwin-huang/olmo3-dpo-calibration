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
