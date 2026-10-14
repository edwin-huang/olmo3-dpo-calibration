# Verification audit (Task 2)

Audited 2026-10-13 against the main write-up "Why DPO Makes Olmo 3 Overconfident" (doc rev 37) and the experiment log. Paths are relative to `olmo-calib/` unless they start with `/workspace` (the rental box). Box-side checks: `train/verify_box.py` (log copied to `verification/verify_box.log`).

## Failures and surprises first

| # | Finding | Status | What it affects |
|---|---|---|---|
| 1 | The 1,000 held-out pairs and 300 judge prompts are **inside `instruct_full`**, the data Ai2's released Instruct-DPO and our full run trained on (1,000/1,000 pairs, 300/300 prompts). They are disjoint from every small-test and 25%-scale training set. | FAIL for full-scale numbers | Released DPO's "learned 57%" and judge .51/.55 are measured on pairs/prompts it trained on. The same will hold for the full run. Small-test and 25%-scale "learned" and judge numbers are clean. The main doc's Limits cites the released model's 55% judge win rate as evidence that the judge is insensitive; that inference is weaker than stated (it was even in-distribution). |
| 2 | D4's "weights moved 21% further from SFT" (relative L2 4.57e-4 vs 3.77e-4; update cosine 0.586) came from a one-off inline script in an earlier session, compared against `q25_all_r2`. Both models have since been deleted. | UNTRACEABLE to a saved file | One clause in the main doc's Why and Mechanism sections and the log §7. Only the session transcript records it. Options: keep with "(from a one-off computation; models no longer available)", or drop. |
| 3 | IFEval: Ai2 reports 81.7 (SFT) / 82.0 (DPO); we get prompt-strict .769 / .789 and instruction-strict .842 / .855. Ai2's number sits between our two variants. | UNCLEAR | Absolute IFEval only; our relative comparisons use one harness throughout. Likely a loose-vs-strict or generation-settings difference; not resolved. |
| 4 | `verify_box.py`'s held-out cross-check first printed 90.0% agreed: its glob picked the 10-pair smoke file `pairs__…__20.json`. The real 1,000-pair file gives 62.8%. | Script bug, no doc impact | Fixed by reading the 2000-entry file; see §e. |
| 5 | Released DPO's MMLU ECE is .205 in the main doc (from `calib_eval.py`, `results_robust/mmlu5_full`) and .2061 in `results_ckpt/released` (`eval_ckpt.py`). Same model, two harness paths that bin identically; difference is in the 4th decimal. | PASS (note only) | None. |

## Later checks (2026-10-13)

| Check | Result | Status |
|---|---|---|
| Full run's weight normalizer (DPO_AGREE_NORM) | Computed on the 25% sample: 0.4135. True mean over all 259,922 pairs (from the full run's reference cache `89203f41147b0c40.pt`): 0.4130, 63.09% agreed. Every pair's loss is scaled by 0.9988 instead of 1; relative weights between pairs are exact. Adam is insensitive to a constant loss scale; the only path to an effect is gradient clipping, whose threshold for this run (DeepSpeed `gradient_clipping: auto`) was not verified | PASS (negligible); clipping setting UNVERIFIED |
| Brief 2 data sets vs held-out pairs and judge prompts | why_hi8, why_lo8, why_pool and s8_a: zero pair, prompt or judge-prompt overlap; why_hi8 and why_lo8 disjoint | PASS |
| Scoring pool vs the scoring model's training data | First pool shared 4 exact-duplicate pairs with q25_all (instruct_full contains duplicate pairs at different rows). Rebuilt before any scoring, excluding duplicates by content; 4 rows swapped; overlap now 0; `train/make_why_sets.py` updated to match | FIXED (before use) |

## a. Harness sanity vs Ai2's reported numbers — PASS (MMLU), UNCLEAR (IFEval)

Source: [Olmo-3-7B-Instruct model card](https://huggingface.co/allenai/Olmo-3-7B-Instruct) (evaluation table: SFT / DPO / final) and the [Olmo 3 technical report](https://arxiv.org/abs/2512.13961). The model card does not state the MMLU format; Ai2's post-training evals follow the Tülu 3 setup, where MMLU is zero-shot chain-of-thought with a chat template ([Tülu 3 report](https://arxiv.org/abs/2411.15124)). I could not extract the exact Olmo 3 config from the PDF; treat the format attribution as likely, not confirmed.

| Benchmark | Ai2 SFT | Ai2 DPO | Ours SFT | Ours DPO | Our format |
|---|---|---|---|---|---|
| MMLU | 67.1 | 69.1 | 66.7 | 67.7 | chat template, chain of thought, 1,000 questions (`results_cot/*.json`, `cot_mmlu.acc`) |
| MMLU | 67.1 | 69.1 | 60.8 | 61.5 | 5-shot, no chat template, A–D letter probabilities, all 14,042 (`results_ckpt/released/*/summary.json`) |
| MMLU | 67.1 | 69.1 | 57.5 | 57.8 | chat template, immediate answer, 1,000 questions (`results_cot`, `direct_mmlu.acc`) |
| IFEval | 81.7 | 82.0 | 76.9 / 84.2 | 78.9 / 85.5 | prompt-strict / instruction-strict, 541 prompts (`results_ckpt/released/*/ifeval.json`) |

Our chain-of-thought format reproduces Ai2's MMLU within 0.4 (SFT) and 1.4 (DPO) points on a 1,000-question subset, so the model loading and prompting are sound. The calibration numbers use 5-shot letter scoring on purpose (it gives a probability over A–D without generation); its lower accuracy is a format difference, not a harness error. The docs should say this in one line.

Also worth knowing for the full-run comparison: Ai2 reports DPO's biggest gains on BigBenchHard (51.0 → 69.3) and AlpacaEval 2 LC (21.8 → 43.3), neither of which we measure.

## b. Inputs for the independent T*/ECE recompute — READY

Files: `verification/recompute_inputs/` (written by `verification/export_recompute_inputs.py`).

- `<model>__mmlu.csv` (14,042 rows) and `<model>__arc.csv` (1,165 rows) for `instruct_sft`, `instruct_dpo_released`, `q25_plain`, `q25_agreement_weighted`. The full run will be added with `python verification/export_recompute_inputs.py full_weighted=results_ckpt/full_w5/final_full`.
- Columns: `qid` (row index in the test split as `calib_eval.load_task` loads it: `cais/mmlu` "all" test; ARC-Challenge test keeping only 4-option questions), `subject`, `answer` (0–3), `p_A … p_D`.
- `reported_values.json`: what our pipeline reported for exactly these files.

**Definitions (as implemented):**

| Item | Definition | Code |
|---|---|---|
| Probabilities | Softmax over the four letter tokens' logits at the answer position (`" A"`…`" D"`); mass on other tokens dropped | `calib_eval.py` run_eval |
| Confidence | max of the four probabilities; prediction = argmax | `eval_ckpt.py` calib() |
| ECE | 15 **equal-width** bins on confidence in [0, 1] (bin = min(int(c·15), 14)); sum over non-empty bins of \|Σconf − Σcorrect\| / N | `temp_scaling.py` ece() |
| Temperature | p_i(T) ∝ exp(log p_i / T), i.e. softmax of logits / T over A–D | `temp_scaling.py` scaled() |
| T* objective | Mean negative log-likelihood of the gold answer under p(T) | `temp_scaling.py` nll() |
| T* optimizer | Golden-section search on log T over [0.2, 10] (chat format: [0.2, 100]), 40 iterations, returns the bracket midpoint | `temp_scaling.py` fit_t() |
| Fit/eval split | Question indices 0..N−1 shuffled with `random.Random(0)`; first ⌊N/2⌋ = fit half; split is by question, not subject; identical for every run on the same question set | `eval_ckpt.py` calib() |
| Reported T* | Fit on the fit half only | `eval_ckpt.py` |
| Reported ECE, AUROC, accuracy | Over **all** N questions, at T = 1 (`eval_ckpt.py`). The older `temp_scaling.py` table reports ECE and AUROC on the **held-out half** instead; the docs' "AUROC .766 → .767" uses that path | both |
| AUROC | Mann–Whitney on confidence, correct vs wrong, ties get average rank | `temp_scaling.py` auroc() |
| Letter scale | c minimizing ‖centered(z_B) − c·centered(z_A)‖² over all questions, logits centered per question over A–D | `why.py` scale_fit() |

Reported values to check against (MMLU, all 14,042):

| Model | T* | ECE | AUROC | Accuracy |
|---|---|---|---|---|
| instruct_sft | 1.5176 | .1153 | .7683 | .6080 |
| instruct_dpo_released | 2.3376 | .2061 | .7659 | .6149 |
| q25_plain | 1.8507 | .1547 | .7679 | .6133 |
| q25_agreement_weighted | 1.4966 | .1079 | .7680 | .6130 |

## c. Agreement-weighting code — PASS

Full diff vs commit 11826255077617a46919ce75cadf1f3d53f30dac (box HEAD verified): `verification/open_instruct.diff` (79 lines, 4 hunks in `dpo_tune_cache.py`, 1 in `model_utils.py`). Source of the patch: `train/patch_open_instruct.py`, patch 5. Line numbers below are in the patched `open_instruct/dpo_tune_cache.py` on the box.

| Question | Answer | Where |
|---|---|---|
| Weights from the cached reference log-probs? | Yes: `ref_lp = reference_cache[batch["index"]]` | dpo_tune_cache.py:594 |
| Margin = per-token-averaged chosen − rejected? | Yes. The cache is built with `average_log_prob=args.loss_type.is_average_loss`, which is True for `dpo_norm` | :525 (cache), :595 (margin) |
| Weights detached? | Yes in effect: the reference cache holds precomputed tensors with no graph; the weight never touches policy outputs | :594–596 |
| Normalization | **Global constant**, not per batch or device: `DPO_AGREE_NORM` (0.4106 for the small-test set, 0.4135 for the 25% and full runs = mean weight over the 65K q25 pairs; `verification/fix_sets_stats.txt`) | :596 |
| Gradient accumulation / multi-GPU | Each micro-batch's per-pair losses are multiplied by their own weights before `losses.mean()`; with 1 pair per GPU per micro-batch, accumulation and data parallelism average weighted losses exactly as plain DPO averages unweighted ones | :597, :601 |
| Is k = 5 a flag? | An environment variable, `DPO_AGREE_K` (0 = off), not a command-line flag; set in `train/fix_wave.sh`, `train/q25_w5.sh`, `train/full_w5.sh` | :592 |

One consequence worth stating: because the normalizer is the dataset mean, the average effective learning rate over an epoch matches plain DPO, but per-step weight sums vary with batch composition.

## d. Data hygiene — PASS (with finding 1)

- **Held-out pairs vs training sets** (sha1 of chosen+rejected messages, and of the prompt): zero pair or prompt overlap with q25_all, all s8/s16 sets, filt8, every rejlo/rejhi/marg split, long8, short8, nomc8/8a, rev8a, swap8a, smoke sets. Full overlap with instruct_full; partial with d4_full, q25_d4 (892), q25_delta (514), q25_judged (486), q25_rand10 (1,000). None of those partial-overlap runs reports held-out metrics, so only finding 1 matters. Judge prompts: same pattern (300/300 in instruct_full).
- **MMLU 3,000-question subset**: one identical (subject, answer) sequence across all 89 `mmlu5_3000` files; one across all 6 `mmlu5_full` files. The subset is `random.Random(0).shuffle(range(14042))[:3000]`, sorted (`calib_eval.py` run_eval).
- **T* split**: fixed by construction (same N, same `Random(0)` shuffle in `calib()`); no per-run seed.

## e. Agreement fraction — PASS

From q25's reference cache (`/workspace/ref_cache/92f0e85ab3b1d79a.pt`, open-instruct's own forward pass at 16K max length): all 65,000 pairs 63.1% agreed (mean margin +0.177); random 500 (seed 2026) 60.2% (±2.2% sampling SE). Independent code path: `eval_ckpt.py`'s own transformers forward on the 1,000 held-out pairs (1,024-token truncation) gives 62.8%. Five example pairs with margins are in `verification/verify_box.log`; two disagreed examples are instructive (a refusal as the rejected response the SFT model finds more likely; a terse rejected code answer vs a long chosen one).

## f. Provenance table

Main write-up, in reading order. "Script" is the producer; "File" is where the value sits. ✓ = checked against the file in this audit or the earlier reviewer audit; ⚠ = see note.

| Value | Section | Run / checkpoint | Script | File | |
|---|---|---|---|---|---|
| T* 1.52 → 2.34; ECE .115 → .205 | Summary, Where | released Instruct-SFT, Instruct-DPO, full MMLU 5-shot | calib_eval.py + temp_scaling.py | results_robust/mmlu5_full; results_ckpt/released/*/summary.json | ✓ |
| DPO = about half of the post-training ECE rise (47%) | Summary, chart 1 | released base, Think-SFT, Instruct-SFT, Instruct-DPO, Instruct | calib_eval.py | results_robust/mmlu5_full | ✓ |
| Letter scale ×1.44 | Summary, Mechanism | SFT → released DPO, 1,000 q | why.py decomp | results_box2/results_why/decomp__…DPO.json `letter_logit_scale_c` | ✓ |
| Transplant 76% / random 1% (and 18/35/53/49/82/100%) | Summary, Mechanism | SFT ← released DPO, 1,000 q | why.py transplant | results_why/transplant__…DPO.json | ✓ |
| Scaling ×1.3–1.5: T* 1.60 → 1.65 vs 2.45; late MLPs 1.81; DPO ÷ → 2.32 | Mechanism | same | why.py gain | results_why/gain__…DPO.json | ✓ |
| 25%: T* 1.50 vs 1.85; ECE .108 / .155; ARC 1.14 / 1.42; chat 1.84 / 2.70; acc .613; AUROC .768; IFEval .782 / .773 | Summary, Fix table | q25_w5, q25_all final_full | run_d25.sh → eval_ckpt.py | results_ckpt/q25_w5/final_full, q25_all/final_full summary.json | ✓ |
| SFT column (1.52/.115, 1.17, 1.90, .608/.768, .769) | Fix table | released SFT, full eval | eval_ckpt.py | results_ckpt/released/Olmo-3-7B-Instruct-SFT | ✓ |
| Learned 30.4% vs 27.4% | Summary, Fix | q25_w5 final_full; q25_all final (as small/fx_q25) | eval_ckpt.py pairs | q25_w5/final_full/pairs.json; small/fx_q25/pairs.json | ✓ |
| 5% longer (448 vs 426 tokens); SFT 371 | Summary, Fix | same; small/fx_sft | eval_ckpt.py gen | gen.json `mean_tokens` | ✓ |
| Judge .57 / .59 (length-adjusted, neutral) | Fix table | fx_q25, q25_w5 vs fx_sft | train/judge_winrate.py (local) | judge_vs_small_fx_sft_neutral.json | ✓ |
| ~63% agreed | Key terms | q25 reference cache | verify_box.py / agreement_analysis.py | verification/verify_box.log | ✓ |
| Small test: 8,192 pairs, 64 steps, 8× LR | Key terms | — | train/run_small.sh | runs/*.log | ✓ |
| 66–85% of ECE removed; 79% for DPO; base 22% | Setup | released checkpoints | temp_scaling.py | results_robust/mmlu5_full (reviewer recompute) | ✓ |
| SFT T* 1.53 on the 3,000-question subset | Setup, charts 2–4 | small/fx_sft | eval_ckpt.py --fast | results_ckpt/small/fx_sft/summary.json | ✓ |
| ±0.02 run noise; ~0.2 subset spread | Setup, Why | pilot8a vs pilot8a_rep; pilot8a–d; q25_all vs q25_all_r2 | run_small.sh, run_d25.sh | results_ckpt/small/pilot8*; q25_all*/ | ✓ |
| AUROC .766 → .767; .797 → .766 | Where | released, held-out half | temp_scaling.py | results_robust/mmlu5_full | ✓ |
| Confidently wrong 5.1% → 14.6% | Where | released, chat format | analyze_perq.py | results_cot/ | ✓ |
| MMLU 0-shot +.111 vs +.108 | Where | released | calib_eval.py | results_robust/mmlu0_full | ✓ |
| Think DPO 20–40% of Instruct DPO's ECE | Where | released, 4 settings | calib_eval.py | results_robust/* | ✓ |
| Letters ×1.54, cloze ×1.10 (full MMLU) | Where | released SFT, DPO | eval_ckpt.py format | results_ckpt/released/*/format.json | ✓ |
| Wikitext top-two gap ×1.08 | Where | released | why.py bridge | results_box2/why_bridge.log; results_why/bridge__…DPO.json | ✓ |
| +0.114 / +0.021 excess over SFT | Where | released, chat | analyze_perq.py | results_cot/ | ✓ |
| q25 rerun T* 1.52 → 1.85 | Where | q25_all | run_d25.sh | results_ckpt/q25_all/final_full | ✓ |
| 91% vs 27% → 3.30 vs 1.80 | Why | marghi8, marglo8 | run_small.sh; agreement_analysis.py | results_ckpt/small/marg*/summary.json; verification/agreement_analysis.out | ✓ |
| 2.37 / 2.37 at equal agreement | Why | rejlo8g, rejhi8g | same | results_ckpt/small/rej*8g | ✓ |
| Swap 2.39 / 1.50 / 1.12; accuracy .61 → .59 | Why | pilot8a, swap8a, rev8a | run_small.sh | results_ckpt/small/{pilot8a,swap8a,rev8a} | ✓ |
| Margin slope 1.25–1.48 agreed / .85–1.01 disagreed | Why | SFT → released DPO, 2,999 pairs | why.py pairs + length split | results_why/pairs_instruct_perpair.json | ⚠ split computed in-session from this file; no saved script |
| Long vs short 2.89 / 1.92 at 62–64% | Why | long8, short8 | run_small.sh | results_ckpt/small/{long8,short8}; agreement_analysis.out | ✓ |
| Random subsets 63–64%, T* 1.95–2.39 | Why | pilot8a–d | same | agreement_analysis.out | ✓ |
| r = .78 (17 runs) | chart 2 | 17 small tests | agreement_analysis.py | verification/agreement_analysis.out | ✓ |
| Rejected-only 1.84 vs full 1.65 at 2e-6 | Why (side note) | neglo8a, pilotlo8a | run_small.sh DPO_DETACH | results_ckpt/small/{neglo8a,pilotlo8a} | ✓ |
| +0.34 vs +0.13 margin growth | Why | q25_all final (fx_q25) | eval_ckpt.py pairs | small/fx_q25/pairs.json | ✓ |
| Think pairs 91% agreed | Why | Think-SFT, 1,000 Think pairs | why.py pairs --data think | results_box2/results_why/pairs_think__….json | ✓ |
| 12.5× lower LR | Why | Ai2 recipe | — | calibration-literature notes (1e-6 vs 8e-8) | ✓ |
| Delta 74% vs judged 54%; removed pairs 48%; 63% → 65% | Why | q25 pairs | agreement_analysis.py | verification/agreement_analysis.out (65% = unflagged 64.7%) | ✓ |
| T* 2.02 (delta-only), 2.02 (D4), predicted 2.04 / 1.88 | Why | q25_delta, q25_d4 | run_d25.sh; fit from agreement_analysis.out | results_ckpt/q25_d4, q25_delta final_full | ✓ (prediction is derived: exp(0.91·Δagree) × 1.85) |
| D4 weights moved 21% further | Why, Mechanism | q25_d4 vs q25_all_r2 | inline session script | session transcript only | ⚠ finding 2 |
| 64% → 31% of weight; 0.08 / 0.82; norm 0.4106 | Fix | s8_a | make_fix_sets.py | verification/fix_sets_stats.txt | ✓ |
| Filter: 23% agreed | Fix | filt8 | make_fix_sets.py | verification/fix_sets_stats.txt | ✓ |
| Small-test fix results (T*, learned, slopes, IFEval, tokens) | Fix, chart 3 | fx_* | fix_wave.sh → eval_ckpt.py | results_ckpt/small/fx_*/{summary,pairs,gen}.json | ✓ |
| ~97% vs 97.6% of agreed pairs kept | Fix | fx_* | eval_ckpt.py pairs | pairs.json `ref_agreed.acc` | ✓ |
| Length 1.35–1.46× vs 1.18× | Fix | fx_* vs fx_sft | eval_ckpt.py gen | gen.json | ✓ |
| Trajectory T* by snapshot | chart 4 | q25_all, q25_w5 snapshots | eval_run.sh | results_ckpt/q25_*/snap_step_*/summary.json | ✓ |
| Letter gap 3.01 → 4.46; +0.60 / +0.86 | Mechanism | released | why.py decomp | decomp__…DPO.json | ✓ |
| Patching 50% at layer 18 | Mechanism | released | why.py patch | results_box2/results_why/patch__…DPO.json | ✓ |
| ×1.27 signal × ×1.14 norm | Mechanism | released | why.py norm | norm__…DPO.json | ✓ |
| Entropy-neuron swap: no effect | Mechanism | released | why.py neurons | neurons__…DPO.json | ✓ |
| ×1.20 / ×1.29 / ×1.44 (small test) | Mechanism | q25_all, q25_d4, fx_base8 | why.py decomp | decomp__…q25_all/q25_d4.json; results_why/fix/decomp__…fx_base8.json | ✓ |
| Fixes ×0.98–1.03 | Mechanism | fx_w5, fx_filt, fx_simpo | why.py decomp | results_why/fix/decomp__*.json | ✓ |
| Free-form commits ×1.44 | Mechanism | released | why.py bridge | bridge__…DPO.json | ✓ |
| Judge 55% (released DPO) | Limits | fx_rel_dpo | judge_winrate.py | small/fx_rel_dpo/judge_vs_small_fx_sft_neutral.json | ⚠ finding 1: prompts were in its training data |
| Fixed models softer: text ×0.93–0.95, first token ×0.72–0.82 | Limits | fx_w5/filt/simpo | why.py bridge | results_why/fix/bridge__*.json | ✓ |
| Chat T* 3.97 | Limits | released DPO | cot_test.py | results_cot/Olmo-3-7B-Instruct-DPO.json | ✓ |
| SFT T* 1.03 → 1.52 | Limits | base, Instruct-SFT | temp_scaling.py | results_robust/mmlu5_full | ✓ |

The experiment log's numbers were checked by the earlier reviewer audit (about 400 values); its 15 mismatches are listed with fixes in `verification/CHANGELOG_NUMBERS.md`.
