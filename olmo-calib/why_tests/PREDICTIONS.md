# Predictions (written before any of these runs; never edited afterwards)

Written 2026-10-13, before the runs below exist. Outcomes go in `WHY_RESULTS.md`, not here. The file's SHA-256 is
recorded in `PREDICTIONS.sha256` at the time of writing.

## The account under test

DPO's per-pair gradient weight is computed relative to the reference model, so at the start of training every pair
gets the same push whether or not the SFT model already prefers the chosen response. Training on pairs the SFT model
already agrees with makes the model overconfident.

## Shared setup

- Start: allenai/Olmo-3-7B-Instruct-SFT. Code: open-instruct @ 11826255 + `train/patch_open_instruct.py` patches 1–6.
- Training: 8,192 pairs, one epoch, batch 128 (64 steps), LR 4e-6 (half the earlier small tests; Ai2's full-scale LR
  is 1e-6), linear schedule, 10% warmup, max 8,192 tokens, 1 GPU per run (ZeRO-1). 3 seeds per condition
  (`--seed 1/2/3`; the seed changes data order).
- Evals (`eval_ckpt.py --fast`, parts letters, format, chat, pairs, gen): MMLU T* on the fixed 3,000-question subset
  (SFT: 1.53), ECE, ARC T*, chat-format T*, accuracy, AUROC; "learned" on the 1,000 held-out pairs (372 SFT gets
  wrong; held out from every training set used here); accuracy on the 628 held-out pairs SFT gets right; mean
  response length. Main figure: MMLU T* vs learned, every run, seed spread shown.
- Agreement = Instruct-SFT's per-token-averaged log-prob of chosen > rejected (open-instruct's reference cache).

## Conditions and hyperparameters

| Run name | Data | Loss | Hyperparameters and source |
|---|---|---|---|
| wy_plain_s{1,2,3} | s8_a (random, 63.5% agreed) | length-normalized DPO | β 5 (Olmo 3 recipe default, dpo_utils.py:99) |
| wy_w5_s{n} | s8_a | agreement-weighted DPO | k 5, weights / 0.4106 (mean over s8_a); ours |
| wy_simpo_s{n} | s8_a | SimPO (open-instruct `simpo`) | β 10, γ/β 0.3 (SimPO README, Llama3-Instruct v0.2, the authors' latest instruct setting) |
| wy_simporef_s{n} | s8_a | SimPO-with-reference: −log σ(β·(h − γ/β)), h = reference-relative per-token margin | same β 10, γ/β 0.3 |
| wy_hi_s{n} / wy_lo_s{n} | why_hi8 (99.1% agreed) / why_lo8 (17.7%), matched on source and length (`train/make_why_sets.py`) | length-normalized DPO | β 5 |
| wy_easy_s{n} | why_easy8: the 8,192 easiest of 16,384 unseen pairs by implicit margin under our 25%-scale plain DPO model (50% keep) | length-normalized DPO | β 5 |
| wy_ipohi_s{n} / wy_ipolo_s{n} | why_hi8 / why_lo8 | IPO on the reference-relative per-token margin: (h − 1/(2τ))² | τ 1 (target 0.5 per token). Deviation: TRL's default τ 0.1 with averaged log-probs puts the target at 5 nats per token, out of reach in 64 steps, so the bound would never act |

## Predictions

Numbers are means over 3 seeds unless stated; "T*" = MMLU T* on the 3,000-question subset; "learned" in percentage
points. Earlier single-seed results at LR 8e-6 are the basis for the expected sizes; at 4e-6 effects should be
smaller (plain DPO at 4e-6 gave T* 1.84, learned 16.9% in one run).

1. **Plain vs agreement-weighted (core fix, replicated).** wy_w5 has lower T* than wy_plain in every seed pair, by at
   least 0.15 on average, with learned no more than 2 points lower on average.
2. **Agreement contrast (core claim).** wy_hi has higher T* than wy_lo in every seed, by at least 0.3 on average.
   (At LR 8e-6, a less well matched 91%/27% split gave 3.30 vs 1.80.)
3. **SimPO vs SimPO-with-reference (decisive).**
   - 3a: wy_simpo sits below plain DPO's calibration–learning trade-off: T* at least 0.1 below wy_plain with learned
     no more than 3 points lower.
   - 3b: adding the reference back removes most of SimPO's advantage: T*(simporef) − T*(simpo) ≥ 0.5 × (T*(plain) −
     T*(simpo)).
4. **Easy subset.** wy_easy is more overconfident than wy_plain at matched learning: T* higher by at least 0.15, or
   equal T* with learned at least 5 points lower. The easy set should be mostly agreed pairs (> 85%).
5. **IPO on the agreement contrast.** IPO lowers T* relative to DPO on the same data (wy_ipohi < wy_hi), but
   wy_ipohi is still clearly above wy_ipolo (by at least 0.15): agreement matters beyond "DPO pushes margins without
   limit".

Not run, predictions recorded for later: WPO should calibrate about like plain DPO (it puts 57% of its starting weight
on agreed pairs vs DPO's 63%); fully on-policy pairs (both responses sampled, then judged) should be about 50% agreed
and give little extra overconfidence; on-policy *rejected* responses alone should be mostly disagreed (SFT prefers its
own samples) and give little overconfidence.

## Stop rule (applied automatically after the 12 runs of predictions 1 and 3)

Computed by `train/why_gate.py` from the runs' summary.json and pairs.json:

- **Refuted → stop**: 3b's ratio (T*(simporef) − T*(simpo)) / (T*(plain) − T*(simpo)) ≤ 0.25, or 3a fails (SimPO
  not at least 0.1 below plain, which leaves 3b untestable). The remaining runs are not queued; results are reported
  first.
- **Confirmed** (ratio ≥ 0.5 and 3a holds) **or inconclusive** (ratio between 0.25 and 0.5): continue with
  predictions 2, 4 and 5, flagging an inconclusive result in the report.

## Reporting

Refuted predictions are reported as prominently as confirmed ones. Every run, including failed ones, appears in the
results table. Seed spread (min–max) is shown for every condition.
