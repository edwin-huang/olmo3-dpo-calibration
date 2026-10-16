# Testing the "why": results

Brief 2 (tmp/brief2.md). The account under test: DPO's per-pair gradient weight is computed relative to the reference
model, so at the start of training every pair gets the same push whether or not the SFT model already prefers the
chosen response; pushing pairs the model already agrees with makes it overconfident.

## 0. Does Olmo 3's DPO use length-normalized or standard DPO?

Length-normalized. Ai2's script `scripts/train/olmo3/7b_instruct_dpo.sh` (open-instruct @ 11826255) passes no
`--loss_type` or `--beta`, so the defaults apply: `open_instruct/dpo_utils.py:101` `loss_type: DPOLossType =
DPOLossType.dpo_norm` and `:99` `beta: float = 5.0`. Our runs use the same. SimPO's per-token margin is therefore on
the same scale as the margin in our DPO runs; SimPO differs only in dropping the reference and adding a target margin.

Side note from the same script: Ai2's own post-training evals use zero-shot chain-of-thought formats
(`bbh:cot::hamish_zs_reasoning_deepseek_v2`, `mmlu:cot::hamish_zs_reasoning_deepseek`), so our 3-shot BBH will not
reproduce their absolute BBH numbers.

## 1. Implied per-pair weights at the start of training (no GPU)

Script: `train/implied_weights.py`; output: `why_tests/implied_weights.txt`. Input: open-instruct's reference cache
for all 259,922 Instruct DPO pairs (`/workspace/ref_cache/89203f41147b0c40.pt`, per-token-averaged log-probs of
Instruct-SFT), 63.1% agreed. At step 0 the policy equals SFT, so each method's weight on a pair is a function of SFT's
log-probs. "Share on agreed" = fraction of the total starting gradient weight that lands on agreed pairs (plain DPO =
the agreed share itself).

| Method (hyperparameters; source) | Share of weight on agreed pairs | Mean weight, agreed / disagreed | Spearman with SFT margin |
|---|---|---|---|
| DPO, length-normalized (β 5; Olmo 3 recipe) | 0.631 | 1.00 | flat |
| IPO | 0.631 | 1.00 | flat |
| Agreement-weighted DPO (k 5; ours) | 0.307 | 0.26 | −1.00 |
| SimPO (β 10, γ/β 0.3; SimPO README, Llama3-Instruct v0.2) | 0.448 | 0.48 | −1.00 |
| SimPO (β 2.5, γ/β 0.55; README, Llama3-Instruct) | 0.524 | 0.65 | −1.00 |
| SimPO (β 2.5, γ/β 0.1; README, Mistral-Instruct) | 0.443 | 0.47 | −1.00 |
| SimPO (β 5, γ/β 0.3; our small test) | 0.451 | 0.48 | −1.00 |
| ORPO odds-ratio term (Hong et al., 2024) | 0.459 | 0.50 | −0.98 |
| WPO (open-instruct `wpo_loss`) | 0.574 | 0.79 | −0.28 |

Reading:

- The Spearman column is −1 by construction for every method whose starting weight is a monotone function of SFT's
  margin (ours, SimPO), so it is reported only as a sanity check. The informative columns are the first two.
- SimPO and ORPO's odds-ratio term already move weight off agreed pairs at step 0 (44–52% of the weight vs DPO's 63%),
  but less than our explicit weighting (31%). SimPO's weights also keep changing during training (they depend on the
  current model's own margin, so agreed pairs lose weight further as their margins grow); the starting share
  understates its effect. This fits SimPO's single-seed small test landing near SFT's calibration (T* 1.64).
- WPO weights pairs mostly by how likely both responses are, not by agreement (correlation −0.28): it barely shifts
  weight off agreed pairs (57%). Prediction to record: WPO should calibrate about like plain DPO.
- By source, our weighting moves weight from delta-learning pairs (48% of DPO's weight → 39%) to LLM-judged pairs
  (48% → 55%), because delta-learning pairs are 74% agreed vs 54%.
- Not yet computed: the Selective DPO / "Less is More" difficulty proxy (implicit margin from our 25%-scale plain DPO
  model on pairs it did not train on). It needs a GPU scoring pass and waits for the GPUs.

## 2. Phase 1: predictions 1 and 3 (12 runs, 3 seeds; finished 2026-10-14 03:05 UTC)

All runs: s8_a (8,192 pairs, 64% agreed), LR 4e-6, 64 steps, seeds 1-3 (data order). T* on the 3,000-question MMLU
subset (SFT 1.53); learned = held-out disagreed pairs now right (372 pairs, one SE about 2 points); BBH = Ai2's
zero-shot protocol on the first 40 questions of each task (1,080; one SE about 1.5 points). An earlier wave was
stopped after one run (wy_plain_s1, T* 1.844); this relaunch reran it (1.839).

| Condition | T* (s1 / s2 / s3) | Mean | ECE | Chat T* | Learned (mean, range) | Agreed kept | Tokens | BBH |
|---|---|---|---|---|---|---|---|---|
| Plain DPO | 1.84 / 1.78 / 1.79 | 1.80 | .142 | 2.66 | 17.7 (17.5-17.7) | .984 | 403 | .472 |
| Agreement-weighted | 1.50 / 1.50 / 1.50 | 1.50 | .104 | 1.93 | 16.2 (15.9-16.7) | .987 | 400 | .479 |
| SimPO | 1.53 / 1.53 / 1.49 | 1.52 | .105 | 1.99 | 19.6 (18.8-21.0) | .979 | 402 | .488 |
| SimPO with reference | 1.89 / 1.85 / 1.84 | 1.86 | .148 | 2.81 | 21.5 (20.2-22.3) | .982 | 406 | .474 |

Accuracy .614-.619 and AUROC .766-.774 in every run.

- **Prediction 1: CONFIRMED.** Weighted T* is lower than plain in every seed pair, by 0.30 on average (needed 0.15);
  learned is 1.4 points lower on average (allowed: 2).
- **Prediction 3: CONFIRMED (stop rule: continue).** 3a: SimPO is 0.28 below plain (needed 0.1) and learns 2.0 points
  more. 3b: ratio (1.86 - 1.52) / (1.80 - 1.52) = 1.21 (needed 0.5). Adding DPO's reference to SimPO restores all
  of plain DPO's overconfidence, while learning the most of the four.
- Not predicted: no length cost for any method at LR 4e-6 (400-406 tokens vs 403), unlike LR 8e-6.

Phase 2 (predictions 2, 4, 5; 15 runs) queued automatically at 03:05 UTC.

## 3. Phase 2: predictions 2, 4 and 5 (15 runs, 3 seeds; finished 2026-10-14 09:55 UTC)

Same settings as phase 1. why_hi8 / why_lo8: 8,192 pairs each from the q25 pool, matched on source mix, task type,
chosen-length decile and rejected-length quantile; agreed 99.1% / 17.7%. why_easy8: the half of an unseen 16,384-pair
pool with the highest implicit margin under our 25%-scale plain DPO model (77.3% agreed). IPO: (h - 1/(2 tau))^2,
tau 1 (first-step loss 0.25 = analytic).

| Condition | T* (s1 / s2 / s3) | Mean | ECE | Chat T* | Learned (mean, range) | Agreed kept | Tokens | BBH |
|---|---|---|---|---|---|---|---|---|
| DPO, high agreement | 2.28 / 2.26 / 2.26 | 2.26 | .187 | 4.11 | 22.9 (22.6-23.1) | .986 | 404 | .471 |
| DPO, low agreement | 1.46 / 1.45 / 1.47 | 1.46 | .093 | 1.78 | 19.3 (18.5-19.9) | .975 | 405 | .485 |
| DPO, easy subset | 1.96 / 2.00 / 1.99 | 1.98 | .164 | 3.18 | 29.2 (28.2-29.8) | .979 | 448 | .477 |
| IPO, high agreement | 1.71 / 1.67 / 1.71 | 1.70 | .130 | 2.34 | 11.9 (11.8-12.1) | .993 | 373 | .457 |
| IPO, low agreement | 1.49 / 1.44 / 1.49 | 1.47 | .099 | 1.77 | 13.7 (13.2-14.2) | .987 | 400 | .459 |

Accuracy .613-.619 and AUROC .765-.774 in every run.

- **Prediction 2: CONFIRMED.** High > low in every seed, by 0.81 on average (needed 0.3). The high-agreement set
  also learns more (22.9 vs 19.3), so the gap is not less training on the low set; length and BBH are equal. The
  low-agreement run ends slightly below SFT (1.46 vs 1.53).
- **Prediction 4: NOT SUPPORTED as stated.** The T* threshold is met (1.98 vs plain 1.80, +0.18 >= 0.15), but the
  prediction was "more overconfident at matched learning" and the easy set learns 11.5 points MORE. On plain DPO's
  learning-rate frontier for s8_a (fix wave: 28.8% learned at T* 2.40), the easy set (29.2% at 1.98) is less
  overconfident per unit of learning, not more. The sub-prediction (> 85% agreed) failed before training (77.3%).
  Different pairs from s8_a, so subset noise (about 0.2 T*) applies; responses 11% longer.
- **Prediction 5: CONFIRMED.** IPO lowers T* on the high-agreement set (1.70 vs DPO's 2.26) and IPO-high stays above
  IPO-low in every seed (gaps 0.22 / 0.23 / 0.22; mean 0.23, needed 0.15). Caveat: IPO learns about half as much as
  DPO (12-14%), and here the low-agreement set learns slightly more than the high one, so the residual gap is not a
  learning artifact. Capping margins removes about 70% of the agreement effect; the rest remains.

## 4. Summary of the wave

| Prediction | Result |
|---|---|
| 1. Agreement weighting lowers T* (>= 0.15) at <= 2 points less learning | Confirmed (0.30; -1.4) |
| 2. High- vs low-agreement, matched source and length (>= 0.3) | Confirmed (0.81, every seed) |
| 3a. SimPO below plain DPO's trade-off | Confirmed (-0.28 T*, +2.0 learned) |
| 3b. Adding the reference back restores >= half the gap | Confirmed (ratio 1.21) |
| 4. Easy subset more overconfident at matched learning | Not supported as stated (T* +0.18 but learns +11.5) |
| 5. IPO lowers T* but agreement still matters (>= 0.15) | Confirmed (1.70 vs 2.26; hi - lo 0.23) |

All 27 models: ehuang5/olmo3-calibration-models, wave/wy_*. Results: results_ckpt/small/wy_*.

## 5. Addendum 1: agreement weighting on the 99%-agreed set (3 runs; finished 2026-10-14 15:43 UTC)

wy_w5hi_s1-s3: why_hi8, LR 4e-6, seeds 1-3, DPO_AGREE_K 5, set-specific DPO_AGREE_NORM 0.1179 (mean raw weight over
why_hi8; mean SFT margin +0.82, so most raw weights are tiny). The 0.9% disagreed pairs carry 4.6% of the weight;
the top 10% of pairs carry 31%. Correction to the addendum's rationale (not a tested prediction): the regular-data
normalizer would have cut the total push to about 29% (0.118 / 0.41), not "roughly a sixth".

| Condition | T* (s1 / s2 / s3) | Mean | ECE | Chat T* | Learned | Agreed kept | Tokens | BBH |
|---|---|---|---|---|---|---|---|---|
| Plain DPO (wy_hi, wave) | 2.28 / 2.26 / 2.26 | 2.26 | .187 | 4.11 | 22.9 | .986 | 404 | .471 |
| Agreement-weighted (wy_w5hi) | 2.18 / 2.19 / 2.20 | 2.19 | .181 | 3.73 | 27.8 | .984 | 456 | .483 |

Accuracy .618-.619, AUROC .768-.770.

- **6a: NOT SUPPORTED.** Mean T* is 0.08 below plain (2.2647 - 2.1893 = 0.075; needed 0.2; seed spread 0.02).
- **6b: holds** (mean 2.19 >= 1.45), trivially.
- Reading (as pre-specified): weighting needs disagreed pairs to shift weight onto. On this set it can only move
  weight from strongly to weakly agreed pairs, and those sharpen almost as much, while learning more (27.8% vs 22.9%).
  What drives the overconfidence is training on pairs SFT already gets right, largely regardless of how strongly it
  prefers them: the sign of SFT's margin, not its size.

## 6. Addendum 2: agreement weighting on a 100%-agreed set (2 runs; finished 2026-10-14 ~17:30 UTC)

why_hiall8: why_hi8 minus its 73 disagreed pairs (8,119 pairs; margins from the 25%-scale reference cache that built
why_hi8; recomputing why_hi8's normalizer from that cache gives 0.1179, identical to the trainer's own cache).
Set-specific DPO_AGREE_NORM 0.1135. **Deviation from the pre-registration:** after the plain check seed and weighted
seed 1, the user stopped the queue (seeds 2-3 not run), since Addendum 1 already predicted the outcome and seeds vary
by about 0.02 in T*. 7b and 7c are therefore checked on seed 1, not on 3-seed means.

| Run (seed 1) | T* | ECE | Chat T* | Learned | Agreed kept | Tokens | BBH |
|---|---|---|---|---|---|---|---|
| Plain DPO (wy_hiall_s1) | 2.32 | .197 | 4.29 | 22.6 | .987 | 402 | .474 |
| Agreement-weighted (wy_w5all_s1) | 2.24 | .187 | 3.88 | 27.7 | .982 | 453 | .494 |

- **7a: holds.** Plain DPO on why_hiall8 (2.32) is within 0.08 of wy_hi_s1 (2.28); dropping the disagreed pairs
  moved it up slightly, as the account expects.
- **7b: NOT SUPPORTED** (single seed). Weighting lowers T* by 0.08 (needed 0.2).
- **7c: holds** (single seed). 2.24 is within 0.1 of Addendum 1's weighted mean (2.19): the 73 disagreed pairs did
  not drive that result.
- Reading: with no disagreed pairs at all, moving the push from strongly to weakly agreed pairs removes only about
  0.08 of T*. Whether SFT already gets a pair right matters far more than by how much (sign, not size); margin size
  has a small effect. Weighting fixes overconfidence by shifting weight onto disagreed pairs, so it does little on
  data that has almost none.
