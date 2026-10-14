# Addendum 1: agreement weighting on the 99%-agreed set

Written 2026-10-14, before any of these runs started. Same conventions as PREDICTIONS.md (T* on the 3,000-question
MMLU subset, SFT 1.53; "learned" in percentage points; means over 3 seeds).

## Runs

wy_w5hi_s1, wy_w5hi_s2, wy_w5hi_s3: why_hi8.parquet (8,192 pairs, 99.1% agreed under SFT), LR 4e-6, 64 steps, seeds
1-3, agreement weighting DPO_AGREE_K = 5 with DPO_AGREE_NORM = the mean of sigma(-5 x SFT margin) over why_hi8 itself
(set-specific; the regular-data value 0.41 would cut the total push to roughly a sixth and confound "weighting" with
"training less"). Evals as in the wave: letters, format, chat, pairs, gen, bbh_ai2 (40 per task).

Comparison: wy_hi_s1-s3 (plain DPO, same data and seeds): T* 2.28 / 2.26 / 2.26, mean 2.26; learned 22.9%.

## Prediction 6

- 6a. Weighting lowers T* on this set: mean T* at least 0.2 below plain (<= 2.06).
- 6b. It does not overshoot below SFT: mean T* >= 1.45. (Undershooting SFT has only happened when most training
  pushes against SFT's preferences: the 18%-agreed set gave 1.46, swapped labels 1.12.)
- Expected value, not a test: about 1.7-1.9, because the weight moves to weakly agreed pairs (plus the 0.9%
  disagreed ones), which still push in SFT's own direction.

No prediction on learned.

## Reading the outcome

- 6a and 6b hold: weighting redistributes the push even when nearly all pairs are agreed.
- 6a fails (T* stays near 2.26): weighting needs disagreed pairs to shift weight onto; state this as the fix's limit.
- 6b fails (T* below 1.45): reweighting alone can overshoot; the fix would need a cap.
