# Addendum 2: agreement weighting on a 100%-agreed set

Written 2026-10-14, before any result of Addendum 1's runs (wy_w5hi_*) or of these runs was seen. Same conventions as
PREDICTIONS.md (T* on the 3,000-question MMLU subset, SFT 1.53; means over 3 seeds).

## Runs

Data why_hiall8: why_hi8 minus its disagreed pairs (SFT margin <= 0 in the 25%-scale reference cache that built
why_hi8), about 8,118 pairs, all agreed. LR 4e-6, 64 steps, same evals as the wave.

- wy_w5all_s1, s2, s3: agreement weighting, DPO_AGREE_K = 5, DPO_AGREE_NORM = mean of sigma(-5 x SFT margin) over
  why_hiall8 itself.
- wy_hiall_s1: plain DPO on why_hiall8, seed 1 only (check that dropping 0.9% of pairs does not move plain DPO).

Comparisons: plain DPO on why_hi8 (wy_hi_s1-s3: 2.28 / 2.26 / 2.26) and Addendum 1's weighted why_hi8 runs.

## Prediction 7

- 7a. Plain DPO barely changes: wy_hiall_s1 is within 0.08 of wy_hi_s1 (2.28).
- 7b. Weighting lowers T* with no disagreed pairs at all: mean T* of wy_w5all at least 0.2 below wy_hiall_s1.
- 7c. The 0.9% disagreed pairs do not drive the weighted result: mean T* of wy_w5all within 0.1 of the mean of
  wy_w5hi (Addendum 1).
- Expected value, not a test: about 1.8-1.95.

No prediction on learned.

## Reading the outcome

- 7b holds: within agreed pairs, how strongly SFT already prefers the chosen response matters, not only whether it
  does; the account should be stated in terms of margin size, not just the agreed share.
- 7b fails while Addendum 1's 6a holds: the few disagreed pairs carry the 99% result.
- 7a fails: removing 0.9% of pairs changes plain DPO, so 7b is compared with wy_hiall_s1, not wy_hi.
