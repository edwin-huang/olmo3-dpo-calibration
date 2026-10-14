# Changelog of reported numbers

Every change to a number in the write-up ("Why DPO Makes Olmo 3 Overconfident") or the experiment log, with old value, new value and reason. Entries before 2026-10-13 are reconstructed from the review session of 2026-10-12 (four reviewer passes, including a numbers audit against the result files); from now on, changes are logged here when they are made.

## 2026-10-12 — review round

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Setup | "a single temperature removes about 80% of ECE at every stage" | "most of the calibration error of post-trained checkpoints (66–85% on MMLU 5-shot; 79% for Instruct-DPO)" | Only post-SFT checkpoints reach ~80%; base is 22%, Instruct-SFT 66% | temp_scaling.py on results_robust/mmlu5_full |
| Log | §1 | same "80%" claim | "66–85% … only 22% for base" | same | same |
| Main | Fix (small tests) | Fixes' responses "1.6–1.95× SFT's", plain "1.24×" | "1.35–1.46×", plain "1.18×" | Old values were the judge's geometric-mean length ratio; switched to mean tokens, the metric used for the 25%-scale "5% longer" | gen.json mean_tokens |
| Main | Limits | same length figures | same change | same | same |
| Main | Mechanism (now Where) | Ordinary-text top-two gap "×1.03" | "×1.08" | No probe gives 1.03; the bridge probe gives 2.554/2.372 = 1.08 | results_box2/why_bridge.log |
| Main | Where | "Excess confidence … is +0.114 for DPO's immediate answers but only +0.021" | Same values, now "DPO adds +0.114 to SFT's excess confidence … only +0.021" | Values are DPO minus SFT (analyze_perq.py), not DPO's own excess (+0.277) | analyze_perq.py on results_cot |
| Log | §2 | same definition | same correction, adds DPO's own +0.277 | same | same |
| Main | Why bullets | Small-test comparisons against SFT "1.52" | "1.53" | Small tests use the 3,000-question subset, where SFT is 1.53 (1.52 is the full set) | results_ckpt/small/fx_sft/summary.json |
| Main | Chart 2 | SFT reference line 1.52 | 1.53 | same | same |
| Log | §9 header | SFT "T* 1.52, ECE .115, chat T* 1.90" | "T* 1.53, ECE .113, chat T* 2.01" (subset) | same | same |
| Main | Fix 25% table, SFT column | acc/AUROC .611/.773, IFEval .754, chat "≈ 2.0" | .608/.768, .769, 1.90 | Column mixed fast-subset values with full-eval columns | results_ckpt/released/Olmo-3-7B-Instruct-SFT |
| Main | Where | Letters "×1.53" vs cloze ×1.10 | "×1.54" | ×1.53 came from a 2,000-question run; on the full set used for cloze, letters give ×1.54 | results_ckpt/released/*/format.json |
| Log | §2 | header "(2,000 MMLU questions)" | notes cloze is full-set, letters ×1.54 there | same | same |
| Main | Mechanism | Heads after the top three: "L24H5, L23H25, L23H20" | "L30H24, L23H25, L24H5, L23H20" | Ranking by change omitted L30H24 and swapped a near-tie | decomp__…DPO.json |
| Log | §8 | "amplified 25–45%" | "19–66%" | L23H20 is +19%, L30H24 +66% | same |
| Main | Fix | Fixes keep "97–98% of agreed held-out pairs (plain DPO 98%)" | "about 97% (plain DPO 97.6%)" | Values are .968–.975 vs .976 | small/fx_*/pairs.json |
| Main | Summary, Where | "Instruct DPO … largest calibration step" (all settings) | "largest single step on MMLU 5-shot and on ARC; on MMLU 0-shot the Think-SFT step is about as large (+.111 vs +.108)" | MMLU 0-shot exception | results_robust/mmlu0_full |
| Log | §1 | "+.065 to +.108 … the largest step each time" | "+.066 to +.108 … on MMLU 0-shot base → Think-SFT adds +.111" | rounding; same exception | results_robust/arc0_full, mmlu0_full |
| Log | §10 | Weighted 25% run T* "between 1.47 and 1.51" at every snapshot | "1.47 and 1.53" | Step 50 is 1.525 | q25_w5/snap_step_50/summary.json |
| Log | §7 table | Accuracy D4 / delta-only ".613 / .613" | ".615 / .612" | Rounding slip | q25_d4, q25_delta final_full |
| Log | §10 table | fx_base4 judge ".55 / .58" | ".55 / .57" | .58 was the strict prompt's length-adjusted value | judge_vs_small_fx_sft_neutral.json |
| Log | §7 | Share of final log-T* gain "9 / 37 / 76 / 95%" | "11 / 37 / 75 / 94%" | Recomputed against full-MMLU SFT (1.518) | q25_all snapshots |
| Log | §8 | first-token same top-1 "62–68%" | "62% … 68% … 89% for short answers" | Short-answer prompts were left out | firsttok__…DPO.json |
| Log | §3 | "LM head and final norm untouched" | "final norm unchanged, LM head changed about as much as the layers (0.11%)" | LM head relative change 0.0011 | results_mech/ |
| Main | Summary, Fix | "learns at least as much", "fixes slightly more (30% vs 27%)" | "no detectable loss of learning", "about as many (30% vs 27%, within noise)" | Difference is about one standard error on 372 pairs, one seed | pairs.json |
| Main | Why | D4 "removing them raised agreement and sharpening" | "raised the agreed share only from 63% to 65%, which predicts about 1.88 (observed 2.02)" | Agreement explains D4's direction, not its size | agreement_analysis.out |

## 2026-10-13 — after the verification audit (decisions by the author)

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Why | D4 "its weights moved 21% further from SFT" | removed ("another, unidentified cause") | Untraceable to a saved file; both models deleted (VERIFICATION.md finding 2) | session transcript only |
| Log | §7, retracted-claims table | "weights moved 21% further (update cosine with the control 0.59)" | removed | same | same |
| Main | Limits | "barely separates even the released DPO model from SFT (55% win rate)" | cites our 25%-scale plain run (56%) instead; states the held-out pairs and judge prompts are training data for the released model and the full run; full-scale comparison moves to BigBenchHard | Held-out set is inside instruct_full (finding 1) | verify_box.log |
| Log | §10 | "released DPO model 57.3% … judge .55" listed as references | same values, marked not held-out | same | same |

## 2026-10-13 — full-scale run, BBH and drift checks added (main doc)

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Summary lead | fix removes the overconfidence "with no detectable loss … (one seed, 25% of the data)" | "on 25% of the data … (one seed); on all of the data it removes about 60% of it, and responses get much longer" | Full-scale run: T* 1.80 vs DPO 2.34, SFT 1.52 (log-T* and ECE both ~61% of the rise removed) | results_ckpt/full_w5/final_full/summary.json |
| Main | Summary, Fix bullet | "keeps the model as calibrated as SFT" (unqualified) | qualified to 25% scale + full-scale sentence (T* 1.80, 2.4× length, best BBH) | same | same + results_ckpt/bbh/*/bbh_ai2.json, small/fx_sft gen.json (371) |
| Main | Fix | — | new "At full scale it holds only partly" subsection: table (T*/ECE, ARC, chat, acc/AUROC, IFEval, BBH Ai2 protocol, lengths), snapshot T* 1.61/1.75/1.87/1.83/1.82, drift checks | New results | full_w5 snapshots, results_why/stale/analysis.txt, results_why/length_weights.out |
| Main | Fix | "Recommendation for the Olmo recipe: weight each pair … before adopting at full scale, check …" | "What this means for the Olmo recipe": not a drop-in change at full scale; presented as a test of the account | Full-scale result | same |
| Main | Mechanism | "The agreement fixes remove it" ×0.98–1.03 | "…up to 25% scale" + ×0.99 at 25%; new bullet: full scale ×1.12, L22H6 +0.14, signal ×1.04, RMS ×1.08 | why.py decomp/norm on q25_w5 and full_w5 | results_why/fixscale/ |
| Main | Limits | fix "on one 25%-scale run, not the full 260K-pair stage"; "confirmed on one 25%-scale run"; "Ai2's standard post-training evals were not run" | "one 25%-scale run and one full-scale run, where it holds only partly"; "tested on one run at each larger scale"; "Apart from BigBenchHard, …" | same | same |
| Main | Limits, softening | small-test softening only | adds full scale ×0.79 (short-answer first tokens), ×0.91 (ordinary text) | why.py bridge on full_w5 | results_why/fixscale/bridge.full_w5.log |
| Main | Related, what's new | fix "removes the effect" | "at 25% scale (about 60% of it at full scale)" | Full-scale result | full_w5 summary |
| — | (chat only, never in docs) | "full-scale capability collapse: BBH .32" | retracted | 3-shot harness truncated at 1,536 tokens; Ai2-protocol BBH .641 (best) | results_ckpt/bbh/full_weighted/ |
| Log | new §11 | — | full-scale table, snapshot table, BBH harness table (Ai2 protocol, fixed extraction, 3-shot, tokens), drift checks (stale weights, length), mechanism-at-scale table | New results; full-data normalizer .4130 vs .4135 used (0.1%) | full_w5 summaries, results_ckpt/bbh/, results_why/stale/, results_why/fixscale/, ref cache 89203f41147b0c40.pt |
| Log | Retracted claims | — | rows: "weighting keeps SFT-level calibration at realistic scale" (partly refuted); "full-scale weighting collapses BBH (.32)" (retracted, chat only) | same | same |

## 2026-10-14 — pre-registered wave results added

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Summary, Why bullet | — | adds "A pre-registered three-seed replication confirmed five of six predictions: … 2.27 vs 1.46" | Wave complete | why_tests/WHY_RESULTS.md §2–4 |
| Main | Why | — | new "Pre-registered replication" subsection with six-row table | same | results_ckpt/small/wy_*/summary.json |
| Main | Limits | "One seed, high learning rate … repeating the contrast … would make the causal claim much stronger" | "Small runs, one dataset": contrasts replicate at LR 4e-6, 3 seeds, matched data; not at 25%/full scale | The suggested replication was run | same |
| Main | Related, what's new | data-level account | adds "with the key contrasts replicated in a pre-registered three-seed test" | same | same |
| Log | new §12 | — | nine-condition table (T* per seed, chat T*, learned, tokens, BBH) + prediction/threshold/outcome table | same | same |
| Log | Code index | — | row "Pre-registered replication" | same | — |
