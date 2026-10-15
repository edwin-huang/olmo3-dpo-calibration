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

## 2026-10-14 — charts replaced; rounding fix

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Why | agreement-share scatter (17 single runs, corr .78) + bullet "What the chart's correlation rests on" | removed; replaced by a dot chart of the 27-run replication (seeds as dots) | Superseded by the matched three-seed contrast | results_ckpt/small/wy_* |
| Main | Fix | calibration–learning frontier chart; "fixes learn more than any plain-DPO run (35.5%, 40.3%, 43.5% vs 28.8%)" | removed; T* values kept; "learn more" claim dropped | Learned favors the fixes by construction; at LR 4e-6 weighting learns about the same as plain (16.2% vs 17.7%) | why_tests/WHY_RESULTS.md §2 |
| Main | Fix, 25% scale | "learning difference … smaller than in the small tests (… 35.5% vs 28.8% there)" | "learning difference (30.4% vs 27.4%) is within noise" | Depends on the removed frontier | — |
| Main | Why replication table | T* column per test | outcome column only (T* now in the chart) | Show each number once | — |
| Main, Log, WHY_RESULTS | high-agreement mean T* | 2.27 | 2.26 | Mean of 2.279, 2.257, 2.258 is 2.2647; 2.27 came from rounding a printed 2.265 again | wy_hi_s*/summary.json |

## 2026-10-14 — two reviewer passes (Ai2-reader + numbers audit) applied

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Related, fix bullet | weighting "or … SimPO … removes the effect at 25% scale (about 60% at full scale)" | weighting at 25%/full; SimPO "in small tests" | SimPO never run at 25% or full scale | results_ckpt/ (no SimPO q25/full run) |
| Main | Replication table, paragraph | SimPO "learns 1.9 points more"; fixes "400–406 tokens" | 2.0; "400–402 tokens vs plain's 403" | 19.624 − 17.652 = 1.97; 406 is SimPO+ref, not a fix | wy_*/summary.json |
| Main | Summary Fix; full-scale paragraph | BBH "best score of the three"; "4× longer"; "about 60%" | adds IFEval −3.5 and "reasons 4.4× longer"; "about 60% on MMLU, about 40% on ARC, all of it in chat format" | 2,685/607 = 4.42; ARC log-T* removed 43%; chat 1.87 < SFT 1.90 | bbh_ai2.json; full_w5 & released summaries |
| Main | Full-scale intro | — | baseline is Ai2's released model, not our rerun (seed, order, hardware uncontrolled) | Unmatched comparison | — |
| Main | Full-scale table | released DPO ECE .206 | .205 | Same value as Summary/Where (calib harness .2053; eval_ckpt .2061) | results_robust/mmlu5_full |
| Main | Why, D4/delta paragraph | "close to the 2.04 the small-test fit predicts"; "which predicts about 1.88" | removed (fit was defined only by the removed scatter) | Undefined fit | verification/agreement_analysis.out |
| Main | Where | AUROC ".766 → .767", base→SFT ".797 → .766" | ".77 in both", ".80 → .77" | Two harnesses (.768/.766 in tables) | released/*/summary.json |
| Main | Fix 8K paragraph | IFEval "vs SFT's .754" | notes SFT scored .754 and .769 in two evaluations (noise ~2 points) | Same model, two runs | fx_sft vs released SFT ifeval.json |
| Main | Mechanism | margin slope "0.84–0.93 vs 1.35" under a 25%-scale heading | adds "in small tests; 0.97 vs 1.27 at 25% scale" | Scope | pairs.json |
| Main | Limits | "half the original learning rate"; judge "56%"; softening "grows"; chat T* "3.97"; on-policy "agreed share near 50%" | "LR 4e-6 (4× Ai2's rate…)"; "56% (57% length-adjusted)"; "persists"; "3.77"; "SFT prefers its own samples → mostly disagreed" | Ambiguity; variant; filter is ×0.72; tables use 3.77 (full set); matches PREDICTIONS.md | various |
| Main | Summary, Why, What this means | "trains on those as hard as"; "more confident everywhere"; recipe-only paragraph | "same starting weight"; "whenever it answers immediately"; adds what the full-scale leftover means for the account (normalization route not explained) and a data-construction sentence | Reviewer: overclaims / missing implication | — |
| Log, WHY_RESULTS | §12 / §2, §4 | SimPO +1.9 learned | +2.0 | Rounding | wy_*/summary.json |

## 2026-10-14 — dense / cuttable / attackable review applied (main doc)

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Summary "Inside the model"; Mechanism transplant bullet | "30 heads and the layer 19–30 MLPs reproduce 76% (30 random heads: 1%)"; top-100 53%, MLPs 49% | top 30 heads 35% vs 1% for 30 random heads; + MLPs 76%; all layers 15–31 100%; "no random-MLP control"; 53%/49% moved to log | Control was not size-matched to the 76% set | results_why/transplant__*.json |
| Main | Where | "top-two gap on ordinary text grows only ×1.08" next to T* ×1.54 | adds same-unit comparison: top-two gap ×1.51 for letters vs ×1.08 wikitext | Different units compared | results_box2/why_bridge.log |
| Main | Fix list | "Agreed pairs drop from 64% to 31% of the total weight" | adds "disagreed pairs get about 1.9× their plain-DPO weight" | Rescaling confound | .69/.36 = 1.92 (s8_a); .693/.369 = 1.88 (full) |
| Main | Full-scale drift | one reading | adds second untested reading (rescaling moves the run further) | same | stale/analysis.txt |
| Main | What this means | "removes the agreement-driven part … at every scale" | "removes most of the gain in the answer signal (about 80% at full scale) … not tested whether the normalization part depends on agreement" | Overclaim | results_why/fixscale/ |
| Main | Why / Mechanism / Limits | D4 filtered run in Why and "×1.29 for the filtered 25% rerun" | removed from Why/Mechanism; one sentence in Limits (unexplained) | Undefined there; does not support the argument | — |
| Main | Why | "Side observation: which loss term" bullet | removed (in the log) | Inconclusive by its own statement | — |
| Main | Setup, Key terms, full scale | GPU list; small test "8× Ai2's LR"; snapshot list | GPU list cut; "2–8×, stated with each result"; "about 1.85 by mid-run" | Density / accuracy | — |
| Main | Setup | chat format undefined | "letter read right after an 'Answer:' prefix, with no reasoning" | Pre-empts the length/reasoning objection to "all of it in chat format" | eval_ckpt.py part_chat |

## 2026-10-14 — new box (RTX PRO 6000): usability check + random-MLP control

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Summary "Inside the model"; Mechanism bullet; Related "A mechanism" | change lives in SFT's answer heads AND late MLPs (heads + MLPs 76%) | heads specific (35% vs 1% random); MLP change spread out (12 random MLP layers give 28–53%; late MLPs 49%); random heads+MLPs 54% vs 76% | Size-matched random-MLP control | results_why/controls/transplant_controls__SFT__DPO.json |
| Main | Summary Fix; full-scale paragraph and table; What this means | no judged-quality number at full scale | LLM judge vs SFT: weighted .40 raw / .50 length-adjusted (neutral), released DPO .55 / .55; finished answers .46 | Regenerated 300 chat answers (old box lost) | results_ckpt/gen/*/judge_vs_small_fx_sft_neutral.json |
| Main | Summary, Key terms, Where heading, Why | "immediate answers", "Our interpretation of the results below", dense margin bullet, "the imbalance", "them" | "commits to an answer"/"Answer token"; "results that follow"; rewritten bullet; explicit referents | User reading pass | — |
| Log | Retracted claims | — | rows: late-answer-MLP claim (refuted), full-scale no judged cost (refuted) | same | same |

## 2026-10-14 — Fix section trimmed (user decision)

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Fix | 8K fix-wave list (weighting, filter, SimPO, controls) + results paragraph (T* 1.51–1.64 vs 2.40; label smoothing 2.14, chosen-likelihood 2.08) + learned-metric caveat paragraph + 8× length (1.35–1.46×) | one paragraph defining agreement-weighted DPO; results deferred to the replication; filter and controls only in the log | Superseded by the three-seed replication; controls need the removed learning-rate frontier | — |
| Main | Key terms, Learned | definition only | adds bias caveat (favors up-weighting disagreed pairs; agreed held-out pairs 97–99% right) | Caveat moved from deleted paragraph | — |
| Main | Mechanism; Limits | letter scale ×0.98–1.03, slope 0.84–0.93; softening ×0.93–0.95 text, ×0.72–0.82 first token | ×0.99–1.03, 0.90–0.93; ×0.94–0.95, ×0.82 | Filter run removed | results_why/fix/*.log, small/fx_*/pairs.json |
| Main | Related | "account predicted which standard variants would work (label smoothing, chosen-likelihood would not)" | pre-registered SimPO with/without reference | Controls removed | — |
| Main | Limits, softening | small-test ×0.94–0.95 / ×0.82 and "may explain their longer responses" | 25% ×0.96 / ×0.86, full ×0.91 / ×0.79; length tied to full scale | Small-test values came from the removed 8× runs; contradicted "no lengthening at 4e-6" | results_why/fixscale/bridge.*.log |
| Main | Mechanism, Why, Limits | wording | letter score defined (logits, letter scale), "answer heads" defined, "between SFT and DPO", Ai2's data-building LLM judge named, Limits rewritten | User reading pass | — |

## 2026-10-14 — Addenda 1-2 (99% / 100% agreed weighting) added

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main | Summary Why; Why (after replication); Fix definition | — | "what matters is whether SFT already gets a pair right, not by how much"; follow-up paragraph (2.26 → 2.19, 3 seeds; 2.32 → 2.24, 1 seed; mixed 1.80 → 1.50); "works by moving weight onto disagreed pairs, so on data with almost none it does little" | New pre-registered results | why_tests/WHY_RESULTS.md §5-6; results_ckpt/small/wy_w5hi_*, wy_hiall_s1, wy_w5all_s1 |
| Log | §12 follow-ups; retracted claims | — | two tables (runs, predictions 6a-7c) + paragraph; row "weighting helps on almost fully agreed data: not supported" | same | same |

## 2026-10-15 — prediction table and chart (user review)

| Doc | Where | Old | New | Reason | Source |
|---|---|---|---|---|---|
| Main, Log, README | Prediction 4 wording | "more overconfident at matched learning" | "more overconfident with no increase in learning" (still Not supported) | Clearer statement of the same pre-registered premise | why_tests/PREDICTIONS.md |
| Main | Replication table | wave predictions only | adds the two follow-up rows (99%: 2.26 → 2.19; 100%, one seed: 2.32 → 2.24; both Not supported); intro sentence and follow-up paragraph adjusted, numbers now only in the table | Predictions in one place | WHY_RESULTS.md §5–6 |
| Main | Replication chart | easy-pairs row carried a "Not supported" note | note removed (predictions kept out of the chart); dashed plain-DPO reference kept | User | — |
| Log, WHY_RESULTS | Addendum 1 effect size | 0.07 (and "about 0.07–0.08") | 0.08 (0.075; 100% set 0.079) | Mean of seeds 2.2647 − 2.1893 = 0.0754, rounded from unrounded means | wy_hi_s*, wy_w5hi_s* summary.json |
| Main | Replication chart | groups: weighting, reference, agreement (DPO, IPO), easy | adds "Weighting (99%)" group (plain 2.26 vs weighted 2.19, Δ 0.08); first group renamed "Weighting (mixed)"; legend now "push shifted onto pairs SFT gets wrong" / "push stays on pairs SFT already gets right" | Shows the follow-up next to the main contrasts; old legend did not fit the weighted 99% run | same |
