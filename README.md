# Nyansapo — RAG-Grounded AI Tutoring System

**Group 1 | KNUST Department of Computer Science | 2025–2026**
**Supervisor: Dr. Eric Opoku Osei**

## Project Overview

Nyansapo is a Corrective Retrieval-Augmented Generation (CRAG) AI tutoring
system that answers KNUST student questions using official lecture content,
preventing hallucination through retrieval grounding and a cost-weighted
intent classifier.

## System Architecture

- **Intent Classifier**: Nyansapo-QIC-WCE (Weighted Cross-Entropy, REPLACE operation)
- **Retrieval**: Hybrid BM25 + BGE-M3 Dense (RRF fusion)
- **Evaluator**: CRAG keyword-overlap groundedness scorer
- **Generator (Online)**: Phi-2 (2.7B, 4-bit NF4) via FastAPI
- **Generator (Offline)**: Phi-4-mini (3.8B) via Ollama — no internet required
- **Interfaces**: React Native mobile app · USSD (Africa's Talking sandbox) · Web app

## Key Results

All numbers below trace to a single commit:
**`2bbfd9aa77d27909f4f72d0c180fd83f78b76555`** — the only commit hash cited
anywhere in this repository's documentation.

| Metric | Nyansapo CRAG / Engineered | Baseline / Comparator | Statistic |
|---|---|---|---|
| QIC-WCE Macro F1 | 0.7074 ± 0.1306 | 0.1976 ± 0.0836 (baseline) | p=0.000088, d=3.0005 |
| Ablation (uniform weights) | 0.1976 ± 0.0836 | identical to baseline | mathematically exact |
| Code-fix-only effect (X5 isolation) | 0.7074 ± 0.1306 | 0.6993 ± 0.1202 (leaked, same env) | +0.0081, p=0.642, d=0.058 — not significant |
| Gold ROUGE-L vs BM25-only | 0.2283 ± 0.1410 | 0.1733 ± 0.1364 | p=0.000001, d=0.396 |
| Gold ROUGE-L vs Vanilla LLM | 0.2283 ± 0.1410 | 0.2080 ± 0.0663 | p=0.056 — CLEARED AS INCONCLUSIVE (see Certificate below) |
| Gold BERTScore F1 | 0.7897 ± 0.0753 | 0.7963 ± 0.0318 (Vanilla) | p=0.325 (not significant) |
| Offline latency | 3.60s (M1 Pro, no internet) | 10.55s (online, T4 GPU) | 2.93× faster offline |
| Student trust (n=298) | 4.1795 ± 0.8343 | vs midpoint 3.0 | t=24.33, p<0.000001 |
| NASA-TLX cognitive load | 1.8761 ± 0.9051 | (lower = better) | — |
| Gender bias in trust | none detected | Male 4.22 vs Female 4.14 | p=0.4307 |

All classifier comparisons: Wilcoxon signed-rank, 20 seeds. All RAG comparisons:
Wilcoxon signed-rank, Bonferroni-corrected α=0.0167, n=200 gold pairs.

**Deployment model**: seed 789 (F1=0.9526), selected post-hoc after all 20 runs
completed, based on test performance. Disclosed as a permanent limitation —
the test set is not a held-out estimate for the deployed model specifically
(Methods §A6). This limitation is specific to the classifier arm and does not
affect the CRAG vs Vanilla LLM RAG-arm claim below.

## X5 — Confound Isolation (Code Fix vs Library Version Change)

An earlier run reported an engineered Macro F1 of 0.6214 (pre-fix, leaked
vectoriser) rising to 0.7074 (post-fix) — a +0.086 shift computed across two
commits in which the scikit-learn/PyTorch runtime also changed. To isolate
the cause, the pre-split leak was deliberately reproduced and run directly
alongside the fix, same environment, same 20 seeds:

- Leaked (reproduced, current environment): 0.6993 ± 0.1202
- Fixed (train-only fit): 0.7074 ± 0.1306
- **Isolated code-fix-only effect: +0.0081 (p=0.642, d=0.058) — not significant**

The historical +0.086 gap is attributable primarily to the concurrent library
version change, not the C1 methodological fix. The fix remains methodologically
necessary — it closes a genuine test-set leak — regardless of its numerical
effect. The primary baseline-vs-engineered result (0.1976 vs 0.7074) is
unaffected, as both conditions were computed in the same environment.
Full data: `data/x5_isolation_v3.json`.

## Null Clearance Certificate — CRAG vs Vanilla LLM

| Condition | Check | Observed | Verdict |
|---|---|---|---|
| C1 | Sign stability (2000-resample bootstrap) | flip fraction = 0.009 | PASS |
| C2 | Leave-one-fold-out stability (5-fold) | same sign, 5/5 folds | PASS |
| C3 | Minimum detectable effect / power | achieved power = 0.4436, n≥413 required | **FAIL** |
| C4 | Baseline tuning budget parity | 0 search constructs, either arm | PASS |
| C5 | Test set scored once | single evaluation pass | PASS |
| C6 | Seed-selection dependency | not applicable to RAG arm | PASS |

**Result: CLEARED AS INCONCLUSIVE.** 5 of 6 structural conditions pass; only
statistical power fails, at the current sample size (n=200). This is a
sample-size limitation, not a structural or methodological one — the claim
is neither confirmed significant nor evidence of no effect.
Full data: `data/null_clearance_certificate_v3.json`.

## Zero-ROUGE-L (Non-Empty) Residue

Answers that are non-empty but score exactly 0.0 ROUGE-L against the gold
answer — a generation-quality failure distinct from an empty response:

| System | Empty | Zero-ROUGE (non-empty) |
|---|---|---|
| CRAG | 0 | 10 |
| Naive RAG | 0 | 11 |
| BM25 only | 0 | 27 |
| Vanilla LLM | 0 | 0 |

Manual inspection of the 27 BM25 cases shows the predominant failure mode is
truncated list-fragment generation (e.g. `"1.18"`, `"1. TCP/IP"`,
`"Solution 0:"`) — the generator begins a numbered-list response, likely
triggered by keyword-matched chunks containing list headers or slide
numbering, and truncates almost immediately. Full data and examples:
`data/zero_rouge_residue_v3.json`.

## Repository Structure

```
Nyansapo-RAG-Research/
├── README.md
├── requirements.txt
├── .gitignore                          # explicit !data/*.json, !data/*.pkl, !data/*.pt exceptions
├── 00_CLOSEOUT_GROUP_1_20260913.py     # supervisor's audit script, v3-patched
├── Nyansapo_Poster.pptx                # conference poster
├── notebooks/
│   └── Nyansapo_RAG_Pipeline.ipynb     # single canonical pipeline, v3 — writes all results to
│                                        # data/ and figures/ (retrieval assets stay on Drive)
├── data/
│   ├── eval_results.json               # source queries for classifier training set
│   ├── gold_qa_pairs.csv               # 200 human-annotated QA pairs
│   ├── gold_system_answers.csv         # all 4 systems' generated answers
│   ├── gold_rouge_v3.json
│   ├── gold_bertscore_v3.json          # per-item BERTScore F1, 4 systems, n=200
│   ├── gold_eval_checkpoint_v3.json
│   ├── qic_results_v3.json
│   ├── perturbation_v3.json
│   ├── user_study_anonymized.csv       # n=298, timestamp/comments stripped
│   ├── user_study_v3.json              # trust-study statistics computed by the notebook
│   ├── x5_isolation_v3.json            # confound isolation experiment
│   ├── null_clearance_certificate_v3.json
│   ├── zero_rouge_residue_v3.json
│   ├── qic_vectorizer_v2.pkl           # deployment classifier (seed 789)
│   └── qic_wce_model_v2.pt
├── figures/
│   ├── system_architecture.png
│   ├── classifier_architecture.png
│   ├── data_partition_v3.png           # corrected split, 101/22/22; producer = citation
│   ├── confusion_matrices_v3.png
│   ├── convergence_v3.png
│   ├── pr_roc_v3.png
│   ├── feature_importance_v3.png
│   ├── perturbation_v3.png
│   └── gold_eval_v3.png
├── docs/
│   ├── Group1_Method_and_Results.docx  # merged manuscript, current round
│   ├── Group1_CoverSheet.docx
│   └── Group1_All_Feedback_Reports.pdf # complete chronological archive
└── offline/
    └── nyansapo_offline.py             # offline deployment (Ollama/Phi-4-mini)
```

**Not committed, hosted externally on Google Drive** (file size): the FAISS
retrieval index (12,311 chunks), chunk text (`chunks_v2.pkl`), and BM25 index
(`bm25_v2.pkl`). This is stated here explicitly — the repository alone does
not permit independent regeneration of the retrieval layer, only of the
classifier, RAG evaluation, and user study components.

## Reproducibility

Every number in this README and in `docs/Group1_Method_and_Results.docx`
traces to a single notebook run committed at
`2bbfd9aa77d27909f4f72d0c180fd83f78b76555`. This is enforced as a single-hash
policy: no other commit hash appears anywhere in this repository's
documentation. This is the commit containing the notebook, data files,
figures and model artefacts. The manuscript, cover sheet and README were
committed afterwards in a documentation-only commit, which by construction
cannot cite its own hash. The notebook was re-run top to bottom with the repository cloned. Every
cell that produces a result file or figure writes to the committed `data/`
or `figures/` path. A text scan of the committed notebook for the Google
Drive path returns exactly three lines, all in Cell 4 (the retrieval assets
listed above as hosted externally).

One canonical training function (`train_model_v2`) is used for baseline,
engineered, and ablation conditions. The TF-IDF vectoriser and class weights
are fit inside each seeded split, on the training partition only — verified
by a vocabulary-SET self-test (149 of 500 tokens present only in the
full-dataset fit are confirmed absent from the training-only fit).

Framework versions are pinned in `requirements.txt` to the actual Colab
runtime under which these results were computed. The effect of this runtime
difference from the originally-planned pins (scikit-learn 1.3.0/torch 2.1.2)
on the headline classifier result is isolated and quantified above (X5), not
left as an unattributed confound.

## Team

- Agyei Michael Addai (20918643) — Lead Researcher
- Akyen Abban Maxwell (20921083) — Data Engineering
- Ajuitey Prince Awonaab (20921418) — Mobile & USSD
- Anorchie Michelle Frimpomaah (20919895) — HCAI & User Study
- Opoku Kwaku Kelvin (20922337) — Statistics & QA

## Target Journal

Computers & Education: Artificial Intelligence (Elsevier, Q1)