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

## Key Results (commit `9cbb4d3443768b145025d039c9fd090f1f0324f8`)

| Metric | Nyansapo CRAG / Engineered | Baseline / Comparator | Statistic |
|---|---|---|---|
| QIC-WCE Macro F1 | 0.7074 ± 0.1306 | 0.1976 ± 0.0836 (baseline) | p=0.000088, d=3.0005 |
| Ablation (uniform weights) | 0.1976 ± 0.0836 | identical to baseline | mathematically exact |
| Gold ROUGE-L vs BM25-only | 0.2283 ± 0.1410 | 0.1733 ± 0.1364 | p=0.000001, d=0.396 |
| Gold ROUGE-L vs Vanilla LLM | 0.2283 ± 0.1410 | 0.2080 ± 0.0663 | p=0.056, **inconclusive** (power=0.4436, n≥413 needed) |
| Gold BERTScore F1 | 0.7897 ± 0.0753 | 0.7963 ± 0.0318 (Vanilla) | p=0.325 (not significant) |
| Offline latency | 3.60s (M1 Pro, no internet) | 10.55s (online, T4 GPU) | 2.93× faster offline |
| Student trust (n=298) | 4.1795 ± 0.8343 | vs midpoint 3.0 | t=24.33, p<0.000001 |
| NASA-TLX cognitive load | 1.8761 ± 0.9051 | (lower = better) | — |
| Gender bias in trust | none detected | Male 4.22 vs Female 4.14 | p=0.4307 |

All classifier comparisons: Wilcoxon signed-rank, 20 seeds. All RAG comparisons:
Wilcoxon signed-rank, Bonferroni-corrected α=0.0167, n=200 gold pairs. MDE and
power figures are the supervisor's own audit script's computed values
(`00_CLOSEOUT_GROUP_1_20260913.py`, Block V1c), not an independent recalculation.

**Deployment model**: seed 789 (F1=0.9526), selected post-hoc after all 20 runs
completed, based on test performance. This is disclosed as a permanent
limitation — the test set is not a held-out estimate for the deployed model
specifically (see Methods §A6, and Limitations in the manuscript).

## Independent Verification

This repository includes `00_CLOSEOUT_GROUP_1_20260913.py`, the supervisor's
own read-only audit script. Its most recent run against this codebase returned:

**CLOSED: 6 · STILL OPEN: 1 · UNVERIFIABLE: 1**

- Training-path leak scan (C1): CLOSED for the production path (`train_model_v2`,
  confirmed INSIDE-FOLD). STILL OPEN only for the leak-free self-test's own
  diagnostic comparison object, which is never passed to any scored model —
  annotated in the manuscript with exact cell/line citation.
- Perturbation pairing (F8), 5-fold CV and fairness-by-topic tables (F1),
  empty-generation count (C3), and trust/workload analysis (C4): all CLOSED.
- Gold artefact staging (C4): UNVERIFIABLE by the script itself (it checks
  local file presence, not git history) — resolved by citing the commit hash
  under which the data was committed (`710f75765fb482f370a7bb6b359438d7b3e7c6a3`).

## Repository Structure

```
Nyansapo-RAG-Research/
├── README.md
├── requirements.txt
├── .gitignore
├── 00_CLOSEOUT_GROUP_1_20260913.py     # supervisor's audit script, v3-patched
├── notebooks/
│   └── Nyansapo_RAG_Pipeline.ipynb     # single canonical pipeline, v3
├── data/
│   ├── gold_qa_pairs.csv               # 200 human-annotated QA pairs
│   ├── gold_system_answers.csv         # all 4 systems' generated answers
│   └── user_study_anonymized.csv       # n=298, timestamp/comments stripped
├── figures/
│   ├── system_architecture.png
│   ├── classifier_architecture.png
│   ├── data_partition_v3.png           # corrected split, 101/22/22
│   ├── confusion_matrices_v3.png
│   ├── convergence_v3.png
│   ├── pr_roc_v3.png
│   ├── feature_importance_v3.png
│   ├── perturbation_v3.png             # deployment-seed self-test verified
│   └── gold_eval_v3.png
├── docs/
│   ├── Group1_Method_and_Results.docx  # merged manuscript, current round
│   ├── Group1_CoverSheet.docx
│   └── Group1_All_Feedback_Reports.pdf # complete chronological archive (6 reports)
└── offline/
    └── nyansapo_offline.py             # offline deployment (Ollama/Phi-4-mini)
```

## Reproducibility

Every number above traces to a single notebook run
(`notebooks/Nyansapo_RAG_Pipeline.ipynb`), verified against the supervisor's
own audit script at commit `2db04900c0f05ca79c95cd36b364f5ce54dab8d6`, with
subsequent housekeeping commits (lock-file removal, hash-citation corrections)
bringing the repository to its current state at `9cbb4d3443768b145025d039c9fd090f1f0324f8`.

One canonical training function (`train_model_v2`) is used for baseline,
engineered, and ablation conditions — no duplicate implementations, no global
state shared between cells. The TF-IDF vectoriser and class weights are fit
inside each seeded split, on the training partition only; this is verified by
a vocabulary-SET self-test in the notebook (149 of 500 tokens present only in
the full-dataset fit are confirmed absent from the training-only fit — a set
comparison, not a size comparison, since both fits hit the same
`max_features=500` cap).

Framework versions are pinned in `requirements.txt` to the actual Colab
runtime under which these results were computed (see file header for details
on why the originally-pinned scikit-learn 1.3.0 / torch 2.1.2 could not be
reproduced).

## Team

- Agyei Michael Addai (20918643) — Lead Researcher
- Akyen Abban Maxwell (20921083) — Data Engineering
- Ajuitey Prince Awonaab (20921418) — Mobile & USSD
- Anorchie Michelle Frimpomaah (20919895) — HCAI & User Study
- Opoku Kwaku Kelvin (20922337) — Statistics & QA

## Target Journal

Computers & Education: Artificial Intelligence (Elsevier, Q1)