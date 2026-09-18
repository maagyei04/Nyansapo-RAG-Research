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

## Key Results (commit `9bb5d95`)

| Metric | Nyansapo CRAG / Engineered | Baseline / Comparator | Statistic |
|---|---|---|---|
| QIC-WCE Macro F1 | 0.7074 ± 0.1306 | 0.1976 ± 0.0836 (baseline) | p=0.000088, d=3.0005 |
| Ablation (uniform weights) | 0.1976 ± 0.0836 | identical to baseline | mathematically exact |
| Gold ROUGE-L vs BM25-only | 0.2283 ± 0.1410 | 0.1733 ± 0.1364 | p=0.000001, d=0.396 |
| Gold ROUGE-L vs Vanilla LLM | 0.2283 ± 0.1410 | 0.2080 ± 0.0663 | p=0.056, **inconclusive** (power=0.446, n≥411 needed) |
| Gold BERTScore F1 | 0.7897 ± 0.0753 | 0.7963 ± 0.0318 (Vanilla) | p=0.325 (not significant) |
| Offline latency | 3.60s (M1 Pro, no internet) | 10.55s (online, T4 GPU) | 2.93× faster offline |
| Student trust (n=298) | 4.1795 ± 0.8343 | vs midpoint 3.0 | t=24.33, p<0.000001 |
| NASA-TLX cognitive load | 1.8761 ± 0.9051 | (lower = better) | — |
| Gender bias in trust | none detected | Male 4.22 vs Female 4.14 | p=0.431 |

All classifier comparisons: Wilcoxon signed-rank, 20 seeds. All RAG comparisons:
Wilcoxon signed-rank, Bonferroni-corrected α=0.0167, n=200 gold pairs.

**Deployment model**: seed 789 (F1=0.9526), selected post-hoc after all 20 runs
completed, based on test performance. This is disclosed as a permanent
limitation — the test set is not a held-out estimate for the deployed model
specifically (see Methods, and Limitations in the manuscript).

## Repository Structure

```
Nyansapo-RAG-Research/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Nyansapo_RAG_Pipeline.ipynb     # single canonical pipeline, v3
├── data/
│   ├── gold_qa_pairs.csv               # 200 human-annotated QA pairs
│   └── gold_system_answers.csv         # all 4 systems' generated answers
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
│   ├── Group1_CoverSheet.pdf
│   └── Group1_All_Feedback_Reports.pdf # complete chronological archive
└── offline/
    └── nyansapo_offline.py             # offline deployment (Ollama/Phi-4-mini)
```

## Reproducibility

Every number above traces to a single notebook run (`notebooks/Nyansapo_RAG_Pipeline.ipynb`),
committed at `9bb5d95a13aa0628e756b863af84004589629fdc`. One canonical training
function (`train_model_v2`) is used for baseline, engineered, and ablation
conditions — no duplicate implementations, no global state shared between cells.
The TF-IDF vectoriser and class weights are fit inside each seeded split, on the
training partition only; this is verified by a vocabulary-set self-test in the
notebook (149 of 500 tokens present only in the full-dataset fit are confirmed
absent from the training-only fit).

Framework versions are pinned in `requirements.txt` to the actual Colab runtime
under which these results were computed (see file header for details).

## Team

- Agyei Michael Addai (20918643) — Lead Researcher
- Akyen Abban Maxwell (20921083) — Data Engineering
- Ajuitey Prince Awonaab (20921418) — Mobile & USSD
- Anorchie Michelle Frimpomaah (20919895) — HCAI & User Study
- Opoku Kwaku Kelvin (20922337) — Statistics & QA

## Target Journal

Computers & Education: Artificial Intelligence (Elsevier, Q1)
