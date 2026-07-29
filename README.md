# Nyansapo — RAG-Grounded AI Tutoring System
**Group 1 | KNUST Department of Computer Science | 2025–2026**
**Supervisor: Dr. Eric Opoku Osei**

## Project Overview
Nyansapo is a Corrective Retrieval-Augmented Generation (CRAG) AI tutoring 
system that answers KNUST student questions using official lecture content, 
preventing hallucination through retrieval grounding and intent classification.

## System Architecture
- **Intent Classifier**: Nyansapo-QIC-WCE (Weighted Cross-Entropy)
- **Retrieval**: Hybrid BM25 + BGE-M3 Dense (RRF fusion)
- **Evaluator**: CRAG keyword-overlap quality scorer
- **Generator (Online)**: Phi-2 (2.7B, 4-bit NF4) via FastAPI
- **Generator (Offline)**: Phi-4-mini (3.8B) via Ollama — no internet required
- **Interface**: React Native mobile app + USSD (Africa's Talking sandbox)

## Key Results
| Metric | Nyansapo CRAG | Vanilla LLM | Improvement |
|--------|--------------|-------------|-------------|
| BERTScore F1 | 1.000 | 0.762 | +0.238 |
| ROUGE-L | 1.000 | 0.299 | +0.701 |
| RAGAS Faithfulness | 0.591 | 0.467 | +0.124 |
| QIC-WCE Macro F1 | 0.8249 | 0.1802 (baseline) | +0.6447 |
| Offline latency | 3.60s | — | 2.93x faster than online |

All comparisons: Wilcoxon signed-rank p < 0.0001, Bonferroni corrected.
Cohen's d = 5.44 (model engineering), d = 1.90 (CRAG vs Vanilla LLM).

## Repository Structure
Nyansapo-RAG-Research/
├── Nyansapo_RAG_Pipeline.ipynb # Main experiment notebook
├── requirements.txt # All dependencies
├── README.md # This file
├── figures/ # Publication-ready result figures
│ ├── confusion_matrices.png
│ ├── qic_engineering_results.png
│ ├── results_chart.png
│ ├── final_results_chart.png
│ ├── latency_chart.png
│ └── complete_evaluation.png
└── docs/
├── Group1_Method.docx
└── Group1_Results.docx

## Team
- Agyei Michael Addai (20918643) — Lead Researcher
- Akyen Abban Maxwell (20921083) — Data Engineering
- Ajuitey Prince Awonaab (20921418) — Mobile & USSD
- Anorchie Michelle Frimpomaah (20919895) — HCAI & User Study
- Opoku Kwaku Kelvin (20922337) — Statistics & QA

## Target Journal
Computers & Education: Artificial Intelligence (Elsevier, Q1)