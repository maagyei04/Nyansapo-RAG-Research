# Nyansapo — RAG-Grounded AI Tutoring System
**Group 1 | KNUST Department of Computer Science | 2025–2026**
**Supervisor: Dr. Eric Opoku Osei**

## Repository Structure
- `notebooks/` — v1 (audit trail) and v2 (current clean pipeline)
- `figures/` — all 13 publication-ready experiment figures
- `docs/` — Methods R01, Results R01, Cover Sheet, Group 1 Previous Feedback
- `Nyansapo Poster` — a full A1-portrait poster of Nyansapo

## Project Overview
Nyansapo is a Corrective Retrieval-Augmented Generation (CRAG) AI tutoring 
system that answers KNUST student questions using official lecture content, 
preventing hallucination through retrieval grounding and intent classification.

## Key Results
| Metric | Result |
|--------|--------|
| QIC-WCE Macro F1 | 0.621 ± 0.156 (vs baseline 0.198 ± 0.082) |
| Wilcoxon p-value | 0.000002 |
| Cohen's d | 2.175 (very large) |
| CRAG vs BM25 ROUGE-L | p=0.000001, d=0.396 |
| Offline latency | 3.60s mean (no internet, M1 Pro) |
| User trust | 4.18/5.0 (n=298, p<0.0001) |

## Team
- Agyei Michael Addai (20918643) — Lead Researcher
- Akyen Abban Maxwell (20921083) — Data Engineering
- Ajuitey Prince Awonaab (20921418) — Mobile & USSD
- Anorchie Michelle Frimpomaah (20919895) — HCAI & User Study
- Opoku Kwaku Kelvin (20922337) — Statistics & QA

## Target Journal
Computers & Education: Artificial Intelligence (Elsevier, Q1)