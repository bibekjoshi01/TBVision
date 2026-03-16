# 🫁 TB-Vision

**Explainable, uncertainty-aware tuberculosis screening for low-resource clinics.**

## Table of Contents

1. [Problem & Vision](#problem--vision)
2. [System at a Glance](#system-at-a-glance)
3. [Model & Architecture](#model--architecture)
4. [Key Features](#key-features)

---

## Problem & Vision

Tuberculosis remains a major health inequity in regions without reliable radiology. TB-Vision delivers:

- **Offline-first inference** via a lightweight CNN ensemble (DenseNet121, EfficientNet-B3, ResNet50).
- **Explainable insights** with Grad-CAM and uncertainty quantification.
- **Evidence-grounded reports** referencing WHO guidance and storing every interaction as a shareable report.
- **Conversational follow-up** so clinicians or patients can ask questions about the AI-generated findings.

This hybrid workflow blends local speed with optional cloud validation, making it ideal for hackathons, pilots, and clinics that demand transparency and reliability.

## System at a Glance

```mermaid
flowchart LR
    A[CXR Upload (Frontend / Clinician)]
    A --> B[ClassifierService<br>(DenseNet / EfficientNet / ResNet ensemble)]
    B --> C[Grad-CAM + Monte Carlo Uncertainty]
    C --> D[Analyzer Service]
    D --> E{Internet Available?}
    E -- Yes --> F[RetrievalService (Qdrant + WHO PDFs)]
    F --> G[GenerationService (Gemini + Mistral synthesis)]
    D --> H[SQLite report store<br>(report_id + summary + follow-up history)]
    G --> I[/api/predict (report + report_id)]
    I --> J[/api/follow-up (question + history)]
    J --> H
```

## Model & Architecture

| Component               | Description                                                                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Classifier Ensemble** | DenseNet121, EfficientNet-B3, and ResNet50 share logits through `TBEnsemble`. MC-Dropout can be enabled for uncertainty estimates and Grad-CAM heatmaps guide explainability. |
| **Analyzer**            | Combines classifier output, metadata, Grad-CAM, and embeddings to orchestrate retrieval and LLM generation.                                                                   |
| **Retrieval Service**   | Qdrant vector store over WHO/clinical PDFs, chunked and embedded at ingest. Used only when online to ground explanations.                                                     |
| **Generation Service**  | Gemini for validation + Mistral for synthesis/follow-up. Prompts in `tbvision/prompts/` keep messaging consistent.                                                            |
| **Stateful Chat**       | SQLite-backed report store holds summaries, evidence, uncertainty, and follow-up history so the frontend can resume conversations after reloads.                              |

The repository includes helper scripts (e.g., `scripts/load_knowledge.py`) to manually populate the vector DB and separate logic for training the classifier (`tbvision/xraytb_net`).

## Key Features

- **Explainability**: Grad-CAM overlays + regional descriptions highlight suspicious lung zones.
- **Uncertainty-aware**: Stochastic dropout and metrics (sensitivity, specificity, AUC) make recommendations safer.
- **Evidence grounding**: Retrieved WHO/clinical text feeds the synthesis prompt.
- **Conversational follow-ups**: Save report context via SQLite, then let clinicians query `/api/follow-up` with history-aware prompts.
- **Offline-first**: Models live locally; only retrieval & LLMs optionally hit the cloud when configured.
- **Modular training/inference stack**: `tbvision/xraytb_net` handles dataset prep, training, and evaluation; `tbvision/main.py` wires up FastAPI services.