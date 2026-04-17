# NutriAI — AI-Powered Food Recognition for College Mess

[![Model](https://img.shields.io/badge/Model-ConvNeXt--Tiny-blue)](https://arxiv.org/abs/2201.03545)
[![Accuracy](https://img.shields.io/badge/Balanced%20Accuracy-91.6%25-brightgreen)]()
[![Classes](https://img.shields.io/badge/Classes-22-orange)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)]()
[![API](https://img.shields.io/badge/API-FastAPI-009688)]()

> **B.Tech Mini Project — KIIT University, Bhubaneswar**  
> My contribution: End-to-end AI/ML pipeline — dataset curation, model training, inference API

---

## What This Does

KIIT students photograph their mess (cafeteria) food → the model identifies the dish → returns calorie and macro nutrition data instantly.

```
📸 Photo → ConvNeXt-Tiny → Dal Fry (78.4%) → 131 kcal | 7.8g protein | 18.2g carbs
```

The full app (built by a 3-person team) includes a React Native frontend, Node.js auth, and a FastAPI backend with LangChain meal planning and RAG-based nutrition Q&A. My scope was the food classifier and inference API.

---

## Results

| Metric | Score |
|--------|-------|
| **Test Balanced Accuracy** | **91.6%** |
| Test Flat Accuracy | ~91% |
| Macro F1 | ~90% |
| Baseline (EfficientNet-B2) | 73.8% |
| **Improvement over baseline** | **+17.8 pts** |

Balanced accuracy is the primary metric — class sizes vary significantly (burger: 100 imgs vs ghuguni: 58 imgs), so flat accuracy would be misleading.

---

## Model & Architecture

**ConvNeXt-Tiny** (27.8M parameters, ImageNet pretrained via `timm`)

### Why ConvNeXt over EfficientNet or ViT?
- **7×7 depthwise convolutions** capture food texture signals better than 3×3 (grainy poha, crispy dosa, chunky ghuguni)
- Stronger than EfficientNet on small datasets — confirmed empirically (+17.8 points)
- ViT requires far more data to generalize — not suitable at ~2,200 images
- ConvNeXt-Tiny fits comfortably on a free Colab T4 GPU (batch_size=32, 224×224)

---

## Training Pipeline

### 3-Stage Progressive Fine-Tuning

```
Stage 1 (8 epochs)   LR = 1e-2    Head only            → fast class alignment
Stage 2 (7 epochs)   LR = 5e-4    Unfreeze stages 2+3  → careful backbone adaptation
Stage 3 (30 epochs)  LR = 1e-4    Full model + LLRD    → final convergence
```

**Stage 2 drops to ~3% accuracy at epoch 9 — this is expected** when 26M frozen params suddenly unfreeze. Recovery happens by epoch 15, then surpasses Stage 1 peak.

### Key Techniques

**Layer-wise LR Decay (LLRD, decay=0.65)**
Earlier layers already have strong ImageNet features — they get smaller LR so those features aren't destroyed:
```
stages.0: 1.79e-05  →  stages.3: 6.50e-05  →  head: 1.00e-04
```
Bug fixed: tracked parameters via `id(p)` to prevent any parameter appearing in multiple optimizer groups (duplicate param bug in AdamW).

**MixUp Augmentation (α=0.4)**
Applied to 50% of batches in Stage 3 only. Interpolates two images + their labels — forces the model to learn softer decision boundaries. Not used in Stages 1–2 (head needs clean signal first).

**Food-Safe Augmentation Rules**
```python
# NEVER use these for food:
RandomVerticalFlip()   # food is never upside down
GaussianBlur()         # destroys texture signals (poha grain, dosa crispiness)

# Safe to use:
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=15)       # not more — food isn't tilted much
ColorJitter(hue=0.05)            # hue is a key signal, keep it tight
```

**Class-Weighted Loss + Label Smoothing**
```python
criterion = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
# Rare classes (ghuguni: 58 imgs) get higher weight than common ones (burger: 100)
```

**Test-Time Augmentation (TTA)**
4 augmented views at inference (original + hflip + slight zoom + slight rotation), logits averaged. Adds ~0.5–1% accuracy at zero training cost.

---

## Dataset

| Source | Images | Notes |
|--------|--------|-------|
| Self-collected (KIIT mess) | ~200 | **Most valuable** — real mess lighting, steel trays |
| Kaggle: Indian Food Images | ~900 | Good for common Indian dishes |
| Food-101 (selective) | ~800 | Capped at 100/class to prevent imbalance |
| Bing scraping (icrawler) | ~300 | Filled gaps for rare classes |
| **Total** | **~2,200** | 22 classes, 70/15/15 train/val/test split |

### The Domain Gap Problem
This was the biggest real-world challenge. A model trained only on clean scraped internet images performs poorly on actual mess photos — different lighting (fluorescent), different presentation (steel thalis, mixed dishes), different angles (overhead phone shots).

**Fix:** Self-collected KIIT mess photos directly in training. Even 10–15 images per class from the real environment dramatically improved inference accuracy on live photos.

### Unique Classes (not in any public dataset)
`ghuguni`, `fish_masala`, `dahibara`, `pitha`, `odia_fish_curry` — Odia cuisine. No Kaggle equivalent exists. Collected by photographing the KIIT mess daily and scraping Instagram hashtags (#odishafood, #pakhala, #odiacuisine).

---

## Inference API

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bash
# POST /detect-food
curl -X POST http://localhost:8000/detect-food \
  -F "file=@dal_fry.jpg"
```

```json
{
  "success": true,
  "status": "ok",
  "top_dish": "dal_fry",
  "confidence": 78.4,
  "predictions": [
    {"dish": "dal_fry",     "confidence": 78.4},
    {"dish": "dal_makhani", "confidence": 12.1},
    {"dish": "chole",       "confidence":  5.2}
  ],
  "nutrition": {
    "calories": 131, "protein": 7.8, "carbs": 18.2, "fat": 3.4
  }
}
```

If `confidence < 62.5%` → `status: "low_confidence"` → app prompts manual entry with top-3 suggestions shown.

---

## Repo Structure

```
NutriAI/
├── FitAI_v4_Clean.ipynb       ← full training pipeline (runnable on Colab)
├── main.py                    ← FastAPI inference server
├── classes.json               ← class list + model config
├── nutrition_db.json          ← calorie/macro data for 22 dishes
├── requirements.txt
├── convnext_tiny_best.pth     ← model weights (Git LFS, ~114MB)
└── docs/
    ├── FitAI_FullProjectDoc_Mar9.txt
    └── FitAI_ExpandedScope_65Classes.docx
```

---

## Run the Notebook

Open `FitAI_v4_Clean.ipynb` in Google Colab:

1. Runtime → Change runtime type → **T4 GPU**
2. Mount your Google Drive
3. Run cells top to bottom — dataset builds automatically from Drive paths
4. Training takes ~45–55 min on T4

> Model weights (`convnext_tiny_best.pth`) are tracked via Git LFS.  
> Run `git lfs pull` after cloning to download them.

---

## What's Next (v2 — 65 Classes)

Expanding from 22 → 65 classes in 4 tiers:

| Tier | Classes | Target Accuracy |
|------|---------|----------------|
| T1: KIIT Mess | 20 | 90%+ |
| T2: Common Indian | 20 | 85%+ |
| T3: Outside/Junk food | 15 | 82%+ |
| T4: Odia Specialty | 10 | 75%+ |

At 65 classes, upgrade to **ConvNeXt-Small** (50M params).

---

## Team

This is my individual AI/ML contribution to a 4-person team project.  
Full project (frontend + backend + auth): [Billu-sMiniProject/Project](https://github.com/Billu-sMiniProject/Project)

| Member | Role |
|--------|------|
| Sanjam Das         | DL/ML — food classifier, inference API |
| Uddipt Shankar     | AI/ML — FastAPI, LangChain, RAG, nutrition pipeline |
| Yashraj Singh      | Backend — PostgresSQL, Supabase, Node.js |
| Samanway Dutta Roy | Frontend - React, JavaScript, Flutter |

---

## Tech Stack

`Python` `PyTorch` `timm` `FastAPI` `Google Colab` `ConvNeXt` `scikit-learn` `icrawler`