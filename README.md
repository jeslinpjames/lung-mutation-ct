# lung-mutation-ct

> **Multimodal ML for lung cancer analysis on an 8 GB GPU.**  
> Two research tracks: LUAD growth-pattern grading via Attention-MIL on whole-slide images, and EGFR mutation "virtual biopsy" from routine CT + clinical data.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Research Tracks](#research-tracks)
  - [Track 1 — LUAD Growth-Pattern Grading (DHMC)](#track-1--luad-growth-pattern-grading-dhmc)
  - [Track 2 — EGFR Virtual Biopsy (NSCLC-Radiogenomics)](#track-2--egfr-virtual-biopsy-nsclc-radiogenomics)
- [Datasets](#datasets)
- [Architecture](#architecture)
  - [Track 1: Gated Attention-MIL](#track-1-gated-attention-mil)
  - [Track 2: Image Encoder Comparison](#track-2-image-encoder-comparison)
  - [Track 2: Full FusionNet](#track-2-full-fusionnet-v32)
- [Results](#results)
  - [EGFR Experiment Progression](#egfr-experiment-progression)
  - [Key Findings](#key-findings)
- [Setup](#setup)
- [Usage](#usage)
- [Data Leakage Rules](#data-leakage-rules)
- [Known Gotchas](#known-gotchas)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository contains two independent ML research pipelines targeting clinically actionable problems in lung cancer.

**Track 1 — LUAD Grading:** Predict the predominant growth pattern of lung adenocarcinoma (LUAD) from whole-slide images (WSI) using a Gated Attention Multiple Instance Learning (MIL) model on frozen [UNI](https://huggingface.co/MahmoodLab/UNI) features. The 5-class pattern prediction maps to the 3-tier IASLC grade at inference.

**Track 2 — EGFR Virtual Biopsy:** Predict EGFR mutation status (Mutant vs. Wildtype) from a routine CT scan fused with clinical variables, using the NSCLC-Radiogenomics dataset (TCIA). EGFR-mutant tumours respond to targeted TKI therapy (e.g., osimertinib), but molecular confirmation requires slow, expensive NGS sequencing. A CT-based predictor triages which patients to sequence first — it is a pre-screen, not a replacement for molecular testing.

**Core engineering constraint:** The entire pipeline runs on an 8 GB GPU. This is achieved by never training on raw gigapixel/3D images directly — frozen foundation models precompute features once, cache to disk, and a small model trains on the cached representations.

---

## Repository Structure

```
LungCancerGrading/
│
├── data/                                      # Raw datasets (gitignored)
│   ├── nsclc_radiogenomics/                   # TCIA NSCLC-Radiogenomics DICOM
│   │   └── AMC-xxx / R01-xxx/                 # Per-patient DICOM series
│   ├── AIM_files_updated-11-10-2020/
│   │   └── AIM_files_updated-11-10-2020/      # Radiologist AIM markup XMLs (190 files)
│   └── NSCLCR01Radiogenomic_DATA_LABELS_*.csv # Clinical CSV (211 × 40)
│
├── egfr_cache/                                # Preprocessed CT tumour cubes (.npy, gitignored)
│   ├── seg1_32x64x64_pad16/                   # v1/v2: SEG-only crops (111 patients)
│   └── seg0_aim1_32x64x64_pad16/             # v3.1+: SEG + AIM-coord crops (139 patients)
│
├── egfr_out/                                  # Model checkpoints and feature CSVs (gitignored)
│   ├── egfr_fusion_v2.pt
│   ├── egfr_fusion_v31.pt
│   ├── egfr_fusion_v32.pt
│   └── radiomics_*.csv
│
├── dhmc_out/                                  # UNI feature .h5 files from TRIDENT (gitignored)
│   └── features_uni_v1/
│
├── AIM_Global_Verification_Sandbox.ipynb      # AIM coordinate resolver with dynamic tumour-box sizing
├── LUAD_Grading_Training.ipynb                # Track 1: Gated Attention-MIL on DHMC WSIs
│
├── EGFR_Multimodal_Training.ipynb             # v1: 3D CNN + clinical MLP baseline
├── EGFR_Multimodal_Training_v2.ipynb          # v2: pretrained 2.5D ResNet18 + cache safety
├── EGFR_Multimodal_Training_v3.ipynb          # v3: ViT-over-slices (experimental, abandoned)
├── EGFR_Multimodal_Training_v3_1.ipynb        # v3.1: AIM-coordinate cohort recovery + repeated CV
├── EGFR_Multimodal_Training_v3_2.ipynb        # v3.2: hand-crafted radiomics ablation (current)
│
├── output.png                                 # Sample AIM tumour crop visualisation
├── .gitignore
└── README.md
```

---

## Research Tracks

### Track 1 — LUAD Growth-Pattern Grading (DHMC)

**Notebook:** `LUAD_Grading_Training.ipynb`

**Task:** Classify the predominant growth pattern of a LUAD whole-slide image into one of 5 classes (lepidic, acinar, papillary, micropapillary, solid), then map to a 3-tier IASLC grade (G1/G2/G3) at inference.

**Why MIL?** Each WSI contains ~2,000 tiles but carries only one slide-level label. We don't know which tiles show the predominant pattern, so the problem is framed as Multiple Instance Learning — the model learns a per-tile attention weight and takes a weighted average, rather than treating all tiles equally (which drowns signal in background tissue).

**Pipeline:**

```
WSI (.tif)
  └─► TRIDENT + frozen UNI (ViT-L, 1024-d)     [one-time, CLI]
        └─► per-slide .h5 feature bag [N_tiles × 1024]
              └─► Gated Attention-MIL head       [trained here]
                    └─► 5-class pattern → 3-tier grade
```

**5 → 3 mapping at inference:** Class probabilities are *summed* by group (lepidic → G1; acinar + papillary → G2; micropapillary + solid → G3). Summing grouped probabilities is more principled than relabelling the top-1 prediction.

**Dataset reality (DHMC, 143 slides):**

| Pattern | Count |
|---|---|
| Acinar | 59 |
| Solid | 51 |
| Lepidic | 19 |
| Micropapillary | 9 |
| Papillary | 5 |

The rare classes (papillary = 5, micropapillary = 9) are too sparse for reliable 5-class learning. The mapped 3-tier grade (G1 = 19, G2 = 64, G3 = 60) is the trustworthy headline metric. Target: beat inter-pathologist kappa of ~0.485. Published DHMC model: ~0.525.

**Prerequisite (one-time, run in terminal):**

```bash
python run_batch_of_slides.py \
    --task all \
    --wsi_dir ./dhmc_slides \
    --job_dir ./dhmc_out \
    --patch_encoder uni_v1 \
    --mag 20 --patch_size 256 --overlap 0
```

Requires TRIDENT installed and UNI access approved on Hugging Face.

---

### Track 2 — EGFR Virtual Biopsy (NSCLC-Radiogenomics)

**Current notebook:** `EGFR_Multimodal_Training_v3_2.ipynb`

**Task:** Binary classification — predict EGFR mutation status (Mutant vs. Wildtype) from a routine CT scan + clinical variables available at scan time.

**Clinical motivation:** EGFR mutations drive ~15% of NSCLC in Western populations, enriched in never-smokers, women, East-Asian ancestry, and adenocarcinoma. Mutant tumours respond well to TKI targeted therapy. A CT-based pre-screen triages who to prioritise for expensive NGS sequencing.

#### Notebook version history

| Notebook | Key changes | Cohort | Positives |
|---|---|---|---|
| `v1` | 3D CNN from scratch + clinical MLP; centre-crop fallback | 172 (Run A), 111 (Run B) | 43 / 21 |
| `v2` | Pretrained 2.5D ResNet18; settings-stamped cache; ethnicity feature; cosine LR | 111 | 21 |
| `v3` | ViT-over-slices image branch | 111 | 21 |
| `v3.1` | **AIM-coordinate cohort recovery** (111 → 139); repeated 5×5 CV; fold-wise mean ± std as primary metric | 139 | 33 |
| `v3.2` | **Hand-crafted radiomics ablation** (first-order + shape + GLCM); full alongside-vs-instead comparison | 139 | 33 |

#### AIM-Coordinate Tumour Recovery (`v3.1`+)

Patients without a DICOM SEG are no longer dropped. The radiologist's tumour markup is parsed from the AIM XML (`TwoDimensionCircle`: pixel x/y centre + radius + SOP Instance UID of the annotated slice). A global SOP→z lookup searches all CT series in the patient folder (not just the largest series), recovers the exact slice, and crops a cube around the annotated centre. The `AIM_Global_Verification_Sandbox.ipynb` notebook implements a refinement: instead of a fixed crop radius, a lung-gated region-grow from the markup centre estimates the actual tumour extent, producing tumour-proportional cubes consistent with the SEG-based crops.

**Recovery result:** 111 → 139 patients, 21 → 33 positives (+57%). The unresolved 20 patients had markups referencing a different CT reconstruction series than the one selected by the preprocessing step — the global search is the fix.

**AIM file structure note:** Only the 28 AMC-site patients carry the radiologist's semantic observation panel (tumour attenuation, nodule type, etc.). The R01-site patients carry only the markup geometry. The semantic-feature "0.89 AUROC ceiling" from the original paper is therefore an AMC-only phenomenon and cannot be reproduced on the full cohort.

---

## Datasets

### NSCLC-Radiogenomics (TCIA)

- 211 patients across two sites: AMC (Amsterdam) and R01 (Stanford)
- DICOM: CT + optional PET + optional DICOM SEG (117 patients have segmentation)
- Clinical CSV: 211 × 40 columns; key fields — EGFR mutation status, age, sex, smoking history, ethnicity, % ground-glass opacity, tumour location
- AIM XMLs: 190 files with radiologist tumour markup geometry; 28 AMC files additionally contain semantic observation panels
- **Usable labels:** 172 (Wildtype 129, Mutant 43; Unknown/Not collected excluded)
- **With SEG crop (v1/v2):** 111 patients, 21 positives (19% prevalence)
- **With AIM recovery (v3.1+):** 139 patients, 33 positives (24% prevalence)
- **Source mix:** AMC is 43% mutant; R01 is 19% mutant — a material site imbalance to account for in stratification and validation

> **Access:** [TCIA NSCLC-Radiogenomics](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/)

### DHMC (Dartmouth)

- 143 LUAD whole-slide images, 45 patients
- Labels: predominant growth pattern (5 classes)

> **Encoder note:** UNI and CONCH are the only safe encoders for this cohort — CTransPath and Phikon were trained on TCGA, which overlaps with the DHMC slides, introducing label leakage.

---

## Architecture

### Track 1: Gated Attention-MIL

```
Tile features [N × 1024]  (from frozen UNI)
     │
     ▼
 proj: Linear(1024→256) + ReLU + Dropout
     │
     ├──► att_V: Linear(256→128) ──tanh──┐
     │                                    × ──► att_w: Linear(128→1) ──softmax──► weights [N]
     └──► att_U: Linear(256→128) ─sigmoid┘
     │
     ▼
 z = Σ (weight_i × h_i)     [1 × 256]  weighted slide vector
     │
     ▼
 head: Dropout + Linear(256→5) ──► 5-class logits
     │
     ▼  (inference only)
 sum groups ──► 3-tier grade probabilities
```

329,606 trainable parameters. The attention weights double as a spatial heatmap of diagnostic relevance.

---

### Track 2: Image Encoder Comparison

Three image encoders were evaluated on the identical cohort and evaluation protocol. Only the box labelled **"Image Encoder"** differs between them — the clinical branch, fusion head, and training loop are byte-for-byte identical.

```
CT Cube [1 × 32 × 64 × 64]
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│  IMAGE ENCODER (one of three, switchable via IMAGE_BACKBONE flag)  │
│                                                                    │
│  "2.5d"  2.5D ResNet18 (SELECTED)                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Take 3 slices (25%, 50%, 75% depth) → stack as RGB        │    │
│  │  Resize → 224×224 → ResNet18 (pretrained, ImageNet)        │    │
│  │  Avg pool → 512-d → Dropout → Linear(512→64) → ReLU        │    │
│  │  11.2M total params; pretrained spatial priors reused       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  "3d"  Small 3D CNN                                                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Conv3D(1→16) + BN + ReLU + MaxPool3D  ×3                  │    │
│  │  AdaptiveAvgPool3D → 64-d → Dropout → Linear(64→64) → ReLU │    │
│  │  84k params; trained from scratch; uses all 32 slices       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  "vit_slices"  Slice Transformer (experimental, abandoned)         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Each of 32 slices → frozen ResNet18 → 512-d token         │    │
│  │  Project → 192-d; prepend CLS + positional embeddings       │    │
│  │  2-layer Transformer (3 heads, ff=384) → CLS → 64-d         │    │
│  │  727k trainable params; failed on 33 positives              │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼  [64-d image vector]
```

**Why 2.5D beats the others at this data size:**
- The 3D CNN ties on AUROC (0.707 vs 0.707) but loses on AUPRC and calibration, at 1/130th the parameters. Either works; 2.5D has better-calibrated probabilities from pretrained features.
- The ViT has no built-in locality or translation-invariance priors — it must learn those from ~26 training positives per fold, so it memorises noise instead. Image-only fell to 0.453 (below chance). Architectural ambition without data to match it causes failure.

---

### Track 2: Full FusionNet (v3.2)

```
                                        ┌─────────────────────┐
CT Cube [1×32×64×64]  ──► Image Encoder │  64-d image vector  │
                                        └──────────┬──────────┘
                                                   │
Clinical vars (25)   ──► MLP(25→64→32) ──► 32-d   │
                                                   │  concat [128-d]
Radiomics (27)       ──► MLP(27→64→32) ──► 32-d   │
                                                   │
                                        ┌──────────▼──────────┐
                                        │  Fusion MLP          │
                                        │  Linear(128→64)      │
                                        │  ReLU + Dropout      │
                                        │  Linear(64→1)        │
                                        └──────────┬──────────┘
                                                   │
                                               sigmoid
                                                   │
                                        P(EGFR-mutant) ∈ [0,1]
```

**Training details:** BCEWithLogitsLoss with `pos_weight` for class imbalance, AdamW, cosine LR annealing (60 epochs), AMP (fp16), train-time augmentation (random H/W flips + intensity jitter ±10%). Radiomics features are standardised with train-fold statistics only (no leakage from validation folds).

**Evaluation:** Repeated stratified 5-fold CV, 5 repeats × 5 folds = 25 fold AUROCs. Stratification is joint on label × site (AMC/R01). Primary metric: **fold-wise mean ± std AUROC**. Pooled OOF AUROC reported as secondary (biased by cross-fold calibration drift — observed gap up to 0.16 on the same model).

---

## Results

### EGFR Experiment Progression

#### v1 / v2 — Establishing baselines (pooled OOF AUROC / AUPRC)

| Model | Run A (N=172, dirty crops) | Run B (N=111, clean SEG crops) | Run C / v2 (N=111, 2.5D + ethnicity) |
|---|---|---|---|
| Clinical-only (logreg) | 0.717 / 0.489 | 0.641 / 0.331 | 0.651 / 0.308 |
| Clinical-only (deep) | 0.756 / 0.515 | 0.562 / 0.224 | 0.589 / 0.252 |
| Image-only (deep) | 0.608 / 0.305 | 0.662 / 0.341 | 0.635 / 0.331 |
| Fusion | 0.760 / 0.512 | 0.676 / 0.322 | 0.652 / 0.337 |

**Note on Run A → B drop:** The 0.76 → 0.68 fall is not a regression. Requiring segmentation changed the patient population (smaller, different site mix, lower mutant prevalence 25% → 19%) and halved the positive count (43 → 21), widening the confidence interval to ±0.10–0.15. The clinical-only logistic regression also dropped (0.717 → 0.641) on the same feature set — proof the population changed, not the model.

#### v3 — ViT experiment (fold-mean ± std, N=111, 21 positives)

| Model | AUROC | Notes |
|---|---|---|
| Clinical-only (logreg) | 0.651 | Unchanged |
| Image-only (deep, ViT) | **0.453 ± 0.095** | Below chance on pooled OOF; overfitting |
| Fusion | **0.473** | Image branch actively drags fusion down |
| Clinical-only (deep) | 0.718 ± 0.016 | Stable; pooled OOF 0.557 (calibration drift) |

The 10× difference in fold std between clinical-only (±0.016) and fusion (±0.164) is the overfitting signature. The ViT experiment was conclusive: more capacity without more data causes failure.

#### v3.1 — AIM cohort recovery (fold-mean ± std, 5×5 CV, N=139, 33 positives)

| Model | AUROC | AUPRC |
|---|---|---|
| Clinical-only (logreg) | 0.726 ± 0.109 | 0.519 ± 0.156 |
| Clinical-only (deep) | **0.756 ± 0.081** | 0.544 ± 0.128 |
| Image-only (deep, 2.5D) | 0.707 ± 0.087 | 0.513 ± 0.131 |
| Fusion (image + clinical) | 0.713 ± 0.100 | 0.540 ± 0.144 |
| Prevalence baseline (AUPRC) | — | 0.237 |

The five seed-means for fusion (0.723, 0.742, 0.697, 0.704, 0.699) barely move — evidence that measurement noise has been substantially tamed by growing from 21 to 33 positives.

**AIM semantic ceiling (AMC-only, 28 patients):** 0.622 ± 0.066. The AMC site's semantic panels (tumour attenuation, lobulation, etc.) do not reproduce the original paper's 0.89 — that result was likely obtained on a different split, possibly with leakage from post-surgery fields.

#### v3.2 — Radiomics ablation (fold-mean ± std, 5×5 CV, N=139, 33 positives)

| Model | AUROC | AUPRC |
|---|---|---|
| Clinical-only (logreg) | 0.726 ± 0.109 | 0.519 ± 0.156 |
| Radiomics-only (logreg) | 0.624 ± 0.135 | 0.438 ± 0.121 |
| Clinical + radiomics (logreg) | 0.698 ± 0.104 | 0.498 ± 0.146 |
| Clinical-only (deep) | 0.739 ± 0.094 | 0.561 ± 0.141 |
| Image-only (deep, 2.5D) | 0.711 ± 0.097 | 0.507 ± 0.128 |
| Image + clinical (prior fusion) | 0.715 ± 0.091 | 0.529 ± 0.110 |
| Image + clinical + radiomics (FULL) | 0.733 ± 0.100 | 0.555 ± 0.121 |

#### Image encoder shootout (v3.1 cohort, N=139, 33 positives)

| Encoder | Params | Image-only AUROC | Fusion AUROC | Notes |
|---|---|---|---|---|
| 2.5D ResNet18 | 11.2M | 0.707 ± 0.087 | 0.713 ± 0.100 | Better AUPRC, better calibration |
| 3D CNN | 84k | 0.707 ± 0.080 | 0.700 ± 0.063 | Tighter variance; 1/130th the size |
| ViT-over-slices | 727k trainable | 0.453 ± 0.095 | 0.473 — | Overfits; abandoned |

Two completely different encoders land on the **exact same image-only AUROC (0.707)**. This is the key diagnostic: the image ceiling is a property of the data and the biology, not the architecture. Further encoder work is not the lever.

---

### Key Findings

1. **Clinical variables are competitive with imaging for EGFR triage.** Simple clinical features (smoking history, ethnicity, % ground-glass opacity, tumour location) predict EGFR status as well as a CT image model, across every encoder tried.

2. **Imaging carries genuine but redundant EGFR signal.** Image-only AUROC ~0.707 is well above chance, but it encodes the same biological signal as the clinical variables (never-smoker, East-Asian, ground-glass adenocarcinoma). Fusing them adds little because the information overlaps rather than complements.

3. **Radiomics does not add to clinical in this setting.** Automated first-order, shape, and GLCM features scored 0.624 alone and slightly hurt the clinical model when combined (0.726 → 0.698). The crude tumour ROI from AIM crops (thresholded, no true mask) and the downsampled cube format reduce the reliability of radiomics features compared to full-resolution mask-based extraction.

4. **Statistical power is the binding constraint, not architecture.** Three versions of the image branch (3D CNN, 2.5D ResNet, ViT) and radiomics all converged to the same ceiling. The ViT overfit at 33 positives, proving that more capacity without more data causes failure. More positives (more patients) moves the result; more model complexity does not.

5. **Pooled OOF AUROC is an unreliable metric at this sample size.** Cross-fold calibration drift caused a 0.16 gap between fold-mean AUROC (0.718) and pooled-OOF AUROC (0.557) on the same model. Fold-wise mean ± std (used from v3.1 onward) is the correct rank-based statistic.

---

## Setup

### Requirements

```bash
# Python 3.10+
pip install "pydicom<3" pydicom-seg SimpleITK scipy scikit-learn scikit-image matplotlib tqdm

# PyTorch (CUDA 12.6 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# For Track 1 only
pip install h5py
# TRIDENT: https://github.com/mahmoodlab/TRIDENT
# UNI access: https://huggingface.co/MahmoodLab/UNI
```

> **Critical:** Pin `pydicom<3`. The `pydicom-seg` library is incompatible with pydicom 3.x and breaks silently.

### Data layout

```
data/
├── nsclc_radiogenomics/               # unzip TCIA download here
│   └── AMC-001/ R01-001/ ...
├── AIM_files_updated-11-10-2020/
│   └── AIM_files_updated-11-10-2020/  # ← nested folder (the zip extracts this way)
│       └── *.xml
└── NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv
```

---

## Usage

### Track 2 — EGFR (entry point: `v3_2`)

Open `EGFR_Multimodal_Training_v3_2.ipynb` and edit the three paths in **Section 3 CONFIG**:

```python
DATA_ROOT    = Path("data/nsclc_radiogenomics")
CLINICAL_CSV = Path("data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")
AIM_DIR      = Path("data/AIM_files_updated-11-10-2020/AIM_files_updated-11-10-2020")
```

Run cells in order. The preprocessing step (Section 7) caches tumour cubes to `egfr_cache/` on first run (~20–40 min for the full cohort). The cache folder name is stamped with the preprocessing settings — changing `TARGET_SHAPE`, `REQUIRE_SEG`, `USE_AIM_COORD_FALLBACK`, or similar flags automatically rebuilds into a new subfolder. Set `FORCE_REBUILD = True` to force a rebuild of the current settings.

**Single-patient inference** (after training):

```python
predict_patient("AMC-005")
# → {"Case ID": "AMC-005", "P(EGFR-mutant)": 0.978, "true_label": "Mutant"}
```

### AIM Visualisation (`AIM_Global_Verification_Sandbox.ipynb`)

Run before committing AIM-recovered crops to the training set. Verifies that AIM markup coordinates resolve to tumour-centred CT slices using the global SOP search, and displays dynamic tumour bounding boxes (lung-gated region-grow sizing). If crops are not tumour-centred, this is the notebook to adjust `crop_from_aim`.

### Track 1 — LUAD Grading

1. Run TRIDENT feature extraction (see prerequisite command above)
2. Open `LUAD_Grading_Training.ipynb`
3. Edit `FEATURE_DIR` and `LABEL_CSV` in **Step 1 Configuration**
4. Run all cells — training runs a few minutes per fold on GPU

---

## Data Leakage Rules

| Rule | Details |
|---|---|
| **Patient-wise splits** | No patient appears in both train and validation in any fold |
| **Site-aware stratification** | Stratify jointly on label × site (AMC/R01) to prevent hospital confounding |
| **Scaler fitting** | StandardScaler fit on train fold only; applied to validation without refitting |
| **Safe clinical features only** | Exclude all post-surgery/pathology fields: stage, invasion depth, recurrence, survival |
| **Histology excluded by default** | `INCLUDE_HISTOLOGY = False` — histology type is borderline-leaky for EGFR prediction |
| **AIM semantic features excluded from model** | Radiologist scores are kept as a reference ceiling cell only; wiring them in makes the image branch redundant and conflates imaging with expert annotation |
| **Foundation model choice (Track 1)** | Use UNI or CONCH only for DHMC. CTransPath and Phikon were trained on TCGA, which overlaps with DHMC → data leakage |
| **Radiomics normalisation** | Standardised inside each CV fold using train-split statistics only |

---

## Known Gotchas

**`pydicom-seg` + pydicom 3.x:** Silent breakage. Always pin `pydicom<3`.

**Cache invalidation (v1 bug, fixed in v2+):** In v1, changing `REQUIRE_SEG` had no effect because preprocessing skipped patients with existing `.npy` files. Fixed via settings-stamped cache folder names. Changing any preprocessing flag now rebuilds into a new subfolder automatically.

**AIM folder nesting:** The TCIA AIM zip extracts to a double-nested directory: `AIM_files_updated-11-10-2020/AIM_files_updated-11-10-2020/`. Point `AIM_DIR` at the inner folder.

**AIM matching vs. parsing:** Filename matching (Case ID as substring of XML filename) works for both AMC and R01 patients. The semantic feature parser returns empty for R01 files because those AIM annotations contain only tumour markup geometry, not the radiologist's written observation panel. This is expected; the semantic ceiling cell will report correctly.

**AIM-unresolved patients:** 20 patients have AIM markup but the annotated SOP Instance UID does not appear in the CT series selected by preprocessing (which picks the series with the most slices). The fix is a global SOP search across all series in the patient folder, implemented in `AIM_Global_Verification_Sandbox.ipynb`. Integrating this into the training pipeline is a planned step (see Roadmap).

**Tabular feature count drift:** `pd.get_dummies` may produce different column counts across runs if the cohort subset changes (e.g., ethnicity categories that appear in some subsets but not others). Harmless for current usage; worth fixing before any deployment.

**Pooled OOF vs. fold-mean AUROC:** These can diverge by up to 0.16 due to cross-fold probability calibration drift. Use fold-wise mean ± std (reported from v3.1 onward) as the primary metric. Pooled OOF is a secondary, noisier number.

**AIM crop quality (AMC-009):** One crop (AMC-009) resolved to a dark cavitary region that may be an airway or bulla rather than solid tumour. It represents 1 of 28 recovered patients and does not dominate results, but is worth auditing if the image branch is refined further.

---

## Roadmap

The "can imaging improve on clinical" question is now thoroughly answered across four encoder types and one radiomics approach. The remaining steps shift focus from improving the score to making the finding robust for publication.

**Next: leave-one-site-out (LOSO) validation.** Train on R01, test on AMC (and vice versa). This is the critical credibility check — AMC is 43% mutant and a distinct scanner population. If clinical-only generalises across sites but image models degrade, it strengthens the story. If everything degrades, the 0.73 may partially reflect site signal rather than biology; better to establish this before writing up.

**Recover the 20 AIM-unresolved patients.** The global SOP search in the sandbox notebook resolves this. It changes N (→ ~155–160) and the cached cubes, so it should be done as a discrete step (v3.3) with a full ablation run to re-establish the reference numbers, then LOSO on top of that.

**Add a second cohort for external validation.** Train on NSCLC-Radiogenomics, test on TCGA-LUAD. Key practical points: join TCGA-LUAD imaging (TCIA) with EGFR calls (cBioPortal) by TCGA barcode; expect a modest usable overlap. A tumour centroid (from an automated detector or manual click) is sufficient for the 2.5D/AIM-style preprocessing — a full painted mask is not required. Apply ComBat harmonisation to radiomics features to remove scanner batch effects before pooling.

**What to explicitly avoid:** additional image encoder experiments, ViT variants, larger CNNs, hyperparameter sweeps. Four independent experiments have confirmed the image ceiling is ~0.707 and is a property of the data and biology, not the architecture.

---

## Acknowledgements

- **Dataset:** [NSCLC-Radiogenomics](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/) — Bakr et al., 2018. Hosted by The Cancer Imaging Archive (TCIA).
- **Foundation model:** [UNI](https://github.com/mahmoodlab/UNI) — Chen et al., 2024 (Nature Medicine). Used under academic licence.
- **Feature extraction:** [TRIDENT](https://github.com/mahmoodlab/TRIDENT) — Mahmood Lab, Harvard Medical School.
- **Original radiogenomics paper:** Gevaert et al., 2012. *Non-small cell lung cancer: identifying prognostic imaging biomarkers by leveraging public gene expression microarray data.* Radiology 264(2):387–396.
