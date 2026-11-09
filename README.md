# Cross-Modal Retrieval: Mobile App Screens and Text Summaries

Individual implementation for Project 2.2 (COMP 1030: Multimodal Data Analytics, Adelaide University), extending the Group 21 proposal (Project 2.1).

**Author:** Tess Santhosh (a2910201)
**Report:** `report/Tess_Santhosh_2.2_Report.pdf` <!-- or a link if the report lives elsewhere -->
**Video:** <link to 3-minute presentation recording>
**Slides:** `slides/Tess_Santhosh_2.2_Slides.pdf`

## Problem

<One or two sentences restating the task: retrieve the matching RICO screen given a caption, and
the matching caption given a screen, in both directions.>

## Key results

<Fill this in once evaluation is complete. A short table the reader can scan before opening any
notebook — this is the single most persuasive part of the README.>

| Technique | Image→Text R@10 | Text→Image R@10 | Params | Train time |
|---|---|---|---|---|
| Ridge regression (baseline) | | | | |
| Early fusion (MLP) | | | | |
| Contrastive dual encoder | | | | |
| CLIP zero-shot (reference) | | | | |

## Dataset

- **Source:** [RICO-Screen2Words](https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words) (Hugging Face)
- **Size:** 22,417 Android screens, 5 captions each (112,085 captions total)
- **Modalities:** image (screenshot) + text (caption)
- **Licence:** CC-BY-4.0 on the annotation dataset card; see the ethics section of the report for the
  caveat on the underlying RICO screenshots themselves.

The dataset is downloaded automatically by `notebooks/01_data_exploration.ipynb` via the
`datasets` library — no manual download needed.

## Repository structure

```
.
├── README.md                          <- you are here
├── requirements.txt                   <- exact package versions
├── notebooks/
│   ├── 01_data_exploration.ipynb      <- load data, quality checks, visualisation
│   ├── 02_preprocessing_features.ipynb<- cleaning, splits, feature extraction (ResNet-50 / BERT)
│   ├── 03_early_fusion.ipynb          <- technique 1: Ridge regression baseline + early fusion (MLP)
│   ├── 04_contrastive_fusion.ipynb    <- technique 2: coordinated / contrastive dual encoder (InfoNCE)
│   └── 05_evaluation_comparison.ipynb <- metrics, comparison table, overfitting check, failure cases
├── src/
│   └── utils.py                       <- shared functions: metrics (Recall@K, AUC), plotting, data loading
├── results/
│   ├── figures/                       <- saved plots referenced by the report (Fig. 1, Fig. 2, ...)
│   └── tables/                        <- saved metrics tables (CSV) referenced by the report
├── report/                            <- final PDF report (not code, kept here for convenience)
└── slides/                            <- final PDF slides
```

## How to reproduce

1. **Environment.** Python 3.10+, a CUDA GPU recommended (tested on a free-tier Colab T4).
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the notebooks in order**, 01 through 05. Each notebook is independently runnable — rather
   than saving and loading model checkpoints between notebooks, every notebook re-extracts the
   frozen ResNet-50/BERT features and retrains its own (fast, lightweight) head(s) from scratch via
   the shared functions in `src/utils.py`. What persists between notebooks is each one's *results*
   (metrics CSVs, figures) in `results/`, not trained weights — model files are git-ignored and
   regenerable, per `.gitignore`.
   - `01_data_exploration.ipynb` — downloads RICO-Screen2Words, explores size/quality/class balance.
   - `02_preprocessing_features.ipynb` — cleans captions, confirms the official train/val/test split
     is already app-disjoint (verified, not assumed — see the notebook), extracts frozen ResNet-50
     image features and BERT text features.
   - `03_early_fusion.ipynb` — Ridge regression baseline and an early-fusion MLP (concatenate +
     project), both evaluated with Recall@K/AUC.
   - `04_contrastive_fusion.ipynb` — the contrastive dual encoder (InfoNCE), the project's
     best-performing technique, with a direct efficiency comparison against early fusion.
   - `05_evaluation_comparison.ipynb` — combines all three techniques' saved results into the
     final comparison table and figures used in the report, plus the overfitting check and
     specific failure-case examples.
3. **Expected runtime:** <fill in once timed, e.g. "~25 minutes end-to-end on a Colab T4">.

## Evaluation methodology

Following the course's Week 7 evaluation practical: scaling statistics are fit on the training
split only, early stopping uses the validation-loss minimum (not a fixed epoch count), and the
test set is touched exactly once, after model/hyperparameter selection is finalised on validation.
Metrics reported: Recall@1 / Recall@5 / Recall@10 in both directions, mean reciprocal rank, and
AUC, alongside parameter count and training/inference time (Week 7's efficiency reporting).
Overfitting is assessed via the train–validation metric gap (as in Week 8's InfoNCE curves).

## Ethics and bias

See Section X of the report for the full assessment (bias sources, fairness-metric results,
privacy review, and mitigations). Summary: the dataset's RICO-side copyright status is flagged
as uncertain and raw screens are kept in access-controlled storage; a manual review of 80 screens
found 3 with readable personal information, which are excluded from any shared output; retrieval
performance is reported per app category to check for fairness disparities given the dataset's
category imbalance.

## Acknowledgements / references

Techniques and evaluation methodology adapted from COMP 1030 Weeks 7–9 practicals and lectures
(Adelaide University, 2026). Dataset: Wang et al. (2021), Screen2Words; RICO: Deka et al. (2017).
Full references in the report's reference list.
