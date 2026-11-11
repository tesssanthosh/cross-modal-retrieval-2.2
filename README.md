# Cross-Modal Retrieval: Mobile App Screens and Text Summaries

Individual implementation for Project 2.2 (COMP 1030: Multimodal Data Analytics, Adelaide University), extending the Group 21 proposal (Project 2.1).

**Author:** Tess Santhosh (a2910201)


## Problem

<Problem statement: retrieve the matching RICO screen for a caption, and the matching caption for a screen, in both directions.>

## Key results

All numbers below are from the test split, touched once, after every hyperparameter decision was finalised on validation (see `notebooks/05_evaluation_comparison.ipynb`). The early-fusion MLP and contrastive dual encoder are each trained from scratch 3 times (seeds 42/123/2024) and reported as mean ± std. Ridge and CLIP are deterministic given their inputs, so a seed sweep doesn't apply to them.

| Technique | Image→Text R@10 | Text→Image R@10 | Params | Train time |
|---|---|---|---|---|
| Random baseline (reference, no training) | 0.16% | 0.19% | - | - (not trained) |
| Ridge regression (baseline) | 3.29% | 3.85% | 3.1M | 11.6 s |
| Early fusion (MLP) | 2.95% ± 0.10% | 2.82% ± 0.10% | 1.5M | 30.4 s/seed (91.3 s total) |
| Contrastive dual encoder | 5.26% ± 0.71% | 5.32% ± 0.83% | 1.7M | 28.7 s/seed (85.9 s total) |
| CLIP zero-shot (reference, not trained on RICO) | 17.89% | 16.43% | 151.3M | 0 s (zero-shot) |

All trained techniques sit well above the random-chance floor (around 0.2% R@10 for this candidate set size), so each one is learning a real cross-modal signal rather than exploiting some trivial dataset artefact. The contrastive dual encoder is the best-performing trained technique on average. This fits with InfoNCE giving a richer per-step training signal than Ridge's closed-form fit or the early-fusion MLP's handful of in-batch negatives, but it's also the most seed-sensitive of the three (test R@10 ranged from 4.29%/4.15% to 5.99%/6.06% across the 3 seeds, versus the MLP's tight 2.83-3.09% range). The ranking between techniques holds regardless, so this variance is reported rather than averaged away. Pretrained CLIP outperforms all three trained techniques with zero RICO-specific training. It's included as a reference point, not a competing technique, since it was pretrained jointly on roughly 400M image-caption pairs (versus about 79K here) and is around 90x larger than the dual encoder. The report's reflection section explains what this implies about closing the gap.

A separate ablation in the same notebook compares mean-pooled vs. [CLS]-token BERT text features via Ridge on validation: mean pooling wins clearly (5.20%/5.03% vs. 3.81%/4.15% R@10), matching the pooling choice used throughout this project.

## Dataset

- **Source:** [RICO-Screen2Words](https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words) (Hugging Face)
- **Size:** 22,417 Android screens, 5 captions each (112,085 captions total)
- **Modalities:** image (screenshot) + text (caption)
- **Licence:** CC-BY-4.0 on the annotation dataset card. The ethics section of the report shows the caveat on the underlying RICO screenshots themselves.

The dataset is downloaded automatically by `notebooks/01_data_exploration.ipynb` via the `datasets` library, no manual download needed.

## Repository structure

```
.
├── README.md                          <- currently opened
├── requirements.txt                   <- exact package versions
├── notebooks/
│   ├── 01_data_exploration.ipynb      <- load data, quality checks (incl. near-duplicate image
│   │   hashing), visualisation
│   ├── 02_preprocessing_features.ipynb<- cleaning, splits, feature extraction (ResNet-50 / BERT)
│   ├── 03_early_fusion.ipynb          <- technique 1: Ridge regression baseline + early fusion (MLP)
│   ├── 04_contrastive_fusion.ipynb    <- technique 2: coordinated / contrastive dual encoder (InfoNCE)
│   └── 05_evaluation_comparison.ipynb <- comparison table (3-seed averaged for MLP/dual encoder),
│       CLIP zero-shot reference, overfitting/efficiency checks, failure cases, category fairness
│       breakdown, mean-pooling-vs-[CLS]-token ablation
├── src/
│   └── utils.py                       <- shared functions: metrics (Recall@K, AUC), plotting, data loading
├── results/
│   ├── figures/                       <- saved plots referenced by the report (Fig. 1, Fig. 2, ...)
│   └── tables/                        <- saved metrics tables (CSV) referenced by the report
```

## How to reproduce

1. **Environment.** Python 3.10+, a CUDA GPU recommended (tested on a free-tier Colab T4).
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the notebooks in order**, 01 through 05. Each notebook runs independently. Rather than saving and loading model checkpoints between notebooks, every notebook re-extracts the frozen ResNet-50/BERT features and retrains its own (fast, lightweight) head(s) from scratch using the shared functions in `src/utils.py`. What persists between notebooks is each one's results (metrics CSVs, figures) in `results/`, not trained weights. Model files are git-ignored and regenerable, per `.gitignore`.
   - `01_data_exploration.ipynb` - downloads RICO-Screen2Words, explores size/quality/class balance, and checks a 500-screen sample for near-duplicate images via a hand-rolled difference hash.
   - `02_preprocessing_features.ipynb` - cleans captions, confirms the official train/val/test split is already app-disjoint, extracts frozen ResNet-50 image features and BERT text features.
   - `03_early_fusion.ipynb` - Ridge regression baseline and an early-fusion MLP (concatenate + project), both evaluated with Recall@K/AUC.
   - `04_contrastive_fusion.ipynb` - the contrastive dual encoder (InfoNCE), the project's best-performing technique, with a direct efficiency comparison against early fusion.
   - `05_evaluation_comparison.ipynb` - retrains all three techniques from scratch (fast, since hyperparameters are already decided), the MLP and dual encoder 3 times each across seeds 42/123/2024 to report mean ± std rather than a single run. Adds a zero-shot CLIP reference point and a mean-pooling-vs-[CLS]-token ablation, and produces the final comparison table, efficiency comparison, overfitting check, failure-case examples, and the per-app-category fairness breakdown used in the report.
3. **Expected runtime:** varies by notebook. `05_evaluation_comparison.ipynb` is the most expensive, since it retrains everything from scratch including 3 seeds each for two techniques. Measured components on a free-tier Colab T4: Ridge about 12s, the early-fusion MLP about 30s/seed (91s for all 3), the dual encoder about 29s/seed (86s for all 3), CLIP zero-shot encoding the full test set about 5.5 minutes, and the pooling ablation's extra [CLS]-pooled BERT pass over train+val captions about 70s. This is on top of the RICO download and ResNet-50/BERT feature extraction shared with notebooks 02-04.

## Evaluation methodology

Scaling statistics are fit on the training split only, early stopping uses the validation-loss minimum rather than a fixed epoch count, and the test set is touched exactly once, after model and hyperparameter selection is finalised on validation. Metrics reported: Recall@1 / Recall@5 / Recall@10 in both directions, mean reciprocal rank (MRR), median rank, and AUC, along with parameter count and training/inference time. A random-baseline floor is reported alongside every trained technique for context. Overfitting is assessed via the train-validation metric gap.

The early-fusion MLP and contrastive dual encoder are each trained and evaluated across 3 seeds (42/123/2024) rather than once, and reported as mean ± std. Ridge and CLIP are deterministic given their inputs, so this doesn't apply to them. A separate ablation compares mean-pooled vs. [CLS]-token BERT text pooling via Ridge, evaluated on validation rather than test, since it compares two representation choices rather than reporting a final result.

## Ethics and bias

Section 6 of the report has the full assessment (bias sources, fairness-metric results, privacy review, and mitigations). Summary: the dataset's RICO-side copyright status is flagged as uncertain and raw screens are kept in access-controlled storage. A manual review of 80 screens found 3 with readable personal information, which are excluded from any shared output. Retrieval performance is reported per app category to check for fairness disparities given the dataset's category imbalance (a real spread was found, see the report). A difference-hash check over a 500-screen sample found 3 near-duplicate image pairs (1.2% of the sample, 0 identical): two are same-app repeats (sequential screen IDs, near-identical layout), and one (screens 434/6201) has non-sequential IDs and visibly different styling, so it more plausibly spans two different apps. That one is flagged in the report as a possible leakage case, since it's a path app-disjoint splitting alone doesn't close.

## Acknowledgements / references

Techniques and evaluation methodology adapted from COMP 1030 Weeks 7-9 practicals and lectures (Adelaide University, 2026). Dataset: Wang et al. (2021), Screen2Words; RICO: Deka et al. (2017). Full references in the report's reference list.
