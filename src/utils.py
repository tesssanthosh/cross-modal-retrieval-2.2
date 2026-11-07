"""
Shared utilities for the cross-modal retrieval project.

Import these from every notebook instead of redefining them — this is what the rubric means by
"modularity" (Criteria 6). Fill in each function as you build the corresponding notebook; the
signatures below follow the pattern used in the Week 7/8 practicals (score matrix in, metrics out).
"""

import re
import time

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


def recall_at_k(scores: torch.Tensor, k: int) -> float:
    """Fraction of queries (rows) whose correct candidate (the diagonal) is in the top k.
    `scores` is a square (n_queries, n_candidates) matrix; row i's correct answer is column i.
    Same approach as Week 7/8: rank = how many wrong candidates outscore the right one.
    """
    ranks = (scores > scores.diag().unsqueeze(1)).sum(dim=1)
    return (ranks < k).float().mean().item()


def retrieval_report(scores: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """Recall@K in both directions plus AUC, matching the Week 7/8 metric suite."""
    out = {}
    for k in ks:
        out[f"R@{k} image->text"] = recall_at_k(scores, k)
        out[f"R@{k} text->image"] = recall_at_k(scores.T, k)
    is_match = torch.eye(len(scores)).flatten().numpy()
    out["AUC"] = roc_auc_score(is_match, scores.flatten().detach().cpu().numpy())
    return out


def info_nce(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, temperature: float = 0.07):
    """Symmetric InfoNCE loss for a contrastive dual encoder (Week 8 practical).
    Both inputs must be L2-normalised, shape (batch, dim); row i of each is the same pair.
    """
    logits = image_embeddings @ text_embeddings.T / temperature
    targets = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --- Frozen backbone feature extraction -------------------------------------------------
# Used by 02 (to build the cached features), then reused as-is by 03/04 (fusion training)
# and 06 (evaluation) so every notebook extracts features the exact same way. Defined once
# here instead of copy-pasted, per this file's own "modularity" rule above.

def build_resnet50_backbone(device):
    """ImageNet-pretrained ResNet-50 with its classification head removed, frozen, eval mode.
    Returns (backbone, preprocess) where `preprocess` is the exact transform that matches the
    pretrained weights (resize/crop/normalise) — always use it, don't hand-roll your own.
    """
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # drop the 1000-way classifier; keep the 2048-d pooled features
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights.transforms()


def build_bert_backbone(device, model_name: str = "bert-base-uncased"):
    """Pretrained BERT encoder + tokenizer, frozen, eval mode."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


@torch.no_grad()
def extract_resnet50_features(images, resnet, preprocess, device, batch_size: int = 64,
                               show_progress: bool = True):
    """Frozen, globally-average-pooled ResNet-50 features (2048-d) for a sequence of PIL images.

    `resnet`/`preprocess` come from build_resnet50_backbone(). Returns (features, elapsed_seconds)
    where features is a (N, 2048) float32 CPU tensor in the same order as `images`.
    """
    feats = []
    t0 = time.time()
    rng = range(0, len(images), batch_size)
    if show_progress:
        rng = tqdm(rng, desc="ResNet-50 features")
    for start in rng:
        batch = images[start:start + batch_size]
        batch_t = torch.stack([preprocess(img.convert("RGB")) for img in batch]).to(device)
        out = resnet(batch_t)
        feats.append(out.float().cpu())
    elapsed = time.time() - t0
    return torch.cat(feats, dim=0), elapsed


@torch.no_grad()
def extract_bert_features(texts, bert, tokenizer, device, batch_size: int = 128,
                           max_length: int = 32, show_progress: bool = True):
    """Frozen, attention-mask-weighted mean-pooled BERT features (768-d) for a list of strings.

    Mean pooling over real (non-padding) tokens is used instead of the [CLS] vector — it's the
    more standard choice for a frozen, not-fine-tuned BERT used as a sentence encoder.
    `bert`/`tokenizer` come from build_bert_backbone(). Returns (features, elapsed_seconds), a
    (N, 768) float32 CPU tensor in the same order as `texts`.
    """
    feats = []
    t0 = time.time()
    rng = range(0, len(texts), batch_size)
    if show_progress:
        rng = tqdm(rng, desc="BERT features")
    for start in rng:
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                         return_tensors="pt").to(device)
        out = bert(**enc).last_hidden_state  # (B, T, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        feats.append(pooled.float().cpu())
    elapsed = time.time() - t0
    return torch.cat(feats, dim=0), elapsed


# --- Full dataset + feature loader -------------------------------------------------------
# Notebook 02 builds these features inline (that notebook's job is specifically to build and
# cache them). From notebook 03 onwards, three different notebooks (03, 04, 06) need the exact
# same features, so — per this file's own rule — the logic is written once here instead of
# copy-pasted a third time. It intentionally mirrors notebook 02's logic exactly.

def clean_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_rico_splits_with_features(device, splits=("train", "val", "test"),
                                    image_batch_size: int = 64, text_batch_size: int = 128,
                                    show_progress: bool = True):
    """Loads RICO-Screen2Words and extracts frozen ResNet-50 / BERT features for each split.

    Returns (data, resnet, bert) where `data` is a dict keyed by split name, each value a dict:
      - screenId:            (N,) tensor, one entry per screen
      - image_features:      (N, 2048) tensor, one row per screen, same order as screenId
      - flat_df:              DataFrame, one row per (screenId, caption) — all ~5 captions/screen,
                              columns: screenId, caption_idx, caption, app_package_name, category
      - flat_text_features:  (M, 768) tensor aligned row-for-row with flat_df (M = len(flat_df))
      - eval_df:              DataFrame, one row per screen (caption_idx == 0 only), in the SAME
                              row order as image_features/screenId — i.e. eval_df.iloc[i] and
                              image_features[i] describe the same screen.
      - eval_text_features:  (N, 768) tensor, flat_text_features filtered to caption_idx == 0,
                              aligned row-for-row with eval_df and image_features.

    `resnet`/`bert` are returned too since 03/04 train small heads on top of these same frozen
    backbones (no need to reload them separately).
    """
    from datasets import load_dataset

    ds = load_dataset("rootsautomation/RICO-Screen2Words")
    resnet, resnet_preprocess = build_resnet50_backbone(device)
    bert, bert_tokenizer = build_bert_backbone(device)

    data = {}
    for split_name in splits:
        split_data = ds[split_name]
        screen_ids = split_data["screenId"]
        images = split_data["image"]

        image_features, _ = extract_resnet50_features(
            images, resnet, resnet_preprocess, device,
            batch_size=image_batch_size, show_progress=show_progress)

        rows = []
        for row in split_data:
            sid, app, cat = row["screenId"], row["app_package_name"], row["category"]
            for idx, cap in enumerate(row["captions"]):
                cap_clean = clean_caption(cap)
                if cap_clean:
                    rows.append({"screenId": sid, "caption_idx": idx, "caption": cap_clean,
                                 "app_package_name": app, "category": cat})
        flat_df = pd.DataFrame(rows)

        flat_text_features, _ = extract_bert_features(
            flat_df["caption"].tolist(), bert, bert_tokenizer, device,
            batch_size=text_batch_size, show_progress=show_progress)

        eval_mask = (flat_df["caption_idx"] == 0).values
        eval_df = flat_df[eval_mask].reset_index(drop=True)
        eval_text_features = flat_text_features[eval_mask]

        data[split_name] = {
            "screenId": torch.tensor(screen_ids),
            "image_features": image_features,
            "flat_df": flat_df,
            "flat_text_features": flat_text_features,
            "eval_df": eval_df,
            "eval_text_features": eval_text_features,
        }

    return data, resnet, bert


def cosine_score_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-normalised cosine similarity matrix: result[i, j] = cosine(a[i], b[j])."""
    a = a / a.norm(dim=1, keepdim=True).clamp(min=1e-9)
    b = b / b.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return a @ b.T


# TODO: add your plotting helpers here too (score heatmap, training curves) once you've
# written them once in a notebook — move them here the second time you need them elsewhere.
