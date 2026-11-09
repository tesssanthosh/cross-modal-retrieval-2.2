"""
Shared utilities for the cross-modal retrieval project.

Import these from every notebook instead of redefining them — this is what the rubric means by
"modularity" (Criteria 6). Fill in each function as you build the corresponding notebook; the
signatures below follow the pattern used in the Week 7/8 practicals (score matrix in, metrics out).
"""

import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
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


# --- Training-pair dataset ---------------------------------------------------------------
# First used in notebook 03 (defined inline there), now needed identically in notebook 04 —
# moved here per this file's own rule.

def build_screen_idx(data: dict) -> dict:
    """screenId -> row-index lookup per split, from load_rico_splits_with_features's output."""
    return {split_name: {int(sid): i for i, sid in enumerate(d["screenId"].tolist())}
            for split_name, d in data.items()}


class PairDataset(torch.utils.data.Dataset):
    """One row per (image, caption) training pair, image repeated across its ~5 captions.

    `flat_df`/`text_features` are a split's flat_df/flat_text_features from
    load_rico_splits_with_features (one row per caption); `image_features` is that split's
    per-screen image features; `screen_idx` is that split's entry from build_screen_idx(data).
    """
    def __init__(self, flat_df, image_features, text_features, screen_idx):
        self.flat_df = flat_df.reset_index(drop=True)
        self.image_features = image_features
        self.text_features = text_features
        self.screen_idx = screen_idx

    def __len__(self):
        return len(self.flat_df)

    def __getitem__(self, i):
        screen_id = int(self.flat_df.iloc[i]["screenId"])
        img = self.image_features[self.screen_idx[screen_id]]
        txt = self.text_features[i]
        return img, txt


# --- Early fusion (technique 1, notebook 03) ---------------------------------------------
# First defined inline in notebook 03, needed identically (model + training loop + the
# pairwise-scoring workaround for not having a standalone embedding) in notebook 05's final
# comparison — moved here per this file's own rule.

class EarlyFusionScorer(nn.Module):
    """Concatenate [image_embedding ; caption_embedding] then score the pair with a small MLP
    ("early fusion" — the two modalities are combined before a single network ever sees them,
    unlike the dual encoder's late fusion via a dot product). Outputs one scalar logit per pair;
    there is no standalone embedding to cache, which is the architectural root of this
    technique's retrieval-time cost (see score_matrix_concat below).
    """
    def __init__(self, image_dim=2048, text_dim=768, hidden=(512, 128), dropout=0.2):
        super().__init__()
        dims = [image_dim + text_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pairwise_batch_loss(model, imgs, txts, criterion, num_negatives=4):
    """One positive + `num_negatives` in-batch-shifted negatives per sample, in one forward pass.
    Shift offsets 1..num_negatives are all distinct (mod batch size), so each negative caption is
    a genuinely different mismatched caption, not a repeat.
    """
    b = imgs.shape[0]
    assert num_negatives < b, "num_negatives must be smaller than the batch size"
    pos_in = torch.cat([imgs, txts], dim=1)
    neg_ins = [torch.cat([imgs, txts.roll(shifts=shift, dims=0)], dim=1)
               for shift in range(1, num_negatives + 1)]
    x = torch.cat([pos_in] + neg_ins, dim=0)
    y = torch.cat([torch.ones(b), torch.zeros(b * num_negatives)]).to(imgs.device)
    logits = model(x)
    return criterion(logits, y)


def train_early_fusion_mlp(train_loader, val_loader, device, image_dim=2048, text_dim=768,
                            hidden=(512, 128), dropout=0.2, num_negatives=4, lr=1e-3,
                            weight_decay=1e-4, max_epochs=20, patience=4, seed=None,
                            verbose=True):
    """Builds and trains an EarlyFusionScorer from scratch, early-stopping on validation loss
    (Week 7 rule — not a fixed epoch count). Generalises notebook 03's inline training loop so
    notebook 05 can retrain this exact technique (same architecture/hyperparameters already
    decided in 03) without duplicating the loop a third time.

    Returns (model, history, best_val_loss); `model` already has the best (restored) epoch's
    weights loaded, and `history` is {"train_loss": [...], "val_loss": [...]}.
    """
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = EarlyFusionScorer(image_dim=image_dim, text_dim=text_dim, hidden=hidden,
                               dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # pos_weight rebalances the 1-positive-to-num_negatives-negatives ratio from pairwise_batch_loss.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(num_negatives), device=device))

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    bad_epochs = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for imgs, txts in train_loader:
            imgs, txts = imgs.to(device), txts.to(device)
            opt.zero_grad()
            loss = pairwise_batch_loss(model, imgs, txts, criterion, num_negatives=num_negatives)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, txts in val_loader:
                imgs, txts = imgs.to(device), txts.to(device)
                val_losses.append(
                    pairwise_batch_loss(model, imgs, txts, criterion,
                                         num_negatives=num_negatives).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"  epoch {epoch:2d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f}).")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_loss


@torch.no_grad()
def score_matrix_concat(model, image_feats, text_feats, device, q_chunk=128, c_chunk=512):
    """Chunked GPU pairwise scorer for an early-fusion model that scores (image, caption) PAIRS
    rather than producing standalone, independently-cacheable embeddings (see EarlyFusionScorer
    above) — this means O(n*m) model calls instead of cosine_score_matrix's single matmul, which
    is exactly the retrieval-time cost this project's efficiency comparison is built around.
    Returns an (n, m) CPU score matrix of raw logits (not probabilities).
    """
    model.eval()
    n, m = image_feats.shape[0], text_feats.shape[0]
    scores = torch.empty(n, m)
    image_feats = image_feats.to(device)
    text_feats = text_feats.to(device)
    for qs in range(0, n, q_chunk):
        qe = min(qs + q_chunk, n)
        img_block = image_feats[qs:qe]
        for cs in range(0, m, c_chunk):
            ce = min(cs + c_chunk, m)
            txt_block = text_feats[cs:ce]
            q, c = img_block.shape[0], txt_block.shape[0]
            img_rep = img_block.unsqueeze(1).expand(q, c, img_block.shape[1]).reshape(q * c, -1)
            txt_rep = txt_block.unsqueeze(0).expand(q, c, txt_block.shape[1]).reshape(q * c, -1)
            logits = model(torch.cat([img_rep, txt_rep], dim=1)).view(q, c)
            scores[qs:qe, cs:ce] = logits.cpu()
    return scores


# --- Contrastive dual encoder (technique 2, notebook 04) ----------------------------------
# First defined inline in notebook 04, needed identically in notebook 05's final comparison —
# moved here per this file's own rule.

class ImageProjector(nn.Module):
    """Projects frozen ResNet-50 features into the shared, L2-normalised contrastive embedding
    space used by the dual encoder."""
    def __init__(self, in_dim=2048, hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TextProjector(nn.Module):
    """Projects frozen BERT features into the shared, L2-normalised contrastive embedding space
    used by the dual encoder."""
    def __init__(self, in_dim=768, hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_dual_encoder(train_loader, val_loader, device, temperature=0.07, embed_dim=256,
                        image_in_dim=2048, text_in_dim=768, hidden=512, dropout=0.1, lr=1e-3,
                        max_epochs=30, patience=4, seed=None, verbose=True):
    """Builds and jointly trains ImageProjector/TextProjector with info_nce (the Week 8 practical
    formula). Early stopping within the run watches validation InfoNCE loss only (Week 7 rule).
    Generalises notebook 04's inline training function (which closed over module-level globals)
    so notebook 05 can retrain this exact technique (same architecture/temperature already
    decided in 04's validation-Recall@10 sweep) without duplicating it a third time.

    Returns (image_proj, text_proj, history, best_val_loss), with both heads' best (restored)
    epoch weights already loaded.
    """
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    image_proj = ImageProjector(in_dim=image_in_dim, hidden=hidden, out_dim=embed_dim,
                                 dropout=dropout).to(device)
    text_proj = TextProjector(in_dim=text_in_dim, hidden=hidden, out_dim=embed_dim,
                               dropout=dropout).to(device)
    opt = torch.optim.Adam(list(image_proj.parameters()) + list(text_proj.parameters()), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        image_proj.train(); text_proj.train()
        train_losses = []
        for imgs, txts in train_loader:
            imgs, txts = imgs.to(device), txts.to(device)
            opt.zero_grad()
            img_emb = F.normalize(image_proj(imgs), dim=1)
            txt_emb = F.normalize(text_proj(txts), dim=1)
            loss = info_nce(img_emb, txt_emb, temperature=temperature)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        image_proj.eval(); text_proj.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, txts in val_loader:
                imgs, txts = imgs.to(device), txts.to(device)
                img_emb = F.normalize(image_proj(imgs), dim=1)
                txt_emb = F.normalize(text_proj(txts), dim=1)
                val_losses.append(info_nce(img_emb, txt_emb, temperature=temperature).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"  epoch {epoch:2d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {
                "image_proj": {k: v.detach().cpu().clone() for k, v in image_proj.state_dict().items()},
                "text_proj": {k: v.detach().cpu().clone() for k, v in text_proj.state_dict().items()},
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best val_loss={best_val:.4f} at epoch {best_epoch}).")
                break

    image_proj.load_state_dict(best_state["image_proj"])
    text_proj.load_state_dict(best_state["text_proj"])
    return image_proj, text_proj, history, best_val


# --- Shared evaluation/data helpers (notebook 05 onward) ----------------------------------
# Each of these was informally reinvented inline more than once (03/04) — consolidated here
# the next time (05) they're needed, per this file's own rule.

def standardize_features(data, splits=("train", "val", "test"), fit_split="train"):
    """Returns a NEW dict with image/text features standardised using `fit_split`'s mean/std
    (train-only statistics, per the Week 7 rule), WITHOUT mutating `data`.

    This matters starting with notebook 05: the dual encoder needs the original, un-standardised
    features at the same time Ridge/the early-fusion MLP need standardised ones. Notebook 03's
    original standardisation mutated `data` in place, which only worked there because that
    notebook never needed both versions to coexist.

    Each split's dict is shallow-copied — screenId/flat_df/eval_df are shared by reference, since
    they are not modified — with image_features/flat_text_features/eval_text_features replaced by
    standardised copies.
    """
    img_mu = data[fit_split]["image_features"].mean(0, keepdim=True)
    img_sd = data[fit_split]["image_features"].std(0, keepdim=True).clamp(min=1e-6)
    txt_mu = data[fit_split]["flat_text_features"].mean(0, keepdim=True)
    txt_sd = data[fit_split]["flat_text_features"].std(0, keepdim=True).clamp(min=1e-6)

    standardized = {}
    for split_name in splits:
        d = data[split_name]
        new_d = dict(d)  # shallow copy: screenId/flat_df/eval_df untouched, shared by reference
        new_d["image_features"] = (d["image_features"] - img_mu) / img_sd
        new_d["flat_text_features"] = (d["flat_text_features"] - txt_mu) / txt_sd
        new_d["eval_text_features"] = (d["eval_text_features"] - txt_mu) / txt_sd
        standardized[split_name] = new_d
    return standardized


def subsample_idx(n_total, max_n, seed=0):
    """Row indices for a reproducible random subsample of size `max_n` out of `n_total` (or all
    `n_total` indices, in original order, if `max_n` is None or already >= n_total). Consolidates
    the informal version each of notebooks 03/04 reinvented inline for capping expensive train-set
    evaluation.
    """
    if max_n is None or n_total <= max_n:
        return torch.arange(n_total)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_total, generator=g)[:max_n]


# --- Plotting helpers (notebook 05 onward) -------------------------------------------------
# The loss-curve figure (03, 04) and the score-heatmap figure (01, 03, 04) have each now been
# hand-written three times; notebook 05 needing them a third/fourth time is exactly the trigger
# this file's own module docstring describes for moving something here instead.

def plot_loss_curve(history, title, save_path, ylabel="loss"):
    """Train-vs-validation loss curve with the restored best (lowest val-loss) epoch marked —
    same figure used by notebooks 03 and 04's training loops."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    ax.plot(epochs, history["train_loss"], label="train loss")
    ax.plot(epochs, history["val_loss"], label="val loss")
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5,
               label=f"best epoch ({best_epoch}, restored)")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_score_heatmap(scores, title, save_path, n_show=40, cbar_label="score"):
    """Score-matrix heatmap diagnostic (first `n_show` rows/columns only, so the diagonal match
    pattern is actually visible) — same figure used by notebooks 01, 03 and 04."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(scores[:n_show, :n_show], cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Caption index"); ax.set_ylabel("Image index")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
