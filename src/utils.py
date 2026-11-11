# Shared utilities for the cross-modal retrieval project
# Metrics, feature extraction, model definitions, training loops, and plotting used across the notebooks

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


# Ranks the correct candidate for each query (row i's correct answer is column i)
def rank_of_correct(scores: torch.Tensor) -> torch.Tensor:
    return (scores > scores.diag().unsqueeze(1)).sum(dim=1)


# Checks if the correct candidate is in the top k, for each query
def recall_at_k(scores: torch.Tensor, k: int) -> float:
    ranks = rank_of_correct(scores)
    return (ranks < k).float().mean().item()


# Averages 1/rank over all queries
def mean_reciprocal_rank(scores: torch.Tensor) -> float:
    ranks = rank_of_correct(scores)
    return (1.0 / (ranks.float() + 1.0)).mean().item()


# Finds the median rank of the correct candidate (1 = always retrieved first)
def median_rank(scores: torch.Tensor) -> float:
    ranks = rank_of_correct(scores)
    return (ranks.float() + 1.0).median().item()


# Builds a report of Recall@K, MRR, median rank, and AUC for a score matrix
def retrieval_report(scores: torch.Tensor, ks=(1, 5, 10)) -> dict:
    out = {}
    for k in ks:
        out[f"R@{k} image->text"] = recall_at_k(scores, k)
        out[f"R@{k} text->image"] = recall_at_k(scores.T, k)
    out["MRR image->text"] = mean_reciprocal_rank(scores)
    out["MRR text->image"] = mean_reciprocal_rank(scores.T)
    out["median_rank image->text"] = median_rank(scores)
    out["median_rank text->image"] = median_rank(scores.T)
    is_match = torch.eye(len(scores)).flatten().numpy()
    out["AUC"] = roc_auc_score(is_match, scores.flatten().detach().cpu().numpy())
    return out


# Computes the symmetric InfoNCE loss between L2-normalised image and text embeddings
def info_nce(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, temperature: float = 0.07):
    logits = image_embeddings @ text_embeddings.T / temperature
    targets = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# Frozen model feature extraction, used by notebooks 02 to 05.

# Loads a frozen ImageNet-pretrained ResNet-50 with the classification head removed
def build_resnet50_backbone(device):
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # drop the classifier and keep the 2048-d pooled features
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights.transforms()


# Loads a frozen pretrained BERT encoder and tokenizer
def build_bert_backbone(device, model_name: str = "bert-base-uncased"):
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


# Extracts frozen ResNet-50 features (2048-d) for a sequence of images
@torch.no_grad()
def extract_resnet50_features(images, resnet, preprocess, device, batch_size: int = 64,
                               show_progress: bool = True):
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


# Extracts frozen BERT features (768-d) for a list of strings, mean or [CLS] pooled
@torch.no_grad()
def extract_bert_features(texts, bert, tokenizer, device, batch_size: int = 128,
                           max_length: int = 32, show_progress: bool = True, pooling: str = "mean"):
    if pooling not in ("mean", "cls"):
        raise ValueError(f"Unknown pooling {pooling!r} - expected 'mean' or 'cls'.")
    feats = []
    t0 = time.time()
    rng = range(0, len(texts), batch_size)
    if show_progress:
        rng = tqdm(rng, desc=f"BERT features ({pooling} pooling)")
    for start in rng:
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                         return_tensors="pt").to(device)
        out = bert(**enc).last_hidden_state  # (B, T, 768)
        if pooling == "cls":
            pooled = out[:, 0, :]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        feats.append(pooled.float().cpu())
    elapsed = time.time() - t0
    return torch.cat(feats, dim=0), elapsed


# Full dataset and feature loader 
# Used by notebooks 03, 04 and 05.

def clean_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Loads RICO-Screen2Words and extracts frozen ResNet-50 / BERT features for each split
def load_rico_splits_with_features(device, splits=("train", "val", "test"),
                                    image_batch_size: int = 64, text_batch_size: int = 128,
                                    show_progress: bool = True):
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


# Computes a row-normalised cosine similarity matrix
def cosine_score_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / a.norm(dim=1, keepdim=True).clamp(min=1e-9)
    b = b / b.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return a @ b.T


# Training-pair dataset 
# Used by notebooks 04 and 05

# Builds a screenId to row-index lookup per split
def build_screen_idx(data: dict) -> dict:
    return {split_name: {int(sid): i for i, sid in enumerate(d["screenId"].tolist())}
            for split_name, d in data.items()}


# One row per (image, caption) training pair, each image repeated across its ~5 captions
class PairDataset(torch.utils.data.Dataset):
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


# Early fusion (technique 1, notebook 03), also used by notebook 05

# Concatenates image and caption embeddings, scores the pair with a small MLP
class EarlyFusionScorer(nn.Module):
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


# Builds one positive pair plus num_negatives in-batch negatives per sample, in one pass
def pairwise_batch_loss(model, imgs, txts, criterion, num_negatives=4):
    b = imgs.shape[0]
    assert num_negatives < b, "num_negatives must be smaller than the batch size"
    pos_in = torch.cat([imgs, txts], dim=1)
    neg_ins = [torch.cat([imgs, txts.roll(shifts=shift, dims=0)], dim=1)
               for shift in range(1, num_negatives + 1)]
    x = torch.cat([pos_in] + neg_ins, dim=0)
    y = torch.cat([torch.ones(b), torch.zeros(b * num_negatives)]).to(imgs.device)
    logits = model(x)
    return criterion(logits, y)


# Builds and trains an EarlyFusionScorer from scratch, with early stopping on validation loss
def train_early_fusion_mlp(train_loader, val_loader, device, image_dim=2048, text_dim=768,
                            hidden=(512, 128), dropout=0.2, num_negatives=4, lr=1e-3,
                            weight_decay=1e-4, max_epochs=20, patience=4, seed=None,
                            verbose=True):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = EarlyFusionScorer(image_dim=image_dim, text_dim=text_dim, hidden=hidden,
                               dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # pos_weight balances the 1-positive vs num_negatives-negative ratio
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


# Scores every (image, caption) pair between two sets of features, in chunks on the GPU
@torch.no_grad()
def score_matrix_concat(model, image_feats, text_feats, device, q_chunk=128, c_chunk=512):
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


# Contrastive dual encoder (technique 2, notebook 04), also used by notebook 05

# Projects frozen ResNet-50 features into the shared contrastive embedding space
class ImageProjector(nn.Module):
    def __init__(self, in_dim=2048, hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# Projects frozen BERT features into the shared contrastive embedding space
class TextProjector(nn.Module):
    def __init__(self, in_dim=768, hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# Builds and trains ImageProjector and TextProjector together using info_nce loss
def train_dual_encoder(train_loader, val_loader, device, temperature=0.07, embed_dim=256,
                        image_in_dim=2048, text_in_dim=768, hidden=512, dropout=0.1, lr=1e-3,
                        max_epochs=30, patience=4, seed=None, verbose=True):
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


# Shared evaluation and data helpers (notebook 05)

# Standardises image and text features using fit_split's mean/std, doesn't modify the original data
def standardize_features(data, splits=("train", "val", "test"), fit_split="train"):
    img_mu = data[fit_split]["image_features"].mean(0, keepdim=True)
    img_sd = data[fit_split]["image_features"].std(0, keepdim=True).clamp(min=1e-6)
    txt_mu = data[fit_split]["flat_text_features"].mean(0, keepdim=True)
    txt_sd = data[fit_split]["flat_text_features"].std(0, keepdim=True).clamp(min=1e-6)

    standardized = {}
    for split_name in splits:
        d = data[split_name]
        new_d = dict(d) 
        new_d["image_features"] = (d["image_features"] - img_mu) / img_sd
        new_d["flat_text_features"] = (d["flat_text_features"] - txt_mu) / txt_sd
        new_d["eval_text_features"] = (d["eval_text_features"] - txt_mu) / txt_sd
        standardized[split_name] = new_d
    return standardized


# Row indices for a reproducible random subsample of size max_n out of n_total
def subsample_idx(n_total, max_n, seed=0):
    if max_n is None or n_total <= max_n:
        return torch.arange(n_total)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_total, generator=g)[:max_n]


# Plotting helpers (notebook 05) 

# Plots train vs validation loss with the best epoch marked
def plot_loss_curve(history, title, save_path, ylabel="loss"):
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


# Heatmap of a score matrix, showing only the first n_show rows/columns
def plot_score_heatmap(scores, title, save_path, n_show=40, cbar_label="score"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(scores[:n_show, :n_show], cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Caption index"); ax.set_ylabel("Image index")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
