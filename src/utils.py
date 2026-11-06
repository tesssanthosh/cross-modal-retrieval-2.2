"""
Shared utilities for the cross-modal retrieval project.

Import these from every notebook instead of redefining them — this is what the rubric means by
"modularity" (Criteria 6). Fill in each function as you build the corresponding notebook; the
signatures below follow the pattern used in the Week 7/8 practicals (score matrix in, metrics out).
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


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


# TODO: add your plotting helpers here too (score heatmap, training curves) once you've
# written them once in a notebook — move them here the second time you need them elsewhere.
