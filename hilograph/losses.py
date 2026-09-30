"""Training objective (Sec. 2.3).

    L = lambda_r L_ROI + lambda_f L_3HG + lambda_KL L_KL
        + lambda_s L_smooth + lambda_t L_contrast + lambda_a L_age          Eq. (5)

The longitudinal terms (L_smooth, L_contrast) only receive contributions from
subjects with at least two visits in the mini-batch; single-visit subjects
contribute to the reconstruction, KL and age terms only.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _subject_codes(subject_ids: Sequence[str], device) -> torch.Tensor:
    lookup = {s: k for k, s in enumerate(dict.fromkeys(subject_ids))}
    return torch.tensor([lookup[s] for s in subject_ids], device=device)


def within_subject_pairs(subject_ids: Sequence[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """All ordered pairs (i, j), i != j, of scans from the same subject."""
    codes = _subject_codes(subject_ids, device)
    same = codes.unsqueeze(0) == codes.unsqueeze(1)
    same.fill_diagonal_(False)
    i, j = same.nonzero(as_tuple=True)
    return i, j


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """D_KL( N(mu, diag sigma^2) || N(0, I) ), summed over latent dims, averaged over scans."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def smoothness_loss(z: torch.Tensor, ages: torch.Tensor, subject_ids: Sequence[str],
                    min_age_gap: float = 0.5) -> torch.Tensor:
    """Eq. (3): mean over within-subject visit pairs of ||(z_2 - z_1) / (a_2 - a_1)||^2.

    Each unordered pair is counted once, oriented so that a_1 <= a_2. The age gap
    is floored at ``min_age_gap`` years so that visits recorded with identical
    (e.g. integer) ages do not divide by zero.
    """
    i, j = within_subject_pairs(subject_ids, z.device)
    keep = (ages[i] < ages[j]) | ((ages[i] == ages[j]) & (i < j))
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return z.new_zeros(())
    gap = (ages[j] - ages[i]).clamp_min(min_age_gap).unsqueeze(1)
    return ((z[j] - z[i]) / gap).pow(2).sum(dim=1).mean()


def contrastive_loss(z: torch.Tensor, subject_ids: Sequence[str],
                     temperature: float = 0.07) -> torch.Tensor:
    """Eq. (4): InfoNCE over within-subject pairs with cosine similarity.

    -1/|P| sum_{(i,j) in P} log( exp(sim(z_i, z_j)/tau) / sum_{k != i} exp(sim(z_i, z_k)/tau) )
    """
    i, j = within_subject_pairs(subject_ids, z.device)
    if i.numel() == 0:
        return z.new_zeros(())
    zn = F.normalize(z, dim=1)
    sim = zn @ zn.t() / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -log_prob[i, j].mean()


def hilograph_loss(out: Dict[str, torch.Tensor], batch, cfg: Dict[str, float]
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Total objective, Eq. (5). ``cfg`` holds the lambda_* weights, the
    contrastive temperature and ``min_age_gap``."""
    subject_ids: List[str] = list(batch.subject_id)
    z, ages = out["z"], batch.age
    terms = {
        "roi": F.mse_loss(out["x_roi_hat"], batch.x_roi),
        "3hg": F.mse_loss(out["x_3hg_hat"], batch.x_3hg),
        "kl": kl_loss(out["mu"], out["logvar"]),
        "smooth": smoothness_loss(z, ages, subject_ids, cfg.get("min_age_gap", 0.5)),
        "contrast": contrastive_loss(z, subject_ids, cfg.get("temperature", 0.07)),
        "age": F.mse_loss(out["age_hat"], ages),
    }
    total = sum(cfg[f"lambda_{name}"] * value for name, value in terms.items())
    parts = {name: float(value.detach()) for name, value in terms.items()}
    parts["total"] = float(total.detach())
    return total, parts
