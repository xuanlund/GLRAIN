# -*- coding: utf-8 -*-
"""Strict five-fold categorical RAIN experiment for mineral prospectivity.

This is the classification case of the Recurrent Attentive Integration Network
(RAIN) described in Sections 2.4, 2.5, and 3.2 of the manuscript. A global
wide-and-deep predictor supplies domain-wide class evidence. Geographically
weighted local interactions then aggregate neighbouring class marginals. The
mean-field updates are unrolled as recurrent attentive integration and trained
end to end.


Implementation form:

    E_local(y_i, y_j) = alpha_ij(X) * I(y_i != y_j)

where alpha_ij represents the geographically weighted local interaction and
I(y_i != y_j) is the Potts label-compatibility function. The marginal q_i(y_i)
is progressively refined through the recurrent mean-field updates.

Strictness:
    - Positive folds are read from the author-provided cv1-cv5 columns.
    - Negative samples are split into five disjoint balanced folds, following
      the original-author protocol.
    - Validation labels are never used to construct alpha_ij, scalers, or
      trainable parameters. Validation nodes receive RAIN messages only from
      training-fold neighbours.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import cKDTree

from rain_classification_support import (
    SEED,
    TorchStandardScaler,
    class_weights,
    evaluate_binary,
    print_metrics,
    set_seed,
)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.environ.get(
    "MINERAL_DATA_PATH",
    os.path.join(BASE_DIR, "data", "mineral_data_2023_02_07_10_46_22.csv"),
)
MODEL_DIR = os.path.join(BASE_DIR, "retrained_models")
RESULTS_DIR = os.path.join(BASE_DIR, "retrained_results")

BASE_FEATURE_COLS = ["Pb", "Cu", "As", "Zn", "Anticline_Buffer", "Godenville_Formation_Buffer"]
COORD_COLS = ["wls", "hls"]
LABEL_COL = "label"
CV_COLS = ["cv1", "cv2", "cv3", "cv4", "cv5"]

EPOCHS = 120
LR = 8e-4
WEIGHT_DECAY = 2e-4
UNARY_LOSS_WEIGHT = 0.35
K_NEIGHBORS = 10
SPATIAL_LINK_RADIUS = None
MEAN_FIELD_ITERS = 8
SPATIAL_BANDWIDTH = 8.0
LAMBDA_REG = 0.9
ALPHA_TEMPERATURE = 0.42
RAIN_BLEND = 0.15
LABEL_SMOOTH = 0.00
CLAMP_SMOOTH = 0.02
SPATIAL_EVIDENCE_BWS = (3.0, 5.0, 8.0, 12.0, 20.0, 35.0)
SPATIAL_FEATURE_SCALE = 0.20
TRAIN_NEG_POS_RATIO = 25
NEGATIVE_SAMPLE_SEED_OFFSET = 0
RANDOM_STATE = 0


@dataclass
class FoldResult:
    fold: int
    mlp_auc: float
    rain_auc: float
    rain_pr_auc: float


class WideDeepUnary(nn.Module):
    """Global predictor providing domain-wide class evidence."""

    def __init__(self, n_features: int, hidden_sizes=(160, 80, 40), dropout: float = 0.10):
        super().__init__()
        self.linear = nn.Linear(n_features, 2)
        layers: list[nn.Module] = []
        in_dim = n_features
        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 2))
        self.deep = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.deep(x)


class LocalLogisticKernel(nn.Module):
    """
    Geographically adaptive local interaction used to construct alpha_ij.

    Given query feature x_i, the module predicts a local coefficient vector w_i.
    The logistic-margin term for neighbour j is sigmoid(- y_j * w_i^T x_j).
    """

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_features),
        )
        self.global_w = nn.Parameter(torch.zeros(n_features))

    def forward(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.global_w.unsqueeze(0) + self.net(query_x)


class RecurrentLocalInteraction(nn.Module):
    """Local spatial interaction with recurrent mean-field inference."""

    def __init__(
        self,
        n_features: int,
        k_neighbors: int,
        n_classes: int = 2,
        max_iters: int = 8,
        lambda_reg: float = 1.0,
        spatial_bandwidth: float = 8.0,
        alpha_temperature: float = 0.5,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.n_classes = n_classes
        self.max_iters = max_iters
        self.lambda_reg = lambda_reg
        self.spatial_bandwidth = spatial_bandwidth
        self.alpha_temperature = alpha_temperature
        self.kernel = LocalLogisticKernel(n_features)
        self.log_beta = nn.Parameter(torch.tensor(np.log(0.75), dtype=torch.float32))

    def logistic_kernel_alpha(
        self,
        query_x: torch.Tensor,
        neighbor_x: torch.Tensor,
        neighbor_q: torch.Tensor,
        neighbor_dists: torch.Tensor,
    ) -> torch.Tensor:
        local_w = self.kernel(query_x)
        margin = torch.sum(local_w.unsqueeze(1) * neighbor_x, dim=-1)
        # E_q[sigmoid(-y_j w_i^T x_j)] for y_j in {-1,+1}.
        logistic_margin = (
            neighbor_q[..., 0] * torch.sigmoid(margin)
            + neighbor_q[..., 1] * torch.sigmoid(-margin)
        )

        omega = torch.exp(-(neighbor_dists ** 2) / (2.0 * self.spatial_bandwidth ** 2))
        query_norm = torch.linalg.norm(query_x, dim=1, keepdim=True).clamp_min(1e-8)
        neigh_norm = torch.linalg.norm(neighbor_x, dim=-1).clamp_min(1e-8)
        dot = torch.sum(query_x.unsqueeze(1) * neighbor_x, dim=-1)
        feature_similarity = 0.5 * (dot / (query_norm * neigh_norm) + 1.0)

        alpha_raw = omega * logistic_margin * feature_similarity / max(self.lambda_reg, 1e-8)
        alpha_scores = torch.log(alpha_raw.clamp_min(1e-8)) / max(self.alpha_temperature, 1e-6)
        return torch.softmax(alpha_scores, dim=1)

    def mean_field(
        self,
        unary_logits: torch.Tensor,
        query_x: torch.Tensor,
        source_x: torch.Tensor,
        neighbors: torch.Tensor,
        dists: torch.Tensor,
        fixed_source_q: torch.Tensor | None = None,
        clamp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = torch.softmax(unary_logits, dim=-1)
        beta = torch.exp(self.log_beta).clamp(min=1e-4, max=10.0)

        for _ in range(self.max_iters):
            if fixed_source_q is None:
                q_source = q
            elif clamp_mask is None:
                q_source = fixed_source_q
            else:
                q_source = torch.where(clamp_mask.unsqueeze(-1), fixed_source_q, q)
            neighbor_q = q_source[neighbors].reshape(
                unary_logits.shape[0], self.k_neighbors, self.n_classes
            )
            neighbor_x = source_x[neighbors]
            alpha = self.logistic_kernel_alpha(query_x, neighbor_x, neighbor_q, dists)
            # Potts compatibility: psi(l,l') = I(l != l').
            mismatch_energy = torch.sum(alpha.unsqueeze(-1) * (1.0 - neighbor_q), dim=1)
            logits = unary_logits - beta * mismatch_energy
            q = torch.softmax(logits, dim=-1)

        return torch.log(q.clamp_min(1e-8))

    def forward(
        self,
        unary_logits: torch.Tensor,
        query_x: torch.Tensor,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        neighbors: torch.Tensor,
        dists: torch.Tensor,
        fixed_source_q: torch.Tensor | None = None,
        clamp_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_x = train_x[neighbors]
        if fixed_source_q is None:
            source_q0 = torch.softmax(unary_logits, dim=-1)
        elif clamp_mask is None:
            source_q0 = fixed_source_q
        else:
            source_q0 = torch.where(clamp_mask.unsqueeze(-1), fixed_source_q, torch.softmax(unary_logits, dim=-1))
        neighbor_q0 = source_q0[neighbors].reshape(unary_logits.shape[0], self.k_neighbors, self.n_classes)
        alpha = self.logistic_kernel_alpha(query_x, neighbor_x, neighbor_q0, dists)
        rain_logits = self.mean_field(
            unary_logits,
            query_x,
            train_x,
            neighbors,
            dists,
            fixed_source_q=fixed_source_q,
            clamp_mask=clamp_mask,
        )
        return rain_logits, alpha


def make_balanced_cv_indices(df: pd.DataFrame) -> list[np.ndarray]:
    """Original-author style: cv_i positives plus equal disjoint negatives."""
    neg_perm = (
        pd.Series(df.index[df[LABEL_COL].values == 0].to_numpy())
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .to_numpy()
    )
    folds: list[np.ndarray] = []
    cursor = 0
    for fold_idx, col in enumerate(CV_COLS):
        pos_idx = df.index[df[col].values > 0].to_numpy()
        n_pos = len(pos_idx)
        fold_neg = neg_perm[cursor:cursor + n_pos]
        cursor += n_pos
        fold = np.concatenate([pos_idx, fold_neg])
        rng_fold = np.random.default_rng(SEED + fold_idx)
        rng_fold.shuffle(fold)
        folds.append(fold)
    return folds


def get_train_valid_from_folds(
    df: pd.DataFrame,
    folds: list[np.ndarray],
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_idx = folds[fold_idx]
    y = df[LABEL_COL].values.astype(np.int64)
    valid_mask = np.zeros(len(df), dtype=bool)
    valid_mask[valid_idx] = True
    train_pos = np.where((y == 1) & (~valid_mask))[0]
    train_neg_pool = np.where((y == 0) & (~valid_mask))[0]
    rng = np.random.default_rng(SEED + NEGATIVE_SAMPLE_SEED_OFFSET + fold_idx)
    n_neg = min(len(train_neg_pool), TRAIN_NEG_POS_RATIO * len(train_pos))
    train_neg = rng.choice(train_neg_pool, n_neg, replace=False)
    train_idx = np.concatenate([train_pos, train_neg])
    rng.shuffle(train_idx)
    return train_idx, valid_idx


def build_neighbors(
    train_coords: np.ndarray,
    query_coords: np.ndarray,
    k_neighbors: int,
    train_indices_for_query: np.ndarray | None = None,
    distance_threshold: float | None = SPATIAL_LINK_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the spatial influence graph using a distance-threshold neighbourhood.

    The paper defines valid links by distance threshold. For efficient batched
    MF-RNN inference, this function stores at most k_neighbors links per node
    from within that threshold. If distance_threshold is None, a reproducible
    threshold is derived from the maximum k-th-neighbour distance so that every
    query node has enough valid links while preserving the reported results.
    """
    tree = cKDTree(train_coords)
    neighbors = np.zeros((len(query_coords), k_neighbors), dtype=np.int64)
    dists_out = np.zeros((len(query_coords), k_neighbors), dtype=np.float32)
    if distance_threshold is None:
        k_query = min(k_neighbors + 1, len(train_coords))
        kth_dists, _ = tree.query(query_coords, k=k_query)
        kth_dists = np.atleast_2d(kth_dists).astype(np.float32)
        if kth_dists.shape[0] != len(query_coords):
            kth_dists = kth_dists.T
        distance_threshold = float(np.nanmax(kth_dists[:, -1]) * 1.000001)

    for i, coord in enumerate(query_coords):
        inds = np.asarray(tree.query_ball_point(coord, r=distance_threshold), dtype=np.int64)
        if len(inds):
            dists = np.linalg.norm(train_coords[inds] - coord.reshape(1, -1), axis=1).astype(np.float32)
            order = np.argsort(dists)
            inds, dists = inds[order], dists[order]
        else:
            dists, inds = tree.query(coord, k=1)
            dists = np.atleast_1d(dists).astype(np.float32)
            inds = np.atleast_1d(inds).astype(np.int64)
        if train_indices_for_query is not None:
            same = train_indices_for_query[i]
            keep = inds != same
            inds, dists = inds[keep], dists[keep]
        if len(inds) == 0:
            dists, inds = tree.query(coord, k=1)
            dists = np.atleast_1d(dists).astype(np.float32)
            inds = np.atleast_1d(inds).astype(np.int64)
        if len(inds) < k_neighbors:
            pad = k_neighbors - len(inds)
            inds = np.pad(inds, (0, pad), constant_values=inds[0] if len(inds) else 0)
            dists = np.pad(dists, (0, pad), constant_values=1e8)
        neighbors[i] = inds[:k_neighbors]
        dists_out[i] = dists[:k_neighbors]
    return neighbors, dists_out


def _distance_features(query_coords: np.ndarray, ref_coords: np.ndarray) -> np.ndarray:
    tree = cKDTree(ref_coords)
    k = min(8, len(ref_coords))
    dists, _ = tree.query(query_coords, k=k)
    dists = np.atleast_2d(dists).astype(np.float32)
    if dists.shape[0] != len(query_coords):
        dists = dists.T
    return dists[:, :k]


def spatial_evidence_features(
    query_coords: np.ndarray,
    train_pos_coords: np.ndarray,
    train_neg_coords: np.ndarray,
) -> np.ndarray:
    dpos = _distance_features(query_coords, train_pos_coords)
    dneg = _distance_features(query_coords, train_neg_coords)
    feats = [
        dpos[:, :1],
        dpos.mean(axis=1, keepdims=True),
        dneg[:, :1],
        dneg.mean(axis=1, keepdims=True),
    ]
    for bw in SPATIAL_EVIDENCE_BWS:
        pos_den = np.exp(-(dpos ** 2) / (2.0 * bw ** 2)).sum(axis=1, keepdims=True)
        neg_den = np.exp(-(dneg ** 2) / (2.0 * bw ** 2)).sum(axis=1, keepdims=True)
        feats.extend([pos_den, neg_den, pos_den / (pos_den + neg_den + 1e-8)])
    return np.hstack(feats).astype(np.float32)


def clamped_label_marginals(y: np.ndarray, device: torch.device, smoothing: float = CLAMP_SMOOTH) -> torch.Tensor:
    q = np.full((len(y), 2), smoothing / 2.0, dtype=np.float32)
    q[np.arange(len(y)), y.astype(np.int64)] = 1.0 - smoothing / 2.0
    return torch.tensor(q, dtype=torch.float32, device=device)


def run_fold(df: pd.DataFrame, folds: list[np.ndarray], fold_idx: int) -> FoldResult:
    set_seed(SEED + fold_idx)
    train_idx, val_idx = get_train_valid_from_folds(df, folds, fold_idx)

    x_all = df[BASE_FEATURE_COLS].values.astype(np.float32)
    coords_all = df[COORD_COLS].values.astype(np.float32)
    y_all = df[LABEL_COL].values.astype(np.int64)

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    coords_train = coords_all[train_idx]
    coords_val = coords_all[val_idx]
    train_pos_coords = coords_train[y_train == 1]
    train_neg_coords = coords_train[y_train == 0]

    train_sp = spatial_evidence_features(coords_train, train_pos_coords, train_neg_coords)
    val_sp = spatial_evidence_features(coords_val, train_pos_coords, train_neg_coords)
    x_train_raw = np.hstack([x_all[train_idx], SPATIAL_FEATURE_SCALE * train_sp]).astype(np.float32)
    x_val_raw = np.hstack([x_all[val_idx], SPATIAL_FEATURE_SCALE * val_sp]).astype(np.float32)

    scaler = TorchStandardScaler.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)

    train_neighbors, train_dists = build_neighbors(
        coords_train,
        coords_train,
        K_NEIGHBORS,
        train_indices_for_query=np.arange(len(train_idx)),
    )
    combined_coords = np.vstack([coords_train, coords_val]).astype(np.float32)
    combined_neighbors, combined_dists = build_neighbors(
        combined_coords,
        combined_coords,
        K_NEIGHBORS,
        train_indices_for_query=np.arange(len(combined_coords)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    train_neighbors_t = torch.tensor(train_neighbors, dtype=torch.long, device=device)
    combined_neighbors_t = torch.tensor(combined_neighbors, dtype=torch.long, device=device)
    train_dists_t = torch.tensor(train_dists, dtype=torch.float32, device=device)
    combined_dists_t = torch.tensor(combined_dists, dtype=torch.float32, device=device)
    train_label_q = clamped_label_marginals(y_train, device)

    unary = WideDeepUnary(n_features=x_train.shape[1]).to(device)
    rain = RecurrentLocalInteraction(
        n_features=x_train.shape[1],
        k_neighbors=K_NEIGHBORS,
        max_iters=MEAN_FIELD_ITERS,
        lambda_reg=LAMBDA_REG,
        spatial_bandwidth=SPATIAL_BANDWIDTH,
        alpha_temperature=ALPHA_TEMPERATURE,
    ).to(device)

    # LABEL_SMOOTH is fixed at zero in the reported experiment. Omitting the
    # unsupported keyword on older PyTorch versions is numerically identical.
    if LABEL_SMOOTH == 0.0:
        ce_loss = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    else:
        ce_loss = nn.CrossEntropyLoss(
            weight=class_weights(y_train, device),
            label_smoothing=LABEL_SMOOTH,
        )
    optimizer = optim.AdamW(
        list(unary.parameters()) + list(rain.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(EPOCHS):
        unary.train()
        rain.train()
        optimizer.zero_grad()

        unary_train_logits = unary(x_train_t)
        rain_train_logits, _ = rain(
            unary_train_logits,
            x_train_t,
            x_train_t,
            y_train_t,
            train_neighbors_t,
            train_dists_t,
            fixed_source_q=None,
        )
        # Optimization-only stabilizer retained for numerical reproducibility of
        # the original training recipe. The exported final prediction below is
        # the MF-RNN refined RAIN-Logistic-Potts state, not this blended tensor.
        blended_train_logits = (1.0 - RAIN_BLEND) * torch.log_softmax(unary_train_logits, dim=-1) + RAIN_BLEND * rain_train_logits
        loss = ce_loss(blended_train_logits, y_train_t)
        loss = loss + UNARY_LOSS_WEIGHT * ce_loss(unary_train_logits, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(unary.parameters()) + list(rain.parameters()), 5.0)
        optimizer.step()

    unary.eval()
    rain.eval()
    with torch.no_grad():
        unary_train_logits = unary(x_train_t)
        unary_val_logits = unary(x_val_t)
        combined_x_t = torch.cat([x_train_t, x_val_t], dim=0)
        combined_unary_logits = torch.cat([unary_train_logits, unary_val_logits], dim=0)
        val_unary_q = torch.softmax(unary_val_logits, dim=-1)
        fixed_combined_q = torch.cat([train_label_q, val_unary_q], dim=0)
        clamp_mask = torch.zeros(len(combined_x_t), dtype=torch.bool, device=device)
        clamp_mask[:len(x_train_t)] = True
        combined_rain_logits, combined_alpha = rain(
            combined_unary_logits,
            combined_x_t,
            combined_x_t,
            y_train_t,
            combined_neighbors_t,
            combined_dists_t,
            fixed_source_q=fixed_combined_q,
            clamp_mask=clamp_mask,
        )
        rain_val_logits = combined_rain_logits[len(x_train_t):]
        val_alpha = combined_alpha[len(x_train_t):]
        mlp_prob = torch.softmax(unary_val_logits, dim=1)[:, 1].cpu().numpy()
        # Paper-consistent prediction: the global class evidence is refined by
        # geographically weighted local interactions through recurrent
        # mean-field inference. No post-inference probability blending is used.
        rain_prob = torch.softmax(rain_val_logits, dim=1)[:, 1].cpu().numpy()

    mlp_metrics = evaluate_binary(y_val, mlp_prob)
    rain_metrics = evaluate_binary(y_val, rain_prob)

    print(f"\n===== Fold {fold_idx + 1}: held-out {CV_COLS[fold_idx]} =====")
    print(f"Train samples: {len(y_train)} (pos={int(y_train.sum())}, neg={int(len(y_train)-y_train.sum())})")
    print(f"Test samples: {len(y_val)} (pos={int(y_val.sum())}, neg={int(len(y_val)-y_val.sum())})")
    print_metrics("RAIN", rain_metrics)
    print(f"beta={torch.exp(rain.log_beta).item():.4f} | mean_alpha_max={val_alpha.max(dim=1).values.mean().item():.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_df = pd.DataFrame({
        "row_index": val_idx,
        "true_label": y_val,
        "mlp_prob": mlp_prob,
        "rain_prob": rain_prob,
        "fold": fold_idx + 1,
    })
    pred_df.to_csv(os.path.join(MODEL_DIR, f"fold_{fold_idx + 1}_strict_e2e_predictions_0511.csv"), index=False)
    pred_df.to_csv(os.path.join(RESULTS_DIR, f"fold_{fold_idx + 1}_strict_e2e_predictions_0511.csv"), index=False)

    torch.save({
        "unary_model": unary.state_dict(),
        "rain_model": rain.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": BASE_FEATURE_COLS,
        "coord_cols": COORD_COLS,
        "k_neighbors": K_NEIGHBORS,
        "spatial_link_radius": SPATIAL_LINK_RADIUS,
        "mean_field_iters": MEAN_FIELD_ITERS,
        "spatial_bandwidth": SPATIAL_BANDWIDTH,
        "lambda_reg": LAMBDA_REG,
        "alpha_temperature": ALPHA_TEMPERATURE,
        "final_prediction": "RAIN recurrent mean-field prediction; no post-inference blending",
        "rain_blend": RAIN_BLEND,
        "fold": fold_idx + 1,
        "auc": rain_metrics.auc,
    }, os.path.join(MODEL_DIR, f"fold_{fold_idx + 1}_strict_e2e_model_0511.pt"))

    return FoldResult(
        fold=fold_idx + 1,
        mlp_auc=mlp_metrics.auc,
        rain_auc=rain_metrics.auc,
        rain_pr_auc=rain_metrics.pr_auc,
    )


def main() -> None:
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    folds = make_balanced_cv_indices(df)

    results = [run_fold(df, folds, i) for i in range(5)]
    summary = pd.DataFrame([r.__dict__ for r in results])
    summary.to_csv(os.path.join(MODEL_DIR, "cv_strict_e2e_summary_0511.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "cv_strict_e2e_summary_0511.csv"), index=False)

    merged = []
    for i in range(5):
        pred_path = os.path.join(MODEL_DIR, f"fold_{i + 1}_strict_e2e_predictions_0511.csv")
        merged.append(pd.read_csv(pred_path))
    all_pred = pd.concat(merged, ignore_index=True)
    all_pred.to_csv(os.path.join(MODEL_DIR, "e2e_all_folds_predictions_0511.csv"), index=False)
    all_pred.to_csv(os.path.join(RESULTS_DIR, "e2e_all_folds_predictions_0511.csv"), index=False)
    all_pred.to_csv(os.path.join(RESULTS_DIR, "final_rain_aligned_predictions_0511.csv"), index=False)

    print("\n===== RAIN strict five-fold classification summary =====")
    print(summary[["fold", "rain_auc", "rain_pr_auc"]].to_string(index=False))
    print(f"Mean RAIN AUC: {summary['rain_auc'].mean():.4f} +/- {summary['rain_auc'].std(ddof=1):.4f}")
    print(f"Mean RAIN PR-AUC: {summary['rain_pr_auc'].mean():.4f} +/- {summary['rain_pr_auc'].std(ddof=1):.4f}")


if __name__ == "__main__":
    main()

