# -*- coding: utf-8 -*-
"""Shared numerical utilities for categorical RAIN experiments.

The terminology follows the manuscript: a global predictor is coupled with
geographically weighted local interactions, and mean-field inference is
unrolled as recurrent attentive integration.


This file keeps the original Torch-only preprocessing, model, and metric
implementations for reproducibility. Some compatibility identifiers retain
historical names because existing checkpoints and result files use them.
"""

from __future__ import annotations

import copy
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import cKDTree


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Metrics:
    auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float


@dataclass
class TorchStandardScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "TorchStandardScaler":
        mean = x.mean(axis=0).astype(np.float32)
        scale = x.std(axis=0).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        return cls(mean_=mean, scale_=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.scale_).astype(np.float32)


class TorchMLP(nn.Module):
    """Local Evidence Module implemented as an MLP unary classifier."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...] = (128, 64, 32), dropout: float = 0.15):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialRAIN(nn.Module):
    """RAIN-Logistic-Potts interaction module with mean-field inference."""

    def __init__(self, k_neighbors: int = 16, n_classes: int = 2, max_iters: int = 15, tol: float = 1e-5):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.n_classes = n_classes
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.mean_field_max_iters = max_iters
        self.mean_field_tol = tol

    def forward(
        self,
        unary_logits: torch.Tensor,
        pairwise_weights: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_source_logits: torch.Tensor | None = None,
        neighbor_source_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        beta = torch.exp(self.log_beta).clamp(max=20.0)
        logits = unary_logits
        q = torch.softmax(logits, dim=-1)
        source_logits = logits if neighbor_source_logits is None else neighbor_source_logits

        for _ in range(self.mean_field_max_iters):
            q_old = q
            source_q = neighbor_source_probs if neighbor_source_probs is not None else torch.softmax(source_logits, dim=-1)
            neighbor_q = source_q[neighbors].reshape(unary_logits.shape[0], self.k_neighbors, self.n_classes)
            pairwise_message = torch.sum(pairwise_weights.unsqueeze(-1) * neighbor_q, dim=1)
            logits = unary_logits + beta * pairwise_message
            q = torch.softmax(logits, dim=-1)

            if torch.max(torch.abs(q - q_old)) < self.mean_field_tol:
                break
            if neighbor_source_logits is None and neighbor_source_probs is None:
                source_logits = logits

        return logits


def stratified_split_indices(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def label_probs_from_y(y: np.ndarray, device: torch.device, smoothing: float = 0.02) -> torch.Tensor:
    probs = np.full((len(y), 2), smoothing / 2.0, dtype=np.float32)
    probs[np.arange(len(y)), y.astype(np.int64)] = 1.0 - smoothing / 2.0
    return torch.tensor(probs, dtype=torch.float32, device=device)


def balanced_indices(y: np.ndarray, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))
    n = min(len(pos), len(neg))
    keep = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    rng.shuffle(keep)
    return keep


def build_neighbors(
    train_coords: np.ndarray,
    query_coords: np.ndarray,
    k_neighbors: int,
    exclude_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(train_coords)
    k_query = min(k_neighbors + int(exclude_self), len(train_coords))
    all_neighbors = np.zeros((len(query_coords), k_neighbors), dtype=np.int64)
    all_dists = np.zeros((len(query_coords), k_neighbors), dtype=np.float32)

    for i, coord in enumerate(query_coords):
        dists, inds = tree.query(coord, k=k_query)
        dists = np.atleast_1d(dists).astype(np.float32)
        inds = np.atleast_1d(inds).astype(np.int64)
        if exclude_self:
            keep = inds != i
            inds = inds[keep]
            dists = dists[keep]
        if len(inds) < k_neighbors:
            pad = k_neighbors - len(inds)
            inds = np.pad(inds, (0, pad), constant_values=inds[0] if len(inds) else 0)
            dists = np.pad(dists, (0, pad), constant_values=1e8)
        all_neighbors[i] = inds[:k_neighbors]
        all_dists[i] = dists[:k_neighbors]

    return all_neighbors, all_dists


def compute_logistic_pairwise_weights_torch(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    neighbors: torch.Tensor,
    dists: torch.Tensor,
    logistic_layer: nn.Linear,
    spatial_bandwidth: float,
    lambda_reg: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable logistic-kernel pairwise weights."""
    train_logits = logistic_layer(train_x).reshape(-1)
    neigh_x = train_x[neighbors]
    neigh_y = train_y[neighbors].float()
    neigh_logits = train_logits[neighbors]

    y_signed = 2.0 * neigh_y - 1.0
    hard_factor = torch.sigmoid(-y_signed * neigh_logits)
    spatial_omega = torch.exp(-(dists ** 2) / (2.0 * spatial_bandwidth ** 2))

    query_norm = torch.linalg.norm(query_x, dim=1, keepdim=True).clamp_min(1e-8)
    neigh_norm = torch.linalg.norm(neigh_x, dim=2).clamp_min(1e-8)
    dot = torch.sum(neigh_x * query_x.unsqueeze(1), dim=2)
    feature_kernel = 0.5 * (dot / (query_norm * neigh_norm) + 1.0)

    raw = spatial_omega * hard_factor * feature_kernel / max(lambda_reg, 1e-8)
    scores = torch.log(raw.clamp_min(1e-8)) / max(temperature, 1e-6)
    return torch.softmax(scores, dim=1)


def roc_auc_score_np(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    order = np.argsort(prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(prob) + 1)
    sorted_prob = prob[order]
    start = 0
    while start < len(prob):
        end = start + 1
        while end < len(prob) and sorted_prob[end] == sorted_prob[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos_ranks = np.sum(ranks[y_true == 1])
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_curve_np(y_true: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int64)
    thresholds = np.unique(prob)[::-1]
    precision = []
    recall = []
    n_pos = max(np.sum(y_true == 1), 1)
    for th in thresholds:
        pred = prob >= th
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / n_pos)
    precision.append(1.0)
    recall.append(0.0)
    return np.array(precision), np.array(recall), thresholds


def youden_threshold_np(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Threshold maximizing Youden's index J = sensitivity + specificity - 1."""
    y_true = y_true.astype(np.int64)
    thresholds = np.unique(prob)[::-1]
    if len(thresholds) == 0:
        return 0.5

    n_pos = max(np.sum(y_true == 1), 1)
    n_neg = max(np.sum(y_true == 0), 1)
    best_threshold = float(thresholds[0])
    best_youden = -np.inf

    for th in thresholds:
        pred = prob >= th
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tpr = tp / n_pos
        fpr = fp / n_neg
        youden = tpr - fpr
        if youden > best_youden:
            best_youden = youden
            best_threshold = float(th)

    return best_threshold


def average_precision_score_np(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    order = np.argsort(-prob)
    y_sorted = y_true[order]
    n_pos = np.sum(y_sorted == 1)
    if n_pos == 0:
        return 0.0
    tp_cumsum = np.cumsum(y_sorted == 1)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision_at_k = tp_cumsum / ranks
    return float(np.sum(precision_at_k[y_sorted == 1]) / n_pos)


def evaluate_binary(y_true: np.ndarray, prob: np.ndarray, threshold: float | None = None) -> Metrics:
    best_threshold = youden_threshold_np(y_true, prob)
    threshold = best_threshold if threshold is None else threshold
    pred = prob >= threshold

    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    tn = np.sum((pred == 0) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    return Metrics(
        auc=roc_auc_score_np(y_true, prob),
        pr_auc=average_precision_score_np(y_true, prob),
        accuracy=(tp + tn) / max(len(y_true), 1),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold=float(threshold),
    )


def print_metrics(name: str, metrics: Metrics) -> None:
    print(
        f"{name}: AUC={metrics.auc:.4f} | PR-AUC={metrics.pr_auc:.4f} | "
        f"ACC={metrics.accuracy:.4f} | P={metrics.precision:.4f} | "
        f"R={metrics.recall:.4f} | F1={metrics.f1:.4f} | TH={metrics.threshold:.4f}"
    )


def print_compare_metrics(name: str, y_true: np.ndarray, mlp_prob: np.ndarray, rain_prob: np.ndarray,
                          rain_threshold: float) -> None:
    print(f"\n===== {name} =====")
    print_metrics("MLP Youden threshold", evaluate_binary(y_true, mlp_prob))
    print_metrics("MLP threshold 0.5", evaluate_binary(y_true, mlp_prob, threshold=0.5))
    print_metrics("RAIN validation threshold", evaluate_binary(y_true, rain_prob, threshold=rain_threshold))
    print_metrics("RAIN threshold 0.5", evaluate_binary(y_true, rain_prob, threshold=0.5))
    print_metrics("RAIN Youden threshold (diagnostic)", evaluate_binary(y_true, rain_prob))


def train_mlp_rain_model(
    data_path: str,
    model_dir: str,
    coord_cols: Iterable[str] = ("X", "Y", "Z"),
    label_col: str = "label",
    feature_cols: Iterable[str] | None = None,
    k_neighbors: int = 15,
    epochs: int = 300,
    lr: float = 1e-3,
    early_stop: int = 30,
    test_size: float = 0.2,
    val_size: float = 0.2,
    spatial_bandwidth: float = 25.0,
    lambda_reg: float = 1.0,
    logistic_aux_weight: float = 0.2,
    observed_label_smoothing: float = 0.02,
) -> tuple[TorchMLP, SpatialRAIN, TorchStandardScaler, Metrics]:
    set_seed(SEED)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    feature_cols = list(feature_cols or ["Pb", "Cu", "As", "Zn", "Anticline_Buffer", "Godenville_Formation_Buffer"])
    coord_cols = list(coord_cols)
    print(f"Feature columns: {feature_cols}")
    print(f"Coordinate columns: {coord_cols}")

    x_raw = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)
    coords = df[coord_cols].values.astype(np.float32)

    idx_train_val, idx_test_local = stratified_split_indices(y, test_size=test_size, seed=SEED)
    idx_train_local, idx_val_local = stratified_split_indices(y[idx_train_val], test_size=val_size, seed=SEED)
    idx_train = idx_train_val[idx_train_local]
    idx_val = idx_train_val[idx_val_local]
    idx_test = idx_test_local

    scaler = TorchStandardScaler.fit(x_raw[idx_train])
    x_train = scaler.transform(x_raw[idx_train])
    x_val = scaler.transform(x_raw[idx_val])
    x_test = scaler.transform(x_raw[idx_test])
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    coords_train, coords_val, coords_test = coords[idx_train], coords[idx_val], coords[idx_test]

    print(f"Split sizes: train={len(y_train)} | val={len(y_val)} | test={len(y_test)}")
    print(f"Positive samples: train={y_train.sum()} | val={y_val.sum()} | test={y_test.sum()}")

    train_neighbors, train_dists = build_neighbors(coords_train, coords_train, k_neighbors, exclude_self=True)
    val_neighbors, val_dists = build_neighbors(coords_train, coords_val, k_neighbors, exclude_self=False)
    test_neighbors, test_dists = build_neighbors(coords_train, coords_test, k_neighbors, exclude_self=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    y_train_float_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    train_neighbors_t = torch.tensor(train_neighbors, dtype=torch.long, device=device)
    val_neighbors_t = torch.tensor(val_neighbors, dtype=torch.long, device=device)
    test_neighbors_t = torch.tensor(test_neighbors, dtype=torch.long, device=device)
    train_dists_t = torch.tensor(train_dists, dtype=torch.float32, device=device)
    val_dists_t = torch.tensor(val_dists, dtype=torch.float32, device=device)
    test_dists_t = torch.tensor(test_dists, dtype=torch.float32, device=device)
    train_label_probs_t = label_probs_from_y(y_train, device=device, smoothing=observed_label_smoothing)

    mlp = TorchMLP(n_features=x_train.shape[1]).to(device)
    logistic_layer = nn.Linear(x_train.shape[1], 1).to(device)
    rain_model = SpatialRAIN(k_neighbors=k_neighbors).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(float(y_train.sum()), 1.0)], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        list(mlp.parameters()) + list(rain_model.parameters()) + list(logistic_layer.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_score = -np.inf
    best_epoch = -1
    best_state = None
    wait = 0

    print("\n===== End-to-end Torch MLP + Logistic-kernel RAIN =====")
    for epoch in range(epochs):
        mlp.train()
        rain_model.train()
        logistic_layer.train()
        optimizer.zero_grad()

        mlp_train_logits = mlp(x_train_t)
        train_weights_t = compute_logistic_pairwise_weights_torch(
            x_train_t, y_train_t, x_train_t, train_neighbors_t, train_dists_t,
            logistic_layer, spatial_bandwidth, lambda_reg
        )
        rain_train_logits = rain_model(
            mlp_train_logits, train_weights_t, train_neighbors_t, neighbor_source_probs=train_label_probs_t
        )
        logistic_logits = logistic_layer(x_train_t).reshape(-1)
        unary_loss = ce_loss(mlp_train_logits, y_train_t)
        rain_loss = ce_loss(rain_train_logits, y_train_t)
        loss = rain_loss + 0.5 * unary_loss + logistic_aux_weight * bce_loss(logistic_logits, y_train_float_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(mlp.parameters()) + list(rain_model.parameters()) + list(logistic_layer.parameters()),
            5.0,
        )
        optimizer.step()

        mlp.eval()
        rain_model.eval()
        logistic_layer.eval()
        with torch.no_grad():
            mlp_val_logits = mlp(x_val_t)
            val_weights_t = compute_logistic_pairwise_weights_torch(
                x_train_t, y_train_t, x_val_t, val_neighbors_t, val_dists_t,
                logistic_layer, spatial_bandwidth, lambda_reg
            )
            rain_val_logits = rain_model(
                mlp_val_logits, val_weights_t, val_neighbors_t, neighbor_source_probs=train_label_probs_t
            )
            val_prob = torch.softmax(rain_val_logits, dim=1)[:, 1].cpu().numpy()

        val_metrics = evaluate_binary(y_val, val_prob)
        score = val_metrics.auc + 0.25 * val_metrics.pr_auc
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                "mlp": copy.deepcopy(mlp.state_dict()),
                "rain": copy.deepcopy(rain_model.state_dict()),
                "logistic": copy.deepcopy(logistic_layer.state_dict()),
                "threshold": val_metrics.threshold,
            }
            wait = 0
        else:
            wait += 1
            if wait >= early_stop:
                print(f": epoch={epoch + 1}, best_epoch={best_epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Loss={loss.item():.4f} | "
                f"Unary={unary_loss.item():.4f} | RAIN={rain_loss.item():.4f} | "
                f"Val AUC={val_metrics.auc:.4f} | Val PR-AUC={val_metrics.pr_auc:.4f} | "
                f"beta={torch.exp(rain_model.log_beta).item():.4f}"
            )

    if best_state is None:
        raise RuntimeError("RAIN did not produce a valid checkpoint.")

    mlp.load_state_dict(best_state["mlp"])
    rain_model.load_state_dict(best_state["rain"])
    logistic_layer.load_state_dict(best_state["logistic"])

    mlp.eval()
    rain_model.eval()
    logistic_layer.eval()
    with torch.no_grad():
        mlp_test_logits = mlp(x_test_t)
        mlp_test_prob = torch.softmax(mlp_test_logits, dim=1).cpu().numpy()
        test_weights_t = compute_logistic_pairwise_weights_torch(
            x_train_t, y_train_t, x_test_t, test_neighbors_t, test_dists_t,
            logistic_layer, spatial_bandwidth, lambda_reg
        )
        rain_test_logits = rain_model(
            mlp_test_logits, test_weights_t, test_neighbors_t, neighbor_source_probs=train_label_probs_t
        )
        final_prob = torch.softmax(rain_test_logits, dim=1)[:, 1].cpu().numpy()

    final_metrics = evaluate_binary(y_test, final_prob, threshold=best_state["threshold"])
    print_compare_metrics("", y_test, mlp_test_prob[:, 1], final_prob, best_state["threshold"])

    bal_idx = balanced_indices(y_test, seed=SEED)
    print_compare_metrics(
        "Balanced test subset",
        y_test[bal_idx],
        mlp_test_prob[bal_idx, 1],
        final_prob[bal_idx],
        best_state["threshold"],
    )

    pd.DataFrame({
        "row_index": idx_test,
        "true_label": y_test,
        "mlp_prob": mlp_test_prob[:, 1],
        "rain_prob": final_prob,
    }).to_csv(os.path.join(model_dir, "test_predictions.csv"), index=False)

    torch.save({
        "mlp_model": mlp.state_dict(),
        "rain_model": rain_model.state_dict(),
        "logistic_layer": logistic_layer.state_dict(),
        "best_threshold": best_state["threshold"],
        "feature_cols": feature_cols,
        "coord_cols": coord_cols,
        "label_col": label_col,
        "k_neighbors": k_neighbors,
        "spatial_bandwidth": spatial_bandwidth,
        "lambda_reg": lambda_reg,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }, os.path.join(model_dir, "EndToEnd_MLP_LogisticRAIN_final.pt"))

    with open(os.path.join(model_dir, "torch_scaler.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler}, f)

    return mlp, rain_model, scaler, final_metrics


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = str(PROJECT_ROOT / "data" / "mineral_data_2023_02_07_10_46_22.csv")
    MODEL_DIR = str(PROJECT_ROOT / "models")

    FEATURE_COLS = ["Pb", "Cu", "As", "Zn", "Anticline_Buffer", "Godenville_Formation_Buffer"]
    COORD_COLS = ["wls", "hls"]
    LABEL_COL = "label"

    K_NEIGHBORS = 16
    EPOCHS = 300
    LR = 1e-3
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    EARLY_STOP = 30
    SPATIAL_BANDWIDTH = 20.0
    LAMBDA_REG = 1.0
    LOGISTIC_AUX_WEIGHT = 0.2
    OBSERVED_LABEL_SMOOTHING = 0.5

    mlp_model, rain_model, scaler, metrics = train_mlp_rain_model(
        data_path=DATA_PATH,
        model_dir=MODEL_DIR,
        coord_cols=COORD_COLS,
        label_col=LABEL_COL,
        feature_cols=FEATURE_COLS,
        k_neighbors=K_NEIGHBORS,
        epochs=EPOCHS,
        lr=LR,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        early_stop=EARLY_STOP,
        spatial_bandwidth=SPATIAL_BANDWIDTH,
        lambda_reg=LAMBDA_REG,
        logistic_aux_weight=LOGISTIC_AUX_WEIGHT,
        observed_label_smoothing=OBSERVED_LABEL_SMOOTHING,
    )
    print(f"Final test AUC: {metrics.auc:.4f}")

