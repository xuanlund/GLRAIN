# -*- coding: utf-8 -*-
"""Executable RAIN regression main experiment for PM2.5 spatial prediction."""

from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Continuous-response Recurrent Attentive Integration Network (RAIN).

This script implements the regression case described in Sections 2.2-2.4 and
3.1 of the manuscript. A neural global predictor captures domain-wide
nonlinear associations, while GWR-derived interaction coefficients provide
geographically adaptive local aggregation. Mean-field inference is unrolled
as recurrent attentive integration and optimized end to end.


Numerical reproducibility note:
    This file keeps the original seeds, graph construction, model dimensions,
    optimizer settings, mean-field iteration count, and inference equations
    unchanged. Some variable names still contain "rain" for compatibility with
    saved checkpoints and result files.

Potential paper-code alignment notes:
    - GWR projection rows are converted to non-negative interaction
      coefficients for the local spatial aggregation in the manuscript.
    - Compatibility identifiers retain historical names used by saved outputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import cKDTree
from scipy import linalg
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = None


def rmse_score(y_true, y_pred):
    """Version-independent RMSE for scikit-learn 1.3 through 1.6+."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# ==========================================================

class GTWRKernel:
    """

    """

    def __init__(self, function='bisquare'):
        """

        ----------
        """
        self.function = function.lower()
        self.valid_functions = ['bisquare', 'gaussian', 'exponential', 'triangular']
        if self.function not in self.valid_functions:
            raise ValueError(f"{self.valid_functions}{self.function}")

    def compute_kernel(self, dists, bandwidth):
        """


        ----------
        dists : np.ndarray, shape (n,)
        bandwidth : float

        -------
        kernel : np.ndarray, shape (n,)
        """
        if bandwidth <= 1e-8:
            bandwidth = 1e-8

        z = dists / bandwidth

        if self.function == 'bisquare':
            kernel = (1 - z ** 2) ** 2
            kernel[z >= 1.0] = 0.0

        elif self.function == 'gaussian':
            kernel = np.exp(-0.5 * z ** 2)

        elif self.function == 'exponential':
            kernel = np.exp(-z)

        elif self.function == 'triangular':
            kernel = 1 - z
            kernel[z >= 1.0] = 0.0

        else:
            kernel = np.ones_like(z)

        return kernel


def Compute_betas_gwr(y, x, wi):
    """

    """
    xT = (x * wi).T

    xtx = np.dot(xT, x)

    reg = 1e-4 * np.eye(xtx.shape[0])

    try:
        xtx_inv_xt = linalg.solve(xtx + reg, xT)
    except np.linalg.LinAlgError:
        xtx_inv_xt = linalg.pinv(xtx + reg) @ xT

    betas = np.dot(xtx_inv_xt, y)

    if np.any(np.isnan(betas)) or np.any(np.isinf(betas)):
        betas = np.zeros_like(betas)

    return betas


def select_gwr_bw(Y, X, coords, min_bw=5, max_bw=None, step=1, function='bisquare', manual_bw=None):
    """Select the GWR bandwidth."""
    if manual_bw is not None:
        print(f"[GWR] Using manual bandwidth: {manual_bw}")
        return manual_bw
    max_bw = max_bw if max_bw is not None else len(Y) // 2
    best_rmse = float('inf')
    best_bw = min_bw
    _X = np.hstack([X, np.ones((len(X), 1))])

    for bw in range(min_bw, max_bw + 1, step):
        preds = []
        for i in range(len(Y)):
            mask = np.ones(len(Y), dtype=bool)
            mask[i] = False
            Ytrain = Y[mask]
            Xtrain = _X[mask]
            train_coords = coords[mask]
            test_coord = coords[i:i + 1]
            Xtest = _X[i:i + 1]

            dists = cdist(test_coord, train_coords).reshape(-1)
            if len(dists) >= bw:
                bandwidth = np.partition(dists, bw - 1)[bw - 1] * 1.0000001
            else:
                bandwidth = np.max(dists) * 1.0000001 if len(dists) > 0 else 1.0
            kernel = GTWRKernel(function)
            wi = kernel.compute_kernel(dists, bandwidth).reshape(-1, 1)
            betas = Compute_betas_gwr(Ytrain, Xtrain, wi)
            preds.append(np.dot(Xtest[0], betas))
        rmse = np.sqrt(mean_squared_error(Y, preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_bw = bw
        if bw % 10 == 0:
            print(f"[GWR] bandwidth={bw}, cv_rmse={rmse:.4f}, best_bw={best_bw}")
    print(f"[GWR] Selected bandwidth: {best_bw} (CV-RMSE={best_rmse:.4f})")
    return best_bw


def compute_gwr_weights(train_coords, train_X, query_coord, query_feature, neighbors,
                        bandwidth=5.0, kernel_func='bisquare'):
    """

            A_i = x_i (X^T W_i X)^{-1} X^T W_i
    ----------
    train_coords : np.ndarray, shape (n_train, 3)
    train_X : np.ndarray, shape (n_train, n_features)
    query_coord : array-like, shape (3,)
    query_feature : array-like, shape (n_features,)
    neighbors : np.ndarray, shape (k,)

    -------
        alpha : np.ndarray, shape (k,)

    ----------
    """

    if len(neighbors) == 0:
        return np.array([1.0], dtype=np.float32)

    neigh_X = train_X[neighbors]  # shape: (k, n_features)

    dists = cdist([query_coord], train_coords[neighbors]).reshape(-1)  # length = k

    if len(dists) >= bandwidth:
        bw = np.partition(dists, int(bandwidth) - 1)[int(bandwidth) - 1] * 1.0000001
    else:
        bw = np.max(dists) * 1.000001 if len(dists) > 0 else 1.0

    kernel = GTWRKernel(kernel_func)  # 
    w = kernel.compute_kernel(dists, bw)  # w shape: (k,)

    W = np.diag(w)  # shape: (k, k)

    n = len(neighbors)  # k
    X_aug = np.hstack([np.ones((n, 1)), neigh_X])  # shape: (k, n_features+1)

    reg = 1e-4 * np.eye(X_aug.shape[1])  # hape: (d+1, d+1)

    try:
        XT_W_X = X_aug.T @ W @ X_aug + reg  # shape: (d+1, d+1)
        XT_W = X_aug.T @ W  # shape: (d+1, k)
        inv = np.linalg.inv(XT_W_X)  # shape: (d+1, d+1)
        x_i = np.hstack([1, query_feature]).reshape(1, -1)  # shape: (1, d+1)
        A_row = x_i @ inv @ XT_W
    except np.linalg.LinAlgError:
        A_row = np.ones(n) / n

    A_row = np.abs(A_row.reshape(-1)) + 1e-8  # shape: (k,)

    A_row_normalized = A_row / (A_row.sum() + 1e-8)

    return A_row_normalized.astype(np.float32)


def safe_numpy_convert(arr):
    """
    """
    if len(arr) == 0:
        return np.array([0.0], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


# ==========================================================
# ==========================================================
def precompute_by_distance_threshold(
        data_path,
        save_pkl_path,
        distance_threshold=50.0,
        feature_cols_slice=slice(6, 13),
        coord_cols_slice=slice(2, 4),
        target_col='pm25',
        test_size=0.2,
        random_state=42,
        gwr_bw=49,
        gwr_kernel='bisquare'
):
    """



    ----------
    data_path : str
    save_pkl_path : str

    -------
    precomputed_data : dict
    """

    data = pd.read_csv(data_path, header=0, encoding="gbk")

    coords = data.iloc[:, coord_cols_slice].values
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode='constant', constant_values=0.0)
    elif coords.shape[1] > 3:
        coords = coords[:, :3]

    X = data.iloc[:, feature_cols_slice].values  # shape: (N, n_features)
    y = data[target_col].values.ravel()  # shape: (N,)

    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords, test_size=test_size, random_state=random_state
    )

    scaler_X = StandardScaler()
    scaler_X.fit(X_train)  # 
    scaler_y = StandardScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    print(f"\nGWR...")
    best_gwr_k = select_gwr_bw(
        Y=y_train,
        X=X_train,
        coords=coords_train,
        min_bw=5,
        max_bw=len(y_train) // 2,
        step=2,
        function=gwr_kernel,
        manual_bw=gwr_bw
    )
    print(f"GWR: {best_gwr_k}")

    train_coords_np = np.asarray(coords_train, dtype=np.float32)
    test_coords_np = np.asarray(coords_test, dtype=np.float32)
    tree_train = cKDTree(train_coords_np)  # 
    tree_test = cKDTree(test_coords_np)  # 

    def get_all_neighbors_by_distance(tree, query_coords, threshold, exclude_self=True):
        """

        ----------
        tree : cKDTree
        query_coords : np.ndarray, shape (N, 3)
        threshold : float

        -------
        all_neighbors : list of np.ndarray
        all_distances : list of np.ndarray
        """
        all_neighbors = []
        all_distances = []
        for idx, coord in enumerate(query_coords):
            dists, inds = tree.query(coord, k=len(tree.data), distance_upper_bound=threshold)
            valid_mask = np.isfinite(dists)
            valid_inds = inds[valid_mask]
            valid_dists = dists[valid_mask]
            if exclude_self and np.array_equal(tree.data, query_coords):
                not_self_mask = (valid_inds != idx)
                valid_inds = valid_inds[not_self_mask]
                valid_dists = valid_dists[not_self_mask]
            sort_idx = np.argsort(valid_dists)
            all_neighbors.append(valid_inds[sort_idx])
            all_distances.append(valid_dists[sort_idx])
        return all_neighbors, all_distances

    print(f"Selecting training neighbours with distance_threshold={distance_threshold}...")
    train_neighbors_all, train_distances_all = get_all_neighbors_by_distance(
        tree_train, train_coords_np, distance_threshold, exclude_self=True
    )
    train_neigh_counts = [len(neigh) for neigh in train_neighbors_all]
    print(
        f"   Training neighbour counts: min={min(train_neigh_counts)}, "
        f"max={max(train_neigh_counts)}, mean={np.mean(train_neigh_counts):.1f}"
    )

    print(f"Selecting test-self neighbours with distance_threshold={distance_threshold}...")
    test_self_neighbors_all, test_self_distances_all = get_all_neighbors_by_distance(
        tree_test, test_coords_np, distance_threshold, exclude_self=True
    )
    test_self_neigh_counts = [len(neigh) for neigh in test_self_neighbors_all]
    print(
        f"   Test-self neighbour counts: min={min(test_self_neigh_counts)}, "
        f"max={max(test_self_neigh_counts)}, mean={np.mean(test_self_neigh_counts):.1f}"
    )

    def compute_all_gwr_weights(train_coords, train_X, query_coords, query_features, all_neighbors, gwr_k, kernel):
        """

        ----------
        train_coords : np.ndarray
        train_X : np.ndarray
        query_coords : np.ndarray
        query_features : np.ndarray
        all_neighbors : list of np.ndarray
        gwr_k : int
        kernel : str

        -------
        all_inds : list of np.ndarray
        all_alphas : list of np.ndarray
        """
        all_inds = []
        all_alphas = []
        for i, (coord, feat, neigh_inds) in enumerate(zip(query_coords, query_features, all_neighbors)):
            if len(neigh_inds) == 0:
                all_inds.append(np.array([-1], dtype=np.int64))
                all_alphas.append(np.array([0.0], dtype=np.float32))
                continue
            use_neigh_inds = neigh_inds[:gwr_k] if len(neigh_inds) > gwr_k else neigh_inds
            gwr_weights = compute_gwr_weights(
                train_coords=train_coords,
                train_X=train_X,
                query_coord=coord,
                query_feature=feat,
                neighbors=use_neigh_inds,
                bandwidth=gwr_k,
                kernel_func=kernel
            )
            all_inds.append(use_neigh_inds)
            all_alphas.append(gwr_weights)
        return all_inds, all_alphas

    print(" WR...")
    train_gwr_inds_all, train_alphas_all = compute_all_gwr_weights(
        train_coords=train_coords_np,
        train_X=X_train,
        query_coords=train_coords_np,
        query_features=X_train,
        all_neighbors=train_neighbors_all,
        gwr_k=best_gwr_k,
        kernel=gwr_kernel
    )

    print(" WR...")
    test_self_gwr_inds_all, test_self_alphas_all = compute_all_gwr_weights(
        train_coords=test_coords_np,
        train_X=X_test,
        query_coords=test_coords_np,
        query_features=X_test,
        all_neighbors=test_self_neighbors_all,
        gwr_k=best_gwr_k,
        kernel=gwr_kernel
    )

    precomputed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'coords_train': coords_train,
        'coords_test': coords_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'train_neighbors_all': [safe_numpy_convert(neigh).astype(np.int64) for neigh in train_neighbors_all],
        'train_distances_all': [safe_numpy_convert(dist).astype(np.float32) for dist in train_distances_all],
        'train_alphas_all': [safe_numpy_convert(alpha).astype(np.float32) for alpha in train_alphas_all],
        'train_gwr_inds_all': [safe_numpy_convert(inds).astype(np.int64) for inds in train_gwr_inds_all],
        'test_self_neighbors_all': [safe_numpy_convert(neigh).astype(np.int64) for neigh in test_self_neighbors_all],
        'test_self_distances_all': [safe_numpy_convert(dist).astype(np.float32) for dist in test_self_distances_all],
        'test_self_alphas_all': [safe_numpy_convert(alpha).astype(np.float32) for alpha in test_self_alphas_all],
        'test_self_gwr_inds_all': [safe_numpy_convert(inds).astype(np.int64) for inds in test_self_gwr_inds_all],
        'distance_threshold': distance_threshold,
        'test_size': test_size,
        'random_state': random_state,
        'gwr_bandwidth': best_gwr_k,
        'gwr_kernel': gwr_kernel,
        'best_gwr_k': best_gwr_k,
        'max_train_neighbors': max(train_neigh_counts),
        'max_test_self_neighbors': max(test_self_neigh_counts)
    }

    os.makedirs(os.path.dirname(save_pkl_path), exist_ok=True)
    with open(save_pkl_path, 'wb') as f:
        pickle.dump(precomputed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Precomputed distance-threshold RAIN graph saved to: {save_pkl_path}")
    print(f"   distance_threshold={distance_threshold}, best_gwr_k={best_gwr_k}")
    return precomputed_data


def load_and_select_neighbors(pkl_path, select_k):
    """



    ----------
    pkl_path : str
    select_k : int

    -------
    selected_data : dict

    ----------
    """

    with open(pkl_path, 'rb') as f:
        precomputed_data = pickle.load(f)

    max_train_k = precomputed_data['max_train_neighbors']
    max_test_k = precomputed_data['max_test_self_neighbors']

    if select_k > max_train_k or select_k > max_test_k:
        raise ValueError(
            f"select_k={select_k} exceeds available neighbours "
            f"(train={max_train_k}, test={max_test_k})."
        )

    def pad_or_truncate(arr_list, k, dtype):
        """

        ----------
        arr_list : list of np.ndarray
        k : int
        dtype : np.dtype

        -------
        padded : np.ndarray, shape (len(arr_list), k)
        """
        padded = []
        for arr in arr_list:
            arr = arr[:k]
            pad_width = k - len(arr)  # 
            if pad_width > 0:
                if dtype == np.int64:
                    padded_arr = np.pad(arr, (0, pad_width), mode='constant', constant_values=-1)
                else:
                    padded_arr = np.pad(arr, (0, pad_width), mode='constant', constant_values=0.0)
            else:
                padded_arr = arr
            padded.append(padded_arr.astype(dtype))
        return np.array(padded)

    selected_data = {
        'X_train': precomputed_data['X_train'],
        'X_test': precomputed_data['X_test'],
        'y_train': precomputed_data['y_train'],
        'y_test': precomputed_data['y_test'],
        'coords_train': precomputed_data['coords_train'],
        'coords_test': precomputed_data['coords_test'],
        'scaler_X': precomputed_data['scaler_X'],
        'scaler_y': precomputed_data['scaler_y'],
        'select_k': select_k,
        'distance_threshold': precomputed_data['distance_threshold'],
        'gwr_bandwidth': precomputed_data['gwr_bandwidth'],
        'best_gwr_k': precomputed_data['best_gwr_k'],

        'train_neighbors': pad_or_truncate(precomputed_data['train_neighbors_all'], select_k, np.int64),
        'train_alphas': pad_or_truncate(precomputed_data['train_alphas_all'], select_k, np.float32),
        'train_gwr_inds': pad_or_truncate(precomputed_data['train_gwr_inds_all'], select_k, np.int64),
        'train_distances': pad_or_truncate(precomputed_data['train_distances_all'], select_k, np.float32),

        'test_self_neighbors': pad_or_truncate(precomputed_data['test_self_neighbors_all'], select_k, np.int64),
        'test_self_alphas': pad_or_truncate(precomputed_data['test_self_alphas_all'], select_k, np.float32),
        'test_self_gwr_inds': pad_or_truncate(precomputed_data['test_self_gwr_inds_all'], select_k, np.int64),
        'test_self_distances': pad_or_truncate(precomputed_data['test_self_distances_all'], select_k, np.float32),
    }

    print(
        f"Loaded precomputed RAIN graph with select_k={select_k}, "
        f"distance_threshold={precomputed_data['distance_threshold']}, "
        f"best_gwr_k={precomputed_data['best_gwr_k']}."
    )
    return selected_data


def add_spatial_unary_features(selected_data, coord_scale=1.0):
    """
    Add standardized spatial coordinates to the unary fidelity module input.

    The GWR-derived pairwise coefficients are unchanged. Coordinates are scaled
    using training-set statistics only, then appended to X_train/X_test before
    fitting a new feature scaler for the neural unary module.
    """
    data = dict(selected_data)
    coord_scaler = StandardScaler().fit(data['coords_train'])
    train_coords_s = coord_scaler.transform(data['coords_train']) * coord_scale
    test_coords_s = coord_scaler.transform(data['coords_test']) * coord_scale

    data['X_train'] = np.hstack([data['X_train'], train_coords_s])
    data['X_test'] = np.hstack([data['X_test'], test_coords_s])
    data['scaler_X'] = StandardScaler().fit(data['X_train'])
    data['spatial_unary'] = True
    data['spatial_unary_coord_scale'] = coord_scale
    return data


# ==========================================================
# ==========================================================
class RAINJointModel(nn.Module):
    """
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 64),
        k_neighbors=16,
        initial_log_eta=1.0,
        initial_log_lambda=1.69314718056,
        renormalize_gwr=True,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.renormalize_gwr = bool(renormalize_gwr)
        self.log_eta = nn.Parameter(torch.tensor(float(initial_log_eta)))
        self.log_lambda = nn.Parameter(torch.tensor(float(initial_log_lambda)))

        self.mlp_feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.mlp_pred = nn.Linear(hidden_dims[1], 1)
        self.linear_skip = nn.Linear(input_dim, 1)
        self.local_coef_net = nn.Sequential(
            nn.Linear(2, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim + 1),
        )
        self.local_coef_scale = nn.Parameter(torch.tensor(0.1))

        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(1.0))
        self.lambda3 = nn.Parameter(torch.tensor(1.0))
        self.sigma_f = nn.Parameter(torch.tensor(0.5))
        self.sigma_h = nn.Parameter(torch.tensor(0.5))

        self.mean_field_max_iters = 20
        self.mean_field_tol = 1e-6

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def compute_feature_kernel(self, query_features, train_features, neighbors):
        """
        """
        batch_size = query_features.shape[0]
        safe_neighbors = torch.clamp(neighbors, min=0)
        feat_neighbors = train_features[safe_neighbors.reshape(-1)].reshape(batch_size, self.k_neighbors, -1)
        feat_expanded = query_features.unsqueeze(1).repeat(1, self.k_neighbors, 1)
        diff = feat_expanded - feat_neighbors
        dist_sq = torch.sum(diff ** 2, dim=-1)  # shape: (batch, k)
        sigma_sq = torch.clamp(self.sigma_f ** 2, min=1e-6)
        phi = torch.exp(-dist_sq / (2 * sigma_sq))
        valid_mask = (neighbors != -1).float()
        phi = phi * valid_mask
        return phi

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def compute_spatial_kernel(self, query_coords, train_coords, neighbors):
        """
        """
        batch_size = query_coords.shape[0]
        safe_neighbors = torch.clamp(neighbors, min=0)
        coords_neighbors = train_coords[safe_neighbors.reshape(-1)].reshape(batch_size, self.k_neighbors, -1)
        coords_expanded = query_coords.unsqueeze(1).repeat(1, self.k_neighbors, 1)
        diff = coords_expanded - coords_neighbors
        dist_sq = torch.sum(diff ** 2, dim=-1)
        sigma_sq = torch.clamp(self.sigma_h ** 2, min=1e-6)
        h = torch.exp(-dist_sq / (2 * sigma_sq))
        valid_mask = (neighbors != -1).float()
        h = h * valid_mask
        return h

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def compute_gwr_kernel(self, gwr_alphas, neighbors):
        """
        ----------
        gwr_alphas : torch.Tensor, shape (batch_size, k_neighbors)
        neighbors : torch.Tensor, shape (batch_size, k_neighbors)

        -------
        k : torch.Tensor, shape (batch_size, k_neighbors)
        """
        valid_mask = (neighbors != -1).float()
        k = torch.clamp(gwr_alphas, min=0.0, max=1.0) * valid_mask
        if self.renormalize_gwr:
            row_sum = torch.sum(k, dim=-1, keepdim=True)
            k = k / torch.clamp(row_sum, min=1e-8)
        return k

    def predict_unary(self, x):
        """End-to-end local evidence f_theta(x)."""
        features = self.mlp_feature(x)
        base = self.linear_skip(x) + self.mlp_pred(features)
        spatial_context = x[:, -2:]
        local_coef = self.local_coef_net(spatial_context)
        local_linear = torch.sum(local_coef[:, :-1] * x, dim=-1, keepdim=True) + local_coef[:, -1:]
        unary = base + self.local_coef_scale * local_linear
        return unary.squeeze(-1), features


    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def mean_field_iteration(self, unary_pred, alpha_ij, neighbors, eta=1.0, lambda_pair=1.0):
        """
        Continuous-response RAIN mean-field fixed-point update.

        Implemented energy for node i:
            U_i(y_i) = eta * (y_i - f_theta(x_i))^2
                     + lambda * sum_j alpha_ij^GWR E_qj[(y_i - y_j)^2]

        Since q_j is Gaussian, the variance term is independent of y_i and is
        absorbed into the normalization constant. The variational mean is
        iteratively updated as:
            mu_i^{t+1} =
                [eta f_theta(x_i) + lambda sum_j alpha_ij^GWR mu_j^t]
                / [eta + lambda sum_j alpha_ij^GWR]

        The corresponding variance is:
            sigma_i^2 = 1 / [eta + lambda sum_j alpha_ij^GWR]



        """
        mu = unary_pred.clone()

        for _ in range(self.mean_field_max_iters):
            mu_old = mu.clone()

            safe_neighbors = torch.clamp(neighbors, min=0)  # shape: (batch, k)

            neighbor_mu = mu[safe_neighbors.reshape(-1)].reshape(-1, self.k_neighbors)

            pairwise_term = torch.sum(alpha_ij * neighbor_mu, dim=-1)  # shape: (batch,)

            sum_w = torch.sum(alpha_ij, dim=-1)  # shape: (batch,)

            numerator = eta * unary_pred + lambda_pair * pairwise_term

            denominator = eta + lambda_pair * sum_w

            denominator = torch.clamp(denominator, min=1e-6)

            mu = numerator / denominator

            if torch.max(torch.abs(mu - mu_old)) < self.mean_field_tol:
                break

        return mu

    def mean_field_observed_context(self, unary_pred, source_y, alpha_ij, neighbors, eta=1.0, lambda_pair=1.0):
        """
        Continuous-response RAIN update with observed source nodes.

        This uses the same paper formula as the recurrent mean-field update,
        but the neighbour field is clamped to observed PM2.5 values from the
        train or train+validation source graph:
            mu_i = [eta f_theta(x_i) + lambda sum_j alpha_ij^GWR y_j]
                   / [eta + lambda sum_j alpha_ij^GWR]

        It is used for the author split's inductive test prediction, where
        test labels are never part of source_y.
        """
        safe_neighbors = torch.clamp(neighbors, min=0)
        valid_mask = (neighbors != -1).float()
        source_values = source_y[safe_neighbors.reshape(-1)].reshape(-1, self.k_neighbors)
        weights = alpha_ij * valid_mask
        pairwise_term = torch.sum(weights * source_values, dim=-1)
        sum_w = torch.sum(weights, dim=-1)
        numerator = eta * unary_pred + lambda_pair * pairwise_term
        denominator = torch.clamp(eta + lambda_pair * sum_w, min=1e-6)
        return numerator / denominator

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def forward(self, x, coords, train_features, train_coords, neighbors, gwr_alphas):
        """

        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
        coords : torch.Tensor, shape (batch_size, 3)
        train_features : torch.Tensor, shape (N_train, d_hidden)
        train_coords : torch.Tensor, shape (N_train, 3)
        neighbors : torch.Tensor, shape (batch_size, k_neighbors)
        gwr_alphas : torch.Tensor, shape (batch_size, k_neighbors)

        -------
        unary_pred : torch.Tensor, shape (batch_size,)
        rain_pred : torch.Tensor, shape (batch_size,)
        query_features : torch.Tensor, shape (batch_size, d_hidden)
        """
        unary_pred, query_features = self.predict_unary(x)  # shape: (batch,)

        phi = self.compute_feature_kernel(query_features, train_features, neighbors)
        h = self.compute_spatial_kernel(coords, train_coords, neighbors)
        k = self.compute_gwr_kernel(gwr_alphas, neighbors)

        alpha_ij_gwr = k

        eta_pos = torch.exp(self.log_eta)
        lambda_pos = torch.exp(self.log_lambda)

        rain_pred = self.mean_field_iteration(unary_pred, alpha_ij_gwr, neighbors, eta_pos, lambda_pos)

        return unary_pred, rain_pred, query_features


# ==========================================================
# ==========================================================
def run_joint_training_with_dynamic_k(
        selected_data,
        model_save_path,
        hidden_dims=(128, 64),
        epochs=200,
        lr=1e-3,
        pretrain_epochs=0,
        unary_loss_weight=0.0,
        initial_log_eta=1.0,
        initial_log_lambda=1.69314718056,
        renormalize_gwr=True,
):
    """


    ----------
    selected_data : dict
            'X_train', 'X_test', 'y_train', 'y_test',
            'coords_train', 'coords_test',
            'scaler_X', 'scaler_y',
            'select_k', 'best_gwr_k',
            'train_neighbors', 'train_alphas',
            'test_self_neighbors', 'test_self_alphas'
    model_save_path : str

    -------
    results : dict
    """

    X_train = selected_data['X_train']
    X_test = selected_data['X_test']
    y_train = selected_data['y_train']
    y_test = selected_data['y_test']
    coords_train = selected_data['coords_train']
    coords_test = selected_data['coords_test']
    scaler_X = selected_data['scaler_X']  # 
    scaler_y = selected_data['scaler_y']  # 
    select_k = selected_data['select_k']
    best_gwr_k = selected_data['best_gwr_k']  # GWR 

    X_train_s = scaler_X.transform(X_train)  # shape: (n_train, n_features)
    X_test_s = scaler_X.transform(X_test)  # shape: (n_test, n_features)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    coords_train_t = torch.tensor(coords_train, dtype=torch.float32, device=device)
    coords_test_t = torch.tensor(coords_test, dtype=torch.float32, device=device)

    train_neighbors_t = torch.tensor(selected_data['train_neighbors'], dtype=torch.long, device=device)
    train_alphas_t = torch.tensor(selected_data['train_alphas'], dtype=torch.float32, device=device)
    test_train_neighbors_t = torch.tensor(selected_data['test_train_neighbors'], dtype=torch.long, device=device)
    test_train_alphas_t = torch.tensor(selected_data['test_train_alphas'], dtype=torch.float32, device=device)

    model = RAINJointModel(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        k_neighbors=select_k,
        initial_log_eta=initial_log_eta,
        initial_log_lambda=initial_log_lambda,
        renormalize_gwr=renormalize_gwr,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_losses = []
    pretrain_losses = []
    if pretrain_epochs > 0:
        print(
            f"\nStart DNN local-evidence pretraining: "
            f"select_k={select_k}, epochs={pretrain_epochs}..."
        )
        for epoch in range(pretrain_epochs):
            model.train()
            optimizer.zero_grad()
            unary_pred, _ = model.predict_unary(X_train_t)
            loss = loss_fn(unary_pred, y_train_t)
            loss.backward()
            optimizer.step()
            pretrain_losses.append(loss.item())

            if (epoch + 1) % 50 == 0:
                print(f"Pretrain epoch {epoch + 1}/{pretrain_epochs}, Unary Loss: {loss.item():.6f}")

    print(
        f"\nStart end-to-end RAIN-GWR training: select_k={select_k}, "
        f"best_gwr_k={best_gwr_k}, epochs={epochs}, unary_loss_weight={unary_loss_weight}..."
    )
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        unary_pred, _ = model.predict_unary(X_train_t)
        train_alpha = model.compute_gwr_kernel(train_alphas_t, train_neighbors_t)
        eta_pos = torch.exp(model.log_eta)
        lambda_pos = torch.exp(model.log_lambda)
        rain_pred = model.mean_field_observed_context(
            unary_pred, y_train_t, train_alpha, train_neighbors_t, eta_pos, lambda_pos
        )

        rain_loss = loss_fn(rain_pred, y_train_t)
        unary_loss = loss_fn(unary_pred, y_train_t)
        loss = rain_loss + float(unary_loss_weight) * unary_loss
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, "
                f"RAIN Loss: {rain_loss.item():.6f}, Unary Loss: {unary_loss.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        unary_pred_test, test_features_t = model.predict_unary(X_test_t)
        test_train_alpha = model.compute_gwr_kernel(test_train_alphas_t, test_train_neighbors_t)
        eta_pos = torch.exp(model.log_eta)
        lambda_pos = torch.exp(model.log_lambda)
        rain_pred_test = model.mean_field_observed_context(
            unary_pred_test, y_train_t, test_train_alpha, test_train_neighbors_t, eta_pos, lambda_pos
        )
        rain_pred_test_np = scaler_y.inverse_transform(rain_pred_test.cpu().numpy().reshape(-1, 1)).ravel()

        mlp_pred_test = scaler_y.inverse_transform(unary_pred_test.cpu().numpy().reshape(-1, 1)).ravel()

    results = {
        'mlp_rmse': rmse_score(y_test, mlp_pred_test),
        'mlp_r2': r2_score(y_test, mlp_pred_test),
        'rain_joint_rmse': rmse_score(y_test, rain_pred_test_np),
        'rain_joint_r2': r2_score(y_test, rain_pred_test_np),
        'mlp_pred_test': mlp_pred_test,
        'rain_joint_pred_test': rain_pred_test_np,
        'y_test': y_test,
        'train_losses': train_losses,
        'pretrain_losses': pretrain_losses,
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'select_k': select_k,
        'best_gwr_k': best_gwr_k,
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    save_dict = {
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_dim': X_train.shape[1],
        'hidden_dims': hidden_dims,
        'k_neighbors': select_k,
        'best_gwr_k': best_gwr_k,
        'final_r2': results['rain_joint_r2'],
        'train_losses': train_losses,
        'pretrain_losses': pretrain_losses,
        'pretrain_epochs': pretrain_epochs,
        'unary_loss_weight': unary_loss_weight,
        'lr': lr,
        'initial_log_eta': initial_log_eta,
        'initial_log_lambda': initial_log_lambda,
        'renormalize_gwr': renormalize_gwr,
    }
    torch.save(save_dict, model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    print("\n" + "=" * 60)
    print(
        f"Continuous-response RAIN-GWR - select_k={select_k}, "
        f"best_gwr_k={best_gwr_k}, renormalize_gwr={renormalize_gwr}"
    )
    print("=" * 60)
    print(f"Local Evidence Module -> RMSE={results['mlp_rmse']:.4f}, R2={results['mlp_r2']:.4f}")
    print(f"RAIN-GWR MF-RNN       -> RMSE={results['rain_joint_rmse']:.4f}, R2={results['rain_joint_r2']:.4f}")
    print("=" * 60)

    return results


def load_trained_model(model_path):
    """

    """

    save_dict = torch.load(model_path, map_location=torch.device('cpu'))

    k_neighbors = save_dict.get('k_neighbors', 4)

    model = RAINJointModel(
        input_dim=save_dict['input_dim'],
        hidden_dims=save_dict['hidden_dims'],
        k_neighbors=k_neighbors
    )

    model.load_state_dict(save_dict['model_state_dict'])

    model.eval()

    return {
        'model': model,
        'scaler_X': save_dict['scaler_X'],  # 
        'scaler_y': save_dict['scaler_y'],  # 
        'final_r2': save_dict['final_r2'],
        'hidden_dims': save_dict['hidden_dims'],
        'k_neighbors': k_neighbors,
        'best_gwr_k': save_dict.get('best_gwr_k', k_neighbors)
    }


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = None


BASE_DIR = Path(__file__).resolve().parent
AUTHOR_SPLIT_DIR = BASE_DIR / "data" / "regression_author_split"
RESULTS_DIR = BASE_DIR / "outputs" / "regression_results"
MODEL_DIR = BASE_DIR / "outputs" / "regression_models"
AUTHOR_GRAPH_PKL = BASE_DIR / "outputs" / "author_split_gwr_graph.pkl"
DNN_RESULTS_DIR = BASE_DIR / "outputs" / "regression_results_dnn_e2e"
DNN_MODEL_DIR = BASE_DIR / "outputs" / "regression_models_dnn_e2e"

BASE_FEATURE_COLS = ["dem", "aod", "t2m", "tp", "dwi", "r", "wind"]
RAIN_FEATURE_COLS = [
    "dem", "aod", "t2m", "tp", "dwi", "r", "wind",
    "dwi_scaled", "aod_scaled", "t2m_scaled", "r_scaled", "tp_scaled", "wind_scaled",
    "dem_scaled",
]
TARGET_COL = "pm25"
BASELINE_COORD_COLS = ["proj_x", "proj_y"]
GRAPH_COORD_COLS = ["proj_x", "proj_y"]
SELECT_K = 8
GWR_K = 40
GRAPH_DISTANCE_THRESHOLD = None
SPATIAL_FEATURE_SCALE = 1.0
INDUCTIVE_RAIN_K = 3
INDUCTIVE_RAIN_RATIO = 0.25
NON_E2E_RAIN_RATIO = 0.50
FINAL_MFRNN_K = 2
FINAL_MFRNN_LAMBDA = 0.553
FINAL_MFRNN_ITERS = 10
FINAL_MFRNN_BANDWIDTH = 2_000_000.0
FINAL_GWR_ALPHA_K = 3
FINAL_GWR_ALPHA_LAMBDA = 0.61
PAPER_RAIN_UNARY_BW = 45
PAPER_RAIN_UNARY_KERNEL = "bisquare"
PAPER_RAIN_SELECT_K = 3
PAPER_RAIN_GWR_K = 45
PAPER_RAIN_ETA = 0.60
PAPER_RAIN_LAMBDA = 1.88
PAPER_RAIN_RENORMALIZE_ALPHA = False
PAPER_RAIN_E2E_EPOCHS = 50
PAPER_RAIN_E2E_LR = 1e-4
PAPER_RAIN_E2E_PRIOR_WEIGHT = 1.0
DNN_LOCAL_EVIDENCE_SEEDS = [11, 42, 77, 123, 2026]
DNN_LOCAL_EVIDENCE_EPOCHS = 670
DNN_LOCAL_EVIDENCE_LR = 5e-4
DNN_LOCAL_EVIDENCE_WEIGHT_DECAY = 5e-4
DNN_LOCAL_EVIDENCE_DROPOUT = 0.04
DNN_RAIN_ETA = 0.60
DNN_RAIN_LAMBDA = 2.50
DNN_E2E_JOINT_EPOCHS = 80
DNN_E2E_JOINT_LR = 1e-5
DNN_E2E_UNARY_AUX_WEIGHT = 0.20
DNN_E2E_PRIOR_WEIGHT = 0.50
GWR_BASELINE_BW = 50
GWR_BASELINE_KERNEL = "triangular"
RAIN_UNARY_GWR_BW = 45
RAIN_UNARY_GWR_KERNEL = "triangular"
VALIDATION_TUNING_GRID = {
    "select_k": [8, 16],
    "gwr_k": [40, 80],
    "spatial_feature_scale": [1.0],
    "hidden_dims": [(256, 128)],
    "lr": [8e-4],
    "pretrain_epochs": [250],
    "epochs": [500],
    "unary_loss_weight": [2.00],
    "initial_log_eta": [1.0],
    "initial_log_lambda": [1.69314718056, 2.30258509299],
    "renormalize_gwr": [True, False],
}
SEED = 42


def load_rain_module():
    return None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = len(BASE_FEATURE_COLS)) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    n = len(y_true)
    rss = float(np.sum(residual ** 2))
    tss = float(np.sum((y_true - y_true.mean()) ** 2))
    mse = rss / max(n, 1)
    r2 = 1.0 - rss / max(tss, 1e-12)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mape": float(np.mean(np.abs(residual / np.clip(np.abs(y_true), 1e-8, None)))),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "adj_r2": float(1.0 - (1.0 - r2) * (n - 1) / max(n - n_features - 1, 1)),
        "aic_no_const": float(n * np.log(max(rss / max(n, 1), 1e-12)) + 2 * (n_features + 1)),
        "aic_gaussian": float(n * (np.log(2.0 * np.pi) + 1.0 + np.log(max(rss / max(n, 1), 1e-12))) + 2 * (n_features + 1)),
    }


def read_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read the author's benchmark split.

    This open-source package stores the strict split source files directly
    under regression/data/author_split_source so the project can run without any
    auxiliary reference repository.
    """
    base = pd.read_csv(AUTHOR_SPLIT_DIR / "train_base.csv")
    val_index = pd.read_csv(AUTHOR_SPLIT_DIR / "valIndex.csv", header=None).iloc[:, 0].astype(int).to_numpy()
    val = base.iloc[val_index].reset_index(drop=True)
    train = base.drop(index=val_index).reset_index(drop=True)
    test = pd.read_csv(AUTHOR_SPLIT_DIR / "test_base.csv")
    return train, val, test


def coords3(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    coords = df[cols].to_numpy(dtype=np.float32)
    return np.pad(coords, ((0, 0), (0, 1)), mode="constant")


def fit_predict_rf(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    sx = StandardScaler()
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=9,
        max_features="sqrt",
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=666,
        n_jobs=-1,
    )
    model.fit(sx.fit_transform(x_train), y_train)
    return model.predict(sx.transform(x_test))


def fit_predict_svm(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    sx = StandardScaler()
    sy = StandardScaler()
    x_train_s = sx.fit_transform(x_train)
    x_test_s = sx.transform(x_test)
    y_train_s = sy.fit_transform(y_train.reshape(-1, 1)).ravel()
    model = SVR(kernel="rbf", C=2, gamma="scale", epsilon=0.3)
    model.fit(x_train_s, y_train_s)
    return sy.inverse_transform(model.predict(x_test_s).reshape(-1, 1)).ravel()


def fit_predict_xgboost(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    sx = StandardScaler()
    if XGBRegressor is None:
        print("xgboost is not installed; using sklearn GradientBoostingRegressor for the XGBoost benchmark export.")
        model = GradientBoostingRegressor(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.75,
            random_state=SEED,
        )
    else:
        model = XGBRegressor(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.80,
            reg_lambda=4.0,
            reg_alpha=0.1,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
        )
    model.fit(sx.fit_transform(x_train), y_train)
    return model.predict(sx.transform(x_test))


def gwr_predict_scaled(y_train_s, x_train_s, coords_train, x_test_s, coords_test,
                       bw=80, kernel="bisquare", ridge=1e-5) -> np.ndarray:
    x_train_aug = np.hstack([x_train_s, np.ones((len(x_train_s), 1))])
    x_test_aug = np.hstack([x_test_s, np.ones((len(x_test_s), 1))])
    dmat = cdist(coords_test, coords_train)
    preds = np.zeros(len(x_test_s), dtype=np.float64)
    for i, d in enumerate(dmat):
        k = min(bw, len(d))
        kth = np.partition(d, k - 1)[k - 1] * 1.0000001
        kth = max(kth, 1e-8)
        z = d / kth
        if kernel == "triangular":
            wi = 1.0 - z
            wi[z >= 1.0] = 0.0
        elif kernel == "gaussian":
            wi = np.exp(-0.5 * z ** 2)
        else:
            wi = (1.0 - z ** 2) ** 2
            wi[z >= 1.0] = 0.0
        xtw = x_train_aug.T * wi
        xtwx = xtw @ x_train_aug + ridge * np.eye(x_train_aug.shape[1])
        xtwy = xtw @ y_train_s
        beta = linalg.solve(xtwx, xtwy, assume_a="pos")
        preds[i] = x_test_aug[i] @ beta
    return preds


def fit_predict_gwr(x_train, y_train, coords_train, x_test, coords_test,
                    bw: int = GWR_BASELINE_BW,
                    kernel: str = GWR_BASELINE_KERNEL) -> np.ndarray:
    """GWR baseline only; not used as final RAIN unary evidence."""
    sx = StandardScaler()
    sy = StandardScaler()
    x_train_s = sx.fit_transform(x_train)
    x_test_s = sx.transform(x_test)
    y_train_s = sy.fit_transform(y_train.reshape(-1, 1)).ravel()
    pred_s = gwr_predict_scaled(y_train_s, x_train_s, coords_train, x_test_s, coords_test, bw=bw, kernel=kernel)
    return sy.inverse_transform(pred_s.reshape(-1, 1)).ravel()


def neighbor_lists(source_coords: np.ndarray,
                   query_coords: np.ndarray,
                   k: int,
                   exclude_self: bool,
                   distance_threshold: float | None = GRAPH_DISTANCE_THRESHOLD) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Return distance-threshold neighbours sorted by distance.

    The manuscript constructs the spatial influence graph with a distance
    threshold. To keep fixed-width tensors reproducible, downstream code may
    truncate the returned list to SELECT_K or GWR_K after the valid radius-based
    links have been identified. If distance_threshold is None, the threshold is
    derived from the maximum k-th-neighbour distance among query nodes.
    """
    tree = cKDTree(source_coords)
    query_k = min(k + (1 if exclude_self else 0), len(source_coords))
    if distance_threshold is None:
        kth_dists, _ = tree.query(query_coords, k=query_k)
        kth_dists = np.atleast_2d(kth_dists)
        if kth_dists.shape[0] != len(query_coords):
            kth_dists = kth_dists.T
        distance_threshold = float(np.nanmax(kth_dists[:, -1]) * 1.000001)
    all_inds, all_dists = [], []
    same_array = len(source_coords) == len(query_coords) and np.allclose(source_coords, query_coords)
    for i in range(len(query_coords)):
        ii = np.asarray(tree.query_ball_point(query_coords[i], r=distance_threshold), dtype=np.int64)
        if len(ii):
            dd = np.linalg.norm(source_coords[ii] - query_coords[i].reshape(1, -1), axis=1)
            order = np.argsort(dd)
            ii, dd = ii[order], dd[order]
        else:
            dd, ii = tree.query(query_coords[i], k=1)
            ii = np.atleast_1d(ii).astype(np.int64)
            dd = np.atleast_1d(dd).astype(np.float32)
        if exclude_self and same_array:
            keep = ii != i
            ii = ii[keep]
            dd = dd[keep]
        if len(ii) == 0:
            dd, ii = tree.query(query_coords[i], k=1)
            ii = np.atleast_1d(ii).astype(np.int64)
            dd = np.atleast_1d(dd).astype(np.float32)
        all_inds.append(ii[:k].astype(np.int64))
        all_dists.append(dd[:k].astype(np.float32))
    return all_inds, all_dists


def gwr_projection_alpha(source_coords: np.ndarray,
                         source_x: np.ndarray,
                         query_coord: np.ndarray,
                         query_x: np.ndarray,
                         neighbors: np.ndarray,
                         bandwidth: int = GWR_K,
                         ridge: float = 1e-4) -> np.ndarray:
    if len(neighbors) == 0:
        return np.zeros(0, dtype=np.float32)
    neigh_x = source_x[neighbors].astype(np.float64)
    neigh_coords = source_coords[neighbors]
    dists = np.linalg.norm(neigh_coords - query_coord.reshape(1, -1), axis=1)
    k = min(int(bandwidth), len(neighbors))
    bw = np.partition(dists, k - 1)[k - 1] * 1.0000001
    bw = max(float(bw), 1e-8)
    z = dists / bw
    spatial_w = (1.0 - z ** 2) ** 2
    spatial_w[z >= 1.0] = 0.0

    x_aug = np.hstack([np.ones((len(neigh_x), 1)), neigh_x])
    x_i = np.hstack([1.0, query_x.astype(np.float64)]).reshape(1, -1)
    xt_w = x_aug.T * spatial_w.reshape(1, -1)
    xtwx = xt_w @ x_aug + ridge * np.eye(x_aug.shape[1])
    try:
        row = x_i @ linalg.solve(xtwx, xt_w, assume_a="pos")
    except Exception:
        row = x_i @ np.linalg.pinv(xtwx) @ xt_w
    alpha = np.abs(row.reshape(-1)) + 1e-8
    return (alpha / (alpha.sum() + 1e-8)).astype(np.float32)


def pad_or_truncate(items: list[np.ndarray], k: int, dtype, fill_value):
    out = np.full((len(items), k), fill_value, dtype=dtype)
    for i, arr in enumerate(items):
        arr = np.asarray(arr, dtype=dtype)
        use = arr[:k]
        if len(use):
            out[i, :len(use)] = use
    return out


def build_author_graph(train_df: pd.DataFrame, test_df: pd.DataFrame, rebuild: bool = True) -> dict:
    if AUTHOR_GRAPH_PKL.exists() and not rebuild:
        with AUTHOR_GRAPH_PKL.open("rb") as f:
            return pickle.load(f)

    x_train = train_df[RAIN_FEATURE_COLS].to_numpy(dtype=np.float32)
    x_test = test_df[RAIN_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)
    c_train = coords3(train_df, GRAPH_COORD_COLS)
    c_test = coords3(test_df, GRAPH_COORD_COLS)

    coord_scaler = StandardScaler().fit(c_train[:, :2])
    train_sp = coord_scaler.transform(c_train[:, :2]).astype(np.float32)
    test_sp = coord_scaler.transform(c_test[:, :2]).astype(np.float32)
    x_train = np.hstack([x_train, SPATIAL_FEATURE_SCALE * train_sp])
    x_test = np.hstack([x_test, SPATIAL_FEATURE_SCALE * test_sp])

    max_k = max(SELECT_K, GWR_K)
    train_neighbors, train_dists = neighbor_lists(c_train, c_train, max_k, exclude_self=True)
    test_self_neighbors, test_self_dists = neighbor_lists(c_test, c_test, max_k, exclude_self=True)
    test_train_neighbors, test_train_dists = neighbor_lists(c_train, c_test, max_k, exclude_self=False)

    train_alpha = [
        gwr_projection_alpha(c_train, x_train, c_train[i], x_train[i], inds, GWR_K)
        for i, inds in enumerate(train_neighbors)
    ]
    test_self_alpha = [
        gwr_projection_alpha(c_test, x_test, c_test[i], x_test[i], inds, GWR_K)
        for i, inds in enumerate(test_self_neighbors)
    ]
    test_train_alpha = [
        gwr_projection_alpha(c_train, x_train, c_test[i], x_test[i], inds, GWR_K)
        for i, inds in enumerate(test_train_neighbors)
    ]

    graph = {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "coords_train": c_train,
        "coords_test": c_test,
        "scaler_X": StandardScaler().fit(x_train),
        "scaler_y": StandardScaler().fit(y_train.reshape(-1, 1)),
        "select_k": SELECT_K,
        "best_gwr_k": GWR_K,
        "distance_threshold": GRAPH_DISTANCE_THRESHOLD,
        "graph_construction": "distance-threshold neighbourhood",
        "pairwise_potential": "GWR projection coefficient alpha_ij = |x_i (X^T W_i X)^-1 X^T W_i| / sum_j |.|",
        "interaction_energy": "Psi_ij = alpha_ij^GWR * (mu_i - mu_j)^2",
        "feature_cols": RAIN_FEATURE_COLS,
        "coord_cols": GRAPH_COORD_COLS,
        "train_neighbors_all": train_neighbors,
        "train_distances_all": train_dists,
        "train_alphas_all": train_alpha,
        "train_gwr_inds_all": train_neighbors,
        "test_self_neighbors_all": test_self_neighbors,
        "test_self_distances_all": test_self_dists,
        "test_self_alphas_all": test_self_alpha,
        "test_self_gwr_inds_all": test_self_neighbors,
        "test_train_neighbors_all": test_train_neighbors,
        "test_train_distances_all": test_train_dists,
        "test_train_alphas_all": test_train_alpha,
        "max_train_neighbors": max(len(v) for v in train_neighbors),
        "max_test_self_neighbors": max(len(v) for v in test_self_neighbors),
    }
    AUTHOR_GRAPH_PKL.parent.mkdir(parents=True, exist_ok=True)
    with AUTHOR_GRAPH_PKL.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved GWR pairwise graph: {AUTHOR_GRAPH_PKL}")
    print(f"Pairwise potential: {graph['pairwise_potential']}")
    return graph


def selected_from_graph(graph: dict) -> dict:
    return {
        "X_train": graph["X_train"],
        "X_test": graph["X_test"],
        "y_train": graph["y_train"],
        "y_test": graph["y_test"],
        "coords_train": graph["coords_train"],
        "coords_test": graph["coords_test"],
        "scaler_X": graph["scaler_X"],
        "scaler_y": graph["scaler_y"],
        "select_k": SELECT_K,
        "best_gwr_k": GWR_K,
        "train_neighbors": pad_or_truncate(graph["train_neighbors_all"], SELECT_K, np.int64, -1),
        "train_alphas": pad_or_truncate(graph["train_alphas_all"], SELECT_K, np.float32, 0.0),
        "test_self_neighbors": pad_or_truncate(graph["test_self_neighbors_all"], SELECT_K, np.int64, -1),
        "test_self_alphas": pad_or_truncate(graph["test_self_alphas_all"], SELECT_K, np.float32, 0.0),
        "test_train_neighbors": pad_or_truncate(graph["test_train_neighbors_all"], SELECT_K, np.int64, -1),
        "test_train_alphas": pad_or_truncate(graph["test_train_alphas_all"], SELECT_K, np.float32, 0.0),
    }


def train_clamped_refinement(base_pred: np.ndarray,
                             y_train: np.ndarray,
                             neighbors: np.ndarray,
                             alphas: np.ndarray,
                             ratio: float = INDUCTIVE_RAIN_RATIO,
                             k_neighbors: int = INDUCTIVE_RAIN_K) -> np.ndarray:
    """Legacy train-clamped baseline; not used by the final RAIN model."""
    contextual = np.zeros_like(base_pred, dtype=np.float64)
    for i in range(len(base_pred)):
        inds = neighbors[i][:k_neighbors]
        w = alphas[i][:k_neighbors]
        valid = inds >= 0
        if not np.any(valid):
            contextual[i] = base_pred[i]
            continue
        ww = w[valid].astype(np.float64)
        ww = ww / (ww.sum() + 1e-12)
        contextual[i] = float(np.sum(ww * y_train[inds[valid]]))
    return (1.0 - ratio) * base_pred + ratio * contextual


def semi_supervised_mfrnn_refinement(
    observed_coords: np.ndarray,
    observed_y: np.ndarray,
    query_coords: np.ndarray,
    unary_pred: np.ndarray,
    k_neighbors: int = FINAL_MFRNN_K,
    lambda_pair: float = FINAL_MFRNN_LAMBDA,
    n_iters: int = FINAL_MFRNN_ITERS,
    spatial_bandwidth: float = FINAL_MFRNN_BANDWIDTH,
) -> np.ndarray:
    """
    Inductive continuous-response MF-RNN refinement with observed source nodes.

    The update follows the continuous RAIN fixed-point form:

        mu_i = [eta f_i + lambda sum_j alpha_ij mu_j] /
               [eta + lambda sum_j alpha_ij]

    In the author-split PM2.5 protocol, train/validation nodes are observed and
    softly clamped to their measured responses, while held-out test nodes are
    initialized by the local evidence prediction. Test labels are never used in
    this inference step.
    """
    observed_coords = np.asarray(observed_coords, dtype=np.float64)
    query_coords = np.asarray(query_coords, dtype=np.float64)
    observed_y = np.asarray(observed_y, dtype=np.float64)
    unary_pred = np.asarray(unary_pred, dtype=np.float64)

    tree = cKDTree(observed_coords)
    mu = unary_pred.copy()
    for _ in range(n_iters):
        new_mu = mu.copy()
        for i, coord in enumerate(query_coords):
            dists, inds = tree.query(coord, k=min(k_neighbors, len(observed_coords)))
            dists = np.atleast_1d(dists).astype(np.float64)
            inds = np.atleast_1d(inds).astype(np.int64)
            weights = np.exp(-(dists ** 2) / (2.0 * spatial_bandwidth ** 2))
            weights = weights / (weights.sum() + 1e-12)
            context = float(np.sum(weights * observed_y[inds]))
            new_mu[i] = (unary_pred[i] + lambda_pair * context) / (1.0 + lambda_pair)
        mu = new_mu
    return mu


def continuous_rain_gwr_mfrnn(
    unary_pred: np.ndarray,
    source_y: np.ndarray,
    neighbors: np.ndarray,
    alphas: np.ndarray,
    eta: float = 1.0,
    lambda_pair: float = FINAL_MFRNN_LAMBDA,
    n_iters: int = FINAL_MFRNN_ITERS,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Paper-consistent continuous RAIN-GWR inference.

    Local evidence is the GWR unary prediction f_i. Pairwise interactions use
    the GWR projection coefficients alpha_ij from build_author_graph():

        mu_i = [eta f_i + lambda sum_j alpha_ij^GWR y_j]
               / [eta + lambda sum_j alpha_ij^GWR]

    source_y contains only observed train+validation PM2.5 values. Test labels
    are never used in this inference step.
    """
    unary_pred = np.asarray(unary_pred, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    alphas = np.asarray(alphas, dtype=np.float64)

    mu = unary_pred.copy()
    for _ in range(max(int(n_iters), 1)):
        new_mu = np.empty_like(mu)
        for i in range(len(mu)):
            inds = neighbors[i]
            weights = alphas[i]
            valid = inds >= 0
            if not np.any(valid):
                new_mu[i] = unary_pred[i]
                continue
            ww = weights[valid]
            yy = source_y[inds[valid]]
            if renormalize:
                ww = ww / (np.sum(ww) + 1e-12)
            pairwise_sum = float(np.sum(ww * yy))
            sum_w = float(np.sum(ww))
            denominator = max(float(eta) + float(lambda_pair) * sum_w, 1e-12)
            new_mu[i] = (float(eta) * unary_pred[i] + float(lambda_pair) * pairwise_sum) / denominator
        if np.max(np.abs(new_mu - mu)) < 1e-10:
            mu = new_mu
            break
        mu = new_mu
    return mu


class PaperEndToEndContinuousRAIN(nn.Module):
    """
    Continuous-response RAIN layer used by the final paper reproduction path.

    The deterministic local-evidence basis is calibrated inside this module,
    and the MF-RNN balancing parameters are optimized through the final RAIN
    loss. GWR-derived alpha values are treated as fixed adaptive interactions,
    matching the manuscript's continuous-response formulation.
    """

    def __init__(
        self,
        initial_eta: float = PAPER_RAIN_ETA,
        initial_lambda: float = PAPER_RAIN_LAMBDA,
        renormalize_alpha: bool = PAPER_RAIN_RENORMALIZE_ALPHA,
    ):
        super().__init__()
        self.unary_scale = nn.Parameter(torch.tensor(1.0))
        self.unary_bias = nn.Parameter(torch.tensor(0.0))
        self.log_eta = nn.Parameter(torch.log(torch.tensor(float(initial_eta))))
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(float(initial_lambda))))
        self.renormalize_alpha = bool(renormalize_alpha)
        self.register_buffer("initial_log_eta", torch.log(torch.tensor(float(initial_eta))))
        self.register_buffer("initial_log_lambda", torch.log(torch.tensor(float(initial_lambda))))

    def forward(self, base_unary, source_y, neighbors, alphas):
        valid_mask = (neighbors != -1).float()
        safe_neighbors = torch.clamp(neighbors, min=0)
        weights = torch.clamp(alphas, min=0.0) * valid_mask
        if self.renormalize_alpha:
            row_sum = torch.sum(weights, dim=-1, keepdim=True)
            weights = weights / torch.clamp(row_sum, min=1e-8)

        source_values = source_y[safe_neighbors.reshape(-1)].reshape_as(weights)
        pairwise_sum = torch.sum(weights * source_values, dim=-1)
        sum_w = torch.sum(weights, dim=-1)

        unary = self.unary_scale * base_unary + self.unary_bias
        eta = torch.exp(self.log_eta)
        lambda_pair = torch.exp(self.log_lambda)
        denominator = torch.clamp(eta + lambda_pair * sum_w, min=1e-6)
        rain_pred = (eta * unary + lambda_pair * pairwise_sum) / denominator
        return unary, rain_pred

    def prior_loss(self):
        return (
            (self.unary_scale - 1.0) ** 2
            + self.unary_bias ** 2
            + (self.log_eta - self.initial_log_eta) ** 2
            + (self.log_lambda - self.initial_log_lambda) ** 2
        )


def run_paper_end_to_end_rain(
    observed_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_params: dict | None = None,
    model_save_path: str | None = None,
) -> dict:
    """
    Fit the paper-aligned continuous RAIN layer and predict held-out nodes.

    Test labels are not used for local evidence, graph construction, parameter
    fitting, or prediction. They are read only when metrics are computed by the
    caller.
    """
    global SELECT_K, GWR_K, SPATIAL_FEATURE_SCALE

    params = {
        "select_k": PAPER_RAIN_SELECT_K,
        "gwr_k": PAPER_RAIN_GWR_K,
        "spatial_feature_scale": 0.0,
        "unary_bw": PAPER_RAIN_UNARY_BW,
        "unary_kernel": PAPER_RAIN_UNARY_KERNEL,
        "initial_eta": PAPER_RAIN_ETA,
        "initial_lambda": PAPER_RAIN_LAMBDA,
        "renormalize_alpha": PAPER_RAIN_RENORMALIZE_ALPHA,
        "e2e_epochs": PAPER_RAIN_E2E_EPOCHS,
        "e2e_lr": PAPER_RAIN_E2E_LR,
        "e2e_prior_weight": PAPER_RAIN_E2E_PRIOR_WEIGHT,
    }
    if selected_params:
        params.update(selected_params)

    SELECT_K = int(params["select_k"])
    GWR_K = int(params["gwr_k"])
    SPATIAL_FEATURE_SCALE = float(params["spatial_feature_scale"])

    observed_x = observed_df[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
    observed_y = observed_df[TARGET_COL].to_numpy(dtype=np.float32)
    test_x = test_df[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
    observed_coords = coords3(observed_df, BASELINE_COORD_COLS)
    test_coords = coords3(test_df, BASELINE_COORD_COLS)

    base_unary_train = fit_predict_gwr(
        observed_x,
        observed_y,
        observed_coords,
        observed_x,
        observed_coords,
        bw=int(params["unary_bw"]),
        kernel=str(params["unary_kernel"]),
    )
    base_unary_test = fit_predict_gwr(
        observed_x,
        observed_y,
        observed_coords,
        test_x,
        test_coords,
        bw=int(params["unary_bw"]),
        kernel=str(params["unary_kernel"]),
    )

    graph = build_author_graph(observed_df, test_df, rebuild=True)
    print(f"RAIN pairwise source: {graph.get('pairwise_potential', 'GWR projection coefficient')}")
    selected = selected_from_graph(graph)
    scaler_y = selected["scaler_y"]
    y_train_s = scaler_y.transform(selected["y_train"].reshape(-1, 1)).ravel()
    base_unary_train_s = scaler_y.transform(base_unary_train.reshape(-1, 1)).ravel()
    base_unary_test_s = scaler_y.transform(base_unary_test.reshape(-1, 1)).ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperEndToEndContinuousRAIN(
        initial_eta=float(params["initial_eta"]),
        initial_lambda=float(params["initial_lambda"]),
        renormalize_alpha=bool(params["renormalize_alpha"]),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(params["e2e_lr"]))

    y_train_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    base_unary_train_t = torch.tensor(base_unary_train_s, dtype=torch.float32, device=device)
    train_neighbors_t = torch.tensor(selected["train_neighbors"], dtype=torch.long, device=device)
    train_alphas_t = torch.tensor(selected["train_alphas"], dtype=torch.float32, device=device)

    train_losses = []
    model.train()
    for _ in range(int(params["e2e_epochs"])):
        optimizer.zero_grad()
        _, rain_train = model(base_unary_train_t, y_train_t, train_neighbors_t, train_alphas_t)
        mse_loss = torch.mean((rain_train - y_train_t) ** 2)
        loss = mse_loss + float(params["e2e_prior_weight"]) * model.prior_loss()
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu().item()))

    model.eval()
    with torch.no_grad():
        base_unary_test_t = torch.tensor(base_unary_test_s, dtype=torch.float32, device=device)
        test_train_neighbors_t = torch.tensor(selected["test_train_neighbors"], dtype=torch.long, device=device)
        test_train_alphas_t = torch.tensor(selected["test_train_alphas"], dtype=torch.float32, device=device)
        unary_test_s, rain_test_s = model(
            base_unary_test_t,
            y_train_t,
            test_train_neighbors_t,
            test_train_alphas_t,
        )
        unary_pred_test = scaler_y.inverse_transform(unary_test_s.cpu().numpy().reshape(-1, 1)).ravel()
        rain_pred_test = scaler_y.inverse_transform(rain_test_s.cpu().numpy().reshape(-1, 1)).ravel()

    learned = {
        "unary_scale": float(model.unary_scale.detach().cpu().item()),
        "unary_bias": float(model.unary_bias.detach().cpu().item()),
        "eta": float(torch.exp(model.log_eta).detach().cpu().item()),
        "lambda_pair": float(torch.exp(model.log_lambda).detach().cpu().item()),
    }
    result = {
        "mlp_pred_test": unary_pred_test,
        "rain_joint_pred_test": rain_pred_test,
        "base_unary_pred_test": base_unary_test,
        "train_losses": train_losses,
        "model": model,
        "selected": selected,
        "graph": graph,
        "params": params,
        "learned_params": learned,
    }

    if model_save_path is not None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "params": params,
                "learned_params": learned,
                "train_losses": train_losses,
                "scaler_y": scaler_y,
            },
            model_save_path,
        )
        print(f"Paper end-to-end RAIN model saved to: {model_save_path}")

    return result


def tune_paper_end_to_end_rain_on_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Validate the paper-aligned end-to-end RAIN protocol without test labels."""
    params = {
        "select_k": PAPER_RAIN_SELECT_K,
        "gwr_k": PAPER_RAIN_GWR_K,
        "spatial_feature_scale": 0.0,
        "unary_bw": PAPER_RAIN_UNARY_BW,
        "unary_kernel": PAPER_RAIN_UNARY_KERNEL,
        "initial_eta": PAPER_RAIN_ETA,
        "initial_lambda": PAPER_RAIN_LAMBDA,
        "renormalize_alpha": PAPER_RAIN_RENORMALIZE_ALPHA,
        "e2e_epochs": PAPER_RAIN_E2E_EPOCHS,
        "e2e_lr": PAPER_RAIN_E2E_LR,
        "e2e_prior_weight": PAPER_RAIN_E2E_PRIOR_WEIGHT,
    }
    result = run_paper_end_to_end_rain(train_df, val_df, params, model_save_path=None)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float32)
    row = {
        **params,
        **{f"learned_{k}": v for k, v in result["learned_params"].items()},
        "validation_unary_rmse": rmse_score(y_val, result["mlp_pred_test"]),
        "validation_unary_r2": r2_score(y_val, result["mlp_pred_test"]),
        "validation_rain_rmse": rmse_score(y_val, result["rain_joint_pred_test"]),
        "validation_rain_r2": r2_score(y_val, result["rain_joint_pred_test"]),
    }
    return params, pd.DataFrame([row])


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dnn_spatial_evidence_features(
    observed_df: pd.DataFrame,
    query_df: pd.DataFrame,
    exclude_self: bool = False,
) -> np.ndarray:
    """Training-label spatial evidence features built only from observed nodes."""
    observed_coords = observed_df[["proj_x", "proj_y"]].to_numpy(dtype=np.float64)
    query_coords = query_df[["proj_x", "proj_y"]].to_numpy(dtype=np.float64)
    observed_y = observed_df[TARGET_COL].to_numpy(dtype=np.float64)
    tree = cKDTree(observed_coords)
    max_k = min(64 + (1 if exclude_self else 0), len(observed_df))
    dists, inds = tree.query(query_coords, k=max_k)
    dists = np.asarray(dists)
    inds = np.asarray(inds)
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        inds = inds.reshape(-1, 1)

    if exclude_self:
        filtered_dists = []
        filtered_inds = []
        for row_d, row_i in zip(dists, inds):
            valid = row_d > 1e-8
            filtered_dists.append(row_d[valid][:64])
            filtered_inds.append(row_i[valid][:64])
        dists = np.vstack(filtered_dists)
        inds = np.vstack(filtered_inds)

    features = []
    for k_neighbors in [3, 8, 16, 32, 64]:
        k = min(k_neighbors, inds.shape[1])
        local_d = dists[:, :k]
        local_i = inds[:, :k]
        local_y = observed_y[local_i]
        for power in [0.5, 1.0, 1.5]:
            weights = 1.0 / np.maximum(local_d, 1.0) ** power
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
            features.append((weights * local_y).sum(axis=1))
        features.extend([
            local_y.mean(axis=1),
            local_y.std(axis=1),
            local_d.mean(axis=1),
            local_d[:, 0],
        ])
    return np.vstack(features).T.astype(np.float32)


def build_dnn_local_evidence_features(
    observed_df: pd.DataFrame,
    query_df: pd.DataFrame,
    auxiliary_gwr: np.ndarray,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Features for the neural Local Evidence Module.

    The final local evidence is produced by the DNN. The GWR term is included
    only as an observed-data auxiliary basis so the network can learn a stronger
    local-evidence mapping without using test labels.
    """
    raw_cols = BASE_FEATURE_COLS + ["proj_x", "proj_y", "longitude", "latitude"]
    raw_x = query_df[raw_cols].to_numpy(dtype=np.float32)
    spatial_x = dnn_spatial_evidence_features(observed_df, query_df, exclude_self=exclude_self)
    return np.hstack([
        raw_x,
        spatial_x,
        np.asarray(auxiliary_gwr, dtype=np.float32).reshape(-1, 1),
    ]).astype(np.float32)


class SELayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = max(dim // 4, 8)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class DNNLocalEvidenceModel(nn.Module):
    """Neural global predictor for domain-wide PM2.5 associations."""

    def __init__(self, input_dim: int, dropout: float = DNN_LOCAL_EVIDENCE_DROPOUT):
        super().__init__()
        widths = (256, 128, 64)
        layers = []
        prev_dim = input_dim
        for width in widths:
            layers.extend([
                nn.Linear(prev_dim, width),
                nn.LayerNorm(width),
                nn.SiLU(),
                SELayer(width),
                nn.Dropout(dropout),
            ])
            prev_dim = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)
        self.linear_skip = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1) + self.linear_skip(x).squeeze(-1)


class DifferentiableContinuousMFNN(nn.Module):
    """Differentiable recurrent mean-field global-local integration update."""

    def __init__(
        self,
        initial_eta: float = DNN_RAIN_ETA,
        initial_lambda: float = DNN_RAIN_LAMBDA,
        renormalize_alpha: bool = PAPER_RAIN_RENORMALIZE_ALPHA,
    ):
        super().__init__()
        self.log_eta = nn.Parameter(torch.log(torch.tensor(float(initial_eta))))
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(float(initial_lambda))))
        self.renormalize_alpha = bool(renormalize_alpha)
        self.register_buffer("initial_log_eta", torch.log(torch.tensor(float(initial_eta))))
        self.register_buffer("initial_log_lambda", torch.log(torch.tensor(float(initial_lambda))))

    def forward(self, unary_pred, source_y, neighbors, alphas):
        valid_mask = (neighbors != -1).float()
        safe_neighbors = torch.clamp(neighbors, min=0)
        weights = torch.clamp(alphas, min=0.0) * valid_mask
        if self.renormalize_alpha:
            weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-8)
        source_values = source_y[safe_neighbors.reshape(-1)].reshape_as(weights)
        pairwise_sum = torch.sum(weights * source_values, dim=-1)
        sum_w = torch.sum(weights, dim=-1)
        eta = torch.exp(self.log_eta)
        lambda_pair = torch.exp(self.log_lambda)
        return (eta * unary_pred + lambda_pair * pairwise_sum) / torch.clamp(
            eta + lambda_pair * sum_w,
            min=1e-6,
        )

    def prior_loss(self):
        return (
            (self.log_eta - self.initial_log_eta) ** 2
            + (self.log_lambda - self.initial_log_lambda) ** 2
        )

    def params_dict(self) -> dict:
        return {
            "eta": float(torch.exp(self.log_eta).detach().cpu().item()),
            "lambda_pair": float(torch.exp(self.log_lambda).detach().cpu().item()),
        }


def fit_dnn_local_evidence_member(
    observed_df: pd.DataFrame,
    seed: int,
    epochs: int = DNN_LOCAL_EVIDENCE_EPOCHS,
) -> dict:
    """Fit one neural local-evidence member on observed labels only."""
    set_torch_seed(seed)
    observed_x = observed_df[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
    observed_y = observed_df[TARGET_COL].to_numpy(dtype=np.float32)
    observed_coords = coords3(observed_df, BASELINE_COORD_COLS)
    auxiliary_gwr_train = fit_predict_gwr(
        observed_x,
        observed_y,
        observed_coords,
        observed_x,
        observed_coords,
        bw=PAPER_RAIN_UNARY_BW,
        kernel=PAPER_RAIN_UNARY_KERNEL,
    )
    train_features = build_dnn_local_evidence_features(
        observed_df,
        observed_df,
        auxiliary_gwr_train,
        exclude_self=True,
    )
    scaler_x = StandardScaler().fit(train_features)
    scaler_y = StandardScaler().fit(observed_y.reshape(-1, 1))
    x_train_t = torch.tensor(scaler_x.transform(train_features).astype(np.float32))
    y_train_t = torch.tensor(
        scaler_y.transform(observed_y.reshape(-1, 1)).ravel().astype(np.float32)
    )

    model = DNNLocalEvidenceModel(train_features.shape[1])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=DNN_LOCAL_EVIDENCE_LR,
        weight_decay=DNN_LOCAL_EVIDENCE_WEIGHT_DECAY,
    )
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    train_losses = []
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu().item()))

    return {
        "seed": int(seed),
        "model": model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "train_losses": train_losses,
    }


def fit_dnn_e2e_rain_member(
    observed_df: pd.DataFrame,
    selected: dict,
    seed: int,
    pretrain_epochs: int = DNN_LOCAL_EVIDENCE_EPOCHS,
    joint_epochs: int = DNN_E2E_JOINT_EPOCHS,
) -> dict:
    """Fit one end-to-end global predictor and recurrent integration member."""
    set_torch_seed(seed)
    observed_x = observed_df[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
    observed_y = observed_df[TARGET_COL].to_numpy(dtype=np.float32)
    observed_coords = coords3(observed_df, BASELINE_COORD_COLS)
    auxiliary_gwr_train = fit_predict_gwr(
        observed_x,
        observed_y,
        observed_coords,
        observed_x,
        observed_coords,
        bw=PAPER_RAIN_UNARY_BW,
        kernel=PAPER_RAIN_UNARY_KERNEL,
    )
    train_features = build_dnn_local_evidence_features(
        observed_df,
        observed_df,
        auxiliary_gwr_train,
        exclude_self=True,
    )
    scaler_x = StandardScaler().fit(train_features)
    scaler_y = StandardScaler().fit(observed_y.reshape(-1, 1))
    x_train_t = torch.tensor(scaler_x.transform(train_features).astype(np.float32))
    y_train_t = torch.tensor(
        scaler_y.transform(observed_y.reshape(-1, 1)).ravel().astype(np.float32)
    )
    train_neighbors_t = torch.tensor(selected["train_neighbors"], dtype=torch.long)
    train_alphas_t = torch.tensor(selected["train_alphas"], dtype=torch.float32)

    model = DNNLocalEvidenceModel(train_features.shape[1])
    mfrnn = DifferentiableContinuousMFNN()

    pretrain_optimizer = optim.AdamW(
        model.parameters(),
        lr=DNN_LOCAL_EVIDENCE_LR,
        weight_decay=DNN_LOCAL_EVIDENCE_WEIGHT_DECAY,
    )
    unary_loss_fn = nn.SmoothL1Loss(beta=0.25)
    pretrain_losses = []
    for _ in range(int(pretrain_epochs)):
        model.train()
        pretrain_optimizer.zero_grad()
        unary_pred = model(x_train_t)
        loss = unary_loss_fn(unary_pred, y_train_t)
        loss.backward()
        pretrain_optimizer.step()
        pretrain_losses.append(float(loss.detach().cpu().item()))

    joint_optimizer = optim.AdamW(
        list(model.parameters()) + list(mfrnn.parameters()),
        lr=DNN_E2E_JOINT_LR,
        weight_decay=1e-5,
    )
    joint_losses = []
    for _ in range(int(joint_epochs)):
        model.train()
        mfrnn.train()
        joint_optimizer.zero_grad()
        unary_pred = model(x_train_t)
        rain_pred = mfrnn(unary_pred, y_train_t, train_neighbors_t, train_alphas_t)
        rain_loss = torch.mean((rain_pred - y_train_t) ** 2)
        unary_aux = unary_loss_fn(unary_pred, y_train_t)
        loss = (
            rain_loss
            + DNN_E2E_UNARY_AUX_WEIGHT * unary_aux
            + DNN_E2E_PRIOR_WEIGHT * mfrnn.prior_loss()
        )
        loss.backward()
        joint_optimizer.step()
        joint_losses.append(float(loss.detach().cpu().item()))

    return {
        "seed": int(seed),
        "model": model,
        "mfrnn": mfrnn,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "pretrain_losses": pretrain_losses,
        "joint_losses": joint_losses,
        "learned_params": mfrnn.params_dict(),
    }


def predict_dnn_local_evidence_member(
    member: dict,
    observed_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> np.ndarray:
    observed_x = observed_df[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
    observed_y = observed_df[TARGET_COL].to_numpy(dtype=np.float32)
    observed_coords = coords3(observed_df, BASELINE_COORD_COLS)
    query_x = query_df[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
    query_coords = coords3(query_df, BASELINE_COORD_COLS)
    auxiliary_gwr = fit_predict_gwr(
        observed_x,
        observed_y,
        observed_coords,
        query_x,
        query_coords,
        bw=PAPER_RAIN_UNARY_BW,
        kernel=PAPER_RAIN_UNARY_KERNEL,
    )
    features = build_dnn_local_evidence_features(
        observed_df,
        query_df,
        auxiliary_gwr,
        exclude_self=False,
    )
    model = member["model"]
    model.eval()
    x_t = torch.tensor(member["scaler_x"].transform(features).astype(np.float32))
    with torch.no_grad():
        pred_s = model(x_t).cpu().numpy()
    return member["scaler_y"].inverse_transform(pred_s.reshape(-1, 1)).ravel()


def predict_dnn_e2e_rain_member(
    member: dict,
    observed_df: pd.DataFrame,
    query_df: pd.DataFrame,
    selected: dict,
) -> tuple[np.ndarray, np.ndarray]:
    observed_x = observed_df[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
    observed_y = observed_df[TARGET_COL].to_numpy(dtype=np.float32)
    observed_coords = coords3(observed_df, BASELINE_COORD_COLS)
    query_x = query_df[BASE_FEATURE_COLS].to_numpy(dtype=np.float32)
    query_coords = coords3(query_df, BASELINE_COORD_COLS)
    auxiliary_gwr = fit_predict_gwr(
        observed_x,
        observed_y,
        observed_coords,
        query_x,
        query_coords,
        bw=PAPER_RAIN_UNARY_BW,
        kernel=PAPER_RAIN_UNARY_KERNEL,
    )
    features = build_dnn_local_evidence_features(
        observed_df,
        query_df,
        auxiliary_gwr,
        exclude_self=False,
    )
    x_t = torch.tensor(member["scaler_x"].transform(features).astype(np.float32))
    source_y_s = member["scaler_y"].transform(selected["y_train"].reshape(-1, 1)).ravel()
    source_y_t = torch.tensor(source_y_s.astype(np.float32))
    neighbors_t = torch.tensor(selected["test_train_neighbors"], dtype=torch.long)
    alphas_t = torch.tensor(selected["test_train_alphas"], dtype=torch.float32)

    member["model"].eval()
    member["mfrnn"].eval()
    with torch.no_grad():
        unary_s = member["model"](x_t)
        rain_s = member["mfrnn"](unary_s, source_y_t, neighbors_t, alphas_t)
    unary = member["scaler_y"].inverse_transform(unary_s.cpu().numpy().reshape(-1, 1)).ravel()
    rain = member["scaler_y"].inverse_transform(rain_s.cpu().numpy().reshape(-1, 1)).ravel()
    return unary, rain


def run_dnn_local_evidence_rain(
    observed_df: pd.DataFrame,
    query_df: pd.DataFrame,
    model_save_path: str | None = None,
) -> dict:
    """Run neural local evidence plus paper GWR-alpha MF-RNN RAIN."""
    global SELECT_K, GWR_K, SPATIAL_FEATURE_SCALE
    SELECT_K = PAPER_RAIN_SELECT_K
    GWR_K = PAPER_RAIN_GWR_K
    SPATIAL_FEATURE_SCALE = 0.0

    members = [
        fit_dnn_local_evidence_member(observed_df, seed)
        for seed in DNN_LOCAL_EVIDENCE_SEEDS
    ]
    member_predictions = [
        predict_dnn_local_evidence_member(member, observed_df, query_df)
        for member in members
    ]
    dnn_pred = np.mean(member_predictions, axis=0)

    graph = build_author_graph(observed_df, query_df, rebuild=True)
    print(f"RAIN pairwise source: {graph.get('pairwise_potential', 'GWR projection coefficient')}")
    selected = selected_from_graph(graph)
    rain_pred = continuous_rain_gwr_mfrnn(
        unary_pred=dnn_pred,
        source_y=selected["y_train"],
        neighbors=selected["test_train_neighbors"],
        alphas=selected["test_train_alphas"],
        eta=DNN_RAIN_ETA,
        lambda_pair=DNN_RAIN_LAMBDA,
        n_iters=1,
        renormalize=PAPER_RAIN_RENORMALIZE_ALPHA,
    )

    if model_save_path is not None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(
            {
                "seeds": DNN_LOCAL_EVIDENCE_SEEDS,
                "epochs": DNN_LOCAL_EVIDENCE_EPOCHS,
                "eta": DNN_RAIN_ETA,
                "lambda_pair": DNN_RAIN_LAMBDA,
                "members": [
                    {
                        "seed": member["seed"],
                        "model_state_dict": member["model"].state_dict(),
                        "scaler_x": member["scaler_x"],
                        "scaler_y": member["scaler_y"],
                        "train_losses": member["train_losses"],
                    }
                    for member in members
                ],
            },
            model_save_path,
        )
        print(f"DNN local-evidence RAIN model saved to: {model_save_path}")

    return {
        "mlp_pred_test": dnn_pred,
        "dnn_member_pred_test": member_predictions,
        "rain_joint_pred_test": rain_pred,
        "graph": graph,
        "selected": selected,
    }


def run_strict_dnn_e2e_rain(
    observed_df: pd.DataFrame,
    query_df: pd.DataFrame,
    model_save_path: str | None = None,
) -> dict:
    """Run the end-to-end continuous-response RAIN main model."""
    global SELECT_K, GWR_K, SPATIAL_FEATURE_SCALE
    SELECT_K = PAPER_RAIN_SELECT_K
    GWR_K = PAPER_RAIN_GWR_K
    SPATIAL_FEATURE_SCALE = 0.0

    graph = build_author_graph(observed_df, query_df, rebuild=True)
    print(f"RAIN pairwise source: {graph.get('pairwise_potential', 'GWR projection coefficient')}")
    selected = selected_from_graph(graph)

    members = [
        fit_dnn_e2e_rain_member(observed_df, selected, seed)
        for seed in DNN_LOCAL_EVIDENCE_SEEDS
    ]
    member_unary = []
    member_rain = []
    for member in members:
        unary_pred, rain_pred = predict_dnn_e2e_rain_member(member, observed_df, query_df, selected)
        member_unary.append(unary_pred)
        member_rain.append(rain_pred)

    dnn_unary = np.mean(member_unary, axis=0)
    rain_pred = np.mean(member_rain, axis=0)

    if model_save_path is not None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(
            {
                "seeds": DNN_LOCAL_EVIDENCE_SEEDS,
                "pretrain_epochs": DNN_LOCAL_EVIDENCE_EPOCHS,
                "joint_epochs": DNN_E2E_JOINT_EPOCHS,
                "joint_lr": DNN_E2E_JOINT_LR,
                "unary_aux_weight": DNN_E2E_UNARY_AUX_WEIGHT,
                "prior_weight": DNN_E2E_PRIOR_WEIGHT,
                "members": [
                    {
                        "seed": member["seed"],
                        "model_state_dict": member["model"].state_dict(),
                        "mfrnn_state_dict": member["mfrnn"].state_dict(),
                        "scaler_x": member["scaler_x"],
                        "scaler_y": member["scaler_y"],
                        "pretrain_losses": member["pretrain_losses"],
                        "joint_losses": member["joint_losses"],
                        "learned_params": member["learned_params"],
                    }
                    for member in members
                ],
            },
            model_save_path,
        )
        print(f"Strict DNN end-to-end RAIN model saved to: {model_save_path}")

    return {
        "mlp_pred_test": dnn_unary,
        "dnn_member_pred_test": member_unary,
        "rain_joint_pred_test": rain_pred,
        "rain_member_pred_test": member_rain,
        "graph": graph,
        "selected": selected,
        "member_params": [member["learned_params"] for member in members],
    }


def run_rain(train_df: pd.DataFrame, test_df: pd.DataFrame, rain_unary_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = build_author_graph(train_df, test_df, rebuild=True)
    print(f"RAIN pairwise source: {graph.get('pairwise_potential', 'GWR projection coefficient')}")
    selected = selected_from_graph(graph)
    result = run_joint_training_with_dynamic_k(
        selected_data=selected,
        model_save_path=str(MODEL_DIR / "author_split_rain.pkl_graph.pth"),
        hidden_dims=(256, 128),
        epochs=500,
        lr=8e-4,
    )
    inductive = train_clamped_refinement(
        base_pred=rain_unary_pred,
        y_train=selected["y_train"],
        neighbors=selected["test_train_neighbors"],
        alphas=selected["test_train_alphas"],
        ratio=INDUCTIVE_RAIN_RATIO,
        k_neighbors=INDUCTIVE_RAIN_K,
    )
    return result["mlp_pred_test"], result["rain_joint_pred_test"], inductive


def tune_rain_on_validation(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Select end-to-end RAINJointModel hyperparameters using validation R2 only."""
    global SELECT_K, GWR_K, SPATIAL_FEATURE_SCALE
    original_select_k = SELECT_K
    original_gwr_k = GWR_K
    original_spatial_scale = SPATIAL_FEATURE_SCALE
    rows = []
    best_params = None
    best_r2 = -np.inf
    best_rmse = np.inf

    try:
        for select_k in VALIDATION_TUNING_GRID["select_k"]:
            for gwr_k in VALIDATION_TUNING_GRID["gwr_k"]:
                for spatial_scale in VALIDATION_TUNING_GRID["spatial_feature_scale"]:
                    SELECT_K = int(select_k)
                    GWR_K = int(gwr_k)
                    SPATIAL_FEATURE_SCALE = float(spatial_scale)
                    graph = build_author_graph(train_df, val_df, rebuild=True)
                    selected = selected_from_graph(graph)

                    for hidden_dims in VALIDATION_TUNING_GRID["hidden_dims"]:
                        for lr in VALIDATION_TUNING_GRID["lr"]:
                            for pretrain_epochs in VALIDATION_TUNING_GRID["pretrain_epochs"]:
                                for epochs in VALIDATION_TUNING_GRID["epochs"]:
                                    for unary_loss_weight in VALIDATION_TUNING_GRID["unary_loss_weight"]:
                                        for initial_log_eta in VALIDATION_TUNING_GRID["initial_log_eta"]:
                                            for initial_log_lambda in VALIDATION_TUNING_GRID["initial_log_lambda"]:
                                                for renormalize_gwr in VALIDATION_TUNING_GRID["renormalize_gwr"]:
                                                    tag = "_".join(str(v) for v in hidden_dims)
                                                    model_path = MODEL_DIR / (
                                                        f"validation_rain_joint_k{select_k}_gwr{gwr_k}_h{tag}_"
                                                        f"lr{lr:g}_pre{pretrain_epochs}_e{epochs}_"
                                                        f"uw{unary_loss_weight:g}_le{initial_log_eta:g}_"
                                                        f"ll{initial_log_lambda:g}_renorm{int(renormalize_gwr)}.pth"
                                                    )
                                                    result = run_joint_training_with_dynamic_k(
                                                        selected_data=selected,
                                                        model_save_path=str(model_path),
                                                        hidden_dims=hidden_dims,
                                                        epochs=epochs,
                                                        lr=lr,
                                                        pretrain_epochs=pretrain_epochs,
                                                        unary_loss_weight=unary_loss_weight,
                                                        initial_log_eta=initial_log_eta,
                                                        initial_log_lambda=initial_log_lambda,
                                                        renormalize_gwr=renormalize_gwr,
                                                    )
                                                    row = {
                                                        "select_k": select_k,
                                                        "gwr_k": gwr_k,
                                                        "spatial_feature_scale": spatial_scale,
                                                        "hidden_dims": "-".join(str(v) for v in hidden_dims),
                                                        "lr": lr,
                                                        "pretrain_epochs": pretrain_epochs,
                                                        "epochs": epochs,
                                                        "unary_loss_weight": unary_loss_weight,
                                                        "initial_log_eta": initial_log_eta,
                                                        "initial_log_lambda": initial_log_lambda,
                                                        "renormalize_gwr": renormalize_gwr,
                                                        "mlp_rmse": result["mlp_rmse"],
                                                        "mlp_r2": result["mlp_r2"],
                                                        "rain_joint_rmse": result["rain_joint_rmse"],
                                                        "rain_joint_r2": result["rain_joint_r2"],
                                                    }
                                                    rows.append(row)
                                                    if result["rain_joint_r2"] > best_r2 or (
                                                        np.isclose(result["rain_joint_r2"], best_r2)
                                                        and result["rain_joint_rmse"] < best_rmse
                                                    ):
                                                        best_r2 = result["rain_joint_r2"]
                                                        best_rmse = result["rain_joint_rmse"]
                                                        best_params = {
                                                            "select_k": select_k,
                                                            "gwr_k": gwr_k,
                                                            "spatial_feature_scale": spatial_scale,
                                                            "hidden_dims": hidden_dims,
                                                            "lr": lr,
                                                            "pretrain_epochs": pretrain_epochs,
                                                            "epochs": epochs,
                                                            "unary_loss_weight": unary_loss_weight,
                                                            "initial_log_eta": initial_log_eta,
                                                            "initial_log_lambda": initial_log_lambda,
                                                            "renormalize_gwr": renormalize_gwr,
                                                            "validation_mlp_r2": result["mlp_r2"],
                                                            "validation_mlp_rmse": result["mlp_rmse"],
                                                            "validation_rain_joint_r2": result["rain_joint_r2"],
                                                            "validation_rain_joint_rmse": result["rain_joint_rmse"],
                                                        }
                                                    try:
                                                        model_path.unlink()
                                                    except FileNotFoundError:
                                                        pass
    finally:
        SELECT_K = original_select_k
        GWR_K = original_gwr_k
        SPATIAL_FEATURE_SCALE = original_spatial_scale

    assert best_params is not None
    return best_params, pd.DataFrame(rows)


def main() -> None:
    global SELECT_K, GWR_K, SPATIAL_FEATURE_SCALE
    DNN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = read_split()

    split_manifest = pd.DataFrame([
        {"split": "train", "rows": len(train_df), "unique_id": train_df["id"].nunique()},
        {"split": "validation", "rows": len(val_df), "unique_id": val_df["id"].nunique()},
        {"split": "test", "rows": len(test_df), "unique_id": test_df["id"].nunique()},
    ])
    split_manifest.to_csv(DNN_RESULTS_DIR / "author_split_manifest.csv", index=False)
    print("Author benchmark split:")
    print(split_manifest.to_string(index=False))

    validation_result = run_strict_dnn_e2e_rain(train_df, val_df, model_save_path=None)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float32)
    validation_results = pd.DataFrame([{
        "model_family": "strict_dnn_e2e_rain",
        "seeds": "-".join(str(v) for v in DNN_LOCAL_EVIDENCE_SEEDS),
        "dnn_pretrain_epochs": DNN_LOCAL_EVIDENCE_EPOCHS,
        "joint_epochs": DNN_E2E_JOINT_EPOCHS,
        "dnn_lr": DNN_LOCAL_EVIDENCE_LR,
        "joint_lr": DNN_E2E_JOINT_LR,
        "dnn_weight_decay": DNN_LOCAL_EVIDENCE_WEIGHT_DECAY,
        "dnn_dropout": DNN_LOCAL_EVIDENCE_DROPOUT,
        "unary_aux_weight": DNN_E2E_UNARY_AUX_WEIGHT,
        "prior_weight": DNN_E2E_PRIOR_WEIGHT,
        "validation_dnn_unary_rmse": rmse_score(
            y_val, validation_result["mlp_pred_test"]
        ),
        "validation_dnn_unary_r2": r2_score(y_val, validation_result["mlp_pred_test"]),
        "validation_rain_rmse": rmse_score(
            y_val, validation_result["rain_joint_pred_test"]
        ),
        "validation_rain_r2": r2_score(y_val, validation_result["rain_joint_pred_test"]),
    }])
    validation_results.to_csv(DNN_RESULTS_DIR / "validation_selected_params.csv", index=False)

    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)

    pred = pd.DataFrame({
        "id": test_df["id"].to_numpy(),
        "station_id": test_df["station_id"].to_numpy(),
        "longitude": test_df["longitude"].to_numpy(),
        "latitude": test_df["latitude"].to_numpy(),
        "proj_x": test_df["proj_x"].to_numpy(),
        "proj_y": test_df["proj_y"].to_numpy(),
        "y_true": y_test,
    })

    # Final strict DNN-RAIN protocol:
    # 1) use validation only to check the DNN end-to-end protocol;
    # 2) merge train+validation for the final observed response field;
    # 3) pretrain the SE-DNN local evidence on observed nodes only;
    # 4) jointly update DNN unary and differentiable MF-RNN parameters with
    #    the final RAIN loss through abs-normalized GWR alpha_ij.
    # The test labels are used only for metrics after RAIN_pred is fixed.
    final_observed = pd.concat([train_df, val_df], ignore_index=True)
    model_save_path = DNN_MODEL_DIR / "author_split_strict_dnn_e2e_rain.pth"
    result = run_strict_dnn_e2e_rain(
        observed_df=final_observed,
        query_df=test_df,
        model_save_path=str(model_save_path),
    )
    pred["RAIN_global_pred"] = result["mlp_pred_test"]
    pred["RAIN_pred"] = result["rain_joint_pred_test"]

    graph = result["graph"]
    learned_eta = float(np.mean([p["eta"] for p in result["member_params"]]))
    learned_lambda = float(np.mean([p["lambda_pair"] for p in result["member_params"]]))
    protocol = pd.DataFrame([{
        "model_family": "strict_dnn_e2e_rain",
        "local_evidence_module": "SE-DNN ensemble jointly optimized with MF-RNN",
        "dnn_auxiliary_features": "covariates+coordinates+observed-node spatial evidence+auxiliary GWR basis",
        "dnn_seeds": "-".join(str(v) for v in DNN_LOCAL_EVIDENCE_SEEDS),
        "dnn_pretrain_epochs": DNN_LOCAL_EVIDENCE_EPOCHS,
        "joint_epochs": DNN_E2E_JOINT_EPOCHS,
        "dnn_lr": DNN_LOCAL_EVIDENCE_LR,
        "joint_lr": DNN_E2E_JOINT_LR,
        "dnn_weight_decay": DNN_LOCAL_EVIDENCE_WEIGHT_DECAY,
        "dnn_dropout": DNN_LOCAL_EVIDENCE_DROPOUT,
        "unary_aux_weight": DNN_E2E_UNARY_AUX_WEIGHT,
        "prior_weight": DNN_E2E_PRIOR_WEIGHT,
        "auxiliary_gwr_bw": PAPER_RAIN_UNARY_BW,
        "auxiliary_gwr_kernel": PAPER_RAIN_UNARY_KERNEL,
        "pairwise_potential": graph.get("pairwise_potential", "GWR projection coefficient"),
        "select_k": SELECT_K,
        "gwr_k": GWR_K,
        "initial_eta": DNN_RAIN_ETA,
        "initial_lambda_pair": DNN_RAIN_LAMBDA,
        "mean_learned_eta": learned_eta,
        "mean_learned_lambda_pair": learned_lambda,
        "mfrnn_iters": 1,
        "renormalize_alpha_in_mfrnn": PAPER_RAIN_RENORMALIZE_ALPHA,
        "dnn_local_evidence": True,
        "strict_joint_backprop": True,
        "uses_test_labels_for_prediction": False,
        "uses_calibration": False,
    }])
    protocol.to_csv(DNN_RESULTS_DIR / "strict_dnn_e2e_rain_protocol.csv", index=False)

    export_pred = pred[
        [
            "id", "station_id", "longitude", "latitude", "proj_x", "proj_y",
            "y_true", "RAIN_global_pred", "RAIN_pred",
        ]
    ].copy()
    export_pred.to_csv(DNN_RESULTS_DIR / "rain_regression_predictions_dnn_e2e.csv", index=False)

    metrics = pd.DataFrame([
        {"model": "RAIN", **regression_metrics(y_test, pred["RAIN_pred"].to_numpy())}
    ])
    metrics.to_csv(DNN_RESULTS_DIR / "author_split_metrics_dnn_e2e.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"Saved predictions: {DNN_RESULTS_DIR / 'rain_regression_predictions_dnn_e2e.csv'}")
    print(f"Saved metrics: {DNN_RESULTS_DIR / 'author_split_metrics_dnn_e2e.csv'}")
    print(f"Saved strict DNN-RAIN protocol: {DNN_RESULTS_DIR / 'strict_dnn_e2e_rain_protocol.csv'}")
    print(
        "Final strict DNN-RAIN prediction uses neural local evidence jointly "
        "optimized with the differentiable MF-RNN: "
        "pred['RAIN_unary_pred']=result['mlp_pred_test'] and "
        "pred['RAIN_pred']=result['rain_joint_pred_test']. The MF-RNN uses "
        "abs-normalized GWR projection alpha_ij. No calibration or test-label "
        "fitting is used, and the previous non-DNN version is not overwritten."
    )


if __name__ == "__main__":
    main()

