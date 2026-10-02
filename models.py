"""
Smog Net GPU Architecture Zoo & Hybrid Stacking Meta-Learner
============================================================
Implements:
  - Baselines: Persistence & Linear Regression
  - Branch A (Tabular/Tree): Random Forest, Gradient Boosting, XGBoost (CUDA accelerated)
  - Branch B (Sequence):
      * SequenceLSTM (2-layer original)
      * BiLSTM+Attention (96 hidden, bidirectional, temporal attention)
      * CNN-BiLSTM+Attention (Conv1d x2 -> BiLSTM -> Temporal Attention)
  - Branch C (Tabular Deep Learning): Tabular MLP with LayerNorm & Cosine Annealing
  - Meta-Learner: Non-negative Ridge regression stacking with 5-fold Out-Of-Fold (OOF)
    training and multiple regularization strengths (alpha in {1, 5, 10, 20}).
  - Supports both log1p-target and raw-target training with automatic inverse transformation.
"""

import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from features import SplitDataset


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    """Ensure strict deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Temporal Attention Pooling Mechanism
# =====================================================================
class TemporalAttention(nn.Module):
    """Soft temporal attention over a sliding window sequence."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        # rnn_out: (batch, seq_len, hidden_dim)
        scores = self.attn(torch.tanh(rnn_out))  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)   # (batch, seq_len, 1)
        context = torch.sum(weights * rnn_out, dim=1)  # (batch, hidden_dim)
        return context


# =====================================================================
# Branch B1: Original 2-Layer LSTM
# =====================================================================
class SmogLSTM(nn.Module):
    """2-layer unidirectional LSTM pooling the final timestep."""
    def __init__(self, input_dim: int = 18, hidden_dim: int = 48, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step).squeeze(-1)


# =====================================================================
# Branch B2: Bidirectional LSTM with Temporal Attention
# =====================================================================
class SmogBiLSTMAttn(nn.Module):
    """Bidirectional LSTM with learned temporal attention pooling (STL-CNN-BiLSTM-AM style)."""
    def __init__(self, input_dim: int = 18, hidden_dim: int = 96, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)
        context = self.attention(out)
        return self.head(context).squeeze(-1)


# =====================================================================
# Branch B3: 1D-CNN + BiLSTM + Attention Hybrid
# =====================================================================
class SmogCNNBiLSTMAttn(nn.Module):
    """Hierarchical spatio-temporal hybrid: 1D temporal convolution + BiLSTM + Attention."""
    def __init__(self, input_dim: int = 18, cnn_channels: int = 64, hidden_dim: int = 96, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, seq_len, channels) -> transpose to (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)
        c1 = self.relu(self.bn1(self.conv1(x_conv)))
        c2 = self.relu(self.bn2(self.conv2(c1)))
        lstm_in = c2.transpose(1, 2)
        out, _ = self.bilstm(lstm_in)
        context = self.attention(out)
        return self.head(context).squeeze(-1)


# =====================================================================
# Branch C: Tabular Deep Learning (MLP with LayerNorm)
# =====================================================================
class SmogTabularMLP(nn.Module):
    """Feed-forward deep network on tabular features with LayerNorm and Dropout."""
    def __init__(self, input_dim: int = 37, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# =====================================================================
# SmogNet Master Pipeline Engine
# =====================================================================
class SmogNetPipeline:
    """
    Unified training and evaluation pipeline for all base learners and stacking meta-learner.
    Supports GPU acceleration, log1p-target conversion, and OOF stacking.
    """
    def __init__(
        self,
        seed: int = 42,
        use_gpu: bool = True,
        use_advanced_branches: bool = True,
        xgb_params: Optional[Dict] = None,
        bilstm_params: Optional[Dict] = None,
    ):
        self.seed = seed
        set_seed(seed)
        self.device = get_device() if use_gpu else torch.device("cpu")
        self.use_advanced_branches = use_advanced_branches
        self.xgb_params = xgb_params or {}
        self.bilstm_params = bilstm_params or {}
        
        self.models: Dict = {}
        self.meta_learner: Ridge = None
        self.meta_alpha: float = 5.0
        
        # Base model catalog
        if use_advanced_branches:
            self.meta_feature_names = [
                "XGBoost",
                "GradientBoosting",
                "RandomForest",
                "TabularMLP",
                "SequenceLSTM",
                "BiLSTM_Attn",
                "CNN_BiLSTM_Attn",
            ]
        else:
            self.meta_feature_names = [
                "XGBoost",
                "GradientBoosting",
                "RandomForest",
                "TabularMLP",
                "SequenceLSTM",
            ]

    def train_baselines_and_trees(self, split: SplitDataset):
        """Train Linear Regression, Random Forest, Gradient Boosting, and XGBoost."""
        y_tr = split.y_train_target

        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(split.X_train_scaled, y_tr)
        self.models["LinearRegression"] = lr

        # 2. Random Forest
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            random_state=self.seed,
            n_jobs=-1,
        )
        rf.fit(split.X_train, y_tr)
        self.models["RandomForest"] = rf

        # 3. Gradient Boosting
        gbr = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            random_state=self.seed,
        )
        gbr.fit(split.X_train, y_tr)
        self.models["GradientBoosting"] = gbr

        # 4. XGBoost (GPU if available)
        xgb_defaults = {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": self.seed,
            "n_jobs": -1,
        }
        if self.device.type == "cuda":
            xgb_defaults["tree_method"] = "hist"
            xgb_defaults["device"] = "cuda"
        xgb_defaults.update(self.xgb_params)

        xgb_model = xgb.XGBRegressor(**xgb_defaults)
        xgb_model.fit(split.X_train, y_tr)
        self.models["XGBoost"] = xgb_model

    def _train_pytorch_model(
        self,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        epochs: int = 35,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> nn.Module:
        """Generic PyTorch training loop with Adam, CosineAnnealing, and validation checkpointing."""
        set_seed(self.seed)
        model = model.to(self.device)
        train_ds = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_weights = None
        val_x_dev = val_x.to(self.device)
        val_y_dev = val_y.to(self.device)

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_x_dev)
                val_loss = criterion(val_pred, val_y_dev).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()
        return model

    def train_deep_models(self, split: SplitDataset, epochs_lstm: int = 30, epochs_mlp: int = 35):
        """Train Tabular MLP, SequenceLSTM, BiLSTM+Attn, and CNN-BiLSTM+Attn on GPU."""
        # 1. Tabular MLP
        train_x_mlp = torch.tensor(split.X_train_scaled, dtype=torch.float32)
        train_y = torch.tensor(split.y_train_target, dtype=torch.float32)
        val_x_mlp = torch.tensor(split.X_val_scaled, dtype=torch.float32)
        val_y = torch.tensor(split.y_val_target, dtype=torch.float32)

        mlp_model = SmogTabularMLP(input_dim=split.X_train_scaled.shape[-1], dropout=0.2)
        self.models["TabularMLP"] = self._train_pytorch_model(
            mlp_model, train_x_mlp, train_y, val_x_mlp, val_y, epochs=epochs_mlp, lr=1e-3
        )

        # 2. Sequence LSTM (Original)
        train_x_seq = torch.tensor(split.X_seq_train, dtype=torch.float32)
        val_x_seq = torch.tensor(split.X_seq_val, dtype=torch.float32)

        lstm_model = SmogLSTM(input_dim=split.X_seq_train.shape[-1], hidden_dim=48, num_layers=2, dropout=0.2)
        self.models["SequenceLSTM"] = self._train_pytorch_model(
            lstm_model, train_x_seq, train_y, val_x_seq, val_y, epochs=epochs_lstm, lr=1e-3
        )

        # Advanced GPU branches
        if self.use_advanced_branches:
            # 3. BiLSTM with Temporal Attention
            hidden_bilstm = self.bilstm_params.get("hidden_dim", 96)
            dropout_bilstm = self.bilstm_params.get("dropout", 0.2)
            lr_bilstm = self.bilstm_params.get("lr", 1e-3)
            
            bilstm_model = SmogBiLSTMAttn(
                input_dim=split.X_seq_train.shape[-1],
                hidden_dim=hidden_bilstm,
                num_layers=2,
                dropout=dropout_bilstm,
            )
            self.models["BiLSTM_Attn"] = self._train_pytorch_model(
                bilstm_model, train_x_seq, train_y, val_x_seq, val_y, epochs=epochs_lstm, lr=lr_bilstm
            )

            # 4. 1D-CNN + BiLSTM with Temporal Attention
            cnn_bilstm_model = SmogCNNBiLSTMAttn(
                input_dim=split.X_seq_train.shape[-1],
                cnn_channels=64,
                hidden_dim=hidden_bilstm,
                dropout=0.2,
            )
            self.models["CNN_BiLSTM_Attn"] = self._train_pytorch_model(
                cnn_bilstm_model, train_x_seq, train_y, val_x_seq, val_y, epochs=epochs_lstm, lr=1e-3
            )

    def predict_base_learner_target_scale(self, model_name: str, split: SplitDataset, subset: str) -> np.ndarray:
        """Predict on target scale (log1p if use_log_target, else real scale)."""
        if subset == "val":
            X_tab = split.X_val
            X_tab_scaled = split.X_val_scaled
            X_seq = split.X_seq_val
        elif subset == "test":
            X_tab = split.X_test
            X_tab_scaled = split.X_test_scaled
            X_seq = split.X_seq_test
        else:
            X_tab = split.X_train
            X_tab_scaled = split.X_train_scaled
            X_seq = split.X_seq_train

        if model_name == "Persistence":
            lag = X_tab["pm25_lag1"].values.astype(np.float32)
            return np.log1p(lag) if split.use_log_target else lag
        elif model_name == "LinearRegression":
            return self.models["LinearRegression"].predict(X_tab_scaled).astype(np.float32)
        elif model_name in ["RandomForest", "GradientBoosting", "XGBoost"]:
            return self.models[model_name].predict(X_tab).astype(np.float32)
        elif model_name == "TabularMLP":
            model = self.models["TabularMLP"]
            model.eval()
            with torch.no_grad():
                tx = torch.tensor(X_tab_scaled, dtype=torch.float32).to(self.device)
                return model(tx).cpu().numpy().astype(np.float32)
        elif model_name in ["SequenceLSTM", "BiLSTM_Attn", "CNN_BiLSTM_Attn"]:
            model = self.models[model_name]
            model.eval()
            with torch.no_grad():
                tx = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
                return model(tx).cpu().numpy().astype(np.float32)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def predict_base_learner(self, model_name: str, split: SplitDataset, subset: str) -> np.ndarray:
        """Predict in real-world PM2.5 units (µg/m³), inverting log1p if necessary."""
        pred = self.predict_base_learner_target_scale(model_name, split, subset)
        if split.use_log_target:
            return np.expm1(pred).clip(min=1.0).astype(np.float32)
        return pred.clip(min=1.0).astype(np.float32)

    def fit_meta_learner_oof(
        self,
        split: SplitDataset,
        n_splits: int = 5,
        alpha: float = 5.0,
    ) -> Dict[str, float]:
        """
        Fit non-negative Ridge regression meta-learner using 5-fold out-of-fold (OOF)
        predictions on the training set. Eliminates single-slice validation overfitting.
        """
        print(f"[Meta-Learner] Generating {n_splits}-fold Out-Of-Fold (OOF) predictions on train set...")
        n_samples = len(split.y_train)
        oof_matrix = np.zeros((n_samples, len(self.meta_feature_names)), dtype=np.float32)

        kf = KFold(n_splits=n_splits, shuffle=False)
        for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(split.X_train)):
            # Fast tree models for OOF stacking
            X_tr, y_tr = split.X_train.iloc[trn_idx], split.y_train_target[trn_idx]
            X_v, y_v = split.X_train.iloc[val_idx], split.y_train_target[val_idx]
            X_tr_sc, X_v_sc = split.X_train_scaled[trn_idx], split.X_train_scaled[val_idx]
            X_seq_tr, X_seq_v = split.X_seq_train[trn_idx], split.X_seq_train[val_idx]

            # Fast sub-models
            sub_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=self.seed, n_jobs=-1)
            sub_xgb.fit(X_tr, y_tr)
            sub_gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=self.seed)
            sub_gbr.fit(X_tr, y_tr)
            sub_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.seed, n_jobs=-1)
            sub_rf.fit(X_tr, y_tr)

            # PyTorch models
            mlp_m = SmogTabularMLP(input_dim=X_tr_sc.shape[-1], dropout=0.2)
            mlp_m = self._train_pytorch_model(
                mlp_m,
                torch.tensor(X_tr_sc, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
                torch.tensor(X_v_sc, dtype=torch.float32),
                torch.tensor(y_v, dtype=torch.float32),
                epochs=20,
            )

            lstm_m = SmogLSTM(input_dim=X_seq_tr.shape[-1], hidden_dim=48, num_layers=2, dropout=0.2)
            lstm_m = self._train_pytorch_model(
                lstm_m,
                torch.tensor(X_seq_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
                torch.tensor(X_seq_v, dtype=torch.float32),
                torch.tensor(y_v, dtype=torch.float32),
                epochs=20,
            )

            oof_preds = [
                sub_xgb.predict(X_v),
                sub_gbr.predict(X_v),
                sub_rf.predict(X_v),
                mlp_m(torch.tensor(X_v_sc, dtype=torch.float32).to(self.device)).cpu().detach().numpy(),
                lstm_m(torch.tensor(X_seq_v, dtype=torch.float32).to(self.device)).cpu().detach().numpy(),
            ]

            if self.use_advanced_branches:
                bilstm_m = SmogBiLSTMAttn(input_dim=X_seq_tr.shape[-1], hidden_dim=64, num_layers=2)
                bilstm_m = self._train_pytorch_model(
                    bilstm_m,
                    torch.tensor(X_seq_tr, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.float32),
                    torch.tensor(X_seq_v, dtype=torch.float32),
                    torch.tensor(y_v, dtype=torch.float32),
                    epochs=20,
                )
                cnn_m = SmogCNNBiLSTMAttn(input_dim=X_seq_tr.shape[-1], cnn_channels=48, hidden_dim=64)
                cnn_m = self._train_pytorch_model(
                    cnn_m,
                    torch.tensor(X_seq_tr, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.float32),
                    torch.tensor(X_seq_v, dtype=torch.float32),
                    torch.tensor(y_v, dtype=torch.float32),
                    epochs=20,
                )
                oof_preds.append(bilstm_m(torch.tensor(X_seq_v, dtype=torch.float32).to(self.device)).cpu().detach().numpy())
                oof_preds.append(cnn_m(torch.tensor(X_seq_v, dtype=torch.float32).to(self.device)).cpu().detach().numpy())

            for j, p in enumerate(oof_preds):
                oof_matrix[val_idx, j] = p

        # Fit non-negative Ridge on OOF matrix
        self.meta_alpha = alpha
        meta = Ridge(alpha=alpha, positive=True, fit_intercept=True)
        meta.fit(oof_matrix, split.y_train_target)
        self.meta_learner = meta

        weights = {name: round(float(w), 4) for name, w in zip(self.meta_feature_names, meta.coef_)}
        weights["Intercept"] = round(float(meta.intercept_), 4)
        return weights

    def predict_smognet(self, split: SplitDataset, subset: str = "test") -> np.ndarray:
        """Generate final stacked ensemble forecast in real-world PM2.5 units (µg/m³)."""
        preds = []
        for name in self.meta_feature_names:
            p = self.predict_base_learner_target_scale(name, split, subset=subset)
            preds.append(p)
        M = np.column_stack(preds)
        stacked_target = self.meta_learner.predict(M).astype(np.float32)

        if split.use_log_target:
            return np.expm1(stacked_target).clip(min=1.0).astype(np.float32)
        return stacked_target.clip(min=1.0).astype(np.float32)
