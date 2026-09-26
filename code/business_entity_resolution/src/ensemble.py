"""
Final Stage: Ensemble Blending Engine for Business Entity Resolution.

Blends probability predictions from:
  - Experiment A: LightGBM (leaf-wise tree scaling)
  - Experiment B: CatBoost (GPU-accelerated symmetric/oblivious trees)
  - Experiment C: XGBoost (histogram-binned tree method)

Weighted Formula:
  P_final = w_A * P_LightGBM + w_B * P_CatBoost + w_C * P_XGBoost
  (Default: w_A=0.30, w_B=0.40, w_C=0.30)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
import pandas as pd

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from classifier import (
        BERClassifier,
        CatBoostBERClassifier,
        export_matching_results,
        compute_instance_macro_f05,
        parse_ground_truth,
    )
except ImportError:
    from .classifier import (
        BERClassifier,
        CatBoostBERClassifier,
        export_matching_results,
        compute_instance_macro_f05,
        parse_ground_truth,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EnsembleBlender:
    """
    Weighted probability ensemble blender for LightGBM, CatBoost, and XGBoost models.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        th_link: float = 0.83,
        th_singleton: float = 0.20,
    ) -> None:
        if weights is None:
            # CatBoost weighted highest due to symmetric tree robustness on noisy text
            self.weights = {"catboost": 0.40, "xgboost": 0.35, "lightgbm": 0.25}
        else:
            total_w = sum(weights.values())
            self.weights = {k: v / total_w for k, v in weights.items()}

        self.th_link = th_link
        self.th_singleton = th_singleton
        self.models: Dict[str, Any] = {}

    def load_model(self, name: str, path: str) -> None:
        """Load serialized model artifact."""
        if not os.path.isfile(path):
            logger.warning(f"Model file for '{name}' not found at {path}. Skipping...")
            return
        logger.info(f"Loading '{name}' model from {path}...")
        try:
            clf = joblib.load(path)
            self.models[name] = clf
            logger.info(f"Loaded '{name}' successfully.")
        except Exception as e:
            logger.error(f"Error loading '{name}' from {path}: {e}")

    def predict_proba_ensemble(self, features_df: pd.DataFrame) -> np.ndarray:
        """Compute weighted probability array across all loaded models."""
        if not self.models:
            raise ValueError("No models loaded in EnsembleBlender.")

        active_weights = {k: self.weights.get(k, 1.0) for k in self.models.keys()}
        total_w = sum(active_weights.values())
        norm_weights = {k: v / total_w for k, v in active_weights.items()}

        logger.info(f"Ensembling across active models: {norm_weights}")
        blended_probs = np.zeros(len(features_df), dtype=np.float32)

        for name, clf in self.models.items():
            w = norm_weights[name]
            p = clf.model.predict_proba(features_df)[:, 1]
            blended_probs += w * p

        return blended_probs

    def predict_matches(
        self,
        meta_df: pd.DataFrame,
        features_df: pd.DataFrame,
        all_s1_ids: Sequence[str],
    ) -> Dict[str, List[str]]:
        """Generate final predictions map using blended probabilities."""
        blended_probs = self.predict_proba_ensemble(features_df)
        meta_with_prob = meta_df.copy()
        meta_with_prob["prob"] = blended_probs

        grouped = meta_with_prob.groupby("source1_entity_id")
        cand_dict = {
            s1_id: (group["target_entity_id"].tolist(), group["prob"].to_numpy())
            for s1_id, group in grouped
        }

        preds_map: Dict[str, List[str]] = {}
        for s1_id in all_s1_ids:
            if s1_id not in cand_dict:
                preds_map[s1_id] = []
                continue

            target_ids, p_arr = cand_dict[s1_id]
            max_p = p_arr.max() if len(p_arr) > 0 else 0.0

            if max_p < self.th_singleton:
                preds_map[s1_id] = []
            else:
                valid_mask = p_arr >= self.th_link
                preds_map[s1_id] = [target_ids[i] for i, valid in enumerate(valid_mask) if valid]

        return preds_map
