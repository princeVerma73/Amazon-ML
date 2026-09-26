"""
Step 3: XGBoost Classification, Live Progress Logging, and 2D Dual-Threshold Grid Search.

Trains an XGBoost GBDT classifier on pairwise feature representations, logs live iteration loss,
auto-checkpoints the optimal model based on validation loss, and executes 2D threshold grid search
over link thresholds [0.65, 0.85] and singleton floors [0.20, 0.40] to maximize instance-level Macro F0.5.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

try:
    from .feature_extraction import (
        FEATURE_COLUMNS,
        build_candidate_feature_matrix,
    )
    from .normalizer import preprocess_dataframe
except (ImportError, ValueError):
    from feature_extraction import (
        FEATURE_COLUMNS,
        build_candidate_feature_matrix,
    )
    from normalizer import preprocess_dataframe

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Parse ground truth TSV into mapping of source1_entity_id -> set of true matching target IDs."""
    gt_df = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_m = str(row["matched_entity_ids"]).strip()
        matches = set(m.strip() for m in raw_m.split(",") if m.strip())
        gt_map[s1_id] = matches
    return gt_map


def compute_instance_macro_f05(
    predictions_map: Dict[str, List[str]],
    ground_truth_map: Dict[str, Set[str]],
    all_s1_ids: Sequence[str],
) -> Tuple[float, float, float]:
    """
    Compute competition Macro F_0.5 score, Precision, and Recall across all S1 entities.
    
    Returns:
        (macro_f05, macro_precision, macro_recall)
    """
    f05_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    for s1_id in all_s1_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = set(predictions_map.get(s1_id, []))

        # Case 1: Ground truth is singleton (0 matches)
        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(1.0)
            continue

        # Case 2: Ground truth has matches, prediction is empty
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        # Case 3: Both have matches
        tp = len(true_set.intersection(pred_set))
        p = tp / len(pred_set)
        r = tp / len(true_set)

        denom = (0.25 * p) + r
        f05 = (1.25 * p * r) / denom if denom > 0 else 0.0

        f05_scores.append(f05)
        precisions.append(p)
        recalls.append(r)

    return (
        float(np.mean(f05_scores)),
        float(np.mean(precisions)),
        float(np.mean(recalls)),
    )


def grid_search_thresholds(
    val_meta_df: pd.DataFrame,
    val_probs: np.ndarray,
    val_s1_ids: Sequence[str],
    ground_truth_map: Dict[str, Set[str]],
    link_range: Optional[np.ndarray] = None,
    singleton_range: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float, List[Dict[str, float]]]:
    """
    2D Grid Search over link and singleton thresholds to maximize Macro F_0.5.
    
    Iterates over link_threshold in [0.65, 0.85] (step 0.02) and singleton_floor in [0.20, 0.40] (step 0.05).
    Prints live step progress and returns top candidate threshold combinations.
    """
    val_meta_df = val_meta_df.copy()
    val_meta_df["prob"] = val_probs

    s1_grouped = val_meta_df.groupby("source1_entity_id")
    s1_cand_data = {
        s1_id: (group["target_entity_id"].tolist(), group["prob"].to_numpy())
        for s1_id, group in s1_grouped
    }

    if link_range is None:
        link_range = np.arange(0.65, 0.8501, 0.02)
    if singleton_range is None:
        singleton_range = np.arange(0.20, 0.4001, 0.05)

    total_steps = len(link_range) * len(singleton_range)
    all_results: List[Dict[str, float]] = []

    print("\n" + "=" * 75)
    print(f"STARTING 2D DUAL-THRESHOLD GRID SEARCH ({total_steps} COMBINATIONS)")
    print(f"  Link Threshold Range:      [{link_range[0]:.2f}, {link_range[-1]:.2f}] (step 0.02)")
    print(f"  Singleton Floor Range:     [{singleton_range[0]:.2f}, {singleton_range[-1]:.2f}] (step 0.05)")
    print("=" * 75)

    step = 0
    for th_sing in singleton_range:
        for th_link in link_range:
            step += 1
            preds_map: Dict[str, List[str]] = {}

            for s1_id in val_s1_ids:
                if s1_id not in s1_cand_data:
                    preds_map[s1_id] = []
                    continue

                target_ids, probs = s1_cand_data[s1_id]
                max_prob = probs.max() if len(probs) > 0 else 0.0

                if max_prob < th_sing:
                    preds_map[s1_id] = []
                else:
                    valid_mask = probs >= th_link
                    preds_map[s1_id] = [target_ids[i] for i, valid in enumerate(valid_mask) if valid]

            f05, p, r = compute_instance_macro_f05(preds_map, ground_truth_map, val_s1_ids)
            result = {
                "step": step,
                "th_link": float(th_link),
                "th_singleton": float(th_sing),
                "macro_f05": float(f05),
                "precision": float(p),
                "recall": float(r),
            }
            all_results.append(result)

            print(
                f"[Step {step:2d}/{total_steps}] link={th_link:.2f}, singleton_floor={th_sing:.2f} "
                f"-> Macro F0.5 = {f05:.4f} (Prec: {p:.4f}, Rec: {r:.4f})"
            )

    all_results.sort(key=lambda x: x["macro_f05"], reverse=True)

    print("\n" + "=" * 75)
    print("TOP 3 BEST CANDIDATE THRESHOLD COMBINATIONS")
    print("=" * 75)
    for rank, res in enumerate(all_results[:3], start=1):
        status = "[BEST - LOCKED]" if rank == 1 else ""
        print(
            f"Rank {rank}: link_threshold={res['th_link']:.2f}, singleton_floor={res['th_singleton']:.2f} "
            f"-> Macro F0.5 = {res['macro_f05']:.4f} (Prec: {res['precision']:.4f}, Rec: {res['recall']:.4f}) {status}"
        )
    print("=" * 75 + "\n")

    best = all_results[0]
    return (
        best["th_link"],
        best["th_singleton"],
        best["macro_f05"],
        best["precision"],
        best["recall"],
        all_results[:3],
    )


class BERClassifier:
    """
    Pairwise classification and inference engine utilizing XGBoost GBDT
    with automatic model checkpointing and 2D threshold optimization.
    """

    __module__ = "classifier"

    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.04,
        max_depth: int = 6,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        tree_method: str = "hist",
        scale_pos_weight: float = 0.6,
        eval_metric: str = "logloss",
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 40,
    ) -> None:
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            scale_pos_weight=scale_pos_weight,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.best_th_link = 0.75
        self.best_th_singleton = 0.30
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.training_duration_seconds_: float = 0.0

    def train_and_evaluate(
        self,
        meta_df: pd.DataFrame,
        features_df: pd.DataFrame,
        ground_truth_map: Dict[str, Set[str]],
        test_size: float = 0.20,
        checkpoint_path: str = "models/xgb_ber_model.joblib",
    ) -> Dict[str, Any]:
        """
        Execute leak-free grouped split training with live XGBoost logging,
        auto-checkpointing the best model, and 2D threshold optimization.
        """
        train_start = time.time()
        logger.info("Constructing binary labels for candidate pairs...")
        labels = [
            1 if target_id in ground_truth_map.get(s1_id, set()) else 0
            for s1_id, target_id in zip(meta_df["source1_entity_id"], meta_df["target_entity_id"])
        ]
        y = np.array(labels, dtype=np.int32)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        logger.info(
            f"Feature dataset: {len(y):,} pairs ({n_pos:,} Positives, {n_neg:,} Negatives | "
            f"Positive Ratio: {n_pos/len(y)*100:.2f}%)"
        )

        # Leak-free grouped split strictly on source1_entity_id
        unique_s1 = np.array(sorted(meta_df["source1_entity_id"].unique()))
        train_s1, val_s1 = train_test_split(unique_s1, test_size=test_size, random_state=42)
        train_s1_set = set(train_s1)
        val_s1_set = set(val_s1)

        train_mask = meta_df["source1_entity_id"].isin(train_s1_set).to_numpy()
        val_mask = meta_df["source1_entity_id"].isin(val_s1_set).to_numpy()

        X_train, y_train = features_df[train_mask], y[train_mask]
        X_val, y_val = features_df[val_mask], y[val_mask]
        val_meta = meta_df[val_mask].reset_index(drop=True)

        logger.info(
            f"Grouped Split: {len(train_s1):,} Train S1 ({len(X_train):,} pairs), "
            f"{len(val_s1):,} Val S1 ({len(X_val):,} pairs)"
        )

        logger.info("\nStarting XGBClassifier training with live iteration loss logging (verbose=50)...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )

        self.training_duration_seconds_ = time.time() - train_start
        self.best_iteration_ = getattr(self.model, "best_iteration", self.model.n_estimators)
        self.best_score_ = getattr(self.model, "best_score", 0.0)

        logger.info(
            f"\nXGBoost training finished in {self.training_duration_seconds_:.2f}s! "
            f"Optimal Iteration: {self.best_iteration_}, Best Validation Loss: {self.best_score_:.5f}"
        )

        # Auto-checkpoint model
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        joblib.dump(self, checkpoint_path)
        logger.info(f"Auto-checkpointed best model based on validation loss to: {checkpoint_path}")

        # Compute validation probabilities using the best iteration
        val_probs = self.model.predict_proba(X_val)[:, 1]

        # 2D Grid Search over link_threshold [0.65, 0.85] and singleton_floor [0.20, 0.40]
        best_link, best_sing, opt_f05, opt_p, opt_r, top_3 = grid_search_thresholds(
            val_meta, val_probs, val_s1, ground_truth_map
        )
        self.best_th_link = best_link
        self.best_th_singleton = best_sing

        # Re-save checkpoint with locked optimal thresholds
        joblib.dump(self, checkpoint_path)
        logger.info(f"Updated checkpoint with locked optimal thresholds: {checkpoint_path}")

        # Feature importances
        try:
            importances = self.model.feature_importances_
            imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances}).sort_values(
                by="importance", ascending=False
            )
            top_features_str = ", ".join([f"{r['feature']} ({r['importance']:.3f})" for _, r in imp_df.head(5).iterrows()])
            logger.info(f"Top 5 Features by Gain: {top_features_str}")
        except Exception:
            pass

        return {
            "training_duration_seconds": self.training_duration_seconds_,
            "optimal_iteration": self.best_iteration_,
            "best_validation_logloss": self.best_score_,
            "best_th_link": self.best_th_link,
            "best_th_singleton": self.best_th_singleton,
            "optimized_macro_f05": opt_f05,
            "optimized_precision": opt_p,
            "optimized_recall": opt_r,
            "top_3_candidates": top_3,
            "checkpoint_path": checkpoint_path,
        }

    def predict_matches(
        self,
        meta_df: pd.DataFrame,
        features_df: pd.DataFrame,
        all_s1_ids: Sequence[str],
    ) -> Dict[str, List[str]]:
        """Generate final predictions map using learned optimal thresholds."""
        probs = self.model.predict_proba(features_df)[:, 1]
        meta_with_prob = meta_df.copy()
        meta_with_prob["prob"] = probs

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

            if max_p < self.best_th_singleton:
                preds_map[s1_id] = []
            else:
                valid_mask = p_arr >= self.best_th_link
                preds_map[s1_id] = [target_ids[i] for i, valid in enumerate(valid_mask) if valid]

        return preds_map


class CatBoostBERClassifier:
    """
    Experiment B: Pairwise classification and inference engine utilizing CatBoost GBDT
    accelerated on NVIDIA RTX GPU (task_type='GPU') with oblivious decision trees and
    asymmetric precision loss tuning for Macro F0.5.
    """

    __module__ = "classifier"

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.04,
        depth: int = 6,
        loss_function: str = "Logloss",
        eval_metric: str = "Logloss",
        class_weights: Optional[List[float]] = None,
        task_type: str = "GPU",
        random_seed: int = 42,
        early_stopping_rounds: int = 50,
        verbose: int = 50,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        # Downweight positive class to 0.6 to prioritize precision for Macro F0.5
        self.class_weights = class_weights if class_weights is not None else [1.0, 0.6]
        self.task_type = task_type
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        try:
            import catboost as cb
            self.model = cb.CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                loss_function=self.loss_function,
                eval_metric=self.eval_metric,
                class_weights=self.class_weights,
                task_type=self.task_type,
                random_seed=self.random_seed,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=self.verbose,
            )
        except Exception as e:
            logger.warning(f"CatBoost GPU initialization encountered ({e}). Falling back to multi-threaded CPU...")
            import catboost as cb
            self.task_type = "CPU"
            self.model = cb.CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                loss_function=self.loss_function,
                eval_metric=self.eval_metric,
                class_weights=self.class_weights,
                task_type="CPU",
                thread_count=-1,
                random_seed=self.random_seed,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=self.verbose,
            )

        self.best_th_link = 0.83
        self.best_th_singleton = 0.20
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.training_duration_seconds_: float = 0.0

    def train_and_evaluate(
        self,
        meta_df: pd.DataFrame,
        features_df: pd.DataFrame,
        ground_truth_map: Dict[str, Set[str]],
        test_size: float = 0.20,
        checkpoint_path: str = "models/catboost_ber_model.joblib",
    ) -> Dict[str, Any]:
        """
        Train CatBoost with live GPU iteration logging, auto-checkpointing,
        and 2D dual-threshold grid search.
        """
        train_start = time.time()
        logger.info("Constructing binary labels for candidate pairs...")
        labels = [
            1 if target_id in ground_truth_map.get(s1_id, set()) else 0
            for s1_id, target_id in zip(meta_df["source1_entity_id"], meta_df["target_entity_id"])
        ]
        y = np.array(labels, dtype=np.int32)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        logger.info(
            f"Feature dataset: {len(y):,} pairs ({n_pos:,} Positives, {n_neg:,} Negatives | "
            f"Positive Ratio: {n_pos/len(y)*100:.2f}%)"
        )

        unique_s1 = np.array(sorted(meta_df["source1_entity_id"].unique()))
        train_s1, val_s1 = train_test_split(unique_s1, test_size=test_size, random_state=42)
        train_s1_set = set(train_s1)
        val_s1_set = set(val_s1)

        train_mask = meta_df["source1_entity_id"].isin(train_s1_set).to_numpy()
        val_mask = meta_df["source1_entity_id"].isin(val_s1_set).to_numpy()

        X_train, y_train = features_df[train_mask], y[train_mask]
        X_val, y_val = features_df[val_mask], y[val_mask]
        val_meta = meta_df[val_mask].reset_index(drop=True)

        logger.info(
            f"Grouped Split: {len(train_s1):,} Train S1 ({len(X_train):,} pairs), "
            f"{len(val_s1):,} Val S1 ({len(X_val):,} pairs) | Device: {self.task_type}"
        )

        logger.info(f"\nStarting CatBoost training on {self.task_type} (iterations={self.iterations}, depth={self.depth})...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=self.verbose,
        )

        self.training_duration_seconds_ = time.time() - train_start
        self.best_iteration_ = getattr(self.model, "get_best_iteration", lambda: self.iterations)()
        best_score_dict = getattr(self.model, "get_best_score", lambda: {})()
        self.best_score_ = best_score_dict.get("validation", {}).get("Logloss", 0.0)

        logger.info(
            f"\nCatBoost training finished in {self.training_duration_seconds_:.2f}s! "
            f"Optimal Iteration: {self.best_iteration_}, Best Validation Loss: {self.best_score_:.5f}"
        )

        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        joblib.dump(self, checkpoint_path)
        logger.info(f"Auto-checkpointed model to: {checkpoint_path}")

        val_probs = self.model.predict_proba(X_val)[:, 1]

        best_link, best_sing, opt_f05, opt_p, opt_r, top_3 = grid_search_thresholds(
            val_meta, val_probs, val_s1, ground_truth_map
        )
        self.best_th_link = best_link
        self.best_th_singleton = best_sing

        joblib.dump(self, checkpoint_path)
        logger.info(f"Updated CatBoost checkpoint with locked optimal thresholds: {checkpoint_path}")

        try:
            importances = self.model.get_feature_importance()
            imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances}).sort_values(
                by="importance", ascending=False
            )
            top_features_str = ", ".join([f"{r['feature']} ({r['importance']:.3f})" for _, r in imp_df.head(5).iterrows()])
            logger.info(f"Top 5 CatBoost Features by Importance: {top_features_str}")
        except Exception:
            pass

        return {
            "training_duration_seconds": self.training_duration_seconds_,
            "optimal_iteration": self.best_iteration_,
            "best_validation_logloss": self.best_score_,
            "best_th_link": self.best_th_link,
            "best_th_singleton": self.best_th_singleton,
            "optimized_macro_f05": opt_f05,
            "optimized_precision": opt_p,
            "optimized_recall": opt_r,
            "top_3_candidates": top_3,
            "checkpoint_path": checkpoint_path,
        }

    def predict_matches(
        self,
        meta_df: pd.DataFrame,
        features_df: pd.DataFrame,
        all_s1_ids: Sequence[str],
    ) -> Dict[str, List[str]]:
        """Generate final predictions map using CatBoost and learned optimal thresholds."""
        probs = self.model.predict_proba(features_df)[:, 1]
        meta_with_prob = meta_df.copy()
        meta_with_prob["prob"] = probs

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

            if max_p < self.best_th_singleton:
                preds_map[s1_id] = []
            else:
                valid_mask = p_arr >= self.best_th_link
                preds_map[s1_id] = [target_ids[i] for i, valid in enumerate(valid_mask) if valid]

        return preds_map


def export_matching_results(
    predictions_map: Dict[str, List[str]],
    all_s1_ids: Sequence[str],
    output_path: str = "output/matching_results.tsv",
) -> None:
    """Export predictions TSV satisfying strict competition schema."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched_list = predictions_map.get(s1_id, [])
            match_str = ",".join(matched_list)
            f.write(f"{s1_id}\t{match_str}\n")
    logger.info(f"Exported {len(all_s1_ids):,} predictions to: {output_path}")


def run_phase3_pipeline(
    s1_path: str = "sample_data/sample_source1.tsv",
    s2_path: str = "sample_data/sample_source2.tsv",
    s3_path: str = "sample_data/sample_source3.tsv",
    gt_path: str = "sample_data/sample_ground_truth.tsv",
    candidate_tsv_path: Optional[str] = None,
    output_matching_path: str = "output/matching_results.tsv",
    checkpoint_path: str = "models/xgb_ber_model.joblib",
) -> Dict[str, Any]:
    """
    Execute complete Step 3 training and threshold optimization on sample dataset.
    """
    t0 = time.time()
    logger.info("=" * 75)
    logger.info("STEP 3: CLASSIFIER TRAINING & 2D DUAL-THRESHOLD OPTIMIZATION")
    logger.info("=" * 75)

    # 1. Load source data
    logger.info("Loading preprocessed source data...")
    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", keep_default_na=False)

    s1_prep = preprocess_dataframe(df_s1)
    s2_prep = preprocess_dataframe(df_s2)
    s3_prep = preprocess_dataframe(df_s3)
    target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

    # 2. Candidate pairs mapping
    if candidate_tsv_path is None or not os.path.isfile(candidate_tsv_path):
        if os.path.isfile("sample_data/sample_candidate_pairs.tsv"):
            candidate_tsv_path = "sample_data/sample_candidate_pairs.tsv"
        elif os.path.isfile("output/candidate_pairs.tsv"):
            candidate_tsv_path = "output/candidate_pairs.tsv"

    if candidate_tsv_path and os.path.isfile(candidate_tsv_path):
        logger.info(f"Reading candidate pairs from {candidate_tsv_path}...")
        cand_df = pd.read_csv(candidate_tsv_path, sep="\t", keep_default_na=False)
        cand_dict: Dict[str, List[str]] = {}
        for _, row in cand_df.iterrows():
            s1_id = row["source1_entity_id"]
            cands = [c.strip() for c in str(row["candidate_entity_ids"]).split(",") if c.strip()]
            cand_dict[s1_id] = cands
    else:
        logger.info("Candidate TSV not found. Generating candidates via blocker...")
        from blocking import DynamicTFIDFBlocker
        blocker = DynamicTFIDFBlocker(top_k=20, min_similarity=0.10)
        cand_dict = blocker.generate_candidates(df_s1, df_s2, df_s3)

    # 3. Extract features with live progress bar
    logger.info(f"Extracting SIMD features for {len(cand_dict):,} S1 queries with live progress...")
    t_feat = time.time()
    meta_df, features_df = build_candidate_feature_matrix(cand_dict, s1_prep, target_prep)
    logger.info(f"Extracted {len(features_df):,} pairwise feature vectors in {time.time() - t_feat:.2f}s")

    # 4. Parse Ground Truth
    gt_map = parse_ground_truth(gt_path)

    # 5. Train XGBoost and Optimize 2D Thresholds
    clf = BERClassifier()
    metrics = clf.train_and_evaluate(
        meta_df=meta_df,
        features_df=features_df,
        ground_truth_map=gt_map,
        test_size=0.20,
        checkpoint_path=checkpoint_path,
    )

    # 6. Predict full dataset
    all_s1_ids = df_s1["entity_id"].tolist()
    final_preds = clf.predict_matches(meta_df, features_df, all_s1_ids)
    export_matching_results(final_preds, all_s1_ids, output_matching_path)

    logger.info(f"Total Step 3 execution time: {time.time() - t0:.2f}s")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3 XGBoost Classifier & 2D Threshold Tuning Engine")
    parser.add_argument("--s1", default="sample_data/sample_source1.tsv")
    parser.add_argument("--s2", default="sample_data/sample_source2.tsv")
    parser.add_argument("--s3", default="sample_data/sample_source3.tsv")
    parser.add_argument("--gt", default="sample_data/sample_ground_truth.tsv")
    parser.add_argument("--candidates", default="sample_data/sample_candidate_pairs.tsv")
    parser.add_argument("--out", default="output/matching_results.tsv")
    parser.add_argument("--checkpoint", default="models/xgb_ber_model.joblib")
    args = parser.parse_args()

    run_phase3_pipeline(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        gt_path=args.gt,
        candidate_tsv_path=args.candidates,
        output_matching_path=args.out,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
