"""
Phase 3: LightGBM GBDT Classification & Macro F_0.5 Threshold Optimization Module.

Trains gradient boosted decision tree classifier on SIMD pairwise features,
performs leak-free group split validation, and executes 2D threshold grid search
to maximize instance-level Macro F_0.5 score on business entity linkages.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
        s1_id = row["source1_entity_id"]
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
                # Correct singleton prediction
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                # False positive link on singleton
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
) -> Tuple[float, float, float, float, float]:
    """
    2D Grid Search over link and singleton thresholds to maximize Macro F_0.5.
    
    Returns:
        (best_th_link, best_th_singleton, best_macro_f05, best_prec, best_rec)
    """
    val_meta_df = val_meta_df.copy()
    val_meta_df["prob"] = val_probs

    # Group predictions by s1
    s1_grouped = val_meta_df.groupby("source1_entity_id")
    s1_cand_data = {
        s1_id: (group["target_entity_id"].tolist(), group["prob"].to_numpy())
        for s1_id, group in s1_grouped
    }

    th_link_range = np.linspace(0.40, 0.85, 10)
    th_singleton_range = np.linspace(0.50, 0.90, 9)

    best_f05 = -1.0
    best_link = 0.50
    best_sing = 0.50
    best_p = 0.0
    best_r = 0.0

    for th_sing in th_singleton_range:
        for th_link in th_link_range:
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

            if f05 > best_f05:
                best_f05 = f05
                best_link = float(th_link)
                best_sing = float(th_sing)
                best_p = p
                best_r = r

    return best_link, best_sing, best_f05, best_p, best_r


class BERClassifier:
    """
    End-to-end LightGBM GBDT pairwise classification and inference model.
    """

    def __init__(
        self,
        learning_rate: float = 0.08,
        num_leaves: int = 31,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        self.model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1,
        )
        self.best_th_link = 0.55
        self.best_th_singleton = 0.65

    def train_and_evaluate(
        self,
        meta_df: pd.DataFrame,
        features_df: pd.DataFrame,
        ground_truth_map: Dict[str, Set[str]],
        test_size: float = 0.20,
    ) -> Dict[str, float]:
        """
        Execute leak-free grouped split training and threshold tuning.
        """
        logger.info("Constructing binary labels for candidate pairs...")
        # Label 1 if target_id in ground_truth_map[s1_id] else 0
        labels = [
            1 if target_id in ground_truth_map.get(s1_id, set()) else 0
            for s1_id, target_id in zip(meta_df["source1_entity_id"], meta_df["target_entity_id"])
        ]
        y = np.array(labels, dtype=np.int32)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        logger.info(f"Feature dataset: {len(y):,} pairs ({n_pos:,} Positives, {n_neg:,} Negatives | Positive Ratio: {n_pos/len(y)*100:.2f}%)")

        # Grouped split on source1_entity_id
        unique_s1 = np.array(sorted(meta_df["source1_entity_id"].unique()))
        train_s1, val_s1 = train_test_split(unique_s1, test_size=test_size, random_state=42)
        train_s1_set = set(train_s1)
        val_s1_set = set(val_s1)

        train_mask = meta_df["source1_entity_id"].isin(train_s1_set).to_numpy()
        val_mask = meta_df["source1_entity_id"].isin(val_s1_set).to_numpy()

        X_train, y_train = features_df[train_mask], y[train_mask]
        X_val, y_val = features_df[val_mask], y[val_mask]
        val_meta = meta_df[val_mask].reset_index(drop=True)

        logger.info(f"Grouped Split: {len(train_s1):,} Train S1 ({len(X_train):,} pairs), {len(val_s1):,} Val S1 ({len(X_val):,} pairs)")

        logger.info("Training LightGBM Classifier with early stopping...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        val_probs = self.model.predict_proba(X_val)[:, 1]

        # Default 0.5 Cutoff Metric
        default_preds: Dict[str, List[str]] = {}
        val_meta_prob = val_meta.copy()
        val_meta_prob["prob"] = val_probs
        for s1_id, group in val_meta_prob.groupby("source1_entity_id"):
            valid = group[group["prob"] >= 0.50]["target_entity_id"].tolist()
            default_preds[s1_id] = valid
        def_f05, def_p, def_r = compute_instance_macro_f05(default_preds, ground_truth_map, val_s1)
        logger.info(f"Baseline (Threshold=0.50): Macro F_0.5 = {def_f05:.4f} (Prec: {def_p:.4f}, Rec: {def_r:.4f})")

        # 2D Grid Search
        logger.info("Executing 2D Threshold Optimization Grid Search for Macro F_0.5...")
        best_link, best_sing, opt_f05, opt_p, opt_r = grid_search_thresholds(
            val_meta, val_probs, val_s1, ground_truth_map
        )
        self.best_th_link = best_link
        self.best_th_singleton = best_sing

        # Feature importances
        importances = self.model.feature_importances_
        imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances}).sort_values(
            by="importance", ascending=False
        )
        top_features_str = ", ".join([f"{r['feature']} ({r['importance']:.1f})" for _, r in imp_df.head(5).iterrows()])
        logger.info(f"Top 5 Features by Gain: {top_features_str}")

        print("\n" + "=" * 70)
        print("PHASE 3 CLASSIFICATION & THRESHOLD TUNING REPORT")
        print("=" * 70)
        print(f"Validation S1 Entities:           {len(val_s1):,}")
        print(f"Default Cutoff (0.50):            Macro F_0.5 = {def_f05:.4f} (Prec: {def_p:.4f}, Rec: {def_r:.4f})")
        print(f"Optimal Threshold Link:           {best_link:.3f}")
        print(f"Optimal Threshold Singleton:      {best_sing:.3f}")
        print(f"Optimized Macro F_0.5 Score:      {opt_f05:.4f} (+{(opt_f05 - def_f05):.4f} gain)")
        print(f"Optimized Precision:              {opt_p:.4f}")
        print(f"Optimized Recall:                 {opt_r:.4f}")
        print("=" * 70 + "\n")

        return {
            "default_macro_f05": def_f05,
            "optimized_macro_f05": opt_f05,
            "optimized_precision": opt_p,
            "optimized_recall": opt_r,
            "best_th_link": best_link,
            "best_th_singleton": best_sing,
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


def export_matching_results(predictions_map: Dict[str, List[str]], all_s1_ids: Sequence[str], output_path: str) -> None:
    """
    Export final matching_results.tsv strictly conforming to the competition specification.
    Header: source1_entity_id\tmatched_entity_ids
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info(f"Writing {len(all_s1_ids):,} matching rows to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = predictions_map.get(s1_id, [])
            match_str = ",".join(matches)
            f.write(f"{s1_id}\t{match_str}\n")
    logger.info(f"Export completed: {output_path}")


def run_phase3_pipeline(
    s1_path: str = "sample_data/sample_source1.tsv",
    s2_path: str = "sample_data/sample_source2.tsv",
    s3_path: str = "sample_data/sample_source3.tsv",
    gt_path: str = "sample_data/sample_ground_truth.tsv",
    candidate_tsv_path: str = "output/candidate_pairs.tsv",
    output_matching_path: str = "output/matching_results.tsv",
) -> Dict[str, float]:
    """
    Execute complete Phase 3 pipeline on candidate pairs.
    """
    t0 = time.time()
    logger.info("Loading preprocessed source data for feature computation...")
    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", keep_default_na=False)
    
    s1_prep = preprocess_dataframe(df_s1)
    s2_prep = preprocess_dataframe(df_s2)
    s3_prep = preprocess_dataframe(df_s3)
    target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

    # Read candidate pairs
    logger.info(f"Reading candidate pairs from {candidate_tsv_path}...")
    cand_df = pd.read_csv(candidate_tsv_path, sep="\t", keep_default_na=False)
    cand_dict: Dict[str, List[str]] = {}
    for _, row in cand_df.iterrows():
        s1_id = row["source1_entity_id"]
        cands = [c.strip() for c in str(row["candidate_entity_ids"]).split(",") if c.strip()]
        cand_dict[s1_id] = cands

    # Extract features
    logger.info(f"Extracting SIMD features for {len(cand_dict):,} S1 candidates...")
    t_feat = time.time()
    meta_df, features_df = build_candidate_feature_matrix(cand_dict, s1_prep, target_prep)
    logger.info(f"Extracted {len(features_df):,} pairwise feature vectors in {time.time() - t_feat:.2f}s")

    # Parse Ground Truth
    gt_map = parse_ground_truth(gt_path)

    # Train and Optimize
    clf = BERClassifier()
    metrics = clf.train_and_evaluate(meta_df, features_df, gt_map, test_size=0.20)

    # Predict full dataset
    all_s1_ids = df_s1["entity_id"].tolist()
    final_preds = clf.predict_matches(meta_df, features_df, all_s1_ids)
    export_matching_results(final_preds, all_s1_ids, output_matching_path)

    logger.info(f"Total Phase 3 execution time: {time.time() - t0:.2f}s")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Classifier & Threshold Optimization Engine")
    parser.add_argument("--s1", default="sample_data/sample_source1.tsv")
    parser.add_argument("--s2", default="sample_data/sample_source2.tsv")
    parser.add_argument("--s3", default="sample_data/sample_source3.tsv")
    parser.add_argument("--gt", default="sample_data/sample_ground_truth.tsv")
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--out", default="output/matching_results.tsv")
    args = parser.parse_args()

    run_phase3_pipeline(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        gt_path=args.gt,
        candidate_tsv_path=args.candidates,
        output_matching_path=args.out,
    )


if __name__ == "__main__":
    main()
