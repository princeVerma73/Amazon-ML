"""
Evaluate calibration and threshold tuning on benchmark data to maximize Macro F0.5.
Tests:
- th_link in [0.75, 0.95]
- th_singleton in [0.30, 0.90]
- relative_margin in [0.70, 0.95]
- max_matches cap in [3, 4, 5, 6, None]
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from classifier import parse_ground_truth, compute_instance_macro_f05
from feature_extraction import build_candidate_feature_matrix
from normalizer import preprocess_dataframe

def main():
    print("=" * 70)
    print("CALIBRATING PRECISION & SINGLETON DECISION ENGINE")
    print("=" * 70)

    # 1. Load sample data
    gt_map = parse_ground_truth("sample_data/sample_ground_truth.tsv")
    s1_df = pd.read_csv("sample_data/sample_source1.tsv", sep="\t", keep_default_na=False).iloc[:20000]
    s2_df = pd.read_csv("sample_data/sample_source2.tsv", sep="\t", keep_default_na=False)
    s3_df = pd.read_csv("sample_data/sample_source3.tsv", sep="\t", keep_default_na=False)
    cand_df = pd.read_csv("sample_data/sample_candidate_pairs.tsv", sep="\t", keep_default_na=False)

    cand_dict = {
        s1: [c.strip() for c in str(cands).split(",") if c.strip()]
        for s1, cands in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"])
        if s1 in set(s1_df["entity_id"])
    }

    # Preprocess
    s1_prep = preprocess_dataframe(s1_df, show_progress=False)
    s2_prep = preprocess_dataframe(s2_df, show_progress=False)
    s3_prep = preprocess_dataframe(s3_df, show_progress=False)
    target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

    # Build features
    print("Extracting feature matrix for 20k validation queries...")
    meta_df, feats_df = build_candidate_feature_matrix(cand_dict, s1_prep, target_prep, show_progress=False)

    # Load CatBoost model
    clf = joblib.load("models/catboost_ber_model.joblib")
    probs = clf.model.predict_proba(feats_df)[:, 1]
    meta_df["prob"] = probs

    grouped = meta_df.groupby("source1_entity_id")
    entity_data = {
        s1_id: (group["target_entity_id"].tolist(), group["prob"].to_numpy())
        for s1_id, group in grouped
    }
    all_s1_ids = s1_df["entity_id"].tolist()

    print(f"Scored {len(feats_df):,} pairs across {len(entity_data):,} entities.")
    print("\n--- Running Grid Search over Decision Rules ---")

    best_f05 = 0.0
    best_params = None

    for th_sing in [0.30, 0.50, 0.70, 0.80, 0.85]:
        for th_link in [0.75, 0.80, 0.85, 0.88, 0.90, 0.92]:
            for rel_margin in [0.75, 0.80, 0.85, 0.90, 1.0]:
                for max_cap in [4, 5, 6, None]:
                    preds = {}
                    for s1_id in all_s1_ids:
                        if s1_id not in entity_data:
                            preds[s1_id] = []
                            continue
                        t_ids, p_arr = entity_data[s1_id]
                        max_p = p_arr.max() if len(p_arr) > 0 else 0.0

                        if max_p < th_sing:
                            preds[s1_id] = []
                        else:
                            # Apply absolute threshold + relative margin
                            valid_indices = [
                                i for i, p in enumerate(p_arr)
                                if p >= th_link and p >= (rel_margin * max_p)
                            ]
                            # Sort by prob desc
                            valid_indices.sort(key=lambda idx: p_arr[idx], reverse=True)
                            if max_cap is not None:
                                valid_indices = valid_indices[:max_cap]
                            preds[s1_id] = [t_ids[idx] for idx in valid_indices]

                    f05, prec, rec = compute_instance_macro_f05(preds, gt_map, all_s1_ids)
                    if f05 > best_f05:
                        best_f05 = f05
                        best_params = (th_sing, th_link, rel_margin, max_cap, f05, prec, rec)
                        print(
                            f"-> NEW BEST: th_sing={th_sing:.2f}, th_link={th_link:.2f}, "
                            f"rel_margin={rel_margin:.2f}, max_cap={max_cap} "
                            f"=> F0.5 = {f05:.4f} (Prec: {prec:.4f}, Rec: {rec:.4f})"
                        )

    print("\n" + "=" * 70)
    print("OPTIMAL CALIBRATION PARAMETERS:")
    print(f"  th_singleton:    {best_params[0]}")
    print(f"  th_link:         {best_params[1]}")
    print(f"  relative_margin: {best_params[2]}")
    print(f"  max_cap:         {best_params[3]}")
    print(f"  Optimal F0.5:    {best_params[4]:.4f}")
    print(f"  Precision:       {best_params[5]:.4f}")
    print(f"  Recall:          {best_params[6]:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
