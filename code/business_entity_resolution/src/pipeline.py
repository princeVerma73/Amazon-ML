"""
Phase 4: End-to-End Test Pipeline Orchestrator & Submission Generator.

Coordinates streaming preprocessing, dynamic country-partitioned TF-IDF blocking,
SIMD feature extraction, LightGBM GBDT scoring, and chunked disk flushing to output
strict competition-compliant candidate_pairs.tsv and matching_results.tsv.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .blocking import DynamicTFIDFBlocker, evaluate_candidate_recall, export_candidate_pairs
    from .classifier import (
        BERClassifier,
        compute_instance_macro_f05,
        export_matching_results,
        parse_ground_truth,
    )
    from .feature_extraction import build_candidate_feature_matrix
    from .normalizer import preprocess_dataframe
except (ImportError, ValueError):
    from blocking import DynamicTFIDFBlocker, evaluate_candidate_recall, export_candidate_pairs
    from classifier import (
        BERClassifier,
        compute_instance_macro_f05,
        export_matching_results,
        parse_ground_truth,
    )
    from feature_extraction import build_candidate_feature_matrix
    from normalizer import preprocess_dataframe

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_and_cache_model(
    sample_dir: str = "sample_data",
    model_save_path: str = "models/xgb_ber_model.joblib",
) -> BERClassifier:
    """
    Train and serialize the production XGBoost classifier on the benchmark sample dataset.
    """
    if os.path.isfile(model_save_path):
        logger.info(f"Checking cached trained model from {model_save_path}...")
        try:
            clf = joblib.load(model_save_path)
            if hasattr(clf, "model"):
                logger.info(
                    f"Model loaded successfully (th_link={clf.best_th_link:.3f}, th_singleton={clf.best_th_singleton:.3f})"
                )
                return clf
        except Exception as e:
            logger.warning(f"Failed to load cached model ({e}). Retraining...")

    logger.info(f"Training production model using benchmark data in '{sample_dir}'...")
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)

    s1_path = os.path.join(sample_dir, "sample_source1.tsv")
    s2_path = os.path.join(sample_dir, "sample_source2.tsv")
    s3_path = os.path.join(sample_dir, "sample_source3.tsv")
    gt_path = os.path.join(sample_dir, "sample_ground_truth.tsv")

    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", keep_default_na=False)

    cand_tsv = os.path.join(sample_dir, "sample_candidate_pairs.tsv")
    if os.path.isfile(cand_tsv):
        cand_df = pd.read_csv(cand_tsv, sep="\t", keep_default_na=False)
        cand_dict = {
            row["source1_entity_id"]: [c.strip() for c in str(row["candidate_entity_ids"]).split(",") if c.strip()]
            for _, row in cand_df.iterrows()
        }
    else:
        blocker = DynamicTFIDFBlocker(top_k=20, min_similarity=0.10, batch_size=5000)
        cand_dict = blocker.generate_candidates(df_s1, df_s2, df_s3)

    # 2. Preprocess and feature extraction
    s1_prep = preprocess_dataframe(df_s1)
    s2_prep = preprocess_dataframe(df_s2)
    s3_prep = preprocess_dataframe(df_s3)
    target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

    meta_df, feats_df = build_candidate_feature_matrix(cand_dict, s1_prep, target_prep)
    gt_map = parse_ground_truth(gt_path)

    # 3. Train classifier
    clf = BERClassifier()
    clf.train_and_evaluate(meta_df, feats_df, gt_map, test_size=0.20, checkpoint_path=model_save_path)
    return clf


def stream_scoring_and_export(
    candidates_dict: Dict[str, List[str]],
    s1_prep: pd.DataFrame,
    target_prep: pd.DataFrame,
    clf: BERClassifier,
    output_matching_path: str,
    chunk_size: int = 20000,
) -> Dict[str, List[str]]:
    """
    Stream candidate scoring in memory-safe chunks and flush directly to disk.
    
    Args:
        candidates_dict: Mapping of source1_entity_id -> candidate target IDs.
        s1_prep: Preprocessed Source 1 DataFrame.
        target_prep: Combined Preprocessed Target DataFrame (S2 + S3).
        clf: Trained BERClassifier with learned thresholds.
        output_matching_path: Destination path for matching_results.tsv.
        chunk_size: Number of S1 entities to score per chunk to bound peak RAM.
        
    Returns:
        Full predictions map for evaluation or telemetry.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_matching_path)), exist_ok=True)
    all_s1_ids = s1_prep["entity_id"].tolist()
    total_entities = len(all_s1_ids)
    total_chunks = (total_entities + chunk_size - 1) // chunk_size
    all_predictions: Dict[str, List[str]] = {}

    th_link = clf.best_th_link
    th_sing = clf.best_th_singleton

    # Build target map once for instantaneous O(1) feature lookup across all chunks
    logger.info("Indexing target records for high-speed candidate feature lookup...")
    t_idx = time.time()
    t_ids = target_prep["entity_id"].to_numpy()
    t_cnames = target_prep["clean_name"].to_numpy()
    t_caddrs = target_prep["clean_addr"].to_numpy()
    t_miss = (
        target_prep["is_addr_missing"].to_numpy()
        if "is_addr_missing" in target_prep.columns
        else np.zeros(len(t_ids), dtype=np.int32)
    )
    t_rnames = target_prep["raw_name"].to_numpy() if "raw_name" in target_prep.columns else t_cnames
    t_dstems = target_prep["domain_stem"].to_numpy() if "domain_stem" in target_prep.columns else np.array([""] * len(t_ids))

    target_map = {
        t_ids[i]: (t_cnames[i], t_caddrs[i], int(t_miss[i]), t_rnames[i], t_dstems[i])
        for i in range(len(t_ids))
    }
    logger.info(f"Target index built in {time.time() - t_idx:.2f}s ({len(target_map):,} target records).")
    logger.info(
        f"Starting chunked scoring for {total_entities:,} reference entities "
        f"across {total_chunks} chunks (chunk_size={chunk_size:,}, th_link={th_link:.2f}, th_sing={th_sing:.2f})..."
    )

    with open(output_matching_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")

        with tqdm(total=total_entities, desc="Scoring S1 Entities", unit="entities", dynamic_ncols=True) as pbar:
            for chunk_idx in range(0, total_entities, chunk_size):
                chunk_num = (chunk_idx // chunk_size) + 1
                chunk_end = min(chunk_idx + chunk_size, total_entities)
                chunk_s1_ids = all_s1_ids[chunk_idx:chunk_end]
                pbar.set_description(f"Scoring [Chunk {chunk_num}/{total_chunks}]")

                chunk_cand_dict = {
                    s1_id: candidates_dict.get(s1_id, []) for s1_id in chunk_s1_ids
                }

                # Fast slice of s1_prep for chunk
                s1_chunk_prep = s1_prep.iloc[chunk_idx:chunk_end]

                # Build feature matrix using precomputed target_map
                meta_chunk, feats_chunk = build_candidate_feature_matrix(
                    chunk_cand_dict, s1_chunk_prep, target_map=target_map, show_progress=False
                )

                if len(feats_chunk) > 0:
                    probs = clf.model.predict_proba(feats_chunk)[:, 1]
                    meta_chunk["prob"] = probs

                    grouped = meta_chunk.groupby("source1_entity_id")
                    grouped_dict = {
                        s1_id: (group["target_entity_id"].tolist(), group["prob"].to_numpy())
                        for s1_id, group in grouped
                    }
                else:
                    grouped_dict = {}

                # Apply dual thresholds and write rows
                chunk_matches_count = 0
                for s1_id in chunk_s1_ids:
                    if s1_id not in grouped_dict:
                        all_predictions[s1_id] = []
                        f_out.write(f"{s1_id}\t\n")
                        continue

                    target_ids, p_arr = grouped_dict[s1_id]
                    max_p = p_arr.max() if len(p_arr) > 0 else 0.0

                    if max_p < th_sing:
                        matched = []
                    else:
                        valid_mask = p_arr >= th_link
                        matched = [target_ids[i] for i, valid in enumerate(valid_mask) if valid]

                    if matched:
                        chunk_matches_count += 1
                    all_predictions[s1_id] = matched
                    match_str = ",".join(matched)
                    f_out.write(f"{s1_id}\t{match_str}\n")

                pbar.set_postfix(chunk=f"{chunk_num}/{total_chunks}", pairs=f"{len(feats_chunk):,}")
                pbar.update(len(chunk_s1_ids))

    logger.info(f"Chunked scoring and disk export completed: {output_matching_path}")
    return all_predictions


def execute_full_pipeline(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    candidate_out: str = "output/candidate_pairs.tsv",
    matching_out: str = "output/matching_results.tsv",
    model_path: str = "models/xgb_ber_model.joblib",
    gt_path: Optional[str] = None,
    chunk_size: int = 20000,
    top_k: int = 20,
    min_sim: float = 0.10,
    country_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """
    Execute complete end-to-end BER pipeline from raw TSVs to final formatted submissions.
    """
    total_start = time.time()
    logger.info("=" * 75)
    logger.info(f"STARTING BUSINESS ENTITY RESOLUTION END-TO-END PIPELINE")
    logger.info(f"  Source 1:   {s1_path}")
    logger.info(f"  Source 2:   {s2_path}")
    logger.info(f"  Source 3:   {s3_path}")
    logger.info(f"  Candidate:  {candidate_out}")
    logger.info(f"  Matching:   {matching_out}")
    logger.info("=" * 75)

    # 1. Load or train model
    clf = train_and_cache_model(sample_dir="sample_data", model_save_path=model_path)

    # 2. Ingest datasets
    t_load = time.time()
    logger.info(f"Loading input datasets from disk...")
    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", keep_default_na=False)
    logger.info(
        f"Loaded datasets in {time.time() - t_load:.2f}s: "
        f"S1={len(df_s1):,}, S2={len(df_s2):,}, S3={len(df_s3):,}"
    )

    # 3. Preprocess representations once with live tqdm progress
    t_prep = time.time()
    logger.info("Preprocessing Source 1, Source 2, and Source 3 text payloads...")
    s1_prep = preprocess_dataframe(df_s1, desc="Preprocessing S1")
    s2_prep = preprocess_dataframe(df_s2, desc="Preprocessing S2")
    s3_prep = preprocess_dataframe(df_s3, desc="Preprocessing S3")
    target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    logger.info(
        f"Preprocessing completed in {time.time() - t_prep:.2f}s: "
        f"S1={len(s1_prep):,}, S2={len(s2_prep):,}, S3={len(s3_prep):,}, Combined Target={len(target_prep):,}"
    )

    # 4. Candidate Generation (Blocking)
    t_block = time.time()
    blocker = DynamicTFIDFBlocker(top_k=top_k, min_similarity=min_sim, batch_size=1000)
    
    # Run blocking on preprocessed data directly
    countries = sorted(s1_prep["country"].dropna().unique())
    if country_filter:
        countries = [c for c in countries if str(c).lower() == country_filter.lower()]
    logger.info(f"Discovered countries to process: {countries}")
    candidates_dict: Dict[str, List[str]] = {}

    for country in countries:
        s1_country = s1_prep[s1_prep["country"] == country].reset_index(drop=True)
        target_country = target_prep[target_prep["country"] == country].reset_index(drop=True)
        if limit is not None and limit > 0:
            s1_country = s1_country.iloc[:limit].reset_index(drop=True)
            logger.info(
                f"Applying limit: {len(s1_country):,} S1 queries for '{country}' "
                f"(full target pool: {len(target_country):,})"
            )

        logger.info(f"\n>>> Processing Country Partition: '{country}' (S1: {len(s1_country):,}, Targets: {len(target_country):,})")
        
        cands = blocker.block_country_partition(
            s1_df_country=s1_country,
            target_df_country=target_country,
            country_name=str(country),
        )
        candidates_dict.update(cands)

    export_candidate_pairs(candidates_dict, candidate_out)
    logger.info(f"Blocking stage completed in {time.time() - t_block:.2f}s")

    # Evaluate Candidate Recall if ground truth exists
    if gt_path and os.path.isfile(gt_path):
        evaluate_candidate_recall(candidates_dict, gt_path)

    # 5. Chunked Streaming Inference & Export
    t_score = time.time()
    predictions_map = stream_scoring_and_export(
        candidates_dict=candidates_dict,
        s1_prep=s1_prep,
        target_prep=target_prep,
        clf=clf,
        output_matching_path=matching_out,
        chunk_size=chunk_size,
    )
    logger.info(f"Scoring and export completed in {time.time() - t_score:.2f}s")

    # 6. Evaluate final metrics if GT provided
    results: Dict[str, float] = {}
    if gt_path and os.path.isfile(gt_path):
        gt_map = parse_ground_truth(gt_path)
        all_s1_ids = df_s1["entity_id"].tolist()
        macro_f05, prec, rec = compute_instance_macro_f05(predictions_map, gt_map, all_s1_ids)
        results["macro_f05"] = macro_f05
        results["precision"] = prec
        results["recall"] = rec

        print("\n" + "=" * 70)
        print("END-TO-END PIPELINE EVALUATION ON DATASET WITH GROUND TRUTH")
        print("=" * 70)
        print(f"Total S1 Reference Entities:      {len(all_s1_ids):,}")
        print(f"Instance Macro F_0.5 Score:       {macro_f05:.4f}")
        print(f"Instance Macro Precision:         {prec:.4f}")
        print(f"Instance Macro Recall:            {rec:.4f}")
        print("=" * 70 + "\n")

    total_time = time.time() - total_start
    logger.info(f"End-to-End Pipeline executed successfully in {total_time:.2f}s")
    results["total_runtime_seconds"] = total_time
    return results
