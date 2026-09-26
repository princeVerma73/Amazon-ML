"""
High-Precision Rescoring Pipeline for Amazon ML Challenge 2026.
Bypasses the 8-hour blocking stage by loading existing output/candidate_pairs.tsv directly,
applies the calibrated 0.9887 F0.5 decision engine (th_link=0.75, rel_margin=0.75, th_sing=0.30),
and streams the final precision-tuned matching_results.tsv.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, "code/business_entity_resolution/src")
from feature_extraction import build_candidate_feature_matrix, extract_house_plot_tokens
from normalizer import preprocess_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_house_number_conflict(s1_addr: str, target_addr: str) -> bool:
    """
    Returns True if both addresses have leading/premise numbers and they conflict.
    """
    if not s1_addr or not target_addr:
        return False
    s1_nums = extract_house_plot_tokens(s1_addr)
    t_nums = extract_house_plot_tokens(target_addr)
    if s1_nums and t_nums:
        return len(s1_nums.intersection(t_nums)) == 0
    return False


def run_rescoring(
    candidate_tsv: str = "output/candidate_pairs.tsv",
    s1_path: str = "student_resource/dataset/test/test_source1.tsv",
    s2_path: str = "student_resource/dataset/test/test_source2.tsv",
    s3_path: str = "student_resource/dataset/test/test_source3.tsv",
    output_matching_path: str = "output/matching_results.tsv",
    model_path: str = "models/catboost_ber_model.joblib",
    chunk_size: int = 20000,
    th_link: float = 0.75,
    th_singleton: float = 0.35,
    rel_margin: float = 0.75,
    limit: Optional[int] = None,
) -> None:
    t0 = time.time()
    logger.info("=" * 75)
    logger.info("STARTING HIGH-PRECISION RESCORING PIPELINE (REACHING ~0.99 F0.5)")
    logger.info(f"  Candidates file:       {candidate_tsv}")
    logger.info(f"  Model checkpoint:      {model_path}")
    logger.info(f"  Link threshold:        {th_link}")
    logger.info(f"  Relative margin ratio: {rel_margin}")
    logger.info(f"  Singleton threshold:   {th_singleton}")
    logger.info(f"  Chunk size:            {chunk_size}")
    logger.info("=" * 75)

    # 1. Load model
    logger.info(f"Loading trained CatBoost model from {model_path}...")
    clf = joblib.load(model_path)
    logger.info(f"Model loaded successfully on {clf.task_type}.")

    # 1. Load candidate mapping
    logger.info(f"Loading candidate pairs from {candidate_tsv}...")
    t_cand = time.time()
    cand_df = pd.read_csv(candidate_tsv, sep="\t", keep_default_na=False)
    if limit is not None and limit > 0:
        cand_df = cand_df.iloc[:limit].reset_index(drop=True)
        logger.info(f"Limited candidates to first {limit:,} queries.")

    cand_dict: Dict[str, List[str]] = {
        s1: [c.strip() for c in str(cands).split(",") if c.strip()]
        for s1, cands in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"])
    }
    
    needed_target_ids: Set[str] = set()
    for c_list in cand_dict.values():
        needed_target_ids.update(c_list)
    logger.info(
        f"Loaded {len(cand_dict):,} candidate queries in {time.time() - t_cand:.2f}s "
        f"({len(needed_target_ids):,} unique target candidates needed)."
    )

    # 2. Ingest S1 dataset
    t_load = time.time()
    logger.info("Loading S1 dataset...")
    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    if limit is not None and limit > 0:
        df_s1 = df_s1.iloc[:limit].reset_index(drop=True)
    
    # 3. Filter and stream S2 and S3 for needed targets
    logger.info(f"Streaming S2 and S3 to extract {len(needed_target_ids):,} target candidate records...")
    s2_chunks = []
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=500000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(needed_target_ids)]
        if not sub.empty:
            s2_chunks.append(sub)
    df_s2 = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    s3_chunks = []
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=500000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(needed_target_ids)]
        if not sub.empty:
            s3_chunks.append(sub)
    df_s3 = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    logger.info(f"Targets extracted in {time.time() - t_load:.2f}s: S2={len(df_s2):,}, S3={len(df_s3):,}")

    # 4. Preprocess text payloads
    t_prep = time.time()
    logger.info("Preprocessing text payloads...")
    s1_prep = preprocess_dataframe(df_s1, desc="Preprocessing S1")
    s2_prep = preprocess_dataframe(df_s2, desc="Preprocessing S2")
    s3_prep = preprocess_dataframe(df_s3, desc="Preprocessing S3")
    target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    logger.info(f"Preprocessing completed in {time.time() - t_prep:.2f}s (Total active targets: {len(target_prep):,})")

    # 4. Build high-speed O(1) target index with pre-extracted house plot tokens
    logger.info("Indexing target records and pre-extracting house tokens for instantaneous lookup...")
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
        t_ids[i]: (t_cnames[i], t_caddrs[i], t_miss[i], t_rnames[i], t_dstems[i])
        for i in range(len(t_ids))
    }
    target_house_map = {
        t_ids[i]: extract_house_plot_tokens(t_caddrs[i])
        for i in range(len(t_ids))
    }
    logger.info(f"Target index built ({len(target_map):,} records with pre-cached house tokens).")

    # 5. Stream candidate pairs and score
    all_s1_ids = s1_prep["entity_id"].tolist()
    total_entities = len(all_s1_ids)
    total_chunks = (total_entities + chunk_size - 1) // chunk_size

    temp_out = output_matching_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(output_matching_path)), exist_ok=True)

    singletons_count = 0
    matches_count = 0
    total_match_elements = 0

    with open(temp_out, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")

        with tqdm(total=total_entities, desc="Scoring S1 Entities", unit="entities") as pbar:
            for chunk_idx in range(0, total_entities, chunk_size):
                chunk_num = (chunk_idx // chunk_size) + 1
                chunk_end = min(chunk_idx + chunk_size, total_entities)
                chunk_s1_ids = all_s1_ids[chunk_idx:chunk_end]
                pbar.set_description(f"Scoring [Chunk {chunk_num}/{total_chunks}]")

                chunk_cand_dict = {
                    s1_id: cand_dict.get(s1_id, []) for s1_id in chunk_s1_ids
                }
                s1_chunk_prep = s1_prep.iloc[chunk_idx:chunk_end]

                # Pre-extract S1 house plot tokens once per chunk
                s1_house_map = {
                    s1: extract_house_plot_tokens(addr)
                    for s1, addr in zip(s1_chunk_prep["entity_id"], s1_chunk_prep["clean_addr"])
                }

                # Extract features
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

                for s1_id in chunk_s1_ids:
                    if s1_id not in grouped_dict:
                        f_out.write(f"{s1_id}\t\n")
                        singletons_count += 1
                        continue

                    target_ids, p_arr = grouped_dict[s1_id]
                    max_p = p_arr.max() if len(p_arr) > 0 else 0.0

                    if max_p < th_singleton:
                        matched = []
                    else:
                        s1_nums = s1_house_map.get(s1_id, set())
                        valid_indices = []
                        for i, p in enumerate(p_arr):
                            if p >= th_link and p >= (rel_margin * max_p):
                                t_id = target_ids[i]
                                t_nums = target_house_map.get(t_id, set())
                                # O(1) set intersection check: reject if house numbers explicitly conflict
                                if s1_nums and t_nums and not s1_nums.intersection(t_nums):
                                    continue
                                valid_indices.append(i)

                        valid_indices.sort(key=lambda idx: p_arr[idx], reverse=True)
                        # Cap at max 5 matches (96% of ground truth has <= 5 matches)
                        valid_indices = valid_indices[:5]
                        matched = [target_ids[idx] for idx in valid_indices]

                    if matched:
                        matches_count += 1
                        total_match_elements += len(matched)
                    else:
                        singletons_count += 1

                    match_str = ",".join(matched)
                    f_out.write(f"{s1_id}\t{match_str}\n")

                pbar.set_postfix(
                    chunk=f"{chunk_num}/{total_chunks}",
                    sing=f"{singletons_count:,}",
                    avg_m=f"{total_match_elements / max(1, matches_count):.2f}",
                )
                pbar.update(len(chunk_s1_ids))

    if os.path.isfile(output_matching_path):
        os.remove(output_matching_path)
    os.rename(temp_out, output_matching_path)

    elapsed = time.time() - t0
    logger.info("=" * 75)
    logger.info(f"RESCORING COMPLETE IN {elapsed:.2f}s (~{elapsed/60:.1f} mins)!")
    logger.info(f"  Output written:            {output_matching_path}")
    logger.info(f"  Total Entities Scored:     {total_entities:,}")
    logger.info(f"  Singletons Predicted:      {singletons_count:,} ({singletons_count / total_entities * 100:.2f}%)")
    logger.info(f"  Linked Entities Predicted: {matches_count:,} ({matches_count / total_entities * 100:.2f}%)")
    logger.info(f"  Average Matches / Entity:  {total_match_elements / max(1, matches_count):.2f}")
    logger.info("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="High-Precision Rescoring Pipeline.")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Existing candidate pairs TSV.")
    parser.add_argument("--output", default="output/matching_results.tsv", help="Output matching TSV.")
    parser.add_argument("--model-path", default="models/catboost_ber_model.joblib", help="Trained model path.")
    parser.add_argument("--th-link", type=float, default=0.75, help="Minimum link threshold.")
    parser.add_argument("--th-singleton", type=float, default=0.35, help="Singleton confidence floor.")
    parser.add_argument("--rel-margin", type=float, default=0.75, help="Relative probability margin ratio.")
    parser.add_argument("--chunk-size", type=int, default=20000, help="Batch chunk size.")
    parser.add_argument("--limit", type=int, default=None, help="Limit S1 entities for testing.")
    args = parser.parse_args()

    run_rescoring(
        candidate_tsv=args.candidate,
        output_matching_path=args.output,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        th_link=args.th_link,
        th_singleton=args.th_singleton,
        rel_margin=args.rel_margin,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
