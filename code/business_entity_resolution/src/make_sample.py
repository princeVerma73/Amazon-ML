"""
Step 2: 50k Hard-Negative Sample Dataset Generator with Calibrated Blocking.

Extracts a representative, non-leaking benchmark slice of 50,000 reference S1 entities
(stratified: 30,000 US, 20,000 India), pulls all true positive ground-truth targets from S2 and S3,
and mines hard negatives with TF-IDF cosine similarity in [0.35, 0.65] under Top-K=20 and
min_similarity=0.10, maintaining a calibrated 3:1 to 4:1 negative-to-positive ratio.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from blocking import DynamicTFIDFBlocker, export_candidate_pairs
    from normalizer import preprocess_dataframe
except ImportError:
    from .blocking import DynamicTFIDFBlocker, export_candidate_pairs
    from .normalizer import preprocess_dataframe

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_matched_ids(gt_df: pd.DataFrame) -> Tuple[Set[str], Set[str], Dict[str, Set[str]], int]:
    """
    Extract sets of true matching S2 and S3 IDs and mapping from ground truth DataFrame.
    
    Returns:
        (s2_matched_ids, s3_matched_ids, gt_map, singleton_count)
    """
    s2_ids: Set[str] = set()
    s3_ids: Set[str] = set()
    gt_map: Dict[str, Set[str]] = {}
    singletons = 0

    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matched_entity_ids"]).strip()
        clean_matches = set(m.strip() for m in raw_matches.split(",") if m.strip())

        gt_map[s1_id] = clean_matches
        if not clean_matches:
            singletons += 1
            continue

        for mid in clean_matches:
            if mid.startswith("S2-"):
                s2_ids.add(mid)
            elif mid.startswith("S3-"):
                s3_ids.add(mid)

    return s2_ids, s3_ids, gt_map, singletons


def load_source_targets_and_distractors(
    source_path: str,
    target_ids: Set[str],
    n_extra_distractors: int = 50000,
    random_state: int = 42,
    chunksize: int = 500000,
) -> pd.DataFrame:
    """
    Stream a train source TSV (Source 2 or 3) to extract all true positive matches
    plus a sampled pool of background distractor records.
    """
    logger.info(f"Streaming {source_path} to collect {len(target_ids):,} true targets + distractors...")
    pos_chunks: List[pd.DataFrame] = []
    distractor_candidates: List[pd.DataFrame] = []

    for chunk in pd.read_csv(source_path, sep="\t", chunksize=chunksize, keep_default_na=False):
        # 1. Capture true targets
        pos_sub = chunk[chunk["entity_id"].isin(target_ids)]
        if not pos_sub.empty:
            pos_chunks.append(pos_sub)

        # 2. Collect sample of non-target rows as distractor pool
        non_target = chunk[~chunk["entity_id"].isin(target_ids)]
        if len(non_target) > 0 and len(distractor_candidates) < 3:
            distractor_candidates.append(non_target.sample(n=min(len(non_target), 30000), random_state=random_state))

    df_pos = pd.concat(pos_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
    logger.info(f"Loaded {len(df_pos):,} / {len(target_ids):,} positive targets from {os.path.basename(source_path)}")

    if distractor_candidates:
        df_dist = pd.concat(distractor_candidates, ignore_index=True).drop_duplicates(subset=["entity_id"])
        sample_dist = df_dist.sample(n=min(len(df_dist), n_extra_distractors), random_state=random_state)
    else:
        sample_dist = pd.DataFrame(columns=df_pos.columns)

    combined = pd.concat([df_pos, sample_dist], ignore_index=True).drop_duplicates(subset=["entity_id"])
    logger.info(f"Total pool for {os.path.basename(source_path)}: {len(combined):,} records ({len(df_pos):,} pos + {len(sample_dist):,} distractors)")
    return combined


def generate_50k_sample_dataset(
    train_dir: str = "student_resource/dataset/train",
    output_dir: str = "sample_data",
    n_s1: int = 50000,
    n_us: int = 30000,
    n_india: int = 20000,
    top_k: int = 20,
    min_similarity: float = 0.10,
    sim_min_hard: float = 0.35,
    sim_max_hard: float = 0.65,
    neg_to_pos_ratio: float = 3.5,
    random_state: int = 42,
) -> Dict[str, int]:
    """
    Generate 50k stratified sample dataset with hard-negative candidate mining.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 75)
    logger.info("EXECUTING STEP 2: 50K HARD-NEGATIVE SAMPLE GENERATION")
    logger.info(f"  Target S1 Entities:     {n_s1:,} ({n_us:,} US, {n_india:,} India)")
    logger.info(f"  Blocking Safeguards:    Top-K = {top_k}, Min Sim Floor = {min_similarity:.2f}")
    logger.info(f"  Hard-Negative Window:   [{sim_min_hard:.2f}, {sim_max_hard:.2f}]")
    logger.info(f"  Target Negative Ratio:  {neg_to_pos_ratio:.1f}:1")
    logger.info("=" * 75)

    # 1. Sample Source 1
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    logger.info(f"Loading Source 1 from {s1_path}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    logger.info(f"Total Source 1 records available: {len(df_s1):,}")

    df_s1_us = df_s1[df_s1["country"] == "US"].sample(n=n_us, random_state=random_state)
    df_s1_in = df_s1[df_s1["country"] == "India"].sample(n=n_india, random_state=random_state)
    sample_s1 = pd.concat([df_s1_us, df_s1_in], ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    sampled_s1_ids = set(sample_s1["entity_id"].tolist())
    logger.info(f"Sampled Source 1: {len(sample_s1):,} records ({n_us:,} US, {n_india:,} India)")

    # 2. Extract Ground Truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    logger.info(f"\nFiltering Ground Truth from {gt_path}...")
    gt_chunks = []
    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=500000, keep_default_na=False):
        sub = chunk[chunk["source1_entity_id"].isin(sampled_s1_ids)]
        if not sub.empty:
            gt_chunks.append(sub)
    sample_gt = pd.concat(gt_chunks, ignore_index=True)

    # Ensure 1-to-1 match with sample_s1 order and fill missing as singletons
    sample_gt = sample_s1[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"}).merge(
        sample_gt, on="source1_entity_id", how="left"
    )
    sample_gt["matched_entity_ids"] = sample_gt["matched_entity_ids"].fillna("")

    s2_matched_ids, s3_matched_ids, gt_map, singleton_count = extract_matched_ids(sample_gt)
    singleton_pct = (singleton_count / len(sample_gt)) * 100
    total_positives = len(s2_matched_ids) + len(s3_matched_ids)

    logger.info(f"Sample Ground Truth: {len(sample_gt):,} records")
    logger.info(f"  - Singletons (0 matches): {singleton_count:,} ({singleton_pct:.2f}%)")
    logger.info(f"  - True S2 positive target IDs: {len(s2_matched_ids):,}")
    logger.info(f"  - True S3 positive target IDs: {len(s3_matched_ids):,}")
    logger.info(f"  - Total Positive Targets:      {total_positives:,}")

    # 3. Stream Source 2 and Source 3 Target Pools
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    pool_s2 = load_source_targets_and_distractors(
        source_path=s2_path,
        target_ids=s2_matched_ids,
        n_extra_distractors=50000,
        random_state=random_state,
    )

    s3_path = os.path.join(train_dir, "train_source3.tsv")
    pool_s3 = load_source_targets_and_distractors(
        source_path=s3_path,
        target_ids=s3_matched_ids,
        n_extra_distractors=50000,
        random_state=random_state,
    )

    # 4. Preprocess DataFrames for High-Speed Blocking
    logger.info("\nPreprocessing text payload representations for blocking...")
    t_prep = time.time()
    s1_prep = preprocess_dataframe(sample_s1)
    s2_prep = preprocess_dataframe(pool_s2)
    s3_prep = preprocess_dataframe(pool_s3)
    target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    logger.info(f"Preprocessing completed in {time.time() - t_prep:.2f}s (Target pool: {len(target_prep):,} records)")

    # 5. Calibrated Blocking & Hard-Negative Mining per Country
    blocker = DynamicTFIDFBlocker(top_k=top_k, min_similarity=min_similarity, batch_size=5000)
    countries = sorted(s1_prep["country"].dropna().unique())
    logger.info(f"\nDiscovered country partitions for candidate mining: {countries}")

    candidates_dict: Dict[str, List[str]] = {}
    all_mined_negative_ids: Set[str] = set()

    for country in countries:
        s1_c = s1_prep[s1_prep["country"] == country].reset_index(drop=True)
        tgt_c = target_prep[target_prep["country"] == country].reset_index(drop=True)
        logger.info(f"\n>>> Mining partition: '{country}' (S1 Queries: {len(s1_c):,}, Target Pool: {len(tgt_c):,})")

        cands, neg_ids = blocker.mine_hard_negatives(
            s1_df_country=s1_c,
            target_df_country=tgt_c,
            gt_map=gt_map,
            country_name=str(country),
            sim_min_hard=sim_min_hard,
            sim_max_hard=sim_max_hard,
            neg_to_pos_ratio=neg_to_pos_ratio,
        )
        candidates_dict.update(cands)
        all_mined_negative_ids.update(neg_ids)

    # 6. Assemble Final S2 and S3 Sample Sets
    # Must contain 100% of true positive matches PLUS all mined hard negatives
    final_s2_ids = s2_matched_ids | {nid for nid in all_mined_negative_ids if nid.startswith("S2-")}
    final_s3_ids = s3_matched_ids | {nid for nid in all_mined_negative_ids if nid.startswith("S3-")}

    sample_s2 = pool_s2[pool_s2["entity_id"].isin(final_s2_ids)].reset_index(drop=True)
    sample_s3 = pool_s3[pool_s3["entity_id"].isin(final_s3_ids)].reset_index(drop=True)

    # Compute candidate pairs statistics
    total_pairs = sum(len(cands) for cands in candidates_dict.values())
    total_hard_negs = len(all_mined_negative_ids)
    avg_pairs_per_query = total_pairs / len(sample_s1) if len(sample_s1) > 0 else 0
    actual_ratio = (total_pairs - total_positives) / max(total_positives, 1)

    logger.info("\n" + "=" * 75)
    logger.info("50K HARD-NEGATIVE SAMPLE STATISTICS SUMMARY")
    logger.info("=" * 75)
    logger.info(f"Reference S1 Entities:             {len(sample_s1):,}")
    logger.info(f"  - US Queries:                    {len(df_s1_us):,}")
    logger.info(f"  - India Queries:                 {len(df_s1_in):,}")
    logger.info(f"Singletons Count:                  {singleton_count:,} ({singleton_pct:.2f}%)")
    logger.info(f"True Positive Matches in GT:       {total_positives:,}")
    logger.info(f"Unique Hard Negatives Mined:       {total_hard_negs:,}")
    logger.info(f"Total Candidate Pairs Generated:   {total_pairs:,}")
    logger.info(f"Average Candidates per S1 Query:   {avg_pairs_per_query:.2f} (cap: {top_k})")
    logger.info(f"Negative-to-Positive Ratio:        {actual_ratio:.2f} : 1")
    logger.info(f"Sample Source 2 Total Rows:        {len(sample_s2):,} ({len(s2_matched_ids):,} true pos + {len(sample_s2) - len(s2_matched_ids):,} negs)")
    logger.info(f"Sample Source 3 Total Rows:        {len(sample_s3):,} ({len(s3_matched_ids):,} true pos + {len(sample_s3) - len(s3_matched_ids):,} negs)")
    logger.info("=" * 75)

    # 7. Save Sample TSVs
    out_s1 = os.path.join(output_dir, "sample_source1.tsv")
    out_s2 = os.path.join(output_dir, "sample_source2.tsv")
    out_s3 = os.path.join(output_dir, "sample_source3.tsv")
    out_gt = os.path.join(output_dir, "sample_ground_truth.tsv")

    logger.info(f"\nWriting sample TSVs to '{output_dir}/'...")
    sample_s1.to_csv(out_s1, sep="\t", index=False, encoding="utf-8")
    sample_s2.to_csv(out_s2, sep="\t", index=False, encoding="utf-8")
    sample_s3.to_csv(out_s3, sep="\t", index=False, encoding="utf-8")
    sample_gt.to_csv(out_gt, sep="\t", index=False, encoding="utf-8")

    # Also export candidate pairs for benchmarking
    out_cand = os.path.join(output_dir, "sample_candidate_pairs.tsv")
    export_candidate_pairs(candidates_dict, out_cand)

    total_time = time.time() - start_time
    logger.info(f"Sample generation and export completed in {total_time:.2f}s!")

    return {
        "s1_count": len(sample_s1),
        "s2_count": len(sample_s2),
        "s3_count": len(sample_s3),
        "gt_count": len(sample_gt),
        "singletons": singleton_count,
        "true_positives": total_positives,
        "hard_negatives": total_hard_negs,
        "candidate_pairs": total_pairs,
        "elapsed_seconds": int(total_time),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 50k stratified sample dataset with hard-negative mining for Business Entity Resolution."
    )
    parser.add_argument(
        "--train-dir",
        default="student_resource/dataset/train",
        help="Path to directory containing train source TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_data",
        help="Destination directory for sample TSV files.",
    )
    parser.add_argument(
        "--n-s1",
        type=int,
        default=50000,
        help="Total S1 entities to sample (default: 50000).",
    )
    parser.add_argument(
        "--n-us",
        type=int,
        default=30000,
        help="Number of US reference entities to sample from S1.",
    )
    parser.add_argument(
        "--n-india",
        type=int,
        default=20000,
        help="Number of India reference entities to sample from S1.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K candidates per entity (default: 20).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.10,
        help="Minimum TF-IDF cosine similarity floor (default: 0.10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    args = parser.parse_args()

    generate_50k_sample_dataset(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        n_s1=args.n_s1,
        n_us=args.n_us,
        n_india=args.n_india,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
