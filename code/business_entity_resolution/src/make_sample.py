"""
Fast Development Benchmark Generator.

Extracts a representative, non-leaking development slice of 25,000 reference S1 entities
(stratified: 15,000 US, 10,000 India), their true positive ground-truth targets from S2 and S3,
and 50,000 random negative distractor records into `sample_data/`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Set, Tuple
import pandas as pd


def extract_matched_ids(gt_df: pd.DataFrame) -> Tuple[Set[str], Set[str], int]:
    """
    Extract sets of true matching S2 and S3 IDs from ground truth DataFrame.
    
    Returns:
        (s2_matched_ids, s3_matched_ids, singleton_count)
    """
    s2_ids: Set[str] = set()
    s3_ids: Set[str] = set()
    singletons = 0
    
    for raw_matches in gt_df["matched_entity_ids"].fillna("").astype(str):
        clean_matches = [m.strip() for m in raw_matches.split(",") if m.strip()]
        if not clean_matches:
            singletons += 1
            continue
        for mid in clean_matches:
            if mid.startswith("S2-"):
                s2_ids.add(mid)
            elif mid.startswith("S3-"):
                s3_ids.add(mid)
                
    return s2_ids, s3_ids, singletons


def generate_sample_dataset(
    train_dir: str = "student_resource/dataset/train",
    output_dir: str = "sample_data",
    n_us: int = 15000,
    n_india: int = 10000,
    n_s2_distractors: int = 25000,
    n_s3_distractors: int = 25000,
    random_state: int = 42,
) -> None:
    """
    Generate stratified sample dataset and save to output_dir.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Generating Lean Development Sample Slice (25k Reference S1 Entities)")
    print("=" * 70)
    
    # 1. Load and sample Source 1
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    print(f"Loading Source 1 from {s1_path}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    print(f"Total Source 1 records: {len(df_s1):,}")
    
    # Stratified sampling
    df_s1_us = df_s1[df_s1["country"] == "US"].sample(n=n_us, random_state=random_state)
    df_s1_in = df_s1[df_s1["country"] == "India"].sample(n=n_india, random_state=random_state)
    sample_s1 = pd.concat([df_s1_us, df_s1_in], ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    
    sampled_s1_ids = set(sample_s1["entity_id"].tolist())
    print(f"Sampled Source 1: {len(sample_s1):,} records ({n_us:,} US, {n_india:,} India)")
    
    # 2. Filter Ground Truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print(f"\nLoading Ground Truth from {gt_path}...")
    df_gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    
    # Keep exact order matching sample_s1
    sample_gt = sample_s1[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"}).merge(
        df_gt, on="source1_entity_id", how="left"
    )
    sample_gt["matched_entity_ids"] = sample_gt["matched_entity_ids"].fillna("")
    
    s2_matched_ids, s3_matched_ids, singleton_count = extract_matched_ids(sample_gt)
    singleton_pct = (singleton_count / len(sample_gt)) * 100
    print(f"Sample Ground Truth: {len(sample_gt):,} records")
    print(f"  - Singletons (0 matches): {singleton_count:,} ({singleton_pct:.2f}%)")
    print(f"  - True S2 positive target IDs: {len(s2_matched_ids):,}")
    print(f"  - True S3 positive target IDs: {len(s3_matched_ids):,}")
    
    # 3. Process Source 2 (Positives + Distractors)
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    print(f"\nProcessing Source 2 from {s2_path}...")
    df_s2 = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    
    pos_s2 = df_s2[df_s2["entity_id"].isin(s2_matched_ids)]
    remaining_s2 = df_s2[~df_s2["entity_id"].isin(s2_matched_ids)]
    distractors_s2 = remaining_s2.sample(
        n=min(n_s2_distractors, len(remaining_s2)), random_state=random_state
    )
    
    sample_s2 = pd.concat([pos_s2, distractors_s2], ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    
    missing_addr_s2 = (sample_s2["business_address"].fillna("").str.strip() == "").sum()
    print(f"Sample Source 2: {len(sample_s2):,} records ({len(pos_s2):,} true matches + {len(distractors_s2):,} distractors)")
    print(f"  - Missing addresses in Sample S2: {missing_addr_s2:,} ({missing_addr_s2/len(sample_s2)*100:.2f}%)")
    
    # 4. Process Source 3 (Positives + Distractors)
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    print(f"\nProcessing Source 3 from {s3_path}...")
    df_s3 = pd.read_csv(s3_path, sep="\t", keep_default_na=False)
    
    pos_s3 = df_s3[df_s3["entity_id"].isin(s3_matched_ids)]
    remaining_s3 = df_s3[~df_s3["entity_id"].isin(s3_matched_ids)]
    distractors_s3 = remaining_s3.sample(
        n=min(n_s3_distractors, len(remaining_s3)), random_state=random_state
    )
    
    sample_s3 = pd.concat([pos_s3, distractors_s3], ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    
    missing_addr_s3 = (sample_s3["business_address"].fillna("").str.strip() == "").sum()
    print(f"Sample Source 3: {len(sample_s3):,} records ({len(pos_s3):,} true matches + {len(distractors_s3):,} distractors)")
    print(f"  - Missing addresses in Sample S3: {missing_addr_s3:,} ({missing_addr_s3/len(sample_s3)*100:.2f}%)")
    
    # 5. Save Sample TSVs
    print(f"\nSaving sample TSV files to '{output_dir}/'...")
    out_s1 = os.path.join(output_dir, "sample_source1.tsv")
    out_s2 = os.path.join(output_dir, "sample_source2.tsv")
    out_s3 = os.path.join(output_dir, "sample_source3.tsv")
    out_gt = os.path.join(output_dir, "sample_ground_truth.tsv")
    
    sample_s1.to_csv(out_s1, sep="\t", index=False, encoding="utf-8")
    sample_s2.to_csv(out_s2, sep="\t", index=False, encoding="utf-8")
    sample_s3.to_csv(out_s3, sep="\t", index=False, encoding="utf-8")
    sample_gt.to_csv(out_gt, sep="\t", index=False, encoding="utf-8")
    
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Sample Dataset Generated Successfully in {elapsed:.2f}s!")
    print(f"  - {out_s1} ({len(sample_s1):,} rows)")
    print(f"  - {out_s2} ({len(sample_s2):,} rows)")
    print(f"  - {out_s3} ({len(sample_s3):,} rows)")
    print(f"  - {out_gt} ({len(sample_gt):,} rows)")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 25k stratified sample dataset for Business Entity Resolution."
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
        "--n-us",
        type=int,
        default=15000,
        help="Number of US reference entities to sample from S1.",
    )
    parser.add_argument(
        "--n-india",
        type=int,
        default=10000,
        help="Number of India reference entities to sample from S1.",
    )
    parser.add_argument(
        "--n-s2-distractors",
        type=int,
        default=25000,
        help="Number of negative distractor records to sample from S2.",
    )
    parser.add_argument(
        "--n-s3-distractors",
        type=int,
        default=25000,
        help="Number of negative distractor records to sample from S3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    args = parser.parse_args()
    
    generate_sample_dataset(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        n_us=args.n_us,
        n_india=args.n_india,
        n_s2_distractors=args.n_s2_distractors,
        n_s3_distractors=args.n_s3_distractors,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
