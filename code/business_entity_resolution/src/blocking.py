"""
Phase 2: Scalable Candidate Generation (Blocking) Module.

Implements country-partitioned, character n-gram TF-IDF sparse matrix dot-product
candidate retrieval for Business Entity Resolution. Dynamically handles open-set countries
(US, India, France) and exports candidate pairs meeting the competition schema.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .normalizer import preprocess_dataframe
except (ImportError, ValueError):
    from normalizer import preprocess_dataframe

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DynamicTFIDFBlocker:
    """
    Scalable, country-partitioned sparse TF-IDF candidate generation engine.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        analyzer: str = "char_wb",
        min_df: int = 2,
        top_k: int = 35,
        min_similarity: float = 0.12,
        batch_size: int = 5000,
        sublinear_tf: bool = True,
        max_features: int = 150000,
    ) -> None:
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.min_df = min_df
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features

    def block_country_partition(
        self,
        s1_df_country: pd.DataFrame,
        target_df_country: pd.DataFrame,
        country_name: str,
    ) -> Dict[str, List[str]]:
        """
        Execute sparse TF-IDF blocking within a single country partition.
        
        Args:
            s1_df_country: Preprocessed Source 1 records for this country.
            target_df_country: Combined preprocessed Source 2 + Source 3 records.
            country_name: Country identifier string.
            
        Returns:
            Dictionary mapping source1_entity_id -> list of candidate target entity_ids.
        """
        n_queries = len(s1_df_country)
        n_targets = len(target_df_country)
        
        if n_queries == 0 or n_targets == 0:
            logger.warning(f"Country {country_name} has empty query ({n_queries}) or target ({n_targets}) pool.")
            return {eid: [] for eid in s1_df_country["entity_id"]}

        logger.info(
            f"[{country_name}] Fitting TF-IDF on {n_targets:,} target records (char_wb {self.ngram_range}, max_features={self.max_features:,})..."
        )
        fit_start = time.time()
        
        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            max_features=self.max_features,
            dtype=np.float32,
        )
        
        # Fit vectorizer on target pool clean_joint text
        target_matrix = vectorizer.fit_transform(target_df_country["clean_joint"].tolist())
        target_matrix_t = target_matrix.T.tocsc()
        fit_time = time.time() - fit_start
        logger.info(f"[{country_name}] Target matrix shape: {target_matrix.shape} (built in {fit_time:.2f}s)")

        s1_ids = s1_df_country["entity_id"].tolist()
        s1_joint_texts = s1_df_country["clean_joint"].tolist()
        target_ids = np.array(target_df_country["entity_id"].tolist())

        candidates_map: Dict[str, List[str]] = {}
        logger.info(f"[{country_name}] Querying {n_queries:,} S1 entities in batches of {self.batch_size:,}...")
        query_start = time.time()

        for batch_idx in range(0, n_queries, self.batch_size):
            batch_end = min(batch_idx + self.batch_size, n_queries)
            batch_texts = s1_joint_texts[batch_idx:batch_end]
            batch_ids = s1_ids[batch_idx:batch_end]

            # Transform query batch
            query_matrix = vectorizer.transform(batch_texts)

            # Sparse dot product: (batch_size x vocab) @ (vocab x n_targets) -> (batch_size x n_targets)
            sim_matrix = query_matrix.dot(target_matrix_t).tocsr()

            # Extract top-K per row
            for row_idx, s1_id in enumerate(batch_ids):
                row_start_ptr = sim_matrix.indptr[row_idx]
                row_end_ptr = sim_matrix.indptr[row_idx + 1]

                if row_start_ptr == row_end_ptr:
                    candidates_map[s1_id] = []
                    continue

                col_indices = sim_matrix.indices[row_start_ptr:row_end_ptr]
                sim_scores = sim_matrix.data[row_start_ptr:row_end_ptr]

                # Filter by minimum similarity
                valid_mask = sim_scores >= self.min_similarity
                valid_indices = col_indices[valid_mask]
                valid_scores = sim_scores[valid_mask]

                n_valid = len(valid_scores)
                if n_valid == 0:
                    candidates_map[s1_id] = []
                    continue

                if n_valid <= self.top_k:
                    # Sort top candidates descending
                    sorted_order = np.argsort(-valid_scores)
                    chosen_target_ids = target_ids[valid_indices[sorted_order]].tolist()
                else:
                    # Partial sort for top-K
                    top_part = np.argpartition(-valid_scores, self.top_k)[: self.top_k]
                    sorted_top = top_part[np.argsort(-valid_scores[top_part])]
                    chosen_target_ids = target_ids[valid_indices[sorted_top]].tolist()

                candidates_map[s1_id] = chosen_target_ids

        total_query_time = time.time() - query_start
        logger.info(
            f"[{country_name}] Candidate generation complete in {total_query_time:.2f}s "
            f"({n_queries / total_query_time:.1f} queries/sec)"
        )
        return candidates_map

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """
        Run complete blocking pipeline dynamically partitioned across all countries present.
        """
        logger.info("=" * 70)
        logger.info("Starting Phase 2 Candidate Generation (Blocking)")
        logger.info("=" * 70)

        # 1. Preprocess dataframes with normalizer
        logger.info("Preprocessing Source 1, Source 2, and Source 3 text payloads...")
        t0 = time.time()
        s1_prep = preprocess_dataframe(df_s1)
        s2_prep = preprocess_dataframe(df_s2)
        s3_prep = preprocess_dataframe(df_s3)
        target_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
        logger.info(
            f"Preprocessing completed in {time.time() - t0:.2f}s: "
            f"S1={len(s1_prep):,}, S2={len(s2_prep):,}, S3={len(s3_prep):,}, Combined Target={len(target_prep):,}"
        )

        # 2. Discover countries dynamically
        countries = sorted(s1_prep["country"].dropna().unique())
        logger.info(f"Discovered countries to process: {countries}")

        all_candidates: Dict[str, List[str]] = {}

        # 3. Process each country partition
        for country in countries:
            s1_country = s1_prep[s1_prep["country"] == country].reset_index(drop=True)
            target_country = target_prep[target_prep["country"] == country].reset_index(drop=True)
            logger.info(f"\n>>> Processing Country Partition: '{country}' (S1: {len(s1_country):,}, Targets: {len(target_country):,})")
            
            country_candidates = self.block_country_partition(
                s1_df_country=s1_country,
                target_df_country=target_country,
                country_name=str(country),
            )
            all_candidates.update(country_candidates)

        return all_candidates


def export_candidate_pairs(candidates_dict: Dict[str, List[str]], output_path: str) -> None:
    """
    Export candidate pairs TSV file strictly matching the official schema.
    Header: source1_entity_id\tcandidate_entity_ids
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info(f"Writing {len(candidates_dict):,} candidate rows to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidates_dict.items():
            cand_str = ",".join(cands)
            f.write(f"{s1_id}\t{cand_str}\n")
            
    logger.info(f"Export completed: {output_path}")


def evaluate_candidate_recall(
    candidates_dict: Dict[str, List[str]],
    ground_truth_path: str,
) -> Dict[str, float]:
    """
    Compute candidate recall ceiling and blocking efficiency metrics against ground truth.
    """
    logger.info(f"Evaluating candidate recall ceiling against {ground_truth_path}...")
    gt_df = pd.read_csv(ground_truth_path, sep="\t", keep_default_na=False)
    
    total_gt_pairs = 0
    captured_gt_pairs = 0
    total_candidates = 0
    s1_count = len(candidates_dict)
    zero_candidates_count = 0
    singletons_count = 0

    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw_m = row["matched_entity_ids"]
        true_matches = set(m.strip() for m in str(raw_m).split(",") if m.strip())
        
        cands = set(candidates_dict.get(s1_id, []))
        total_candidates += len(cands)
        if len(cands) == 0:
            zero_candidates_count += 1

        if not true_matches:
            singletons_count += 1
            continue

        total_gt_pairs += len(true_matches)
        captured_gt_pairs += len(true_matches.intersection(cands))

    recall_ceiling = (captured_gt_pairs / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 100.0
    avg_candidates = total_candidates / s1_count if s1_count > 0 else 0.0

    print("\n" + "=" * 70)
    print("PHASE 2 BLOCKING CANDIDATE EVALUATION REPORT")
    print("=" * 70)
    print(f"Total S1 Entities Evaluated:      {s1_count:,}")
    print(f"Total Ground Truth Matches:       {total_gt_pairs:,}")
    print(f"Captured Ground Truth Matches:    {captured_gt_pairs:,}")
    print(f"Candidate Recall Ceiling:         {recall_ceiling:.3f}%")
    print(f"Average Candidates per S1:        {avg_candidates:.2f}")
    print(f"Singletons in Ground Truth:       {singletons_count:,} ({(singletons_count/s1_count)*100:.2f}%)")
    print(f"Entities with Zero Candidates:    {zero_candidates_count:,} ({(zero_candidates_count/s1_count)*100:.2f}%)")
    print("=" * 70 + "\n")

    return {
        "recall_ceiling": recall_ceiling,
        "avg_candidates": avg_candidates,
        "total_gt_pairs": total_gt_pairs,
        "captured_gt_pairs": captured_gt_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Scalable Sparse TF-IDF Blocking Engine"
    )
    parser.add_argument("--s1", default="sample_data/sample_source1.tsv", help="Source 1 TSV path")
    parser.add_argument("--s2", default="sample_data/sample_source2.tsv", help="Source 2 TSV path")
    parser.add_argument("--s3", default="sample_data/sample_source3.tsv", help="Source 3 TSV path")
    parser.add_argument("--gt", default="sample_data/sample_ground_truth.tsv", help="Ground Truth TSV path (optional)")
    parser.add_argument("--out", default="output/candidate_pairs.tsv", help="Output candidate TSV path")
    parser.add_argument("--top-k", type=int, default=35, help="Top-K candidates per entity")
    parser.add_argument("--min-sim", type=float, default=0.12, help="Minimum cosine similarity threshold")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch query size")
    args = parser.parse_args()

    start_total = time.time()
    logger.info(f"Loading datasets: S1={args.s1}, S2={args.s2}, S3={args.s3}...")
    df_s1 = pd.read_csv(args.s1, sep="\t", keep_default_na=False)
    df_s2 = pd.read_csv(args.s2, sep="\t", keep_default_na=False)
    df_s3 = pd.read_csv(args.s3, sep="\t", keep_default_na=False)

    blocker = DynamicTFIDFBlocker(
        top_k=args.top_k,
        min_similarity=args.min_sim,
        batch_size=args.batch_size,
    )
    candidates_dict = blocker.generate_candidates(df_s1, df_s2, df_s3)

    export_candidate_pairs(candidates_dict, args.out)

    if args.gt and os.path.isfile(args.gt):
        evaluate_candidate_recall(candidates_dict, args.gt)

    logger.info(f"Total blocking runtime: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    main()
