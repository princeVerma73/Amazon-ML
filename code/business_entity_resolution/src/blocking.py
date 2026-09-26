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
from tqdm import tqdm

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
        # Stage 1: Coarse Word Retrieval
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_analyzer: str = "word",
        word_min_df: int = 2,
        word_max_df: float = 0.02,
        coarse_top_k: int = 300,
        # Stage 2: Fine Character Re-ranking
        char_ngram_range: Tuple[int, int] = (3, 4),
        char_analyzer: str = "char_wb",
        char_min_df: int = 2,
        char_max_df: float = 0.25,
        char_max_features: int = 150000,
        # Output ranking
        top_k: int = 20,
        min_similarity: float = 0.10,
        batch_size: int = 1000,
        sublinear_tf: bool = True,
        # Backward compatibility aliases
        ngram_range: Optional[Tuple[int, int]] = None,
        analyzer: Optional[str] = None,
        max_df: Optional[float] = None,
        max_features: Optional[int] = None,
    ) -> None:
        self.word_ngram_range = word_ngram_range
        self.word_analyzer = word_analyzer
        self.word_min_df = word_min_df
        self.word_max_df = max_df if max_df is not None else word_max_df
        self.coarse_top_k = coarse_top_k

        self.char_ngram_range = ngram_range if ngram_range is not None else char_ngram_range
        self.char_analyzer = analyzer if analyzer is not None else char_analyzer
        self.char_min_df = char_min_df
        self.char_max_df = char_max_df
        self.char_max_features = max_features if max_features is not None else char_max_features

        self.top_k = top_k
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        self.sublinear_tf = sublinear_tf

    def block_country_partition(
        self,
        s1_df_country: pd.DataFrame,
        target_df_country: pd.DataFrame,
        country_name: str,
    ) -> Dict[str, List[str]]:
        """
        Execute Two-Stage Hybrid TF-IDF blocking within a single country partition.
        
        Stage 1 (Coarse Retrieval):
          Word n-gram TF-IDF (analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.50).
          Retrieves top 300 candidate targets per query via sparse dot product.
          
        Stage 2 (Fine Re-ranking):
          Character n-gram TF-IDF (analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_df=0.25).
          Re-scores only the 300 shortlisted targets per query, filters by min_similarity (0.10),
          and extracts top_k (20) with deterministic tie-breaking.
        """
        n_queries = len(s1_df_country)
        n_targets = len(target_df_country)
        
        if n_queries == 0 or n_targets == 0:
            logger.warning(f"Country {country_name} has empty query ({n_queries}) or target ({n_targets}) pool.")
            return {eid: [] for eid in s1_df_country["entity_id"]}

        target_texts = target_df_country["clean_joint"].tolist()
        target_ids = np.array(target_df_country["entity_id"].tolist())

        # -----------------------------------------------------------------------
        # 1. Fit Stage 1: Coarse Word TF-IDF Vectorizer
        # -----------------------------------------------------------------------
        w_min_df = self.word_min_df if n_targets >= self.word_min_df else 1
        w_max_df = 1.0 if (isinstance(self.word_max_df, float) and n_targets * self.word_max_df < w_min_df) else self.word_max_df

        c_min_df = self.char_min_df if n_targets >= self.char_min_df else 1
        c_max_df = 1.0 if (isinstance(self.char_max_df, float) and n_targets * self.char_max_df < c_min_df) else self.char_max_df

        logger.info(
            f"[{country_name}] Fitting Stage 1 Word TF-IDF on {n_targets:,} target records "
            f"({self.word_analyzer} {self.word_ngram_range}, max_df={w_max_df})..."
        )
        t_w0 = time.time()
        vec_word = TfidfVectorizer(
            analyzer=self.word_analyzer,
            ngram_range=self.word_ngram_range,
            min_df=w_min_df,
            max_df=w_max_df,
            sublinear_tf=self.sublinear_tf,
            dtype=np.float32,
        )
        target_word_matrix = vec_word.fit_transform(target_texts)
        target_word_matrix_T = target_word_matrix.T.tocsr()
        target_word_matrix_T.sort_indices()
        logger.info(
            f"[{country_name}] Stage 1 Word matrix shape: {target_word_matrix.shape} "
            f"(built in {time.time() - t_w0:.2f}s)"
        )

        # -----------------------------------------------------------------------
        # 2. Fit Stage 2: Fine Character TF-IDF Vectorizer
        # -----------------------------------------------------------------------
        logger.info(
            f"[{country_name}] Fitting Stage 2 Char TF-IDF on {n_targets:,} target records "
            f"({self.char_analyzer} {self.char_ngram_range}, max_df={c_max_df}, max_features={self.char_max_features:,})..."
        )
        t_c0 = time.time()
        vec_char = TfidfVectorizer(
            analyzer=self.char_analyzer,
            ngram_range=self.char_ngram_range,
            min_df=c_min_df,
            max_df=c_max_df,
            max_features=self.char_max_features,
            sublinear_tf=self.sublinear_tf,
            dtype=np.float32,
        )
        target_char_matrix = vec_char.fit_transform(target_texts)
        logger.info(
            f"[{country_name}] Stage 2 Char matrix shape: {target_char_matrix.shape} "
            f"(built in {time.time() - t_c0:.2f}s)"
        )

        self.vec_word = vec_word
        self.vec_char = vec_char
        self.target_word_matrix = target_word_matrix
        self.target_char_matrix = target_char_matrix

        s1_ids = s1_df_country["entity_id"].tolist()
        s1_joint_texts = s1_df_country["clean_joint"].tolist()

        effective_batch = self.batch_size
        candidates_map: Dict[str, List[str]] = {}
        logger.info(
            f"[{country_name}] Two-Stage querying {n_queries:,} S1 entities in batches of {effective_batch:,} "
            f"(coarse_top_k={self.coarse_top_k}, final_top_k={self.top_k})..."
        )
        query_start = time.time()

        with tqdm(total=n_queries, desc=f"Blocking [{country_name}]", unit="queries", dynamic_ncols=True) as pbar:
            for batch_idx in range(0, n_queries, effective_batch):
                batch_end = min(batch_idx + effective_batch, n_queries)
                batch_texts = s1_joint_texts[batch_idx:batch_end]
                batch_ids = s1_ids[batch_idx:batch_end]

                # Transform queries into word and char representations
                query_word = vec_word.transform(batch_texts)
                query_char = vec_char.transform(batch_texts)

                # Stage 1: Sparse dot product on word space
                sim_word = query_word.dot(target_word_matrix_T)
                if not isinstance(sim_word, sparse.csr_matrix):
                    sim_word = sim_word.tocsr()

                # Stage 2: Sliced re-ranking on shortlisted 300 candidates per query
                for row_idx, s1_id in enumerate(batch_ids):
                    start = sim_word.indptr[row_idx]
                    end = sim_word.indptr[row_idx + 1]
                    row_cols = sim_word.indices[start:end]
                    row_vals = sim_word.data[start:end]

                    if len(row_vals) == 0:
                        candidates_map[s1_id] = []
                        continue

                    # Select top coarse_top_k candidates with monotonic ordering
                    if len(row_vals) > self.coarse_top_k:
                        top_coarse = np.argpartition(-row_vals, self.coarse_top_k)[: self.coarse_top_k]
                        coarse_cols = np.sort(row_cols[top_coarse])
                    else:
                        coarse_cols = np.sort(row_cols)

                    # Stage 2: fine character n-gram cosine similarities on shortlisted targets
                    cand_char_sub = target_char_matrix[coarse_cols]
                    scores = cand_char_sub.dot(query_char[row_idx].T).toarray().ravel()

                    # Apply minimum similarity floor
                    mask = scores >= self.min_similarity
                    valid_cols = coarse_cols[mask]
                    valid_scores = scores[mask]

                    n_valid = len(valid_scores)
                    if n_valid == 0:
                        candidates_map[s1_id] = []
                        continue

                    # Select final top_k with deterministic tie-breaking (stable sort)
                    if n_valid > self.top_k:
                        top_fine = np.argpartition(-valid_scores, self.top_k)[: self.top_k]
                        stable_order = top_fine[np.argsort(-valid_scores[top_fine], kind="stable")]
                        selected_cols = valid_cols[stable_order]
                    else:
                        stable_order = np.argsort(-valid_scores, kind="stable")
                        selected_cols = valid_cols[stable_order]

                    candidates_map[s1_id] = target_ids[selected_cols].tolist()

                pbar.update(len(batch_ids))

        total_query_time = time.time() - query_start
        throughput = n_queries / total_query_time if total_query_time > 0 else 0.0
        self.last_query_time = total_query_time
        self.last_query_throughput = throughput
        logger.info(
            f"[{country_name}] Candidate generation complete in {total_query_time:.2f}s "
            f"({throughput:.1f} queries/sec)"
        )
        return candidates_map

    def mine_hard_negatives(
        self,
        s1_df_country: pd.DataFrame,
        target_df_country: pd.DataFrame,
        gt_map: Dict[str, Set[str]],
        country_name: str,
        sim_min_hard: float = 0.35,
        sim_max_hard: float = 0.65,
        neg_to_pos_ratio: float = 3.5,
    ) -> Tuple[Dict[str, List[str]], Set[str]]:
        """
        Mine hard-negative candidate target records with TF-IDF cosine similarities
        between sim_min_hard (0.35) and sim_max_hard (0.65) that are not true matches,
        maintaining an approximate 3:1 to 4:1 negative-to-positive ratio per entity.
        
        Args:
            s1_df_country: Preprocessed Source 1 records for this country.
            target_df_country: Preprocessed Target records (Source 2 + Source 3).
            gt_map: Mapping from source1_entity_id -> set of true matching target entity_ids.
            country_name: Country identifier string.
            sim_min_hard: Lower bound for hard negative cosine similarity (default 0.35).
            sim_max_hard: Upper bound for hard negative cosine similarity (default 0.65).
            neg_to_pos_ratio: Ratio of negative to positive candidates per entity (default 3.5).
            
        Returns:
            (candidate_dict, mined_negative_target_ids)
        """
        n_queries = len(s1_df_country)
        n_targets = len(target_df_country)
        
        if n_queries == 0 or n_targets == 0:
            return {eid: [] for eid in s1_df_country["entity_id"]}, set()

        logger.info(
            f"[{country_name}] Fitting Stage 1 Word and Stage 2 Char TF-IDF for hard-negative mining on {n_targets:,} targets..."
        )
        target_texts = target_df_country["clean_joint"].tolist()
        target_ids = np.array(target_df_country["entity_id"].tolist())

        w_min_df = self.word_min_df if n_targets >= self.word_min_df else 1
        w_max_df = 1.0 if (isinstance(self.word_max_df, float) and n_targets * self.word_max_df < w_min_df) else self.word_max_df

        c_min_df = self.char_min_df if n_targets >= self.char_min_df else 1
        c_max_df = 1.0 if (isinstance(self.char_max_df, float) and n_targets * self.char_max_df < c_min_df) else self.char_max_df

        vec_word = TfidfVectorizer(
            analyzer=self.word_analyzer,
            ngram_range=self.word_ngram_range,
            min_df=w_min_df,
            max_df=w_max_df,
            sublinear_tf=self.sublinear_tf,
            dtype=np.float32,
        )
        target_word_matrix = vec_word.fit_transform(target_texts)
        target_word_matrix_T = target_word_matrix.T.tocsr()
        target_word_matrix_T.sort_indices()

        vec_char = TfidfVectorizer(
            analyzer=self.char_analyzer,
            ngram_range=self.char_ngram_range,
            min_df=c_min_df,
            max_df=c_max_df,
            max_features=self.char_max_features,
            sublinear_tf=self.sublinear_tf,
            dtype=np.float32,
        )
        target_char_matrix = vec_char.fit_transform(target_texts)

        s1_ids = s1_df_country["entity_id"].tolist()
        s1_joint_texts = s1_df_country["clean_joint"].tolist()

        effective_batch = self.batch_size
        candidates_map: Dict[str, List[str]] = {}
        all_mined_negatives: Set[str] = set()

        logger.info(
            f"[{country_name}] Mining hard negatives for {n_queries:,} queries in sparse batches of {effective_batch:,}..."
        )
        query_start = time.time()

        for batch_idx in range(0, n_queries, effective_batch):
            batch_end = min(batch_idx + effective_batch, n_queries)
            batch_texts = s1_joint_texts[batch_idx:batch_end]
            batch_ids = s1_ids[batch_idx:batch_end]

            query_word = vec_word.transform(batch_texts)
            query_char = vec_char.transform(batch_texts)

            sim_word = query_word.dot(target_word_matrix_T)
            if not isinstance(sim_word, sparse.csr_matrix):
                sim_word = sim_word.tocsr()

            for row_idx, s1_id in enumerate(batch_ids):
                start = sim_word.indptr[row_idx]
                end = sim_word.indptr[row_idx + 1]
                row_cols = sim_word.indices[start:end]
                row_vals = sim_word.data[start:end]
                true_pos = gt_map.get(s1_id, set())

                if len(row_vals) == 0:
                    candidates_map[s1_id] = []
                    continue

                if len(row_vals) > self.coarse_top_k:
                    top_coarse = np.argpartition(-row_vals, self.coarse_top_k)[: self.coarse_top_k]
                    coarse_cols = np.sort(row_cols[top_coarse])
                else:
                    coarse_cols = np.sort(row_cols)

                # Stage 2: sliced char scores
                cand_char_sub = target_char_matrix[coarse_cols]
                scores = cand_char_sub.dot(query_char[row_idx].T).toarray().ravel()

                # Valid candidates meeting minimum floor >= 0.10
                mask = scores >= self.min_similarity
                valid_indices = coarse_cols[mask]
                valid_scores = scores[mask]
                valid_target_ids = target_ids[valid_indices]

                # True positive targets captured in candidates
                pos_cands = [tid for tid in valid_target_ids if tid in true_pos]

                # Partition non-GT candidates into hard negatives [0.35, 0.65] and general negatives
                hard_neg_cands = [
                    tid for tid, score in zip(valid_target_ids, valid_scores)
                    if tid not in true_pos and (sim_min_hard <= score <= sim_max_hard)
                ]
                other_neg_cands = [
                    tid for tid, score in zip(valid_target_ids, valid_scores)
                    if tid not in true_pos and not (sim_min_hard <= score <= sim_max_hard)
                ]

                # Calculate negative budget (approx 3:1 to 4:1 ratio)
                n_pos = len(true_pos)
                if n_pos == 0:
                    n_neg_wanted = 4
                else:
                    n_neg_wanted = min(self.top_k - len(pos_cands), max(3, round(n_pos * neg_to_pos_ratio)))

                # Select hard negatives first, then fill from other negatives
                selected_negs = hard_neg_cands[:n_neg_wanted]
                if len(selected_negs) < n_neg_wanted:
                    rem = n_neg_wanted - len(selected_negs)
                    selected_negs.extend(other_neg_cands[:rem])

                # Total candidates per S1 capped at top_k
                chosen = (pos_cands + selected_negs)[: self.top_k]
                candidates_map[s1_id] = chosen
                all_mined_negatives.update(selected_negs)

        logger.info(
            f"[{country_name}] Mining complete in {time.time() - query_start:.2f}s: "
            f"Mined {len(all_mined_negatives):,} unique hard-negative target records."
        )
        return candidates_map, all_mined_negatives

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
        country_filter: Optional[str] = None,
        limit: Optional[int] = None,
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
        if country_filter:
            countries = [c for c in countries if str(c).lower() == country_filter.lower()]
            if not countries:
                raise ValueError(
                    f"Country filter '{country_filter}' not found in data. "
                    f"Available: {sorted(s1_prep['country'].dropna().unique())}"
                )
        logger.info(f"Discovered countries to process: {countries}")

        all_candidates: Dict[str, List[str]] = {}

        # 3. Process each country partition
        for country in countries:
            s1_country = s1_prep[s1_prep["country"] == country].reset_index(drop=True)
            target_country = target_prep[target_prep["country"] == country].reset_index(drop=True)
            if limit is not None and limit > 0:
                s1_country = s1_country.iloc[:limit].reset_index(drop=True)
                logger.info(
                    f"Applying limit: {len(s1_country):,} S1 queries for '{country}' "
                    f"(full target pool: {len(target_country):,})"
                )

            logger.info(
                f"\n>>> Processing Country Partition: '{country}' "
                f"(S1: {len(s1_country):,}, Targets: {len(target_country):,})"
            )
            
            country_candidates = self.block_country_partition(
                s1_df_country=s1_country,
                target_df_country=target_country,
                country_name=str(country),
            )
            all_candidates.update(country_candidates)

        return all_candidates


def export_candidate_pairs(
    candidates_dict: Dict[str, List[str]],
    output_path: str,
    all_s1_ids: Optional[Sequence[str]] = None,
) -> None:
    """
    Export candidate pairs TSV file strictly matching the official schema.
    Header: source1_entity_id\tcandidate_entity_ids
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    keys = all_s1_ids if all_s1_ids is not None else list(candidates_dict.keys())
    logger.info(f"Writing {len(keys):,} candidate rows to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in keys:
            cands = candidates_dict.get(s1_id, [])
            if isinstance(cands, str):
                cand_str = cands
            elif isinstance(cands, (list, tuple, set)):
                cand_str = ",".join(str(c) for c in cands if str(c).strip())
            else:
                cand_str = ""
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
    parser.add_argument("--top-k", type=int, default=20, help="Top-K final candidates per entity (default: 20)")
    parser.add_argument("--coarse-top-k", type=int, default=300, help="Top-K coarse candidate targets in Stage 1 (default: 300)")
    parser.add_argument("--min-sim", type=float, default=0.10, help="Minimum cosine similarity threshold (default: 0.10)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch query size (default: 1000)")
    parser.add_argument("--word-max-df", type=float, default=0.02, help="Stage 1 word max_df (default: 0.02)")
    parser.add_argument("--char-max-df", type=float, default=0.25, help="Stage 2 char max_df (default: 0.25)")
    parser.add_argument("--max-df", type=float, default=None, help="Alias for word-max-df")
    parser.add_argument("--country", default=None, help="Process only specified country (e.g. France)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of S1 queries per country (for sanity check)")
    args = parser.parse_args()

    start_total = time.time()
    logger.info(f"Loading datasets: S1={args.s1}, S2={args.s2}, S3={args.s3}...")
    df_s1 = pd.read_csv(args.s1, sep="\t", keep_default_na=False)
    df_s2 = pd.read_csv(args.s2, sep="\t", keep_default_na=False)
    df_s3 = pd.read_csv(args.s3, sep="\t", keep_default_na=False)

    w_max_df = args.max_df if args.max_df is not None else args.word_max_df
    blocker = DynamicTFIDFBlocker(
        word_max_df=w_max_df,
        coarse_top_k=args.coarse_top_k,
        char_max_df=args.char_max_df,
        top_k=args.top_k,
        min_similarity=args.min_sim,
        batch_size=args.batch_size,
    )
    candidates_dict = blocker.generate_candidates(
        df_s1, df_s2, df_s3, country_filter=args.country, limit=args.limit
    )

    export_candidate_pairs(candidates_dict, args.out)

    if args.gt and os.path.isfile(args.gt):
        evaluate_candidate_recall(candidates_dict, args.gt)

    logger.info(f"Total blocking runtime: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    main()
