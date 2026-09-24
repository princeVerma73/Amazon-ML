"""
Multi-Index Sparse TF-IDF Blocking Module for Business Entity Resolution.

Implements high-recall, country-partitioned candidate generation using
character n-gram TF-IDF sparse matrix dot products. Strictly enforces
cross-country isolation and outputs formatted candidate pairs adhering
to the Amazon ML Challenge 2026 schema.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalizer import normalize_dataframe

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sparse TF-IDF Blocker Class
# ---------------------------------------------------------------------------

class TFIDFBlocker:
    """
    High-performance, country-partitioned TF-IDF sparse blocking engine.
    
    Generates Top-K candidate pairs for each Source 1 query entity by:
      1. Partitioning entities strictly by country.
      2. Constructing unified representation (normalized name + normalized address).
      3. Fitting character n-gram TF-IDF vectorizers on target corpus (Source 2 + Source 3).
      4. Executing batch sparse matrix multiplications (X_query @ X_target.T).
      5. Extracting Top-K candidates meeting minimum cosine similarity threshold.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        analyzer: str = "char_wb",
        min_df: int = 2,
        top_k: int = 35,
        min_similarity: float = 0.15,
        batch_size: int = 10000,
        sublinear_tf: bool = True,
        max_features: Optional[int] = 250000,
    ) -> None:
        """
        Initialize the TFIDFBlocker with hyperparameters.
        
        Args:
            ngram_range: (min_n, max_n) character n-gram range.
            analyzer: Analyzer type ('char_wb' or 'char').
            min_df: Minimum document frequency for vocabulary pruning.
            top_k: Maximum candidate matches to retrieve per Source 1 entity.
            min_similarity: Minimum cosine similarity cutoff for candidate inclusion.
            batch_size: Batch size for sparse matrix multiplication to conserve RAM.
            sublinear_tf: Apply sublinear scaling 1 + log(tf).
            max_features: Maximum vocabulary size to cap memory usage.
        """
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.min_df = min_df
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features

    def _create_vectorizer(self) -> TfidfVectorizer:
        """Create a fresh TfidfVectorizer instance with configured settings."""
        return TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            max_features=self.max_features,
            dtype=np.float32,
            norm="l2",
        )

    def block_country_partition(
        self,
        s1_ids: Sequence[str],
        s1_texts: Sequence[str],
        target_ids: Sequence[str],
        target_texts: Sequence[str],
        country: str = "UNKNOWN",
    ) -> Dict[str, List[str]]:
        """
        Perform blocking on a single country partition.
        
        Args:
            s1_ids: List of Source 1 entity IDs.
            s1_texts: List of normalized matching strings for Source 1.
            target_ids: List of target (Source 2 + Source 3) entity IDs.
            target_texts: List of normalized matching strings for target corpus.
            country: Name of the country partition for logging.
            
        Returns:
            Dictionary mapping source1_entity_id -> list of candidate entity IDs.
        """
        n_s1 = len(s1_ids)
        n_target = len(target_ids)
        
        logger.info(
            f"Blocking partition '{country}': {n_s1:,} S1 entities vs. "
            f"{n_target:,} Target entities."
        )
        
        results: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_ids}
        
        if n_s1 == 0 or n_target == 0:
            logger.warning(
                f"Partition '{country}' is empty on one side (S1: {n_s1}, Target: {n_target})."
            )
            return results
            
        # 1. Fit vectorizer on target corpus (and transform both)
        start_time = time.time()
        vectorizer = self._create_vectorizer()
        
        # Fit-transform target corpus
        X_target = vectorizer.fit_transform(target_texts)
        logger.info(
            f"[{country}] Target TF-IDF matrix: {X_target.shape[0]:,} rows x "
            f"{X_target.shape[1]:,} features ({X_target.nnz:,} non-zeros). "
            f"Vectorization took {time.time() - start_time:.2f}s."
        )
        
        # Transform Source 1 queries
        X_s1 = vectorizer.transform(s1_texts)
        
        # Transpose target matrix once for efficient CSR dot product
        X_target_T = X_target.T.tocsr()
        target_ids_arr = np.asarray(target_ids)
        
        # 2. Batched sparse matrix multiplication
        total_candidates_found = 0
        match_time = time.time()
        
        for start_idx in range(0, n_s1, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_s1)
            X_batch = X_s1[start_idx:end_idx]
            
            # Compute sparse dot product: (batch_size x vocab) @ (vocab x n_target)
            sim_batch: sparse.csr_matrix = X_batch.dot(X_target_T)
            
            # Extract top-K candidates directly from CSR matrix buffers
            indptr = sim_batch.indptr
            indices = sim_batch.indices
            data = sim_batch.data
            
            for row_i in range(end_idx - start_idx):
                global_s1_idx = start_idx + row_i
                s1_id = s1_ids[global_s1_idx]
                
                row_start = indptr[row_i]
                row_end = indptr[row_i + 1]
                
                if row_start == row_end:
                    continue  # Zero overlap with any target document
                
                row_indices = indices[row_start:row_end]
                row_scores = data[row_start:row_end]
                
                # Apply similarity threshold filter
                valid_mask = row_scores >= self.min_similarity
                if not np.any(valid_mask):
                    continue
                    
                valid_indices = row_indices[valid_mask]
                valid_scores = row_scores[valid_mask]
                n_valid = len(valid_indices)
                
                if n_valid <= self.top_k:
                    # Sort top candidates descending by score
                    sort_order = np.argsort(-valid_scores)
                    selected_target_indices = valid_indices[sort_order]
                else:
                    # Efficient partial sort using argpartition
                    partition_idx = np.argpartition(-valid_scores, self.top_k)[:self.top_k]
                    sub_scores = valid_scores[partition_idx]
                    sub_sort = np.argsort(-sub_scores)
                    selected_target_indices = valid_indices[partition_idx[sub_sort]]
                    
                cand_ids = target_ids_arr[selected_target_indices].tolist()
                results[s1_id] = cand_ids
                total_candidates_found += len(cand_ids)
                
        elapsed = time.time() - match_time
        non_empty_count = sum(1 for v in results.values() if v)
        avg_cands = total_candidates_found / max(1, n_s1)
        logger.info(
            f"[{country}] Blocking complete in {elapsed:.2f}s. "
            f"Matched {non_empty_count:,}/{n_s1:,} S1 entities ({avg_cands:.2f} cands/entity)."
        )
        
        return results

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        source2_df: pd.DataFrame,
        source3_df: pd.DataFrame,
        text_column: str = "norm_joint",
        id_column: str = "entity_id",
        country_column: str = "country",
    ) -> Dict[str, List[str]]:
        """
        Generate candidate pairs partitioned strictly by country.
        
        Args:
            source1_df: Normalized DataFrame for Source 1.
            source2_df: Normalized DataFrame for Source 2.
            source3_df: Normalized DataFrame for Source 3.
            text_column: Column name containing the text representation to match.
            id_column: Column name containing the entity IDs.
            country_column: Column name containing the country classification.
            
        Returns:
            Dictionary mapping every Source 1 entity_id -> list of candidate target IDs.
        """
        logger.info("Initializing multi-country candidate generation pipeline...")
        
        # Combine target sources into unified corpus
        target_df = pd.concat(
            [source2_df[[id_column, text_column, country_column]],
             source3_df[[id_column, text_column, country_column]]],
            ignore_index=True,
        )
        
        logger.info(
            f"Total Corpus: Source 1 = {len(source1_df):,} records, "
            f"Target (S2 + S3) = {len(target_df):,} records."
        )
        
        # Unique countries in Source 1
        s1_countries = source1_df[country_column].dropna().unique()
        all_candidates: Dict[str, List[str]] = {}
        
        total_start = time.time()
        for country in s1_countries:
            s1_subset = source1_df[source1_df[country_column] == country]
            target_subset = target_df[target_df[country_column] == country]
            
            s1_ids = s1_subset[id_column].tolist()
            s1_texts = s1_subset[text_column].tolist()
            
            target_ids = target_subset[id_column].tolist()
            target_texts = target_subset[text_column].tolist()
            
            country_candidates = self.block_country_partition(
                s1_ids=s1_ids,
                s1_texts=s1_texts,
                target_ids=target_ids,
                target_texts=target_texts,
                country=str(country),
            )
            all_candidates.update(country_candidates)
            
        # Ensure any missing Source 1 IDs receive empty list
        for s1_id in source1_df[id_column]:
            if s1_id not in all_candidates:
                all_candidates[s1_id] = []
                
        logger.info(
            f"All partitions processed in {time.time() - total_start:.2f}s. "
            f"Total S1 entities indexed: {len(all_candidates):,}."
        )
        return all_candidates


# ---------------------------------------------------------------------------
# Submission-Compliant Export Function
# ---------------------------------------------------------------------------

def export_candidate_pairs(
    candidates_dict: Dict[str, List[str]],
    output_path: str = "output/candidate_pairs.tsv",
    all_s1_ids: Optional[Sequence[str]] = None,
) -> str:
    """
    Export candidate pairs to TSV adhering strictly to the competition format.
    
    Validation Guarantees:
      - Tab-separated header: source1_entity_id\\tcandidate_entity_ids
      - Exactly one row per required Source 1 entity.
      - Candidate IDs formatted as comma-separated list without quotes or spaces.
      - Empty string in candidate column for singletons / unmatched entities.
      - Zero self-matches (S1- IDs).
      - UTF-8 encoding.
      
    Args:
        candidates_dict: Mapping of s1_entity_id -> list of candidate IDs.
        output_path: Destination path for the TSV file.
        all_s1_ids: Optional ordered list of required Source 1 IDs. If provided,
                    the output file will follow this exact row order.
                    
    Returns:
        Absolute path to the created candidate pairs file.
    """
    # Create target directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Determine the complete ordered sequence of S1 IDs
    if all_s1_ids is not None:
        ordered_ids = all_s1_ids
    else:
        ordered_ids = list(candidates_dict.keys())
        
    logger.info(f"Exporting candidate pairs for {len(ordered_ids):,} entities to: {output_path}")
    
    total_written = 0
    non_empty_count = 0
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        # Write exact required header
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in ordered_ids:
            cands = candidates_dict.get(s1_id, [])
            
            # Filter any accidental self-matches (safety check)
            clean_cands = [c for c in cands if not c.startswith("S1-")]
            
            if clean_cands:
                cand_str = ",".join(clean_cands)
                non_empty_count += 1
            else:
                cand_str = ""
                
            f.write(f"{s1_id}\t{cand_str}\n")
            total_written += 1
            
    logger.info(
        f"Export successful. Written {total_written:,} rows "
        f"({non_empty_count:,} non-empty, {total_written - non_empty_count:,} empty singletons)."
    )
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# High-Level Blocking Pipeline Runner
# ---------------------------------------------------------------------------

def run_blocking_pipeline(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str = "output/candidate_pairs.tsv",
    top_k: int = 35,
    min_similarity: float = 0.15,
) -> Dict[str, List[str]]:
    """
    Load raw TSV files, normalize them, run TF-IDF blocking, and export candidate TSV.
    
    Args:
        s1_path: Path to source1.tsv.
        s2_path: Path to source2.tsv.
        s3_path: Path to source3.tsv.
        output_path: Path to candidate_pairs.tsv.
        top_k: Top K candidates per entity.
        min_similarity: Cosine similarity cutoff.
        
    Returns:
        Generated candidates dictionary.
    """
    logger.info(f"Loading data: S1='{s1_path}', S2='{s2_path}', S3='{s3_path}'")
    
    # Load raw TSV files
    df_s1 = pd.read_csv(s1_path, sep="\t", encoding="utf-8")
    df_s2 = pd.read_csv(s2_path, sep="\t", encoding="utf-8")
    df_s3 = pd.read_csv(s3_path, sep="\t", encoding="utf-8")
    
    # Normalize datasets
    logger.info("Normalizing Source 1...")
    df_s1 = normalize_dataframe(df_s1)
    logger.info("Normalizing Source 2...")
    df_s2 = normalize_dataframe(df_s2)
    logger.info("Normalizing Source 3...")
    df_s3 = normalize_dataframe(df_s3)
    
    # Initialize Blocker
    blocker = TFIDFBlocker(
        ngram_range=(3, 4),
        analyzer="char_wb",
        min_df=2,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    
    # Generate candidates
    candidates = blocker.generate_candidates(
        source1_df=df_s1,
        source2_df=df_s2,
        source3_df=df_s3,
    )
    
    # Export compliant TSV
    export_candidate_pairs(
        candidates_dict=candidates,
        output_path=output_path,
        all_s1_ids=df_s1["entity_id"].tolist(),
    )
    
    return candidates
