"""
Phase 3: Pairwise Feature Engineering Module.

Extracts SIMD-accelerated string similarity, phonetic, and numeric overlap features
for candidate pairs generated in Phase 2 blocking.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

# Configure logger
logger = logging.getLogger(__name__)

NUMERIC_REGEX = re.compile(r"\b\d+\b")
PIN_POSTAL_REGEX = re.compile(r"\b\d{5,6}\b")

FEATURE_COLUMNS = [
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_levenshtein_norm",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    "numeric_token_overlap",
    "postal_pin_exact_match",
    "len_diff_name",
    "len_diff_addr",
    "is_addr_missing",
    "blocking_rank",
]


def extract_numeric_tokens(text: str) -> Set[str]:
    """Extract all numeric digit sequences from string."""
    if not text:
        return set()
    return set(NUMERIC_REGEX.findall(text))


def extract_pin_tokens(text: str) -> Set[str]:
    """Extract 5 or 6 digit PIN/ZIP codes from string."""
    if not text:
        return set()
    return set(PIN_POSTAL_REGEX.findall(text))


def compute_pair_features(
    name1: str,
    addr1: str,
    name2: str,
    addr2: str,
    is_addr_missing2: int,
    rank: int,
) -> List[float]:
    """
    Compute 13-dimensional dense feature vector for a single (S1, Target) entity pair.
    """
    # 1. Name String Similarities (SIMD C++ via RapidFuzz)
    n_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
    n_set = fuzz.token_set_ratio(name1, name2) / 100.0
    n_jw = distance.JaroWinkler.similarity(name1, name2)
    n_lev = distance.Levenshtein.normalized_similarity(name1, name2)

    # 2. Address String Similarities
    if addr1 and addr2:
        a_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        a_set = fuzz.token_set_ratio(addr1, addr2) / 100.0
        a_jw = distance.JaroWinkler.similarity(addr1, addr2)
    else:
        a_sort = 0.0
        a_set = 0.0
        a_jw = 0.0

    # 3. Numeric & Postal Token Overlap
    joint1 = f"{name1} {addr1}"
    joint2 = f"{name2} {addr2}"
    nums1 = extract_numeric_tokens(joint1)
    nums2 = extract_numeric_tokens(joint2)
    if nums1 and nums2:
        num_overlap = len(nums1.intersection(nums2)) / len(nums1.union(nums2))
    else:
        num_overlap = 0.0

    pins1 = extract_pin_tokens(addr1)
    pins2 = extract_pin_tokens(addr2)
    if pins1 and pins2 and len(pins1.intersection(pins2)) > 0:
        pin_match = 1.0
    else:
        pin_match = 0.0

    # 4. Length Ratios & Indicators
    len_name1, len_name2 = len(name1), len(name2)
    max_len_name = max(len_name1, len_name2, 1)
    len_diff_n = abs(len_name1 - len_name2) / max_len_name

    len_addr1, len_addr2 = len(addr1), len(addr2)
    max_len_addr = max(len_addr1, len_addr2, 1)
    len_diff_a = abs(len_addr1 - len_addr2) / max_len_addr if (addr1 and addr2) else 1.0

    return [
        n_sort,
        n_set,
        n_jw,
        n_lev,
        a_sort,
        a_set,
        a_jw,
        num_overlap,
        pin_match,
        len_diff_n,
        len_diff_a,
        float(is_addr_missing2),
        float(rank),
    ]


def build_candidate_feature_matrix(
    candidate_dict: Dict[str, List[str]],
    s1_preprocessed: pd.DataFrame,
    target_preprocessed: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix for all candidate pairs in batch.
    
    Args:
        candidate_dict: Mapping from source1_entity_id -> list of candidate target entity_ids.
        s1_preprocessed: Preprocessed S1 DataFrame with 'entity_id', 'clean_name', 'clean_addr'.
        target_preprocessed: Preprocessed Target DataFrame (S2 + S3) with 'entity_id', 'clean_name', 'clean_addr', 'is_addr_missing'.
        
    Returns:
        (pairs_meta_df, features_df)
    """
    # Index reference records
    s1_map = {
        row["entity_id"]: (row["clean_name"], row["clean_addr"])
        for _, row in s1_preprocessed[["entity_id", "clean_name", "clean_addr"]].iterrows()
    }

    target_map = {
        row["entity_id"]: (row["clean_name"], row["clean_addr"], row.get("is_addr_missing", 0))
        for _, row in target_preprocessed[["entity_id", "clean_name", "clean_addr", "is_addr_missing"]].iterrows()
    }

    pairs_s1: List[str] = []
    pairs_target: List[str] = []
    feature_rows: List[List[float]] = []

    for s1_id, candidate_ids in candidate_dict.items():
        if s1_id not in s1_map or not candidate_ids:
            continue
        n1, a1 = s1_map[s1_id]

        for rank, target_id in enumerate(candidate_ids, start=1):
            if target_id not in target_map:
                continue
            n2, a2, missing2 = target_map[target_id]
            feats = compute_pair_features(n1, a1, n2, a2, missing2, rank)

            pairs_s1.append(s1_id)
            pairs_target.append(target_id)
            feature_rows.append(feats)

    meta_df = pd.DataFrame({"source1_entity_id": pairs_s1, "target_entity_id": pairs_target})
    feats_df = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    return meta_df, feats_df
