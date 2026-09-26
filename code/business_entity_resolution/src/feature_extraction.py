"""
Phase 3: Pairwise Feature Engineering Module.

Extracts SIMD-accelerated string similarity, phonetic, numeric/pincode overlap,
domain match, and script mismatch features for candidate pairs generated in Phase 2 blocking.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

try:
    from .normalizer import extract_domain_stem, clean_domain_stem
except (ImportError, ValueError):
    from normalizer import extract_domain_stem, clean_domain_stem

# Configure logger
logger = logging.getLogger(__name__)

# Numeric & Postal / PIN regexes
NUMERIC_REGEX = re.compile(r"\b\d+\b")
PIN_POSTAL_REGEX = re.compile(r"\b\d{5,6}\b")

# Premises, house, and plot number regex patterns
HOUSE_PLOT_PREFIX_REGEX = re.compile(
    r"\b(?:plot|polt|h\.?no|house|flat|f\.?no|d\.?no|door|unit|shop|suite|ste|apt|bldg)\b(?:\s*no\.?)?\s*[:#-]?\s*([a-zA-Z0-9/-]+)",
    flags=re.IGNORECASE,
)
NO_NUM_REGEX = re.compile(
    r"\bno\.?\s*[:#-]?\s*(\d+[a-zA-Z0-9/-]*)",
    flags=re.IGNORECASE,
)
LEADING_HOUSE_NUM_REGEX = re.compile(r"(?:^|,\s*)(\d{1,5}[a-zA-Z]?)\b")
COMPOUND_HOUSE_NUM_REGEX = re.compile(r"\b(\d{1,4}[-/]\d{1,4}(?:[-/]\d{1,4})?)\b")

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
    "numeric_pincode_match",
    "domain_match",
    "script_mismatch",
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


def extract_house_plot_tokens(text: str) -> Set[str]:
    """
    Extract house, plot, flat, unit, and premise numbers from address text.
    """
    if not text:
        return set()
    tokens: Set[str] = set()

    # 1. Explicit prefix matches (e.g., 'Plot No. 14', 'H.No 162', 'Flat No. 301')
    for match in HOUSE_PLOT_PREFIX_REGEX.finditer(text):
        val = match.group(1).lower().strip(".-#/ ")
        if val and any(c.isdigit() for c in val):
            tokens.add(val)
            tokens.update(re.findall(r"\d+", val))

    # 2. 'No. 52' patterns
    for match in NO_NUM_REGEX.finditer(text):
        val = match.group(1).lower().strip(".-#/ ")
        if val:
            tokens.add(val)
            tokens.update(re.findall(r"\d+", val))

    # 3. Leading house/street numbers (e.g., '3309 Terrier Ln', '611 South 38th St')
    for match in LEADING_HOUSE_NUM_REGEX.finditer(text):
        val = match.group(1).lower().strip()
        if val:
            tokens.add(val)
            tokens.add(val.lstrip("0") or "0")

    # 4. Compound numbers (e.g., '4-61/28')
    for match in COMPOUND_HOUSE_NUM_REGEX.finditer(text):
        val = match.group(1).lower().strip(".-#/ ")
        if val:
            tokens.add(val)
            tokens.update(re.findall(r"\d+", val))

    return tokens


def extract_numeric_pincode_tokens(text: str) -> Set[str]:
    """
    Extract 5-6 digit postal codes and house/plot numbers from address string.
    
    Combines postal PIN/ZIP codes and specific house/plot tokens for robust
    address identifier matching across jurisdictions.
    """
    if not text:
        return set()
    pins = extract_pin_tokens(text)
    houses = extract_house_plot_tokens(text)
    return pins.union(houses)


def compute_numeric_pincode_match(tokens1: Set[str], tokens2: Set[str]) -> float:
    """
    Numeric/pincode matching: 1.0 if any overlap between extracted
    postal codes and house/plot numbers, else 0.0.
    """
    if tokens1 and tokens2 and len(tokens1.intersection(tokens2)) > 0:
        return 1.0
    return 0.0


def compute_domain_match(
    domain1: Optional[str],
    domain2: Optional[str],
    name1: Optional[str] = None,
    name2: Optional[str] = None,
) -> float:
    """
    Domain match logic: 1.0 if domain stems extracted from entity_1 and
    entity_2 match and at least one entity contains a domain stem, else 0.0.
    
    Supports matching between two domain stems (e.g., 'www.trustedin.com' vs 'trustedin.com')
    as well as brand-name-to-domain stems (e.g., 'helainasorrellclean' vs 'helainasorrellclean.com').
    """
    d1 = domain1.strip().lower() if (domain1 and isinstance(domain1, str)) else ""
    d2 = domain2.strip().lower() if (domain2 and isinstance(domain2, str)) else ""
    has1 = bool(d1)
    has2 = bool(d2)

    if not has1 and not has2:
        return 0.0

    # If both have explicit domain stems
    if has1 and has2:
        return 1.0 if d1 == d2 else 0.0

    # If one has domain stem and the other has brand name
    s1 = d1 if has1 else re.sub(r"[^a-z0-9]", "", (name1 or "").lower())
    s2 = d2 if has2 else re.sub(r"[^a-z0-9]", "", (name2 or "").lower())
    if s1 and s2 and s1 == s2:
        return 1.0

    return 0.0


def compute_script_mismatch(text1: str, text2: str) -> float:
    """
    Script mismatch indicator: 1.0 if one text has non-ASCII characters
    (e.g., Hindi/Telugu/Indic) and the other is pure ASCII, else 0.0.
    """
    if not text1 or not text2:
        return 0.0
    ascii1 = text1.isascii()
    ascii2 = text2.isascii()
    return 1.0 if (ascii1 != ascii2) else 0.0


def compute_pair_features(
    name1: str,
    addr1: str,
    name2: str,
    addr2: str,
    is_addr_missing2: int,
    rank: int,
    raw_name1: Optional[str] = None,
    raw_name2: Optional[str] = None,
    domain1: Optional[str] = None,
    domain2: Optional[str] = None,
    nums1: Optional[Set[str]] = None,
    nums2: Optional[Set[str]] = None,
    pins1: Optional[Set[str]] = None,
    pins2: Optional[Set[str]] = None,
    pincode_house1: Optional[Set[str]] = None,
    pincode_house2: Optional[Set[str]] = None,
) -> List[float]:
    """
    Compute dense feature vector for a single (S1, Target) entity pair.
    
    Includes SIMD string similarities, address similarities, numeric token Jaccard,
    postal PIN exact match, postal+house number overlap match, domain stem match,
    and script mismatch indicator. Supports pre-cached token sets for ultra-high throughput.
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

    # 3. Numeric & Postal Token Overlap (using pre-cached tokens when available)
    if nums1 is None:
        nums1 = extract_numeric_tokens(f"{name1} {addr1}")
    if nums2 is None:
        nums2 = extract_numeric_tokens(f"{name2} {addr2}")
    if nums1 and nums2:
        num_overlap = len(nums1.intersection(nums2)) / len(nums1.union(nums2))
    else:
        num_overlap = 0.0

    if pins1 is None:
        pins1 = extract_pin_tokens(addr1)
    if pins2 is None:
        pins2 = extract_pin_tokens(addr2)
    pin_match = 1.0 if (pins1 and pins2 and len(pins1.intersection(pins2)) > 0) else 0.0

    # 4. Numeric / Pincode & House/Plot Overlap (1.0 if any overlap, else 0.0)
    if pincode_house1 is None:
        pincode_house1 = extract_numeric_pincode_tokens(addr1)
    if pincode_house2 is None:
        pincode_house2 = extract_numeric_pincode_tokens(addr2)
    num_pin_match = compute_numeric_pincode_match(pincode_house1, pincode_house2)

    # 5. Domain Match Logic (1.0 if non-empty domain stems match, else 0.0)
    dom_stem1 = domain1 if domain1 is not None else extract_domain_stem(raw_name1 or name1)
    dom_stem2 = domain2 if domain2 is not None else extract_domain_stem(raw_name2 or name2)
    dom_match = compute_domain_match(dom_stem1, dom_stem2, name1=name1, name2=name2)

    # 6. Script Mismatch Indicator (1.0 if one text has non-ASCII and other is pure ASCII)
    str1 = raw_name1 if raw_name1 is not None else name1
    str2 = raw_name2 if raw_name2 is not None else name2
    script_mismatch = compute_script_mismatch(str1, str2)

    # 7. Length Ratios & Indicators
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
        num_pin_match,
        dom_match,
        script_mismatch,
        len_diff_n,
        len_diff_a,
        float(is_addr_missing2),
        float(rank),
    ]


def build_candidate_feature_matrix(
    candidate_dict: Dict[str, List[str]],
    s1_preprocessed: pd.DataFrame,
    target_preprocessed: Optional[pd.DataFrame] = None,
    target_map: Optional[Dict[str, Tuple[str, str, int, str, str]]] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix for all candidate pairs in batch.
    
    Args:
        candidate_dict: Mapping from source1_entity_id -> list of candidate target entity_ids.
        s1_preprocessed: Preprocessed S1 DataFrame with 'entity_id', 'clean_name', 'clean_addr', 'raw_name', 'domain_stem'.
        target_preprocessed: Preprocessed Target DataFrame (S2 + S3).
        target_map: Optional precomputed target lookup dictionary for chunked streaming inference.
        show_progress: Whether to show a progress bar.
        
    Returns:
        (pairs_meta_df, features_df)
    """
    # Fast vectorized indexing of reference records
    s1_ids = s1_preprocessed["entity_id"].to_numpy()
    s1_cnames = s1_preprocessed["clean_name"].to_numpy()
    s1_caddrs = s1_preprocessed["clean_addr"].to_numpy()
    s1_rnames = s1_preprocessed["raw_name"].to_numpy() if "raw_name" in s1_preprocessed.columns else s1_cnames
    s1_dstems = s1_preprocessed["domain_stem"].to_numpy() if "domain_stem" in s1_preprocessed.columns else np.array([""] * len(s1_ids))

    s1_map = {
        s1_ids[i]: (s1_cnames[i], s1_caddrs[i], s1_rnames[i], s1_dstems[i])
        for i in range(len(s1_ids))
    }

    if target_map is None:
        if target_preprocessed is None:
            raise ValueError("Either target_preprocessed or target_map must be provided.")
        t_ids = target_preprocessed["entity_id"].to_numpy()
        t_cnames = target_preprocessed["clean_name"].to_numpy()
        t_caddrs = target_preprocessed["clean_addr"].to_numpy()
        t_miss = (
            target_preprocessed["is_addr_missing"].to_numpy()
            if "is_addr_missing" in target_preprocessed.columns
            else np.zeros(len(t_ids), dtype=np.int32)
        )
        t_rnames = target_preprocessed["raw_name"].to_numpy() if "raw_name" in target_preprocessed.columns else t_cnames
        t_dstems = target_preprocessed["domain_stem"].to_numpy() if "domain_stem" in target_preprocessed.columns else np.array([""] * len(t_ids))

        target_map = {
            t_ids[i]: (t_cnames[i], t_caddrs[i], int(t_miss[i]), t_rnames[i], t_dstems[i])
            for i in range(len(t_ids))
        }

    pairs_s1: List[str] = []
    pairs_target: List[str] = []
    feature_rows: List[List[float]] = []

    # Pre-extract token sets for reference S1 entities
    s1_token_cache: Dict[str, Tuple[Set[str], Set[str], Set[str]]] = {}
    for s1_id in candidate_dict.keys():
        if s1_id in s1_map:
            n1, a1, _, _ = s1_map[s1_id]
            s1_token_cache[s1_id] = (
                extract_numeric_tokens(f"{n1} {a1}"),
                extract_pin_tokens(a1),
                extract_numeric_pincode_tokens(a1),
            )

    # Dynamic target token cache: populated once per unique target seen
    target_token_cache: Dict[str, Tuple[Set[str], Set[str], Set[str]]] = {}

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                candidate_dict.items(),
                desc="Extracting Features",
                unit="queries",
                dynamic_ncols=True,
            )
        except ImportError:
            iterator = candidate_dict.items()
    else:
        iterator = candidate_dict.items()

    for s1_id, candidate_ids in iterator:
        if s1_id not in s1_map or not candidate_ids:
            continue
        n1, a1, raw_n1, dom1 = s1_map[s1_id]
        nums1, pins1, pinh1 = s1_token_cache[s1_id]

        for rank, target_id in enumerate(candidate_ids, start=1):
            if target_id not in target_map:
                continue
            n2, a2, missing2, raw_n2, dom2 = target_map[target_id]

            if target_id not in target_token_cache:
                target_token_cache[target_id] = (
                    extract_numeric_tokens(f"{n2} {a2}"),
                    extract_pin_tokens(a2),
                    extract_numeric_pincode_tokens(a2),
                )
            nums2, pins2, pinh2 = target_token_cache[target_id]

            feats = compute_pair_features(
                name1=n1,
                addr1=a1,
                name2=n2,
                addr2=a2,
                is_addr_missing2=missing2,
                rank=rank,
                raw_name1=raw_n1,
                raw_name2=raw_n2,
                domain1=dom1,
                domain2=dom2,
                nums1=nums1,
                nums2=nums2,
                pins1=pins1,
                pins2=pins2,
                pincode_house1=pinh1,
                pincode_house2=pinh2,
            )

            pairs_s1.append(s1_id)
            pairs_target.append(target_id)
            feature_rows.append(feats)

    meta_df = pd.DataFrame({"source1_entity_id": pairs_s1, "target_entity_id": pairs_target})
    feats_df = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    return meta_df, feats_df


__all__ = [
    "FEATURE_COLUMNS",
    "extract_numeric_tokens",
    "extract_pin_tokens",
    "extract_house_plot_tokens",
    "extract_numeric_pincode_tokens",
    "compute_numeric_pincode_match",
    "compute_domain_match",
    "compute_script_mismatch",
    "compute_pair_features",
    "build_candidate_feature_matrix",
]
