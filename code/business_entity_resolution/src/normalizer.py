"""
Phase 1: Data Normalization & Cleaning Module for Business Entity Resolution.

Provides deterministic string cleaning, Unicode NFKD diacritics stripping (supporting
English, Indian, and French datasets), legal entity suffix removal, address contraction
expansion, missing address fallback handling, and memory-efficient DataFrame preprocessing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Union
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Precompiled Regex Patterns for Maximum Throughput
# ---------------------------------------------------------------------------

# Legal Entity Suffixes to remove from business names across US, India, and France
# Sorted by descending token length to ensure longer phrases (e.g., 'private limited')
# are matched prior to single-token substrings (e.g., 'ltd')
LEGAL_SUFFIXES = [
    r"private\s+limited",
    r"pvt\s+ltd",
    r"incorporated",
    r"corporation",
    r"enterprises",
    r"associates",
    r"industries",
    r"solutions",
    r"services",
    r"limited",
    r"company",
    r"pvt",
    r"ltd",
    r"inc",
    r"corp",
    r"l\.?l\.?c\.?",
    r"l\s+l\s+c",
    r"llc",
    r"l\.?l\.?p\.?",
    r"l\s+l\s+p",
    r"llp",
    r"p\.?l\.?l\.?c\.?",
    r"pllc",
    r"s\.?a\.?r\.?l\.?",
    r"s\s+a\s+r\s+l",
    r"sarl",
    r"s\.?a\.?s\.?u\.?",
    r"s\s+a\s+s\s+u",
    r"sasu",
    r"e\.?u\.?r\.?l\.?",
    r"e\s+u\s+r\s+l",
    r"eurl",
    r"s\.?a\.?s\.?",
    r"s\s+a\s+s",
    r"sas",
    r"gmbh",
    r"s\.?a\.?",
    r"s\s+a",
    r"sa",
    r"sci",
    r"snc",
    r"gie",
    r"c\s+o",
    r"co",
]

LEGAL_SUFFIX_REGEX = re.compile(
    rf"\b({'|'.join(LEGAL_SUFFIXES)})\b",
    flags=re.IGNORECASE,
)

# Address contractions and directional expansions
ADDRESS_CONTRACTIONS = {
    # Primary Road / Street types
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bpkg\b": "parking",
    r"\bhwy\b": "highway",
    r"\brte\b": "route",
    r"\br\b": "rue",
    # Landmark and unit indicators
    r"\bopp\b": "opposite",
    r"\bb/h\b": "behind",
    r"\bnr\b": "near",
    r"\bapt\b": "apartment",
    r"\bflt\b": "flat",
    r"\bste\b": "suite",
    r"\bbldg\b": "building",
    r"\bfl\b": "floor",
    r"\bh\.?no\.?\b": "house number",
    r"\bhno\b": "house number",
    # Cardinal directions
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bne\b": "northeast",
    r"\bnw\b": "northwest",
    r"\bse\b": "southeast",
    r"\bsw\b": "southwest",
}

ADDRESS_CONTRACTION_PATTERNS = [
    (re.compile(pattern, flags=re.IGNORECASE), replacement)
    for pattern, replacement in ADDRESS_CONTRACTIONS.items()
]

# Character cleaning & whitespace regexes
AMPERSAND_REGEX = re.compile(r"&")
NON_ALPHANUMERIC_REGEX = re.compile(r"[^a-z0-9\s]")
WHITESPACE_REGEX = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Domain Stem Regex Patterns & Extraction
# ---------------------------------------------------------------------------

# Domain URL token pattern: matches protocol, www, domain label, and common TLDs
DOMAIN_TOKEN_REGEX = re.compile(
    r"(?:https?://)?(?:www\d*\.)?[a-zA-Z0-9][-a-zA-Z0-9.]*\.(?:co\.in|org\.in|com|in|org|fr|net|co|io|biz|info|gov|edu|eu|us)\b(?:/[^\s]*)?",
    flags=re.IGNORECASE,
)
DOMAIN_PROTOCOL_WWW_REGEX = re.compile(
    r"^(?:https?://)?(?:www\d*\.)?",
    flags=re.IGNORECASE,
)
DOMAIN_TLD_SUFFIX_REGEX = re.compile(
    r"\.(?:co\.in|org\.in|com|in|org|fr|net|co|io|biz|info|gov|edu|eu|us)$",
    flags=re.IGNORECASE,
)
DOMAIN_INVALID_CHARS_REGEX = re.compile(r"[^a-z0-9-]")


def extract_domain_stem(s: Optional[Union[str, float]]) -> str:
    """
    Clean and extract domain stems from business names, text payloads, or URLs.
    
    Strips protocol ('https://', 'http://'), 'www.' prefixes, paths, and common
    TLDs (e.g., '.com', '.in', '.org', '.fr', '.co.in', '.net', etc.) using regex
    to isolate the script- and domain-invariant brand stem.
    
    Args:
        s: Raw input text string or missing value.
        
    Returns:
        Extracted lowercase domain stem, or empty string if no valid domain found.
    """
    if s is None or not isinstance(s, str):
        return ""
    
    s_clean = s.strip()
    tokens = DOMAIN_TOKEN_REGEX.findall(s_clean)
    if not tokens:
        return ""
    
    candidate = tokens[0]
    # Strip protocol and www
    candidate = DOMAIN_PROTOCOL_WWW_REGEX.sub("", candidate)
    # Strip path / query
    candidate = candidate.split("/")[0].split("?")[0]
    # Strip known TLD extension
    stem = DOMAIN_TLD_SUFFIX_REGEX.sub("", candidate)
    # If subdomains exist (e.g. shop.brand), select core brand domain stem
    parts = stem.split(".")
    domain = parts[-1]
    
    cleaned = DOMAIN_INVALID_CHARS_REGEX.sub("", domain.lower()).strip("-")
    return cleaned


def clean_domain_stem(s: Optional[Union[str, float]]) -> str:
    """
    Clean and extract domain stem from input text.
    Alias for extract_domain_stem.
    """
    return extract_domain_stem(s)


# ---------------------------------------------------------------------------
# Core Normalization Functions
# ---------------------------------------------------------------------------

def normalize_text(s: Optional[Union[str, float]]) -> str:
    """
    Normalize text via Unicode NFKD decomposition, lowercase conversion,
    ampersand replacement, and non-alphanumeric removal.
    
    Handles accented Latin characters in French test records (e.g., 'é' -> 'e', 'ç' -> 'c').
    
    Args:
        s: Raw input text string or missing value.
        
    Returns:
        Cleaned lowercase ASCII-alphanumeric string.
    """
    if s is None or not isinstance(s, str):
        return ""
    
    # NFKD decomposition to separate base letters from diacritical marks
    decomposed = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    
    # Lowercase conversion
    text = stripped.lower()
    
    # Replace ampersands with 'and'
    text = AMPERSAND_REGEX.sub(" and ", text)
    
    # Replace non-alphanumeric characters with spaces
    text = NON_ALPHANUMERIC_REGEX.sub(" ", text)
    
    # Collapse multiple whitespace characters into a single space
    return WHITESPACE_REGEX.sub(" ", text).strip()


def clean_legal_suffixes(s: Optional[Union[str, float]]) -> str:
    """
    Remove corporate designations and legal entity suffixes across US, Indian,
    and French business jurisdictions using word boundary matching.
    
    Args:
        s: Business name string.
        
    Returns:
        Business name with legal entity suffixes removed.
    """
    if not s or not isinstance(s, str):
        return ""
    cleaned = LEGAL_SUFFIX_REGEX.sub(" ", s)
    return WHITESPACE_REGEX.sub(" ", cleaned).strip()


def clean_address(s: Optional[Union[str, float]]) -> str:
    """
    Clean business address, expanding contractions and handling NaN/missing values.
    
    Args:
        s: Raw address string or missing value.
        
    Returns:
        Cleaned and expanded address string, or empty string if missing.
    """
    if s is None or not isinstance(s, str):
        return ""
    
    # NFKD decomposition and lowercase
    decomposed = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    text = stripped.lower()
    
    # Replace ampersands
    text = AMPERSAND_REGEX.sub(" and ", text)
    
    # Expand address contractions
    for pattern, replacement in ADDRESS_CONTRACTION_PATTERNS:
        text = pattern.sub(replacement, text)
        
    # Replace non-alphanumeric punctuation
    text = NON_ALPHANUMERIC_REGEX.sub(" ", text)
    
    # Collapse whitespace
    return WHITESPACE_REGEX.sub(" ", text).strip()


def preprocess_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    addr_col: str = "business_address",
    inplace: bool = False,
    desc: str = "Preprocessing",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Preprocess an entity DataFrame with vectorized high-speed string normalization.
    
    Adds the following processed columns:
      - 'clean_name': Normalized name stripped of legal entity suffixes.
      - 'clean_addr': Normalized and expanded address (empty string if NaN).
      - 'is_addr_missing': Binary indicator (1 if original address was null/blank, 0 otherwise).
      - 'clean_joint': Weighted composite string ('clean_name clean_name clean_addr') for TF-IDF blocking.
      
    Args:
        df: Input DataFrame containing entity records.
        name_col: Column name containing business names.
        addr_col: Column name containing business addresses.
        inplace: Whether to mutate input DataFrame or return a shallow copy.
        desc: Description label for tqdm progress bar.
        show_progress: Whether to show live tqdm progress bar.
        
    Returns:
        Preprocessed DataFrame with cleaned text and indicator columns.
    """
    target_df = df if inplace else df.copy(deep=False)
    
    # 1. Address missing indicator
    if addr_col in target_df.columns:
        addr_series = target_df[addr_col]
        target_df["is_addr_missing"] = (
            addr_series.isna() | (addr_series.astype(str).str.strip() == "") | (addr_series.astype(str).str.lower() == "nan")
        ).astype(int)
        raw_addrs = addr_series.fillna("").astype(str).tolist()
    else:
        target_df["is_addr_missing"] = 1
        raw_addrs = [""] * len(target_df)
        
    # 2. Extract and process names
    if name_col in target_df.columns:
        raw_names = target_df[name_col].fillna("").astype(str).tolist()
    else:
        raw_names = [""] * len(target_df)
        
    total_rows = len(target_df)
    chunk_size = 100000

    if show_progress and total_rows > chunk_size:
        clean_names: List[str] = []
        clean_addrs: List[str] = []
        domain_stems: List[str] = []
        has_non_ascii: List[bool] = []
        clean_joint: List[str] = []

        with tqdm(total=total_rows, desc=desc, unit="records", dynamic_ncols=True) as pbar:
            for i in range(0, total_rows, chunk_size):
                end_i = min(i + chunk_size, total_rows)
                sub_names = raw_names[i:end_i]
                sub_addrs = raw_addrs[i:end_i]

                c_n = [clean_legal_suffixes(normalize_text(name)) for name in sub_names]
                c_a = [clean_address(addr) for addr in sub_addrs]
                d_s = [extract_domain_stem(name) or extract_domain_stem(addr) for name, addr in zip(sub_names, sub_addrs)]
                h_na = [not (name.isascii() if isinstance(name, str) else True) for name in sub_names]
                c_j = [f"{n} {n} {a}".strip() if a else f"{n} {n}".strip() for n, a in zip(c_n, c_a)]

                clean_names.extend(c_n)
                clean_addrs.extend(c_a)
                domain_stems.extend(d_s)
                has_non_ascii.extend(h_na)
                clean_joint.extend(c_j)
                pbar.update(end_i - i)
    else:
        clean_names = [clean_legal_suffixes(normalize_text(name)) for name in raw_names]
        clean_addrs = [clean_address(addr) for addr in raw_addrs]
        domain_stems = [extract_domain_stem(name) or extract_domain_stem(addr) for name, addr in zip(raw_names, raw_addrs)]
        has_non_ascii = [not (name.isascii() if isinstance(name, str) else True) for name in raw_names]
        clean_joint = [f"{n} {n} {a}".strip() if a else f"{n} {n}".strip() for n, a in zip(clean_names, clean_addrs)]

    target_df["clean_name"] = clean_names
    target_df["clean_addr"] = clean_addrs
    target_df["raw_name"] = raw_names
    target_df["domain_stem"] = domain_stems
    target_df["has_non_ascii"] = has_non_ascii
    target_df["clean_joint"] = clean_joint
    
    return target_df


# ---------------------------------------------------------------------------
# Compatibility Aliases for Downstream Modules
# ---------------------------------------------------------------------------

def normalize_business_name(text: Optional[str]) -> str:
    """Compatibility alias for clean_legal_suffixes(normalize_text(text))."""
    return clean_legal_suffixes(normalize_text(text))


def normalize_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    addr_col: str = "business_address",
    create_joint: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compatibility wrapper mapping to preprocess_dataframe with alias column names.
    """
    res = preprocess_dataframe(df, name_col=name_col, addr_col=addr_col, inplace=inplace)
    res["norm_business_name"] = res["clean_name"]
    res["norm_business_address"] = res["clean_addr"]
    if create_joint:
        res["norm_joint"] = res["clean_joint"]
    return res


__all__ = [
    "normalize_text",
    "clean_legal_suffixes",
    "clean_address",
    "extract_domain_stem",
    "clean_domain_stem",
    "preprocess_dataframe",
    "normalize_business_name",
    "normalize_dataframe",
]
