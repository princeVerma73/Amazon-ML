"""
Text normalization module for Business Entity Resolution.

Provides deterministic cleaning, Unicode diacritics stripping (NFKD),
legal entity suffix removal, address abbreviation expansions, and
memory-efficient vectorized application over pandas DataFrames.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Precompiled Regex Patterns for High Performance
# ---------------------------------------------------------------------------

# Legal Entity Suffixes to remove from business names
# Ordered by descending length to prevent partial prefix/substring collisions
LEGAL_SUFFIXES = [
    r"private\s+limited",
    r"pvt\s+ltd",
    r"incorporated",
    r"corporation",
    r"enterprises",
    r"associates",
    r"limited",
    r"company",
    r"pvt",
    r"ltd",
    r"inc",
    r"corp",
    r"llc",
    r"llp",
    r"sarl",
    r"sasu",
    r"eurl",
    r"sas",
    r"gmbh",
    r"sa",
    r"co",
]
LEGAL_SUFFIX_REGEX = re.compile(
    rf"\b({'|'.join(LEGAL_SUFFIXES)})\b",
    flags=re.IGNORECASE,
)

# Address abbreviations and directional expansions
# Applied with word boundary matches to avoid false positives
ADDRESS_ABBREVIATIONS = {
    # Road / Street types
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
    # Unit / Sub-division tokens
    r"\bapt\b": "apartment",
    r"\bflt\b": "flat",
    r"\bste\b": "suite",
    r"\bbldg\b": "building",
    r"\bfl\b": "floor",
    r"\bh\.?no\.?\b": "house number",
    r"\bhno\b": "house number",
    r"\bb/h\b": "behind",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    # Compass directions
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bne\b": "northeast",
    r"\bnw\b": "northwest",
    r"\bse\b": "southeast",
    r"\bsw\b": "southwest",
}

ADDRESS_EXPANSION_PATTERNS = [
    (re.compile(pattern, flags=re.IGNORECASE), replacement)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items()
]

# Common symbol replacements
SYMBOL_REPLACEMENTS = [
    (re.compile(r"&"), " and "),
    (re.compile(r"@"), " at "),
    (re.compile(r"\+"), " plus "),
    (re.compile(r"[/\\_]"), " "),
]

# Non-alphanumeric strip and whitespace collapse
NON_ALPHANUMERIC_REGEX = re.compile(r"[^a-z0-9\s]")
WHITESPACE_REGEX = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core Normalization Functions
# ---------------------------------------------------------------------------

def strip_diacritics(text: str) -> str:
    """
    Decompose Unicode characters into ASCII base characters and accents (NFKD),
    then discard combining diacritical marks.
    
    Handles French accented characters in test set (e.g., 'École' -> 'Ecole', 'Château' -> 'Chateau').
    """
    if not text:
        return ""
    # Normalize with NFKD decomposition and filter non-spacing marks
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_business_name(text: Optional[str]) -> str:
    """
    Normalize business entity name.
    
    Steps:
      1. Handle nulls and convert to string.
      2. Unicode NFKD decomposition to strip accents / diacritics.
      3. Lowercase text and expand symbols ('&' -> 'and', '@' -> 'at', etc.).
      4. Strip legal suffixes (e.g., LLC, Pvt Ltd, Inc, SARL, SAS, GmbH).
      5. Remove special punctuation while preserving alphanumerics and single spaces.
      6. Collapse whitespace and strip leading/trailing spaces.
    
    Args:
        text: Raw business name string.
        
    Returns:
        Cleaned, canonical business name string.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # 1. Unicode decomposition & diacritics removal
    text = strip_diacritics(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Symbol replacements
    for pattern, replacement in SYMBOL_REPLACEMENTS:
        text = pattern.sub(replacement, text)
        
    # 4. Remove legal entity suffixes using word boundaries
    text = LEGAL_SUFFIX_REGEX.sub(" ", text)
    
    # 5. Remove special characters / punctuation
    text = NON_ALPHANUMERIC_REGEX.sub(" ", text)
    
    # 6. Collapse multiple spaces
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    
    return text


def normalize_address(text: Optional[str]) -> str:
    """
    Normalize business address.
    
    Steps:
      1. Handle nulls and convert to string.
      2. Unicode NFKD decomposition to strip accents / diacritics.
      3. Lowercase text and expand symbols.
      4. Expand common address contractions (e.g., rd -> road, st -> street, blvd -> boulevard).
      5. Remove commas, periods, and special punctuation.
      6. Collapse whitespace and strip leading/trailing spaces.
      
    Args:
        text: Raw address string.
        
    Returns:
        Cleaned, canonical address string.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # 1. Unicode decomposition & diacritics removal
    text = strip_diacritics(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Symbol replacements
    for pattern, replacement in SYMBOL_REPLACEMENTS:
        text = pattern.sub(replacement, text)
        
    # 4. Expand address abbreviations
    for pattern, replacement in ADDRESS_EXPANSION_PATTERNS:
        text = pattern.sub(replacement, text)
        
    # 5. Remove non-alphanumeric punctuation
    text = NON_ALPHANUMERIC_REGEX.sub(" ", text)
    
    # 6. Collapse multiple spaces
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    
    return text


def normalize_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    addr_col: str = "business_address",
    create_joint: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Memory-efficient vectorized normalization of business names and addresses
    across a pandas DataFrame.
    
    Uses optimized list comprehensions (2x faster than Series.apply with less memory overhead)
    and handles missing/null values gracefully.
    
    Args:
        df: Input DataFrame containing entity records.
        name_col: Name of column containing business name.
        addr_col: Name of column containing business address.
        create_joint: If True, creates a concatenated 'norm_joint' column for blocking.
        inplace: Whether to modify input DataFrame in place or return a shallow copy.
        
    Returns:
        DataFrame with normalized columns:
          - 'norm_business_name'
          - 'norm_business_address'
          - 'norm_joint' (if create_joint is True)
    """
    target_df = df if inplace else df.copy(deep=False)
    
    # Extract string values with null handling
    names = target_df[name_col].fillna("").astype(str).tolist() if name_col in target_df.columns else []
    addresses = target_df[addr_col].fillna("").astype(str).tolist() if addr_col in target_df.columns else []
    
    # Process business names
    if names:
        target_df["norm_business_name"] = [normalize_business_name(name) for name in names]
    else:
        target_df["norm_business_name"] = ""
        
    # Process addresses
    if addresses:
        target_df["norm_business_address"] = [normalize_address(addr) for addr in addresses]
    else:
        target_df["norm_business_address"] = ""
        
    # Create combined joint string for fast TF-IDF indexing
    if create_joint:
        norm_names = target_df["norm_business_name"].tolist()
        norm_addrs = target_df["norm_business_address"].tolist()
        target_df["norm_joint"] = [
            f"{n} {a}".strip() for n, a in zip(norm_names, norm_addrs)
        ]
        
    return target_df
