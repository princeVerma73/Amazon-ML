"""
Business Entity Resolution package.
"""

from .normalizer import (
    normalize_text,
    clean_legal_suffixes,
    clean_address,
    extract_domain_stem,
    clean_domain_stem,
    preprocess_dataframe,
)
from .blocking import DynamicTFIDFBlocker, export_candidate_pairs
from .feature_extraction import (
    FEATURE_COLUMNS,
    extract_numeric_tokens,
    extract_pin_tokens,
    extract_house_plot_tokens,
    extract_numeric_pincode_tokens,
    compute_numeric_pincode_match,
    compute_domain_match,
    compute_script_mismatch,
    compute_pair_features,
    build_candidate_feature_matrix,
)
from .classifier import (
    BERClassifier,
    compute_instance_macro_f05,
    export_matching_results,
)

__all__ = [
    "normalize_text",
    "clean_legal_suffixes",
    "clean_address",
    "extract_domain_stem",
    "clean_domain_stem",
    "preprocess_dataframe",
    "DynamicTFIDFBlocker",
    "export_candidate_pairs",
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
    "BERClassifier",
    "compute_instance_macro_f05",
    "export_matching_results",
]
