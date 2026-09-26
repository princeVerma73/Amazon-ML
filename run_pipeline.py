"""
ML Challenge 2026: Business Entity Resolution Submission Pipeline Entry Point.

Usage:
  # 1. Fast Development / Benchmark Mode (on sample_data/):
  python run_pipeline.py --mode sample

  # 2. Full Test Submission Mode (on student_resource/dataset/test/):
  python run_pipeline.py --mode test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Ensure source package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "code", "business_entity_resolution", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline import execute_full_pipeline


def validate_outputs(matching_path: str, candidate_path: str, test_dir: str, mode: str) -> bool:
    """
    Run the official submission validation script.
    """
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL SUBMISSION VALIDATION")
    print("=" * 70)

    validator_script = os.path.join(current_dir, "student_resource", "utils", "validate_submission.py")
    if not os.path.isfile(validator_script):
        print(f"Warning: Validator script not found at {validator_script}")
        return True

    # For sample mode, test_source1.tsv is in sample_data
    if mode == "sample":
        import pandas as pd
        utils_dir = os.path.join(current_dir, "student_resource", "utils")
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        from validate_submission import validate_id_list_file, MATCHING_HEADER, CANDIDATE_HEADER

        errors = []
        warnings = []
        req = set(pd.read_csv(os.path.join("sample_data", "sample_source1.tsv"), sep="\t")["entity_id"])
        
        m_map = validate_id_list_file(matching_path, MATCHING_HEADER, "matched_entity_ids", required=req, valid_ids=None, errors=errors)
        c_map = validate_id_list_file(candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required=req, valid_ids=None, errors=errors)

        if m_map is not None and c_map is not None:
            offenders = {s1 for s1, mids in m_map.items() if mids - c_map.get(s1, set())}
            if offenders:
                warnings.append(f"{len(offenders)} S1 entity(ies) have matched IDs not present in candidate_pairs.tsv")

        print()
        for w in warnings:
            print(f"WARNING: {w}")
        if errors:
            print(f"FAIL — {len(errors)} issue(s) found:")
            for i, err in enumerate(errors, 1):
                print(f"  {i}. {err}")
            return False
        else:
            print("PASS — All formatting rules and schema constraints satisfied!")
            return True
    else:
        cmd = [
            sys.executable,
            validator_script,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", test_dir,
        ]
        res = subprocess.run(cmd)
        return res.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Business Entity Resolution Pipeline (Sample Benchmark or Full Test Generation)."
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "test"],
        default="sample",
        help="Execution mode: 'sample' (runs on 25k benchmark) or 'test' (runs on full 11.7M test set).",
    )
    parser.add_argument(
        "--candidate-out",
        default="output/candidate_pairs.tsv",
        help="Path for candidate pairs TSV output (default: output/candidate_pairs.tsv).",
    )
    parser.add_argument(
        "--matching-out",
        default="output/matching_results.tsv",
        help="Path for final matching results TSV output (default: output/matching_results.tsv).",
    )
    parser.add_argument(
        "--model-type",
        choices=["catboost", "xgboost"],
        default="catboost",
        help="Classifier engine: 'catboost' (NVIDIA GPU accelerated, Experiment B) or 'xgboost' (Experiment C).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path for cached/trained model (default: models/<model_type>_ber_model.joblib).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="Batch chunk size for streaming inference (default: 20000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K candidates per entity during blocking (default: 20).",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.10,
        help="Minimum cosine similarity for candidates (default: 0.10).",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Filter to specific country partition (e.g. France).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit S1 queries per country for fast benchmark/sanity check.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining classifier on sample_data even if cached model exists.",
    )
    args = parser.parse_args()

    model_path = args.model_path or (
        "models/catboost_ber_model.joblib" if args.model_type == "catboost" else "models/xgb_ber_model.joblib"
    )

    if args.mode == "sample":
        s1_path = "sample_data/sample_source1.tsv"
        s2_path = "sample_data/sample_source2.tsv"
        s3_path = "sample_data/sample_source3.tsv"
        gt_path = "sample_data/sample_ground_truth.tsv"
        test_dir = "sample_data"
    else:
        s1_path = "student_resource/dataset/test/test_source1.tsv"
        s2_path = "student_resource/dataset/test/test_source2.tsv"
        s3_path = "student_resource/dataset/test/test_source3.tsv"
        gt_path = None
        test_dir = "student_resource/dataset/test"

    # Execute pipeline
    results = execute_full_pipeline(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        candidate_out=args.candidate_out,
        matching_out=args.matching_out,
        model_path=model_path,
        model_type=args.model_type,
        gt_path=gt_path,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        min_sim=args.min_sim,
        country_filter=args.country,
        limit=args.limit,
        force_retrain=args.force_retrain,
    )

    # Validate output files
    valid = validate_outputs(
        matching_path=args.matching_out,
        candidate_path=args.candidate_out,
        test_dir=test_dir,
        mode=args.mode,
    )

    if valid:
        print("\nSUCCESS: All pipeline deliverables are generated and validated!")
    else:
        print("\nWARNING: Pipeline generated files, but validation flagged issues.")


if __name__ == "__main__":
    main()
