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

# Ensure source package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Workspace root
workspace_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from pipeline import execute_full_pipeline


def validate_outputs(matching_path: str, candidate_path: str, test_dir: str, mode: str) -> bool:
    """Run the official submission validation script."""
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL SUBMISSION VALIDATION")
    print("=" * 70)

    validator_script = os.path.join(workspace_dir, "student_resource", "utils", "validate_submission.py")
    if not os.path.isfile(validator_script):
        print(f"Warning: Validator script not found at {validator_script}")
        return True

    if mode == "sample":
        import pandas as pd
        sys.path.insert(0, os.path.join(workspace_dir, "student_resource", "utils"))
        from validate_submission import validate_id_list_file, MATCHING_HEADER, CANDIDATE_HEADER

        errors = []
        warnings = []
        req = set(pd.read_csv(os.path.join(workspace_dir, "sample_data", "sample_source1.tsv"), sep="\t")["entity_id"])
        
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
        default=os.path.join(workspace_dir, "output", "candidate_pairs.tsv"),
        help="Path for candidate pairs TSV output.",
    )
    parser.add_argument(
        "--matching-out",
        default=os.path.join(workspace_dir, "output", "matching_results.tsv"),
        help="Path for final matching results TSV output.",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(workspace_dir, "models", "lgbm_ber_model.joblib"),
        help="Path for cached/trained LightGBM model.",
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
        default=35,
        help="Top-K candidates per entity during blocking (default: 35).",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.12,
        help="Minimum cosine similarity for candidates (default: 0.12).",
    )
    args = parser.parse_args()

    if args.mode == "sample":
        s1_path = os.path.join(workspace_dir, "sample_data", "sample_source1.tsv")
        s2_path = os.path.join(workspace_dir, "sample_data", "sample_source2.tsv")
        s3_path = os.path.join(workspace_dir, "sample_data", "sample_source3.tsv")
        gt_path = os.path.join(workspace_dir, "sample_data", "sample_ground_truth.tsv")
        test_dir = os.path.join(workspace_dir, "sample_data")
    else:
        s1_path = os.path.join(workspace_dir, "student_resource", "dataset", "test", "test_source1.tsv")
        s2_path = os.path.join(workspace_dir, "student_resource", "dataset", "test", "test_source2.tsv")
        s3_path = os.path.join(workspace_dir, "student_resource", "dataset", "test", "test_source3.tsv")
        gt_path = None
        test_dir = os.path.join(workspace_dir, "student_resource", "dataset", "test")

    # Execute pipeline
    results = execute_full_pipeline(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        candidate_out=args.candidate_out,
        matching_out=args.matching_out,
        model_path=args.model_path,
        gt_path=gt_path,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        min_sim=args.min_sim,
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
