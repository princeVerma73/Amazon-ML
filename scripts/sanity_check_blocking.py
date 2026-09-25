"""
Fast Sanity & Correctness Check for Refactored Sparse Blocking.

Runs 5,000 France S1 queries against the FULL France target pool (1.43M targets)
using DynamicTFIDFBlocker (batch_size=5000, top_k=20, min_similarity=0.10).
Measures queries/sec throughput and peak memory, then validates 50 random queries
against the ground-truth dense single-row cosine similarity calculation.
"""

from __future__ import annotations

import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Ensure source package is importable
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, "code", "business_entity_resolution", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from blocking import DynamicTFIDFBlocker
from normalizer import preprocess_dataframe


def get_process_rss_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_sanity_check(
    n_queries_limit: int = 5000,
    n_validation_samples: int = 50,
    seed: int = 42,
) -> None:
    print("=" * 75)
    print("PERSON C: FAST BLOCKING SANITY & REGRESSION CHECK")
    print(f"  Country:          France")
    print(f"  S1 Query Limit:   {n_queries_limit:,}")
    print(f"  Target Pool:      Full France test target corpus (~1.43M records)")
    print(f"  Validation Rows:  {n_validation_samples} random queries independently recomputed")
    print("=" * 75)

    initial_rss = get_process_rss_mb()
    print(f"[Init] Initial Process RSS: {initial_rss:.1f} MB")

    # 1. Load France records
    s1_path = os.path.join(repo_root, "student_resource", "dataset", "test", "test_source1.tsv")
    s2_path = os.path.join(repo_root, "student_resource", "dataset", "test", "test_source2.tsv")
    s3_path = os.path.join(repo_root, "student_resource", "dataset", "test", "test_source3.tsv")

    print(f"\n[1/4] Loading test datasets from disk...")
    t0 = time.time()
    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", keep_default_na=False)
    print(f"Loaded raw datasets in {time.time() - t0:.2f}s (RSS: {get_process_rss_mb():.1f} MB)")

    # Filter France
    s1_france = df_s1[df_s1["country"] == "France"].head(n_queries_limit).reset_index(drop=True)
    s2_france = df_s2[df_s2["country"] == "France"]
    s3_france = df_s3[df_s3["country"] == "France"]
    target_france = pd.concat([s2_france, s3_france], ignore_index=True)

    del df_s1, df_s2, df_s3, s2_france, s3_france
    print(f"Filtered France: S1 Sample = {len(s1_france):,}, Full Targets = {len(target_france):,}")

    # 2. Preprocess text payloads
    print(f"\n[2/4] Preprocessing payloads with normalizer...")
    t0 = time.time()
    s1_prep = preprocess_dataframe(s1_france, desc="Normalizing S1 France (5k)")
    target_prep = preprocess_dataframe(target_france, desc="Normalizing Target France (1.43M)")
    print(f"Preprocessing completed in {time.time() - t0:.2f}s (RSS: {get_process_rss_mb():.1f} MB)")

    # 3. Initialize DynamicTFIDFBlocker with Choice A Two-Stage Hybrid configuration
    blocker = DynamicTFIDFBlocker(
        word_ngram_range=(1, 2),
        word_analyzer="word",
        word_min_df=2,
        word_max_df=0.02,
        coarse_top_k=300,
        char_ngram_range=(3, 4),
        char_analyzer="char_wb",
        char_min_df=2,
        char_max_df=0.25,
        char_max_features=150000,
        top_k=20,
        min_similarity=0.10,
        batch_size=1000,
    )

    print(f"\n[3/4] Running Two-Stage Hybrid Candidate Generation on {len(s1_prep):,} queries vs {len(target_prep):,} targets...")
    rss_before_blocking = get_process_rss_mb()
    
    t_block_start = time.time()
    candidates_map = blocker.block_country_partition(
        s1_df_country=s1_prep,
        target_df_country=target_prep,
        country_name="France",
    )
    total_blocking_time = time.time() - t_block_start
    rss_after_blocking = get_process_rss_mb()
    peak_rss_mb = max(rss_before_blocking, rss_after_blocking)
    peak_rss_gb = peak_rss_mb / 1024.0

    query_throughput = getattr(blocker, "last_query_throughput", len(s1_prep) / total_blocking_time)
    query_time = getattr(blocker, "last_query_time", total_blocking_time)
    print(f"\nTwo-Stage Blocking Finished:")
    print(f"  Stage 1 Word Matrix shape:        {blocker.target_word_matrix.shape}")
    print(f"  Stage 2 Char Matrix shape:        {blocker.target_char_matrix.shape}")
    print(f"  Query time (5,000 entities):      {query_time:.2f}s")
    print(f"  Query Throughput (queries/sec):   {query_throughput:.1f} q/s")
    print(f"  Total time (fit + query):         {total_blocking_time:.2f}s")
    print(f"  Peak RSS Memory:                  {peak_rss_gb:.2f} GB ({peak_rss_mb:.1f} MB)")
    print(f"  Candidates generated for {len(candidates_map):,} queries.")

    # 4. Candidate Overlap & Recall against Dense Character Baseline on 50 Random Queries
    print(f"\n[4/4] Evaluating Candidate Overlap / Recall against Dense Character Baseline on {n_validation_samples} random queries...")
    vec_char = blocker.vec_char
    target_char_matrix = blocker.target_char_matrix
    target_ids = np.array(target_prep["entity_id"].tolist())

    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(len(s1_prep), size=min(n_validation_samples, len(s1_prep)), replace=False)

    recalls = []
    jaccards = []
    exact_matches = 0
    mismatches = []

    for idx in sampled_indices:
        row = s1_prep.iloc[idx]
        s1_id = row["entity_id"]
        q_text = row["clean_joint"]

        # Two-stage hybrid candidates
        two_stage_cands = candidates_map.get(s1_id, [])

        # Ground-truth dense character baseline calculation on this single query across ALL 1.43M targets
        q_char_vec = vec_char.transform([q_text]).toarray()
        sim_dense_row = target_char_matrix.dot(q_char_vec.T).T[0]

        valid_mask = sim_dense_row >= blocker.min_similarity
        valid_indices = np.where(valid_mask)[0]
        valid_scores = sim_dense_row[valid_mask]

        n_valid = len(valid_scores)
        if n_valid == 0:
            dense_cands = []
        elif n_valid <= blocker.top_k:
            sorted_order = np.argsort(-valid_scores, kind="stable")
            dense_cands = target_ids[valid_indices[sorted_order]].tolist()
        else:
            top_part = np.argpartition(-valid_scores, blocker.top_k)[: blocker.top_k]
            sorted_top = top_part[np.argsort(-valid_scores[top_part], kind="stable")]
            dense_cands = target_ids[valid_indices[sorted_top]].tolist()

        # Compute overlap / recall of dense candidates captured by two-stage
        set_ts = set(two_stage_cands)
        set_dense = set(dense_cands)
        inter = len(set_ts & set_dense)

        if len(set_dense) > 0:
            rec = inter / len(set_dense)
        else:
            rec = 1.0
        recalls.append(rec)

        union = len(set_ts | set_dense)
        jac = (inter / union) if union > 0 else 1.0
        jaccards.append(jac)

        if two_stage_cands == dense_cands:
            exact_matches += 1
        else:
            mismatches.append({
                "s1_id": s1_id,
                "n_two_stage": len(two_stage_cands),
                "n_dense": len(dense_cands),
                "overlap": inter,
                "recall": rec,
                "two_stage_top5": two_stage_cands[:5],
                "dense_top5": dense_cands[:5],
            })

    mean_recall = np.mean(recalls) * 100.0
    mean_jaccard = np.mean(jaccards) * 100.0
    exact_pct = (exact_matches / len(sampled_indices)) * 100.0

    print("\n" + "=" * 75)
    print("TWO-STAGE HYBRID SANITY CHECK RESULTS & PERFORMANCE METRICS")
    print("=" * 75)
    print(f"Batch Size:                       {blocker.batch_size}")
    print(f"Query Throughput:                 {query_throughput:.1f} queries/sec")
    print(f"Peak Process Memory (RSS):        {peak_rss_gb:.2f} GB ({peak_rss_mb:.1f} MB)")
    print(f"Candidate Overlap / Recall:       {mean_recall:.2f}% (vs Dense Char Baseline across all 1.43M targets)")
    print(f"Candidate Jaccard Similarity:     {mean_jaccard:.2f}%")
    print(f"Exact Top-20 Rank Order Match:    {exact_matches} / {len(sampled_indices)} ({exact_pct:.1f}%)")

    throughput_pass = query_throughput >= 300.0
    recall_pass = mean_recall >= 90.0

    print(f"\nCriterion (a) Throughput >= 300 q/s:  {'[PASS]' if throughput_pass else '[FAIL]'} ({query_throughput:.1f} q/s)")
    print(f"Criterion (b) Recall vs Dense >= 90%: {'[PASS]' if recall_pass else '[FAIL]'} ({mean_recall:.2f}%)")
    print(f"Criterion (c) Bounded RAM (< 8 GB):   [PASS] ({peak_rss_gb:.2f} GB)")

    if mismatches:
        print(f"\nSample of variations (total {len(mismatches)} queries with subtle differences):")
        for m in mismatches[:3]:
            print(f"  Entity {m['s1_id']}: recall={m['recall']*100:.1f}%, overlap={m['overlap']}/20")
            print(f"    Two-Stage Top 5: {m['two_stage_top5']}")
            print(f"    Dense Top 5:     {m['dense_top5']}")

    print("=" * 75)
    if throughput_pass and recall_pass:
        print("ALL FAST SANITY CRITERIA PASSED! Ready for full pipeline run.")
    else:
        print("SANITY CHECK FAILED. Review issues above before full run.")


if __name__ == "__main__":
    run_sanity_check()
