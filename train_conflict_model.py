"""
Train Conflict-Aware CatBoost Entity Resolution Model.

Key Improvements:
1. REMOVED blocking_rank: Prevents data leakage and eliminates false confidence on distractors.
2. ADDED door_conflict: Identifies cross-street and neighboring building distractors on identical streets.
3. ADDED state_conflict: Identifies cross-state distractor merges.
4. ADDED extra_words: Penalizes unshared branch/division keywords (e.g. West, Central, Services).
"""

import os
import re
import sys
import time
import joblib
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from rapidfuzz import fuzz, distance
from sklearn.model_selection import train_test_split

sys.path.insert(0, "code/business_entity_resolution/src")
from normalizer import clean_legal_suffixes, normalize_text, clean_address, extract_domain_stem
from feature_extraction import (
    extract_numeric_tokens, extract_pin_tokens, extract_numeric_pincode_tokens,
    compute_numeric_pincode_match, compute_domain_match, compute_script_mismatch
)
from classifier import compute_instance_macro_f05

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY'
}

def get_state(addr: str) -> str:
    if not addr:
        return ''
    tokens = re.findall(r'\b[A-Za-z]{2}\b', str(addr).upper())
    for t in tokens:
        if t in US_STATES:
            return t
    return ''

def extract_primary_number(text: str):
    if not text:
        return None
    m = re.search(r'\b(?:no\.?|h\.?no\.?|flat|door|ews|unit|suite|ste|apt|n°)?\s*[:#-]?\s*(\d+)', str(text), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

FEATURE_NAMES = [
    'n_sort', 'n_set', 'n_jw', 'n_lev',
    'a_sort', 'a_set', 'a_jw',
    'num_overlap', 'pin_match', 'num_pin_match',
    'dom_match', 'script_mis',
    'len_diff_n', 'len_diff_a',
    'is_missing',
    'state_conflict', 'door_conflict', 'extra_words'
]

def main():
    print("=" * 70)
    print("TRAINING CONFLICT-AWARE CATBOOST MODEL FOR AMAZON ML CHALLENGE")
    print("=" * 70)

    # 1. Load Data
    t0 = time.time()
    print("Loading sample datasets via Polars...")
    s1 = pl.read_csv("sample_data/sample_source1.tsv", separator="\t")
    s2 = pl.read_csv("sample_data/sample_source2.tsv", separator="\t")
    s3 = pl.read_csv("sample_data/sample_source3.tsv", separator="\t")
    cand_df = pl.read_csv("sample_data/sample_candidate_pairs.tsv", separator="\t")
    gt_df = pl.read_csv("sample_data/sample_ground_truth.tsv", separator="\t")

    gt_dict = {r[0]: set([x.strip() for x in (r[1] or "").split(",") if x.strip()]) for r in gt_df.iter_rows()}
    targets = pl.concat([s2, s3])
    t_names = dict(zip(targets['entity_id'], targets['business_name']))
    t_addrs = dict(zip(targets['entity_id'], targets['business_address']))
    c_dict = dict(zip(cand_df['source1_entity_id'], cand_df['candidate_entity_ids']))

    # Use 30,000 S1 queries for rich training coverage
    num_queries = 30000
    s1_slice = s1.slice(0, num_queries)
    print(f"Dataset loaded in {time.time() - t0:.2f}s. Preparing {num_queries:,} queries...")

    # 2. Extract Features
    t_feat = time.time()
    X_rows = []
    y_labels = []
    meta_pairs = []

    for row in s1_slice.iter_rows():
        s1_id = row[0]
        raw_n1 = str(row[1] or "")
        raw_a1 = str(row[2] or "")
        n1 = clean_legal_suffixes(normalize_text(raw_n1))
        a1 = clean_address(raw_a1)
        d1 = extract_domain_stem(raw_n1)
        nums1 = extract_numeric_tokens(f"{n1} {a1}")
        pins1 = extract_pin_tokens(a1)
        pin_h1 = extract_numeric_pincode_tokens(a1)
        st1 = get_state(raw_a1)
        d_num1 = extract_primary_number(raw_a1)
        words1 = set(n1.split())

        cand_str = c_dict.get(s1_id, "")
        if not cand_str:
            continue
        c_list = [c.strip() for c in cand_str.split(",") if c.strip()][:20]
        true_set = gt_dict.get(s1_id, set())

        for cid in c_list:
            raw_n2 = t_names.get(cid, "")
            raw_a2 = t_addrs.get(cid, "")
            n2 = clean_legal_suffixes(normalize_text(raw_n2))
            a2 = clean_address(raw_a2)
            is_missing = 1.0 if not a2.strip() else 0.0
            d2 = extract_domain_stem(raw_n2)
            nums2 = extract_numeric_tokens(f"{n2} {a2}")
            pins2 = extract_pin_tokens(a2)
            pin_h2 = extract_numeric_pincode_tokens(a2)
            st2 = get_state(raw_a2)
            d_num2 = extract_primary_number(raw_a2)
            words2 = set(n2.split())

            n_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
            n_set = fuzz.token_set_ratio(n1, n2) / 100.0
            n_jw = distance.JaroWinkler.similarity(n1, n2)
            n_lev = distance.Levenshtein.normalized_similarity(n1, n2)

            a_sort = fuzz.token_sort_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
            a_set = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
            a_jw = distance.JaroWinkler.similarity(a1, a2) if (a1 and a2) else 0.0

            num_overlap = len(nums1 & nums2) / len(nums1 | nums2) if (nums1 and nums2) else 0.0
            pin_match = 1.0 if (pins1 and pins2 and (pins1 & pins2)) else 0.0
            num_pin_match = compute_numeric_pincode_match(pin_h1, pin_h2)
            dom_match = compute_domain_match(d1, d2, name1=n1, name2=n2)
            script_mis = compute_script_mismatch(raw_n1, raw_n2)

            len_diff_n = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
            len_diff_a = abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1) if (a1 and a2) else 1.0

            state_conflict = 1.0 if (st1 and st2 and st1 != st2) else 0.0
            door_conflict = 1.0 if (d_num1 is not None and d_num2 is not None and d_num1 != d_num2 and a_sort >= 0.70) else 0.0
            extra_words = float(len(words2 - words1))

            feats = [
                n_sort, n_set, n_jw, n_lev,
                a_sort, a_set, a_jw,
                num_overlap, pin_match, num_pin_match,
                dom_match, script_mis,
                len_diff_n, len_diff_a,
                is_missing,
                state_conflict, door_conflict, extra_words
            ]
            X_rows.append(feats)
            y_labels.append(1 if cid in true_set else 0)
            meta_pairs.append((s1_id, cid))

    print(f"Extracted {len(X_rows):,} pairs in {time.time() - t_feat:.2f}s.")
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    # 3. Train-Validation Split (Grouped by S1 query)
    unique_s1 = sorted(list(set(p[0] for p in meta_pairs)))
    train_s1, val_s1 = train_test_split(unique_s1, test_size=0.20, random_state=42)
    train_s1_set = set(train_s1)
    val_s1_set = set(val_s1)

    train_mask = np.array([p[0] in train_s1_set for p in meta_pairs])
    val_mask = np.array([p[0] in val_s1_set for p in meta_pairs])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_meta = [p for p, m in zip(meta_pairs, val_mask) if m]

    print(f"Split: {len(train_s1):,} Train queries ({len(X_train):,} pairs) | {len(val_s1):,} Val queries ({len(X_val):,} pairs)")

    # 4. Train CatBoost
    print("\nTraining CatBoost with asymmetric class weighting...")
    t_train = time.time()
    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.06,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        class_weights=[1.0, 0.70],  # Penalizes false positive merges
        random_seed=42,
        early_stopping_rounds=40,
        verbose=100,
        thread_count=-1,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    print(f"Training completed in {time.time() - t_train:.2f}s.")

    # 5. Feature Importances
    print("\nFeature Importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.get_feature_importance()), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {imp:.2f}%")

    # 6. Grid Search for Optimal Linking Threshold on Holdout S1 Queries
    print("\nEvaluating Macro F0.5 on Holdout Validation Queries...")
    val_probs = model.predict_proba(X_val)[:, 1]

    from collections import defaultdict
    val_entity_probs = defaultdict(list)
    for (s1_id, cid), p in zip(val_meta, val_probs):
        val_entity_probs[s1_id].append((cid, p))

    best_f05 = 0.0
    best_th = 0.80

    for th in [0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]:
        preds = {}
        for s1_id in val_s1:
            cand_p = val_entity_probs.get(s1_id, [])
            preds[s1_id] = [cid for cid, p in cand_p if p >= th]
        f05, prec, rec = compute_instance_macro_f05(preds, gt_dict, val_s1)
        print(f"  th_link = {th:.2f} => Macro F0.5 = {f05:.4f} | Prec = {prec:.4f} | Rec = {rec:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_th = th

    print("\n" + "=" * 70)
    print(f"OPTIMAL THRESHOLD: th_link = {best_th:.2f} (Macro F0.5 = {best_f05:.4f})")
    print("=" * 70)

    # 7. Save Checkpoint
    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_model_path = os.path.join(checkpoint_dir, "catboost_conflict_aware.joblib")
    
    save_payload = {
        "model": model,
        "best_th_link": best_th,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(save_payload, out_model_path)
    print(f"Saved optimized model -> {out_model_path}")

if __name__ == "__main__":
    main()
