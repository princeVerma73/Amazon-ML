"""
Retrain CatBoost on REAL training data (20k entities, 400k pairs).
Uses output/train_candidate_pairs_20k.tsv + train_ground_truth.tsv

This replaces models/catboost_conflict_aware.joblib with a better model.

Run: python retrain_on_real_data.py
Expected time: ~15-20 minutes
"""

import re, sys, time, joblib, os
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz, distance

sys.path.insert(0, "code/business_entity_resolution/src")
from normalizer import clean_legal_suffixes, normalize_text, clean_address, extract_domain_stem
from feature_extraction import (
    extract_numeric_tokens, extract_pin_tokens, extract_numeric_pincode_tokens,
    compute_numeric_pincode_match, compute_domain_match, compute_script_mismatch
)
from train_conflict_model import get_state, extract_primary_number

# ─────────────────────────────────────────────────────────────────────────────
N_EVAL      = 20_000
TRAIN_S1    = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2    = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3    = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT    = "student_resource/dataset/train/train_ground_truth.tsv"
TRAIN_CANDS = "output/train_candidate_pairs_20k.tsv"
MODEL_OUT   = "models/catboost_real_20k.joblib"
BACKUP_MODEL = "models/catboost_conflict_aware.joblib"

# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'n_sort', 'n_set', 'n_jw', 'n_lev',
    'a_sort', 'a_set', 'a_jw',
    'num_overlap', 'pin_match', 'num_pin_match',
    'dom_match', 'script_mis',
    'len_diff_n', 'len_diff_a', 'is_missing',
    'state_conflict', 'door_conflict', 'extra_words'
]

def clean_name(s):
    return clean_legal_suffixes(normalize_text(str(s or "")))

def macro_f05(preds, gt_map, all_ids):
    f05_sum = prec_sum = rec_sum = 0.0
    for sid in all_ids:
        ps = set(preds.get(sid, []))
        ts = gt_map.get(sid, set())
        if not ps and not ts:
            f05_sum += 1; prec_sum += 1; rec_sum += 1; continue
        tp = len(ps & ts)
        p = tp / len(ps) if ps else 0.0
        r = tp / len(ts) if ts else 0.0
        d = 0.25 * p + r
        f05_sum += (1.25 * p * r / d if d > 0 else 0.0)
        prec_sum += p; rec_sum += r
    n = len(all_ids)
    return f05_sum / n, prec_sum / n, rec_sum / n

def featurize(s1rn, s1a, trn, ta, s1dom, tdom, s1nums, tnums,
              s1pins, tpins, s1ph, tph, s1st, tst, s1dnum, tdnum, s1words, twords):
    n1 = clean_name(s1rn); n2 = clean_name(trn)
    a1 = clean_address(str(s1a or "")); a2 = clean_address(str(ta or ""))
    STOP = {'the','a','an','of','and','&','or','for','in','at','to','by',
            'le','la','les','de','du','des'}
    door = 0.0
    if s1dnum is not None and tdnum is not None and abs(s1dnum - tdnum) > 1:
        ss, ts2 = str(s1dnum), str(tdnum)
        if not (ss.startswith(ts2) or ts2.startswith(ss)):
            door = 1.0
    extra = 0.0
    if s1words and twords:
        sg = s1words - twords - STOP; tg = twords - s1words - STOP
        if sg and tg:
            extra = min(1.0, (len(sg) + len(tg)) / max(len(s1words | twords), 1))
    return [
        fuzz.token_sort_ratio(n1, n2) / 100,
        fuzz.token_set_ratio(n1, n2) / 100,
        distance.JaroWinkler.normalized_similarity(n1, n2),
        distance.Levenshtein.normalized_similarity(n1, n2),
        fuzz.token_sort_ratio(a1, a2) / 100,
        fuzz.token_set_ratio(a1, a2) / 100,
        distance.JaroWinkler.normalized_similarity(a1, a2),
        len(s1nums & tnums) / max(len(s1nums | tnums), 1) if (s1nums and tnums) else 0.0,
        1.0 if (s1pins and tpins and s1pins & tpins) else 0.0,
        compute_numeric_pincode_match(s1ph, tph),
        compute_domain_match(s1dom, tdom),
        compute_script_mismatch(s1rn, trn),
        abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
        abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1),
        1.0 if not a2.strip() else 0.0,
        1.0 if (s1st and tst and s1st != tst) else 0.0,
        door, extra
    ]

def main():
    print("=" * 70)
    print("RETRAINING CATBOOST ON REAL TRAINING DATA (20k entities)")
    print("=" * 70)

    # 1. Load data
    t0 = time.time()
    s1 = pl.read_csv(TRAIN_S1, separator="\t", null_values=[""]).head(N_EVAL)
    s2 = pl.read_csv(TRAIN_S2, separator="\t", null_values=[""])
    s3 = pl.read_csv(TRAIN_S3, separator="\t", null_values=[""])
    gt = pl.read_csv(TRAIN_GT, separator="\t", null_values=[""])

    all_s1_ids = s1["entity_id"].to_list()
    s1_ids_set = set(all_s1_ids)

    # GT map
    gt_map = {}
    for r in gt.iter_rows(named=True):
        sid = r["source1_entity_id"]
        if sid not in s1_ids_set: continue
        m = str(r.get("matched_entity_ids") or "").strip()
        gt_map[sid] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()

    # Target lookup
    raw_t = {}
    for r in pl.concat([s2, s3]).iter_rows(named=True):
        raw_t[r["entity_id"]] = (
            str(r.get("business_name") or ""),
            str(r.get("business_address") or "")
        )
    print(f"Data loaded in {time.time()-t0:.1f}s | "
          f"S1={len(s1)} | S2={len(s2):,} | S3={len(s3):,} | GT={len(gt_map):,}")

    # 2. Load candidates
    cand_df = pl.read_csv(TRAIN_CANDS, separator="\t", null_values=[""])
    cand_map = {}
    for r in cand_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        if sid not in s1_ids_set: continue
        c = str(r.get("candidate_entity_ids") or "").strip()
        cand_map[sid] = [x.strip() for x in c.split(",") if x.strip()] if c else []

    # 3. Cache target features
    print("Caching target features...")
    needed = set(tid for cands in cand_map.values() for tid in cands)
    t_cache = {}
    for tid in needed:
        if tid not in raw_t: continue
        rn, ra = raw_t[tid]
        cn = clean_name(rn); ca = clean_address(ra)
        t_cache[tid] = (
            cn, ca, rn, extract_domain_stem(rn),
            extract_numeric_tokens(f"{cn} {ca}"),
            extract_pin_tokens(ca),
            extract_numeric_pincode_tokens(ca),
            get_state(ra), extract_primary_number(ra),
            set(cn.split()),
            1.0 if not ca.strip() else 0.0
        )
    print(f"Cached {len(t_cache):,} targets")

    # 4. Build feature matrix with labels
    print("\nBuilding feature matrix...")
    ts = time.time()
    X, y, pair_ids = [], [], []

    s1_info = {}
    for r in s1.iter_rows(named=True):
        sid = r["entity_id"]; rn = str(r.get("business_name") or "")
        ra = str(r.get("business_address") or ""); cn = clean_name(rn); ca = clean_address(ra)
        s1_info[sid] = (
            cn, ca, rn, extract_domain_stem(rn),
            extract_numeric_tokens(f"{cn} {ca}"),
            extract_pin_tokens(ca), extract_numeric_pincode_tokens(ca),
            get_state(ra), extract_primary_number(ra), set(cn.split())
        )

    pos_count = 0; neg_count = 0
    for sid in all_s1_ids:
        if sid not in cand_map or sid not in s1_info: continue
        s1n,s1a,s1rn,s1dom,s1nums,s1pins,s1ph,s1st,s1dnum,s1words = s1_info[sid]
        true_set = gt_map.get(sid, set())

        for tid in cand_map[sid]:
            ti = t_cache.get(tid)
            if not ti: continue
            tn,ta,trn,tdom,tnums,tpins,tph,tst,tdnum,twords,miss = ti
            feat = featurize(s1rn,s1a,trn,ta,s1dom,tdom,s1nums,tnums,
                             s1pins,tpins,s1ph,tph,s1st,tst,s1dnum,tdnum,
                             s1words,twords)
            label = 1 if tid in true_set else 0
            X.append(feat); y.append(label); pair_ids.append((sid, tid))
            if label == 1: pos_count += 1
            else: neg_count += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Feature matrix: {len(X):,} pairs in {time.time()-ts:.2f}s")
    print(f"  Positive (true match): {pos_count:,} ({pos_count/len(y)*100:.1f}%)")
    print(f"  Negative (distractor): {neg_count:,} ({neg_count/len(y)*100:.1f}%)")
    print(f"  Class ratio: 1:{neg_count//max(pos_count,1):.0f}")

    # 5. Entity-level train/val split (80/20)
    # IMPORTANT: split by entity, not by pair, so eval entities have all 20 candidates
    all_entity_ids_with_cands = list(ent_data_build.keys()) if False else []
    unique_entities = list({sid for sid, tid in pair_ids})
    np.random.seed(42)
    np.random.shuffle(unique_entities)
    split = int(0.80 * len(unique_entities))
    train_ents = set(unique_entities[:split])
    val_ents   = set(unique_entities[split:])

    train_idx = [i for i, (sid, _) in enumerate(pair_ids) if sid in train_ents]
    val_idx   = [i for i, (sid, _) in enumerate(pair_ids) if sid in val_ents]

    X_tr = X[train_idx]; y_tr = y[train_idx]
    X_val = X[val_idx];  y_val = y[val_idx]
    val_pair_ids_local = [pair_ids[i] for i in val_idx]
    print(f"\nTrain: {len(X_tr):,} pairs ({len(train_ents):,} entities) | "
          f"Val: {len(X_val):,} pairs ({len(val_ents):,} entities)")

    # 6. Train CatBoost
    print("\nTraining CatBoost...")
    scale_pos = neg_count / max(pos_count, 1)
    print(f"  scale_pos_weight = {scale_pos:.2f}")

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
        use_best_model=True,
    )
    t_train = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=100,
    )
    print(f"\nTraining done in {time.time()-t_train:.1f}s")
    print(f"Best iteration: {model.best_iteration_}")

    # 7. Feature importance
    importances = model.get_feature_importance()
    print("\nFeature Importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp / 2)
        print(f"  {name:<20} {imp:6.2f}  {bar}")

    # 8. Evaluate on validation set with grid search (entity-level)
    val_probs = model.predict_proba(X_val)[:, 1]

    # Group by entity — each val entity has ALL its candidates visible
    ent_val = {}
    for (sid, tid), p, label in zip(val_pair_ids_local, val_probs, y_val):
        ent_val.setdefault(sid, []).append((tid, float(p), label))

    val_s1_ids = list(ent_val.keys())
    val_gt = {sid: gt_map.get(sid, set()) for sid in val_s1_ids}
    print(f"Val entities: {len(val_s1_ids):,} | Val GT matches: {sum(len(v) for v in val_gt.values()):,}")

    print("\n--- Validation Grid Search ---")
    best_f05 = 0.0; best_cfg = None; results = []
    for th_l in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90]:
        for mrg in [0.75, 0.80, 0.85, 0.88, 0.90, 0.95]:
            for cap in [4, 5, 6, None]:
                preds = {}; tot = 0
                for sid in val_s1_ids:
                    csl = ent_val[sid]
                    mx = max(p for _,p,_ in csl)
                    if mx < 0.30:
                        preds[sid] = []; continue
                    ms = []
                    for tid,p,_ in sorted(csl, key=lambda x:-x[1]):
                        if p >= th_l and p >= mrg*mx:
                            ms.append(tid)
                    if cap: ms=ms[:cap]
                    preds[sid]=ms; tot+=len(ms)
                avg = tot/max(len(val_s1_ids),1)
                f05,prec,rec = macro_f05(preds, val_gt, val_s1_ids)
                results.append((f05,prec,rec,avg,th_l,mrg,cap))
                if f05 > best_f05:
                    best_f05=f05; best_cfg=results[-1]
                    print(f"  VAL BEST th_l={th_l} mrg={mrg} cap={cap} "
                          f"F0.5={f05:.4f} P={prec:.4f} R={rec:.4f} avg={avg:.3f}")

    print(f"\nBest validation F0.5: {best_f05:.4f}")
    f05,prec,rec,avg,th_l,mrg,cap = best_cfg
    print(f"  th_link={th_l}, rel_margin={mrg}, max_cap={cap}")
    print(f"  Precision={prec:.4f}, Recall={rec:.4f}, AvgM={avg:.3f}")

    # 9. Save model
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "best_th_link": th_l,
        "best_rel_margin": mrg,
        "best_max_cap": cap,
        "val_f05": best_f05,
        "trained_on": f"{N_EVAL}_real_train_entities",
        "training_pairs": len(X),
        "pos_pairs": int(pos_count),
        "neg_pairs": int(neg_count),
    }
    joblib.dump(payload, MODEL_OUT, compress=3)
    print(f"Model saved to: {MODEL_OUT}")

    # 10. Compare with old model
    print("\n" + "=" * 70)
    print("COMPARISON: Old Model vs New Model")
    print("=" * 70)
    old_payload = joblib.load(BACKUP_MODEL)
    old_model = old_payload["model"] if isinstance(old_payload, dict) else old_payload
    old_probs = old_model.predict_proba(X_val)[:, 1]

    ent_val_old = {}
    for (sid, tid), p, label in zip(val_pair_ids_local, old_probs, y_val):
        ent_val_old.setdefault(sid, []).append((tid, float(p), label))

    # Use same optimal thresholds on old model
    preds_old = {}
    for sid in val_s1_ids:
        csl = ent_val_old[sid]
        mx = max(p for _,p,_ in csl)
        if mx < 0.30:
            preds_old[sid] = []; continue
        ms = []
        for t,p,_ in sorted(csl, key=lambda x:-x[1]):
            if p >= th_l and p >= mrg*mx:
                ms.append(t)
        if cap: ms=ms[:cap]
        preds_old[sid] = ms

    f05_old, p_old, r_old = macro_f05(preds_old, val_gt, val_s1_ids)
    print(f"  Old model (sample_data trained):  F0.5={f05_old:.4f} P={p_old:.4f} R={r_old:.4f}")
    print(f"  New model (20k real data trained): F0.5={best_f05:.4f} P={prec:.4f} R={rec:.4f}")
    improvement = (best_f05 - f05_old) / max(f05_old, 0.001) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    if best_f05 > f05_old:
        print("[BETTER] New model wins -- replacing main model")
        import shutil
        shutil.copy(BACKUP_MODEL, BACKUP_MODEL + ".prev")
        joblib.dump(payload, BACKUP_MODEL, compress=3)
        print(f"   Old model backed up to: {BACKUP_MODEL}.prev")
        print(f"   New model saved to:     {BACKUP_MODEL}")
    else:
        print(f"[WARN] Old model is still better -- new model kept in {MODEL_OUT} only")

    print("\n" + "=" * 70)
    print("NEXT STEP: Run full rescore with new model")
    print(f"  th_link={th_l}, rel_margin={mrg}, max_cap={cap}")
    print("=" * 70)

if __name__ == "__main__":
    main()
