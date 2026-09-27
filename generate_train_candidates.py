"""
Step 1: Generate blocking candidates for N_EVAL training entities.
Step 2: Score with CatBoost and find optimal thresholds via grid search.

Run: python generate_train_candidates.py
Expected time: ~20-30 minutes total.
"""

import re, sys, time, joblib, os
import numpy as np
import polars as pl
import pandas as pd
from rapidfuzz import fuzz, distance

sys.path.insert(0, "code/business_entity_resolution/src")
from normalizer import clean_legal_suffixes, normalize_text, clean_address, extract_domain_stem
from feature_extraction import (
    extract_numeric_tokens, extract_pin_tokens, extract_numeric_pincode_tokens,
    compute_numeric_pincode_match, compute_domain_match, compute_script_mismatch
)
from train_conflict_model import get_state, extract_primary_number
from blocking import DynamicTFIDFBlocker, export_candidate_pairs

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
N_EVAL      = 20_000    # Training entities to evaluate (20k = ~20 min blocking)
TRAIN_S1    = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2    = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3    = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT    = "student_resource/dataset/train/train_ground_truth.tsv"
TRAIN_CANDS_OUT = "output/train_candidate_pairs_20k.tsv"
MODEL_PATH  = "models/catboost_conflict_aware.joblib"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: GENERATE CANDIDATES VIA BLOCKING
# ─────────────────────────────────────────────────────────────────────────────
def generate_candidates():
    if os.path.exists(TRAIN_CANDS_OUT):
        print(f"[SKIP] {TRAIN_CANDS_OUT} already exists — skipping blocking step.")
        print("       Delete it to re-run blocking.")
        return

    print("=" * 70)
    print("PHASE 1: BLOCKING — Generating candidates for training entities")
    print("=" * 70)
    t0 = time.time()

    df_s1 = pd.read_csv(TRAIN_S1, sep="\t", keep_default_na=False).head(N_EVAL)
    df_s2 = pd.read_csv(TRAIN_S2, sep="\t", keep_default_na=False)
    df_s3 = pd.read_csv(TRAIN_S3, sep="\t", keep_default_na=False)

    print(f"Loaded: S1={len(df_s1):,}  S2={len(df_s2):,}  S3={len(df_s3):,}")

    blocker = DynamicTFIDFBlocker(
        top_k=20,
        coarse_top_k=300,
        min_similarity=0.10,
        batch_size=1000,
    )
    candidates = blocker.generate_candidates(df_s1, df_s2, df_s3)

    all_s1_ids = df_s1["entity_id"].tolist()
    export_candidate_pairs(candidates, TRAIN_CANDS_OUT, all_s1_ids=all_s1_ids)

    covered = sum(1 for sid in all_s1_ids if candidates.get(sid))
    avg_cands = sum(len(v) for v in candidates.values()) / max(len(all_s1_ids), 1)
    print(f"\nBlocking done in {time.time()-t0:.1f}s")
    print(f"  Entities with candidates: {covered:,}/{len(all_s1_ids):,}")
    print(f"  Avg candidates per entity: {avg_cands:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def calibrate():
    print("\n" + "=" * 70)
    print("PHASE 2: CALIBRATION — Grid search for optimal thresholds")
    print("=" * 70)

    # Load model
    payload = joblib.load(MODEL_PATH)
    model = payload["model"] if isinstance(payload, dict) else payload
    print(f"Model: {model.tree_count_} trees")

    # Load data
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
    print(f"Data loaded in {time.time()-t0:.1f}s")

    # Load candidates
    cand_df = pl.read_csv(TRAIN_CANDS_OUT, separator="\t", null_values=[""])
    cand_map = {}
    for r in cand_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        if sid not in s1_ids_set: continue
        c = str(r.get("candidate_entity_ids") or "").strip()
        cand_map[sid] = [x.strip() for x in c.split(",") if x.strip()] if c else []

    covered = sum(1 for sid in s1_ids_set if cand_map.get(sid))
    print(f"Candidates: {covered:,}/{len(s1_ids_set):,} entities have candidates ({covered/len(s1_ids_set)*100:.1f}%)")

    # Recall ceiling check
    total_gt = sum(len(v) for v in gt_map.values())
    captured = sum(
        len(gt_map.get(sid, set()) & set(cand_map.get(sid, [])))
        for sid in s1_ids_set
    )
    print(f"Blocking recall ceiling: {captured}/{total_gt} = {captured/max(total_gt,1)*100:.2f}%")

    # Cache targets
    needed = set(tid for cands in cand_map.values() for tid in cands)
    t_cache = {}
    for tid in needed:
        if tid not in raw_t: continue
        rn, ra = raw_t[tid]
        cn = clean_name(rn); ca = clean_address(ra)
        t_cache[tid] = (
            cn, ca, rn,
            extract_domain_stem(rn),
            extract_numeric_tokens(f"{cn} {ca}"),
            extract_pin_tokens(ca),
            extract_numeric_pincode_tokens(ca),
            get_state(ra), extract_primary_number(ra),
            set(cn.split()),
            1.0 if not ca.strip() else 0.0
        )

    # S1 info
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

    # Score pairs
    print("\nScoring candidate pairs with CatBoost...")
    ts = time.time()
    pairs = []; feats = []; misses = []

    for sid in all_s1_ids:
        if sid not in cand_map or sid not in s1_info: continue
        s1n,s1a,s1rn,s1dom,s1nums,s1pins,s1ph,s1st,s1dnum,s1words = s1_info[sid]
        for tid in cand_map[sid]:
            ti = t_cache.get(tid)
            if not ti: continue
            tn,ta,trn,tdom,tnums,tpins,tph,tst,tdnum,twords,miss = ti
            feats.append(featurize(s1rn,s1a,trn,ta,s1dom,tdom,s1nums,tnums,
                                   s1pins,tpins,s1ph,tph,s1st,tst,s1dnum,tdnum,
                                   s1words,twords))
            pairs.append((sid, tid)); misses.append(miss)

    probs = model.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
    print(f"Scored {len(probs):,} pairs in {time.time()-ts:.2f}s")

    ent_data = {}
    for (sid, tid), p, miss in zip(pairs, probs, misses):
        ent_data.setdefault(sid, []).append((tid, float(p), float(miss)))

    # Show probability distribution
    print(f"\nProb distribution: "
          f"p>0.9: {(probs>0.9).sum():,}  "
          f"p>0.7: {(probs>0.7).sum():,}  "
          f"p>0.5: {(probs>0.5).sum():,}  "
          f"p<0.3: {(probs<0.3).sum():,}")

    # Grid search
    print("\n" + "=" * 70)
    print("GRID SEARCH...")
    print("=" * 70)
    SING  = [0.20, 0.30, 0.40, 0.50, 0.60]
    LINK  = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90]
    MARG  = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.95]
    CAPS  = [4, 5, 6, 7, None]
    best = 0.0; best_p = None; results = []

    total = len(SING)*len(LINK)*len(MARG)*len(CAPS)
    print(f"Testing {total:,} configurations...\n")

    for th_s in SING:
        for th_l in LINK:
            for mrg in MARG:
                for cap in CAPS:
                    preds = {}; tot = 0; sings = 0
                    for sid in all_s1_ids:
                        csl = ent_data.get(sid)
                        if not csl:
                            preds[sid] = []; sings += 1; continue
                        mx = max(p for _, p, _ in csl)
                        if mx < th_s:
                            preds[sid] = []; sings += 1; continue
                        ms = []; nc = 0
                        for tid, p, miss in sorted(csl, key=lambda x: -x[1]):
                            if p >= th_l and p >= mrg * mx:
                                if miss == 1.0:
                                    if nc >= 1: continue
                                    nc += 1
                                ms.append(tid)
                        if cap: ms = ms[:cap]
                        preds[sid] = ms; tot += len(ms)
                        if not ms: sings += 1
                    avg = tot / max(len(all_s1_ids), 1)
                    f05, prec, rec = macro_f05(preds, gt_map, all_s1_ids)
                    results.append((f05, prec, rec, avg, sings, th_s, th_l, mrg, cap))
                    if f05 > best:
                        best = f05; best_p = results[-1]
                        print(f"  BEST th_s={th_s} th_l={th_l} mrg={mrg} cap={cap} "
                              f"F0.5={f05:.4f} P={prec:.4f} R={rec:.4f} avg={avg:.3f} sings={sings:,}")

    print("\n" + "=" * 70)
    print("TOP 20 CONFIGURATIONS:")
    print(f"{'F0.5':>7} {'Prec':>7} {'Rec':>7} {'AvgM':>6} {'Sings':>7}  th_s  th_l  marg  cap")
    print("-" * 70)
    for r in sorted(results, key=lambda x: -x[0])[:20]:
        f05, p, rc, av, sg, ts2, tl, mg, cp = r
        print(f"{f05:7.4f} {p:7.4f} {rc:7.4f} {av:6.3f} {sg:7,}  "
              f"{ts2:.2f}  {tl:.2f}  {mg:.2f}  {str(cp):>4}")

    print("\n" + "=" * 70)
    print("OPTIMAL RESULT:")
    f05, p, rc, av, sg, ts2, tl, mg, cp = best_p
    print(f"  th_singleton = {ts2}")
    print(f"  th_link      = {tl}")
    print(f"  rel_margin   = {mg}")
    print(f"  max_cap      = {cp}")
    print(f"  F0.5         = {f05:.4f}")
    print(f"  Precision    = {p:.4f}")
    print(f"  Recall       = {rc:.4f}")
    print(f"  Avg matches  = {av:.3f}  (target: 3.46)")
    print(f"  Singletons   = {sg:,}  (target: ~{int(0.056*N_EVAL):,})")
    print("=" * 70)

    # GT stats
    total_gt_m = sum(len(v) for v in gt_map.values())
    gt_avg = total_gt_m / max(len(all_s1_ids), 1)
    gt_sings = sum(1 for v in gt_map.values() if not v)
    print(f"\nGround Truth reference:")
    print(f"  Avg matches  = {gt_avg:.3f}")
    print(f"  Singletons   = {gt_sings:,} ({gt_sings/N_EVAL*100:.1f}%)")

    if av > 4.5:
        print(f"\n*** WARNING: avg {av:.2f} > 4.5 — too many false positives, precision will suffer ***")
    elif av < 2.0:
        print(f"\n*** WARNING: avg {av:.2f} < 2.0 — too few matches, recall will suffer ***")
    else:
        print(f"\n[OK] avg matches {av:.2f} is in acceptable range [2.0 - 4.5]")


if __name__ == "__main__":
    generate_candidates()   # ~20 min if running for the first time
    calibrate()             # ~5 min
