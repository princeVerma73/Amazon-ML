"""
Calibration Grid Search on REAL Training Ground Truth.
Run: python calibrate_thresholds.py
"""
import re, sys, time, joblib
import numpy as np
import polars as pl
from rapidfuzz import fuzz, distance

sys.path.insert(0, "code/business_entity_resolution/src")
from normalizer import clean_legal_suffixes, normalize_text, clean_address, extract_domain_stem
from feature_extraction import (
    extract_numeric_tokens, extract_pin_tokens, extract_numeric_pincode_tokens,
    compute_numeric_pincode_match, compute_domain_match, compute_script_mismatch
)
from train_conflict_model import get_state, extract_primary_number

N_EVAL      = 50_000
TRAIN_S1    = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2    = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3    = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT    = "student_resource/dataset/train/train_ground_truth.tsv"
TRAIN_CANDS = "sample_data/sample_candidate_pairs.tsv"
MODEL_PATH  = "models/catboost_conflict_aware.joblib"

def clean_name(s):
    return clean_legal_suffixes(normalize_text(str(s or "")))

def macro_f05(preds, gt_map, all_ids):
    f05_sum = prec_sum = rec_sum = 0.0
    for sid in all_ids:
        ps = set(preds.get(sid, [])); ts = gt_map.get(sid, set())
        if not ps and not ts:
            f05_sum+=1; prec_sum+=1; rec_sum+=1; continue
        tp = len(ps & ts)
        p = tp/len(ps) if ps else 0.0
        r = tp/len(ts) if ts else 0.0
        d = 0.25*p + r
        f05_sum += (1.25*p*r/d if d>0 else 0.0)
        prec_sum += p; rec_sum += r
    n = len(all_ids)
    return f05_sum/n, prec_sum/n, rec_sum/n

def featurize(s1rn,s1a,trn,ta,s1dom,tdom,s1nums,tnums,
              s1pins,tpins,s1ph,tph,s1st,tst,s1dnum,tdnum,s1words,twords):
    n1=clean_name(s1rn); n2=clean_name(trn)
    a1=clean_address(str(s1a or "")); a2=clean_address(str(ta or ""))
    STOP={'the','a','an','of','and','&','or','for','in','at','to','by',
          'le','la','les','de','du','des'}
    door=0.0
    if s1dnum is not None and tdnum is not None and abs(s1dnum-tdnum)>1:
        ss,ts2=str(s1dnum),str(tdnum)
        if not (ss.startswith(ts2) or ts2.startswith(ss)):
            door=1.0
    extra=0.0
    if s1words and twords:
        sg=s1words-twords-STOP; tg=twords-s1words-STOP
        if sg and tg:
            extra=min(1.0,(len(sg)+len(tg))/max(len(s1words|twords),1))
    return [
        fuzz.token_sort_ratio(n1,n2)/100, fuzz.token_set_ratio(n1,n2)/100,
        distance.JaroWinkler.normalized_similarity(n1,n2),
        distance.Levenshtein.normalized_similarity(n1,n2),
        fuzz.token_sort_ratio(a1,a2)/100, fuzz.token_set_ratio(a1,a2)/100,
        distance.JaroWinkler.normalized_similarity(a1,a2),
        len(s1nums&tnums)/max(len(s1nums|tnums),1) if (s1nums and tnums) else 0.0,
        1.0 if (s1pins and tpins and s1pins&tpins) else 0.0,
        compute_numeric_pincode_match(s1ph,tph),
        compute_domain_match(s1dom,tdom),
        compute_script_mismatch(s1rn,trn),
        abs(len(n1)-len(n2))/max(len(n1),len(n2),1),
        abs(len(a1)-len(a2))/max(len(a1),len(a2),1),
        1.0 if not a2.strip() else 0.0,
        1.0 if (s1st and tst and s1st!=tst) else 0.0,
        door, extra
    ]

def main():
    print("="*70)
    print("THRESHOLD CALIBRATION ON REAL TRAINING DATA")
    print("="*70)
    payload = joblib.load(MODEL_PATH)
    model = payload["model"] if isinstance(payload,dict) else payload
    print(f"Model: {model.tree_count_} trees")

    t0=time.time()
    s1=pl.read_csv(TRAIN_S1,separator="\t",null_values=[""]).head(N_EVAL)
    s2=pl.read_csv(TRAIN_S2,separator="\t",null_values=[""])
    s3=pl.read_csv(TRAIN_S3,separator="\t",null_values=[""])
    gt=pl.read_csv(TRAIN_GT,separator="\t",null_values=[""])
    s1_ids=set(s1["entity_id"].to_list())

    gt_map={}
    for r in gt.iter_rows(named=True):
        sid=r["source1_entity_id"]
        if sid not in s1_ids: continue
        m=str(r.get("matched_entity_ids") or "").strip()
        gt_map[sid]=set(x.strip() for x in m.split(",") if x.strip()) if m else set()

    raw_t={}
    for r in pl.concat([s2,s3]).iter_rows(named=True):
        raw_t[r["entity_id"]]=(str(r.get("business_name") or ""),
                                str(r.get("business_address") or ""))
    print(f"Loaded in {time.time()-t0:.1f}s | S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    cand_df=pl.read_csv(TRAIN_CANDS,separator="\t",null_values=[""])
    cand_map={}
    for r in cand_df.iter_rows(named=True):
        sid=r["source1_entity_id"]
        if sid not in s1_ids: continue
        c=str(r.get("candidate_entity_ids") or "").strip()
        cand_map[sid]=[x.strip() for x in c.split(",") if x.strip()] if c else []
    print(f"Candidates loaded for {len(cand_map):,} entities")

    # Cache targets
    t_cache={}
    needed=set(tid for cands in cand_map.values() for tid in cands)
    for tid in needed:
        if tid not in raw_t: continue
        rn,ra=raw_t[tid]
        cn=clean_name(rn); ca=clean_address(ra)
        t_cache[tid]=(cn,ca,rn,extract_domain_stem(rn),
                      extract_numeric_tokens(f"{cn} {ca}"),
                      extract_pin_tokens(ca),
                      extract_numeric_pincode_tokens(ca),
                      get_state(ra),extract_primary_number(ra),
                      set(cn.split()),1.0 if not ca.strip() else 0.0)
    print(f"Cached {len(t_cache):,} targets")

    # Build features
    print("Scoring pairs...")
    ts=time.time()
    s1_info={}
    for r in s1.iter_rows(named=True):
        sid=r["entity_id"]; rn=str(r.get("business_name") or "")
        ra=str(r.get("business_address") or ""); cn=clean_name(rn)
        ca=clean_address(ra)
        s1_info[sid]=(cn,ca,rn,extract_domain_stem(rn),
                      extract_numeric_tokens(f"{cn} {ca}"),
                      extract_pin_tokens(ca),
                      extract_numeric_pincode_tokens(ca),
                      get_state(ra),extract_primary_number(ra),set(cn.split()))

    all_ids=s1["entity_id"].to_list()
    pairs=[]; feats=[]; misses=[]
    for sid in all_ids:
        if sid not in s1_info or sid not in cand_map: continue
        s1n,s1a,s1rn,s1dom,s1nums,s1pins,s1ph,s1st,s1dnum,s1words=s1_info[sid]
        for tid in cand_map[sid]:
            ti=t_cache.get(tid)
            if not ti: continue
            tn,ta,trn,tdom,tnums,tpins,tph,tst,tdnum,twords,miss=ti
            feats.append(featurize(s1rn,s1a,trn,ta,s1dom,tdom,s1nums,tnums,
                                   s1pins,tpins,s1ph,tph,s1st,tst,s1dnum,tdnum,
                                   s1words,twords))
            pairs.append((sid,tid)); misses.append(miss)

    probs=model.predict_proba(np.array(feats,dtype=np.float32))[:,1]
    print(f"Scored {len(probs):,} pairs in {time.time()-ts:.2f}s")

    ent_data={}
    for (sid,tid),p,miss in zip(pairs,probs,misses):
        ent_data.setdefault(sid,[]).append((tid,float(p),float(miss)))

    # Grid search
    print("\n"+"="*70)
    print("GRID SEARCH...")
    print("="*70)
    SING=[0.30,0.40,0.50,0.60,0.70]
    LINK=[0.75,0.80,0.83,0.85,0.87,0.88,0.90,0.92]
    MARG=[0.80,0.85,0.88,0.90,0.92,0.95,1.00]
    CAPS=[4,5,6,7,None]
    best=0.0; best_p=None; results=[]

    for th_s in SING:
      for th_l in LINK:
        for mrg in MARG:
          for cap in CAPS:
            preds={}; tot=0; sings=0
            for sid in all_ids:
                csl=ent_data.get(sid)
                if not csl:
                    preds[sid]=[]; sings+=1; continue
                mx=max(p for _,p,_ in csl)
                if mx<th_s:
                    preds[sid]=[]; sings+=1; continue
                ms=[]; nc=0
                for tid,p,miss in sorted(csl,key=lambda x:-x[1]):
                    if p>=th_l and p>=mrg*mx:
                        if miss==1.0:
                            if nc>=1: continue
                            nc+=1
                        ms.append(tid)
                if cap: ms=ms[:cap]
                preds[sid]=ms; tot+=len(ms)
                if not ms: sings+=1
            avg=tot/max(len(all_ids),1)
            f05,prec,rec=macro_f05(preds,gt_map,all_ids)
            results.append((f05,prec,rec,avg,sings,th_s,th_l,mrg,cap))
            if f05>best:
                best=f05; best_p=results[-1]
                print(f"  BEST th_s={th_s} th_l={th_l} mrg={mrg} cap={cap} "
                      f"F0.5={f05:.4f} P={prec:.4f} R={rec:.4f} avg={avg:.3f}")

    print("\n"+"="*70+"  TOP 20  "+"="*70)
    print(f"{'F0.5':>7} {'Prec':>7} {'Rec':>7} {'AvgM':>6} {'Sings':>7}  th_s  th_l  marg  cap")
    for r in sorted(results,key=lambda x:-x[0])[:20]:
        f05,p,rc,av,sg,ts2,tl,mg,cp=r
        print(f"{f05:7.4f} {p:7.4f} {rc:7.4f} {av:6.3f} {sg:7,}  "
              f"{ts2:.2f}  {tl:.2f}  {mg:.2f}  {str(cp):>4}")

    print("\n"+"="*70)
    print("OPTIMAL RESULT:")
    f05,p,rc,av,sg,ts2,tl,mg,cp=best_p
    print(f"  th_singleton = {ts2}")
    print(f"  th_link      = {tl}")
    print(f"  rel_margin   = {mg}")
    print(f"  max_cap      = {cp}")
    print(f"  F0.5={f05:.4f}  Precision={p:.4f}  Recall={rc:.4f}")
    print(f"  Avg matches  = {av:.3f}  (target: 3.46)")
    print(f"  Singletons   = {sg:,}  (target: ~{int(0.056*N_EVAL):,})")
    if av>4.0: print(f"\nWARNING: avg {av:.2f}>4.0, precision will suffer on test set")
    elif av<2.5: print(f"\nWARNING: avg {av:.2f}<2.5, recall may suffer")
    else: print(f"\nOK: avg matches {av:.2f} is in target range [2.5-4.0]")

if __name__=="__main__":
    main()
