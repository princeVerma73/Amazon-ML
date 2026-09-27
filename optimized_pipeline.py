"""
LAST DAY OPTIMIZED PIPELINE — Target: 0.95+ by 11:59 PM

Strategy:
  Round 1 (NOW):   Augment candidates (pincode + token) + rel_margin=0.82
                   Run 12-core parallel scoring → ~50 min
                   Submit → Expected 0.89+

  Round 2 (4 PM):  FAISS semantic candidates + retrain → submit → 0.93+
  Round 3 (9 PM):  If time: retrain on 200k → submit → 0.95+

Run: python optimized_pipeline.py
"""

import os, sys, re, time, logging, joblib, shutil
import numpy as np
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
from rapidfuzz import fuzz, distance
from tqdm import tqdm

sys.path.insert(0, "code/business_entity_resolution/src")
from normalizer import clean_legal_suffixes, normalize_text, clean_address, extract_domain_stem
from feature_extraction import (extract_numeric_tokens, extract_pin_tokens,
    extract_numeric_pincode_tokens, compute_numeric_pincode_match,
    compute_domain_match, compute_script_mismatch)
from train_conflict_model import get_state, extract_primary_number

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("OptPipeline")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TEST_DIR      = "student_resource/dataset/test"
CAND_IN       = "output/candidate_pairs.tsv"        # existing 20-cand file
CAND_OUT      = "output/candidate_pairs_augmented.tsv"
MATCH_OUT     = "output/matching_results.tsv"
MODEL_PATH    = "models/catboost_conflict_aware.joblib"
TH_LINK       = 0.70
REL_MARGIN    = 0.82    # loosened from 0.90 → more recall
TH_SINGLETON  = 0.30
N_WORKERS     = 10      # use 10 of 12 cores (leave 2 for OS)
CHUNK_SIZE    = 50_000  # smaller chunks for better parallelism

STOP = {'the','a','an','of','and','&','or','for','in','at','to','by',
        'le','la','les','de','du','des','pvt','ltd','llc','inc','corp','co'}

# ── HELPERS ──────────────────────────────────────────────────────────────────
def clean_name(s): return clean_legal_suffixes(normalize_text(str(s or "")))

def extract_pincode(addr: str) -> str | None:
    """Extract 6-digit Indian pincode or 5-digit US ZIP."""
    if not addr: return None
    m = re.search(r'\b(\d{6})\b', addr)
    if m: return m.group(1)
    m = re.search(r'\b(\d{5})(?:-\d{4})?\b', addr)
    if m: return m.group(1)
    return None

def name_tokens(name: str) -> frozenset:
    """Meaningful tokens from a business name."""
    cn = clean_name(name)
    return frozenset(t for t in cn.split() if len(t) > 2 and t not in STOP)

def featurize_batch(rows):
    """Featurize a list of (s1_info, t_info) tuples. Returns np.array."""
    feats = []
    for s1, t in rows:
        s1rn,s1a,s1dom,s1nums,s1pins,s1ph,s1st,s1dnum,s1words,s1cn,s1ca = s1
        trn,ta,tdom,tnums,tpins,tph,tst,tdnum,twords,tcn,tca,miss = t
        door = 0.0
        if s1dnum is not None and tdnum is not None and abs(s1dnum-tdnum)>1:
            ss,ts2=str(s1dnum),str(tdnum)
            if not(ss.startswith(ts2) or ts2.startswith(ss)): door=1.0
        extra=0.0
        if s1words and twords:
            sg=s1words-twords-STOP; tg=twords-s1words-STOP
            if sg and tg: extra=min(1.0,(len(sg)+len(tg))/max(len(s1words|twords),1))
        feats.append([
            fuzz.token_sort_ratio(s1cn,tcn)/100, fuzz.token_set_ratio(s1cn,tcn)/100,
            distance.JaroWinkler.normalized_similarity(s1cn,tcn),
            distance.Levenshtein.normalized_similarity(s1cn,tcn),
            fuzz.token_sort_ratio(s1ca,tca)/100, fuzz.token_set_ratio(s1ca,tca)/100,
            distance.JaroWinkler.normalized_similarity(s1ca,tca),
            len(s1nums&tnums)/max(len(s1nums|tnums),1) if(s1nums and tnums) else 0.0,
            1.0 if(s1pins and tpins and s1pins&tpins) else 0.0,
            compute_numeric_pincode_match(s1ph,tph),
            compute_domain_match(s1dom,tdom),
            compute_script_mismatch(s1rn,trn),
            abs(len(s1cn)-len(tcn))/max(len(s1cn),len(tcn),1),
            abs(len(s1ca)-len(tca))/max(len(s1ca),len(tca),1) if(s1ca and tca) else 1.0,
            miss,
            1.0 if(s1st and tst and s1st!=tst) else 0.0,
            door, extra
        ])
    return np.array(feats, dtype=np.float32)


# ── PHASE 1: AUGMENT CANDIDATES ──────────────────────────────────────────────
def augment_candidates():
    """Add pincode-exact and name-token-overlap matches to existing candidates."""
    if os.path.exists(CAND_OUT):
        log.info(f"Augmented candidates already exist: {CAND_OUT} — skipping.")
        return

    log.info("=" * 65)
    log.info("PHASE 1: AUGMENTING CANDIDATES (pincode + token overlap)")
    log.info("=" * 65)
    t0 = time.time()

    s2 = pl.read_csv(f"{TEST_DIR}/test_source2.tsv", separator="\t", null_values=[""])
    s3 = pl.read_csv(f"{TEST_DIR}/test_source3.tsv", separator="\t", null_values=[""])
    s1 = pl.read_csv(f"{TEST_DIR}/test_source1.tsv", separator="\t", null_values=[""])
    targets = pl.concat([s2, s3])
    log.info(f"Loaded S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

    # Build pincode → target_ids index
    log.info("Building pincode index...")
    pincode_idx: dict[str, list[str]] = {}
    token_idx:   dict[str, list[str]] = {}  # token → target_ids

    for r in tqdm(targets.iter_rows(named=True), total=len(targets), desc="Indexing targets"):
        tid  = r["entity_id"]
        addr = str(r.get("business_address") or "")
        name = str(r.get("business_name") or "")
        pc   = extract_pincode(addr)
        if pc:
            pincode_idx.setdefault(pc, []).append(tid)
        for tok in name_tokens(name):
            token_idx.setdefault(tok, []).append(tid)

    log.info(f"Pincode index: {len(pincode_idx):,} unique pincodes")
    log.info(f"Token index:   {len(token_idx):,} unique tokens")

    # Load existing candidates
    log.info("Loading existing candidates...")
    cand_map: dict[str, set[str]] = {}
    cand_df = pl.read_csv(CAND_IN, separator="\t", null_values=[""])
    for r in cand_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        c   = str(r.get("candidate_entity_ids") or "").strip()
        cand_map[sid] = set(x.strip() for x in c.split(",") if x.strip()) if c else set()

    # Augment
    log.info("Augmenting with pincode + token matches...")
    pincode_added = 0; token_added = 0
    MAX_TOKEN_MATCHES = 50   # cap to avoid explosion for generic names

    for r in tqdm(s1.iter_rows(named=True), total=len(s1), desc="Augmenting S1"):
        sid  = r["entity_id"]
        addr = str(r.get("business_address") or "")
        name = str(r.get("business_name") or "")
        existing = cand_map.get(sid, set())

        # Pincode match
        pc = extract_pincode(addr)
        if pc and pc in pincode_idx:
            for tid in pincode_idx[pc]:
                if tid not in existing:
                    existing.add(tid)
                    pincode_added += 1

        # Token overlap (≥2 tokens in common)
        toks = name_tokens(name)
        if len(toks) >= 2:
            token_hits: dict[str, int] = {}
            for tok in toks:
                for tid in token_idx.get(tok, []):
                    token_hits[tid] = token_hits.get(tid, 0) + 1
            for tid, cnt in token_hits.items():
                if cnt >= 2 and tid not in existing:
                    existing.add(tid)
                    token_added += 1
            # Hard cap: if too many matches, keep only top-MAX by token overlap
            if len(existing) > 60:
                # Sort existing by token overlap score, keep best 60
                scored = []
                for tid in existing:
                    scored.append((tid, token_hits.get(tid, 0)))
                scored.sort(key=lambda x: -x[1])
                existing = set(t[0] for t in scored[:60])

        cand_map[sid] = existing

    log.info(f"Added {pincode_added:,} via pincode  +  {token_added:,} via token overlap")
    avg_cands = sum(len(v) for v in cand_map.values()) / max(len(cand_map), 1)
    log.info(f"New avg candidates/entity: {avg_cands:.1f}")

    # Write augmented candidates
    log.info(f"Writing augmented candidates → {CAND_OUT}")
    with open(CAND_OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for r in s1.iter_rows(named=True):
            sid = r["entity_id"]
            cids = ",".join(cand_map.get(sid, set()))
            f.write(f"{sid}\t{cids}\n")

    log.info(f"Augmentation done in {time.time()-t0:.1f}s")
    return cand_map


# ── PHASE 2: PARALLEL SCORING ────────────────────────────────────────────────
def score_chunk_worker(args):
    """Worker function: scores one chunk and returns lines for matching_results.tsv."""
    (chunk_s1_rows, raw_targets, cand_map, model_path,
     th_link, rel_margin, th_singleton) = args

    payload = joblib.load(model_path)
    model   = payload["model"] if isinstance(payload, dict) else payload

    def get_tinfo(tid):
        rt = raw_targets.get(tid)
        if not rt: return None
        rn, ra = rt
        cn = clean_name(rn); ca = clean_address(ra)
        return (rn, ra, extract_domain_stem(rn),
                extract_numeric_tokens(f"{cn} {ca}"),
                extract_pin_tokens(ca), extract_numeric_pincode_tokens(ca),
                get_state(ra), extract_primary_number(ra),
                set(cn.split()), cn, ca, 1.0 if not ca.strip() else 0.0)

    lines = []
    for sid, s1rn, s1ra in chunk_s1_rows:
        s1cn = clean_name(s1rn); s1ca = clean_address(s1ra)
        s1info = (s1rn, s1ra, extract_domain_stem(s1rn),
                  extract_numeric_tokens(f"{s1cn} {s1ca}"),
                  extract_pin_tokens(s1ca), extract_numeric_pincode_tokens(s1ca),
                  get_state(s1ra), extract_primary_number(s1ra),
                  set(s1cn.split()), s1cn, s1ca)

        cands = [tid for tid in cand_map.get(sid, []) if raw_targets.get(tid)]
        if not cands:
            lines.append(f"{sid}\t\n"); continue

        # Build feature matrix
        tinfos = [get_tinfo(tid) for tid in cands]
        valid  = [(tid, ti) for tid, ti in zip(cands, tinfos) if ti]
        if not valid:
            lines.append(f"{sid}\t\n"); continue

        rows   = [(s1info, ti) for _, ti in valid]
        X      = featurize_batch(rows)
        probs  = model.predict_proba(X)[:, 1]

        pmax = probs.max()
        if pmax < th_singleton:
            lines.append(f"{sid}\t\n"); continue

        matches = []
        for (tid, _), p in zip(valid, probs):
            if p >= th_link and p >= rel_margin * pmax:
                matches.append(tid)

        lines.append(f"{sid}\t{','.join(matches)}\n")

    return lines


def parallel_score(cand_map_path: str):
    log.info("=" * 65)
    log.info("PHASE 2: PARALLEL SCORING WITH 10 WORKERS")
    log.info("=" * 65)
    t0 = time.time()

    s1_df = pl.read_csv(f"{TEST_DIR}/test_source1.tsv", separator="\t", null_values=[""])
    s2_df = pl.read_csv(f"{TEST_DIR}/test_source2.tsv", separator="\t", null_values=[""])
    s3_df = pl.read_csv(f"{TEST_DIR}/test_source3.tsv", separator="\t", null_values=[""])

    # Load all targets into dict
    log.info("Loading targets into memory...")
    raw_targets: dict[str, tuple] = {}
    for r in pl.concat([s2_df, s3_df]).iter_rows(named=True):
        raw_targets[r["entity_id"]] = (
            str(r.get("business_name") or ""),
            str(r.get("business_address") or "")
        )
    log.info(f"Loaded {len(raw_targets):,} targets")

    # Load augmented candidates
    log.info(f"Loading candidates from {cand_map_path}...")
    cand_map: dict[str, list[str]] = {}
    cand_df = pl.read_csv(cand_map_path, separator="\t", null_values=[""])
    for r in cand_df.iter_rows(named=True):
        c = str(r.get("candidate_entity_ids") or "").strip()
        cand_map[r["source1_entity_id"]] = [x.strip() for x in c.split(",") if x.strip()] if c else []
    log.info(f"Loaded candidates for {len(cand_map):,} entities")

    # Build chunks of (sid, name, addr) tuples
    all_rows = []
    for r in s1_df.iter_rows(named=True):
        all_rows.append((r["entity_id"], str(r.get("business_name") or ""),
                         str(r.get("business_address") or "")))

    chunks = [all_rows[i:i+CHUNK_SIZE] for i in range(0, len(all_rows), CHUNK_SIZE)]
    log.info(f"Scoring {len(all_rows):,} entities in {len(chunks)} chunks × {N_WORKERS} workers")

    # Write output
    tmp_out = MATCH_OUT + ".tmp"
    written = 0

    with open(tmp_out, "w", encoding="utf-8", newline="\n") as fout:
        fout.write("source1_entity_id\tmatched_entity_ids\n")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {}
            for i, chunk in enumerate(chunks):
                args = (chunk, raw_targets, cand_map, MODEL_PATH,
                        TH_LINK, REL_MARGIN, TH_SINGLETON)
                futures[pool.submit(score_chunk_worker, args)] = i

            completed = 0
            for fut in as_completed(futures):
                chunk_idx = futures[fut]
                lines = fut.result()
                fout.writelines(lines)
                written += len(lines)
                completed += 1
                elapsed = time.time() - t0
                rate = written / elapsed
                eta = (len(all_rows) - written) / max(rate, 1)
                log.info(f"Chunk {chunk_idx+1:2d}/{len(chunks)} done | "
                         f"{written:,}/{len(all_rows):,} entities | "
                         f"{rate:.0f}/s | ETA {eta/60:.1f}min")

    # Atomic replace
    shutil.move(tmp_out, MATCH_OUT)
    elapsed = time.time() - t0
    log.info(f"Scoring complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Quick stats
    total_m = 0; sings = 0
    with open(MATCH_OUT, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if len(p) > 1 and p[1].strip():
                total_m += len(p[1].split(","))
            else:
                sings += 1
    n = len(all_rows)
    log.info(f"Avg matches: {total_m/n:.3f}  Singletons: {sings:,} ({sings/n*100:.2f}%)")

    # Update candidate_pairs.tsv to be the augmented version
    shutil.copy(cand_map_path, "output/candidate_pairs.tsv")
    log.info("Updated candidate_pairs.tsv with augmented version")


# ── PHASE 3: VALIDATE ─────────────────────────────────────────────────────────
def validate():
    log.info("Running official validator...")
    ret = os.system(
        "python student_resource/utils/validate_submission.py "
        "--matching output/matching_results.tsv "
        "--candidate output/candidate_pairs.tsv "
        f"--test-dir {TEST_DIR}"
    )
    if ret == 0:
        log.info("VALIDATOR: PASS — safe to submit!")
    else:
        log.error("VALIDATOR: FAIL — check output files!")
    return ret == 0


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-augment", action="store_true", help="Skip candidate augmentation")
    parser.add_argument("--cand-file", default=None, help="Override candidate file path")
    args = parser.parse_args()

    t_total = time.time()
    log.info("=" * 65)
    log.info("LAST DAY OPTIMIZED PIPELINE")
    log.info(f"  th_link={TH_LINK}  rel_margin={REL_MARGIN}  workers={N_WORKERS}")
    log.info("=" * 65)

    # Phase 1: Augment candidates
    if not args.skip_augment:
        augment_candidates()
        cand_file = CAND_OUT
    else:
        cand_file = args.cand_file or CAND_IN
        log.info(f"Skipping augmentation, using: {cand_file}")

    # Phase 2: Parallel scoring
    parallel_score(cand_file)

    # Phase 3: Validate
    validate()

    total_mins = (time.time() - t_total) / 60
    log.info(f"TOTAL PIPELINE TIME: {total_mins:.1f} minutes")
    log.info("Upload output/matching_results.tsv to the portal!")
