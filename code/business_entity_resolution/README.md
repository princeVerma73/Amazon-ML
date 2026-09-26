# Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

## Overview
This package implements the end-to-end Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**. It resolves entity linkages between reference Source 1 ($S_1$) records and noisy, heterogeneous target pools ($S_2, S_3$) across multiple countries (United States, India, and France as an open-set test country), directly optimizing the competition's instance-level **Macro $F_{0.5}$** metric.

---

## Hardware & Environment Requirements
- **OS**: Windows / Linux / macOS
- **Python**: >= 3.10 (tested on Python 3.12)
- **GPU (Recommended)**: NVIDIA GPU with CUDA support (e.g. GeForce RTX 3050 Laptop GPU / T4 / V100 / A100).
- **RAM**: Minimum 8 GB (16 GB recommended for 11.7M record streaming inference).

---

## Installation

Install all pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── normalizer.py         # Text preprocessing, legal suffix stripping, address expansion
│   ├── make_sample.py        # Development benchmark slice generator with hard-negative mining
│   ├── blocking.py           # Dynamic country-partitioned TF-IDF sparse candidate generation
│   ├── feature_extraction.py # RapidFuzz SIMD 18-feature extraction with token caching
│   ├── classifier.py         # CatBoost GPU & LightGBM GBDT models with 2D dual-threshold tuning
│   ├── ensemble.py           # Weighted probability ensemble blender
│   └── pipeline.py           # Streaming chunked end-to-end inference orchestrator
├── requirements.txt          # Pinned library dependencies
├── README.md                 # Reproduction instructions
└── run_pipeline.py           # Top-level executable CLI
```

---

## How to Reproduce End-to-End

### 1. Fast Development Benchmark (100k Entities with Metric Evaluation)
Generates candidates, trains CatBoost on GPU, searches 2D optimal thresholds, and evaluates against ground truth:

```bash
# Generate 100k hard-negative benchmark slice
python src/make_sample.py --n-s1 100000 --n-us 60000 --n-india 40000

# Run end-to-end sample benchmark
python run_pipeline.py --mode sample --model-type catboost
```

### 2. Full Test Set Inference (11.7M Records / 1.73M S1 Queries)
Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` on the full competition test set:

```bash
python run_pipeline.py --mode test --model-type catboost --chunk-size 20000 --top-k 20 --min-sim 0.10
```

### 3. Submission Validation
Validate formatting against official competition rules:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
