import re
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "code/business_entity_resolution/src")
from classifier import parse_ground_truth, compute_instance_macro_f05
from feature_extraction import extract_house_plot_tokens

def check_house_number_conflict(s1_addr: str, target_addr: str) -> bool:
    if not s1_addr or not target_addr:
        return False
    s1_nums = extract_house_plot_tokens(s1_addr)
    t_nums = extract_house_plot_tokens(target_addr)
    if s1_nums and t_nums:
        return len(s1_nums.intersection(t_nums)) == 0
    return False

def main():
    print("Testing House Number Conflict Rule on Ground Truth...")
    gt_map = parse_ground_truth("sample_data/sample_ground_truth.tsv")
    s1_df = pd.read_csv("sample_data/sample_source1.tsv", sep="\t", keep_default_na=False).iloc[:10000]
    cand_df = pd.read_csv("sample_data/sample_candidate_pairs.tsv", sep="\t", keep_default_na=False).iloc[:10000]
    s2 = pd.read_csv("sample_data/sample_source2.tsv", sep="\t", keep_default_na=False)
    s3 = pd.read_csv("sample_data/sample_source3.tsv", sep="\t", keep_default_na=False)
    target_all = pd.concat([s2, s3], ignore_index=True)
    t_map = dict(zip(target_all["entity_id"], target_all["business_address"]))
    s1_map = dict(zip(s1_df["entity_id"], s1_df["business_address"]))

    # Test conflicting house numbers on true ground truth
    conflicts_in_gt = 0
    total_gt_pairs = 0
    for s1_id, matches in gt_map.items():
        s1_a = s1_map.get(s1_id, "")
        for mid in matches:
            t_a = t_map.get(mid, "")
            total_gt_pairs += 1
            if check_house_number_conflict(s1_a, t_a):
                conflicts_in_gt += 1

    print(f"Total GT pairs checked: {total_gt_pairs:,}")
    print(f"True matches with conflicting house number: {conflicts_in_gt:,} ({conflicts_in_gt/max(1, total_gt_pairs)*100:.3f}%)")
    print(f"-> House number consistency holds for {(1 - conflicts_in_gt/max(1, total_gt_pairs))*100:.2f}% of true matches!")

if __name__ == "__main__":
    main()
