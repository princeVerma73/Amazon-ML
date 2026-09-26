"""
Package final competition deliverables into <team_name>_submission.zip
strictly conforming to the official competition directory hierarchy:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import time
import zipfile

def package_submission(team_name: str = "BER-NeuralForge") -> str:
    zip_filename = f"{team_name}_submission.zip"
    print("=" * 70)
    print(f"BUILDING OFFICIAL SUBMISSION PACKAGE: {zip_filename}")
    print("=" * 70)
    t0 = time.time()

    # Verify input deliverables exist
    required_files = [
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
        "student_resource/Documentation_template.md",
        "code/business_entity_resolution/requirements.txt",
        "code/business_entity_resolution/README.md",
    ]
    for rf in required_files:
        if not os.path.isfile(rf):
            raise FileNotFoundError(f"Missing required deliverable: {rf}")

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        # 1. Output folder
        print("Adding output/matching_results.tsv...")
        zf.write("output/matching_results.tsv", "output/matching_results.tsv")
        print("Adding output/candidate_pairs.tsv...")
        zf.write("output/candidate_pairs.tsv", "output/candidate_pairs.tsv")

        # 2. Methodology documentation at root
        print("Adding Documentation_template.md...")
        zf.write("student_resource/Documentation_template.md", "Documentation_template.md")

        # 3. Code directory
        print("Adding code/business_entity_resolution/...")
        for root, dirs, files in os.walk("code/business_entity_resolution"):
            # Exclude pycache and hidden directories
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for f in files:
                if f.endswith(".pyc") or f.startswith("."):
                    continue
                file_path = os.path.join(root, f)
                arcname = file_path.replace("\\", "/")
                print(f"  -> {arcname}")
                zf.write(file_path, arcname)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print("=" * 70)
    print(f"SUCCESS: Package created: {zip_filename}")
    print(f"  Total Compressed Size: {size_mb:.2f} MB")
    print(f"  Packaging Duration:    {elapsed:.2f}s")
    print("=" * 70)
    return zip_filename

if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else "BER-NeuralForge"
    package_submission(team)
