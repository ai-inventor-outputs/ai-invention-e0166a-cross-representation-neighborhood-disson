# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "scikit-learn>=1.3.0",
# ]
# ///
"""
Data standardization script for Clinical Text Dataset for CRND Triage Analysis.

Loads 2 chosen datasets from temp/datasets/, standardizes them to the
exp_sel_data_out.json schema, adds stratified 5-fold cross-validation splits,
and saves to full_data_out.json.

Selected datasets (best 2 for CRND triage analysis):
1. TimSchopf/medical_abstracts — 14,438 medical abstracts, 5 disease categories
   (published NLPIR 2022 paper, long text ideal for TF-IDF + embedding)
2. tchebonenko/MedicalTranscriptions — 5K clinical transcriptions, 5 specialty groups
   (genuine clinical notes, rich text ~3K chars avg, natural class overlap)
"""

import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

# --- Resource limits (14GB RAM, 1hr CPU) ---
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# --- Paths ---
WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"

# Optional: limit examples per dataset for debugging
MAX_EXAMPLES_PER_DATASET = int(sys.argv[1]) if len(sys.argv) > 1 else None

# --- Label mappings ---
MEDICAL_ABSTRACTS_LABELS = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

# Group 40 medical specialties into 5 major categories for CRND analysis
SPECIALTY_TO_GROUP = {}
_GROUPS = {
    "surgical": [
        "surgery", "orthopedic", "neurosurgery", "ophthalmology",
        "ent - otolaryngology", "urology", "cosmetic / plastic surgery",
        "podiatry", "dentistry", "bariatrics",
    ],
    "internal_medicine": [
        "general medicine", "gastroenterology", "nephrology",
        "endocrinology", "hematology - oncology", "allergy / immunology",
        "rheumatology", "ime-qme-work comp etc.", "hospice - palliative care",
        "physical medicine - rehab", "pain management", "sleep medicine",
        "letters", "diets and nutritions", "chiropractic", "speech - language",
        "autopsy", "office notes",
    ],
    "cardiovascular_pulmonary": [
        "cardiovascular / pulmonary",
    ],
    "neurology_psychiatry": [
        "neurology", "psychiatry / psychology",
    ],
    "radiology_consults_obgyn": [
        "radiology", "consult - history and phy.",
        "soap / chart / progress notes", "discharge summary",
        "emergency room reports", "lab medicine - pathology",
        "obstetrics / gynecology", "pediatrics - neonatal", "dermatology",
    ],
}
for group_name, specialties in _GROUPS.items():
    for spec in specialties:
        SPECIALTY_TO_GROUP[spec] = group_name


def load_json(filename: str) -> list[dict]:
    """Load a JSON file from temp/datasets/."""
    path = DATASETS_DIR / filename
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assign_folds(labels: list[str], n_splits: int = 5, random_state: int = 42) -> list[int]:
    """Assign stratified k-fold indices. Falls back to hash-based if too few per class."""
    if len(labels) < n_splits:
        return [i % n_splits for i in range(len(labels))]

    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    if min_count < n_splits:
        import hashlib
        folds = []
        for i, lbl in enumerate(labels):
            h = int(hashlib.md5(f"{lbl}_{i}_{random_state}".encode()).hexdigest(), 16)
            folds.append(h % n_splits)
        return folds

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_assignments = [0] * len(labels)
    dummy_X = list(range(len(labels)))
    for fold_idx, (_, test_indices) in enumerate(skf.split(dummy_X, labels)):
        for idx in test_indices:
            fold_assignments[idx] = fold_idx
    return fold_assignments


def limit(items: list) -> list:
    """Optionally limit items for debugging."""
    if MAX_EXAMPLES_PER_DATASET is not None:
        return items[:MAX_EXAMPLES_PER_DATASET]
    return items


# ============================================================
# Dataset 1: TimSchopf/medical_abstracts
# ============================================================
def process_medical_abstracts() -> dict | None:
    """TimSchopf/medical_abstracts: train+test combined, 5 disease categories."""
    print("  Loading TimSchopf/medical_abstracts (train + test)...")
    train_data = load_json("full_TimSchopf_medical_abstracts_default_train.json")
    test_data = load_json("full_TimSchopf_medical_abstracts_default_test.json")
    all_data = train_data + test_data
    print(f"    Loaded {len(train_data)} train + {len(test_data)} test = {len(all_data)} total")

    # Filter: valid text and label
    valid = [
        row for row in all_data
        if (row.get("medical_abstract") or "").strip()
        and row.get("condition_label") in MEDICAL_ABSTRACTS_LABELS
    ]
    print(f"    Valid after filtering: {len(valid)}")
    valid = limit(valid)

    labels = [MEDICAL_ABSTRACTS_LABELS[row["condition_label"]] for row in valid]
    folds = assign_folds(labels)

    examples = []
    for i, row in enumerate(valid):
        label_text = MEDICAL_ABSTRACTS_LABELS[row["condition_label"]]
        examples.append({
            "input": row["medical_abstract"].strip(),
            "output": label_text,
            "metadata_fold": folds[i],
            "metadata_task_type": "classification",
            "metadata_n_classes": 5,
            "metadata_row_index": i,
            "metadata_source": "TimSchopf/medical_abstracts",
            "metadata_original_label": row["condition_label"],
        })

    label_dist = Counter(e["output"] for e in examples)
    print(f"    Label dist: {dict(label_dist)}")
    return {"dataset": "medical_abstracts", "examples": examples}


# ============================================================
# Dataset 2: tchebonenko/MedicalTranscriptions
# ============================================================
def process_medical_transcriptions() -> dict | None:
    """tchebonenko/MedicalTranscriptions: 5K transcriptions, 40 specialties grouped into 5."""
    print("  Loading tchebonenko/MedicalTranscriptions...")
    all_data = load_json("full_tchebonenko_MedicalTranscriptions_train.json")
    print(f"    Loaded {len(all_data)} total")

    valid = []
    unmapped = Counter()
    for row in all_data:
        # Prefer transcription text (longer, richer); fall back to description
        text = (row.get("transcription") or "").strip()
        if not text or len(text) < 50:
            text = (row.get("description") or "").strip()
        if not text or len(text) < 50:
            continue

        specialty = (row.get("medical_specialty") or "").strip().lower()
        group = SPECIALTY_TO_GROUP.get(specialty)
        if group is None:
            unmapped[specialty] += 1
            group = "internal_medicine"  # catch-all for unmapped

        valid.append({"text": text, "group": group, "original_specialty": (row.get("medical_specialty") or "").strip()})

    if unmapped:
        print(f"    Unmapped specialties → internal_medicine: {dict(unmapped)}")
    print(f"    Valid after filtering (>=50 chars): {len(valid)}")
    valid = limit(valid)

    labels = [v["group"] for v in valid]
    folds = assign_folds(labels)

    examples = []
    for i, item in enumerate(valid):
        examples.append({
            "input": item["text"],
            "output": item["group"],
            "metadata_fold": folds[i],
            "metadata_task_type": "classification",
            "metadata_n_classes": len(set(labels)),
            "metadata_row_index": i,
            "metadata_source": "tchebonenko/MedicalTranscriptions",
            "metadata_original_specialty": item["original_specialty"],
        })

    label_dist = Counter(e["output"] for e in examples)
    print(f"    Label dist: {dict(label_dist)}")
    return {"dataset": "medical_transcriptions", "examples": examples}


# ============================================================
# Main
# ============================================================
def main():
    start = time.time()
    print("=" * 60)
    print("Clinical Text Dataset Standardization for CRND Triage Analysis")
    print(f"MAX_EXAMPLES_PER_DATASET: {MAX_EXAMPLES_PER_DATASET or 'unlimited'}")
    print("=" * 60)

    processors = [
        process_medical_abstracts,
        process_medical_transcriptions,
    ]

    all_datasets = []
    total_examples = 0
    for proc_fn in processors:
        print(f"\nProcessing: {proc_fn.__name__}...")
        t0 = time.time()
        try:
            result = proc_fn()
            if result and result["examples"]:
                all_datasets.append(result)
                n = len(result["examples"])
                total_examples += n
                print(f"  -> {n:,} examples in {time.time() - t0:.1f}s")
            else:
                print(f"  -> SKIPPED (no valid examples)")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()

    output = {"datasets": all_datasets}

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for ds in all_datasets:
        n = len(ds["examples"])
        labels = set(ex["output"] for ex in ds["examples"])
        print(f"  {ds['dataset']}: {n:,} examples, {len(labels)} classes")
    print(f"  TOTAL: {total_examples:,} examples across {len(all_datasets)} datasets")

    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Total time: {elapsed:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
