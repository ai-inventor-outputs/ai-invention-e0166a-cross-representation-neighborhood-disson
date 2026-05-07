# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
# ]
# ///
"""
data.py — Standardize 5 clinical triage datasets into exp_sel_data_out.json schema.

Loads from temp/datasets/, processes each dataset, outputs full_data_out.json
with the schema: {"datasets": [{"dataset": "...", "examples": [{...}]}]}

For large datasets, subsamples to at most 15,000 rows (stratified by label)
to keep CPU-only CRND processing within 1 hour.

Datasets:
  1. medical_abstracts   — 14K PubMed abstracts, 5 disease categories
  2. mimic_iv_ed_demo    — 207 ED triage records, ESI acuity levels
  3. clinical_patient_triage_nl — 31 synthetic triage vignettes, 6 severity levels
  4. ohsumed_single      — 7.4K MEDLINE abstracts, 23 MeSH categories
  5. mental_health_conditions — 15K social media posts, 7 conditions
"""
import json
import csv
import resource
import sys
import time
from pathlib import Path

# Resource limits: 14GB RAM, 1h CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

WORKSPACE = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260216_170044/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"

MAX_SAMPLES_PER_DATASET = 15000

start_time = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - start_time
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def load_json(path: Path) -> list:
    """Load a JSON file that contains a top-level array."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_subsample(examples: list, max_n: int) -> list:
    """Subsample examples stratified by 'output' field."""
    if len(examples) <= max_n:
        return examples

    import random
    random.seed(42)

    # Group by output label
    by_label: dict[str, list] = {}
    for ex in examples:
        label = ex["output"]
        by_label.setdefault(label, []).append(ex)

    # Calculate per-label allocation (proportional)
    n_labels = len(by_label)
    result = []
    remaining = max_n
    labels = sorted(by_label.keys())

    for i, label in enumerate(labels):
        group = by_label[label]
        if i == len(labels) - 1:
            n_take = remaining
        else:
            n_take = max(1, round(len(group) / len(examples) * max_n))
            n_take = min(n_take, remaining, len(group))
        random.shuffle(group)
        result.extend(group[:n_take])
        remaining -= min(n_take, len(group))

    random.shuffle(result)
    return result


def process_medical_abstracts() -> dict:
    """Process TimSchopf/medical_abstracts (Priority 4 - PRIMARY)."""
    log("Processing TimSchopf/medical_abstracts...")

    label_map = {
        1: "Neoplasms",
        2: "Digestive_system_diseases",
        3: "Nervous_system_diseases",
        4: "Cardiovascular_diseases",
        5: "General_pathological_conditions",
    }

    examples = []
    for split_name, fold_id in [("train", 0), ("test", 1)]:
        fpath = DATASETS_DIR / f"full_TimSchopf_medical_abstracts_{split_name}.json"
        if not fpath.exists():
            log(f"  WARNING: {fpath.name} not found, skipping")
            continue
        data = load_json(fpath)
        log(f"  Loaded {split_name}: {len(data)} rows")
        for idx, row in enumerate(data):
            text = row.get("medical_abstract", "")
            label_num = row.get("condition_label")
            if not text or label_num is None:
                continue
            label_str = label_map.get(int(label_num), f"Unknown_{label_num}")
            examples.append({
                "input": text.strip(),
                "output": label_str,
                "metadata_fold": fold_id,
                "metadata_task_type": "classification",
                "metadata_n_classes": 5,
                "metadata_row_index": idx,
                "metadata_source": "huggingface/TimSchopf/medical_abstracts",
            })

    examples = stratified_subsample(examples, max_n=MAX_SAMPLES_PER_DATASET)
    log(f"  Final: {len(examples)} examples")
    return {"dataset": "medical_abstracts", "examples": examples}


def process_mimic_iv_ed_demo() -> dict:
    """Process MIMIC-IV-ED Demo triage CSV (Priority 2)."""
    log("Processing MIMIC-IV-ED Demo...")

    triage_path = (
        DATASETS_DIR
        / "mimic_iv_ed_demo"
        / "physionet.org"
        / "files"
        / "mimic-iv-ed-demo"
        / "2.2"
        / "ed"
        / "triage.csv"
    )

    if not triage_path.exists():
        log(f"  WARNING: {triage_path} not found")
        return {"dataset": "mimic_iv_ed_demo", "examples": []}

    examples = []
    with open(triage_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            cc = row.get("chiefcomplaint", "").strip()
            acuity = row.get("acuity", "").strip()
            if not cc or not acuity:
                continue

            # Build combined clinical text with vitals
            parts = [f"Chief complaint: {cc}"]
            vitals = []
            if row.get("heartrate", "").strip():
                vitals.append(f"HR {row['heartrate'].strip()}")
            if row.get("resprate", "").strip():
                vitals.append(f"RR {row['resprate'].strip()}")
            if row.get("sbp", "").strip() and row.get("dbp", "").strip():
                vitals.append(f"BP {row['sbp'].strip()}/{row['dbp'].strip()}")
            if row.get("temperature", "").strip():
                vitals.append(f"Temp {row['temperature'].strip()}")
            if row.get("o2sat", "").strip():
                vitals.append(f"O2 {row['o2sat'].strip()}%")
            pain_val = row.get("pain", "").strip()
            if pain_val and pain_val not in ("", "UA", "ua", "uta", "unable", "o", "ett", "Critical"):
                vitals.append(f"Pain {pain_val}/10")
            if vitals:
                parts.append("Vitals: " + ", ".join(vitals))

            text = ". ".join(parts)

            examples.append({
                "input": text,
                "output": f"ESI-{acuity}",
                "metadata_fold": 2,  # validation_mimic fold
                "metadata_task_type": "classification",
                "metadata_n_classes": 5,
                "metadata_row_index": idx,
                "metadata_source": "physionet/mimic-iv-ed-demo/2.2",
                "metadata_subject_id": row.get("subject_id", ""),
                "metadata_stay_id": row.get("stay_id", ""),
            })

    log(f"  Final: {len(examples)} examples")
    return {"dataset": "mimic_iv_ed_demo", "examples": examples}


def process_shoriful_triage() -> dict:
    """Process Shoriful025/clinical_patient_triage_natural_language."""
    log("Processing Shoriful025/clinical_patient_triage_natural_language...")

    fpath = DATASETS_DIR / "full_Shoriful025_clinical_patient_triage_natural_language_train.json"
    if not fpath.exists():
        log(f"  WARNING: {fpath.name} not found")
        return {"dataset": "clinical_patient_triage_nl", "examples": []}

    data = load_json(fpath)
    log(f"  Loaded: {len(data)} rows")

    examples = []
    for idx, row in enumerate(data):
        note = row.get("clinical_note", "").strip()
        severity = row.get("severity_level", "").strip()
        if not note or not severity:
            continue

        # Enrich with available metadata
        parts = [note]
        age = row.get("patient_age")
        if age is not None:
            parts.append(f"Age: {age}")
        vitals_status = row.get("vital_signs_status", "").strip()
        if vitals_status:
            parts.append(f"Vital signs: {vitals_status}")
        complaint = row.get("primary_complaint", "").strip()
        if complaint:
            parts.append(f"Primary complaint: {complaint}")

        text = ". ".join(parts)

        examples.append({
            "input": text,
            "output": severity,
            "metadata_fold": 0,
            "metadata_task_type": "classification",
            "metadata_row_index": idx,
            "metadata_source": "huggingface/Shoriful025/clinical_patient_triage_natural_language",
            "metadata_record_id": row.get("record_id", ""),
        })

    log(f"  Final: {len(examples)} examples")
    return {"dataset": "clinical_patient_triage_nl", "examples": examples}


def process_ohsumed_single() -> dict:
    """Process joao-luz/ohsumed-single (23 cardiovascular disease categories)."""
    log("Processing joao-luz/ohsumed-single...")

    # Label map from the dataset card
    label_names = {
        0: "Bacterial_Infections_and_Mycoses",
        1: "Virus_Diseases",
        2: "Parasitic_Diseases",
        3: "Neoplasms",
        4: "Musculoskeletal_Diseases",
        5: "Digestive_System_Diseases",
        6: "Stomatognathic_Diseases",
        7: "Respiratory_Tract_Diseases",
        8: "Otorhinolaryngologic_Diseases",
        9: "Nervous_System_Diseases",
        10: "Eye_Diseases",
        11: "Urologic_and_Male_Genital_Diseases",
        12: "Female_Genital_Diseases_and_Pregnancy_Complications",
        13: "Cardiovascular_Diseases",
        14: "Hemic_and_Lymphatic_Diseases",
        15: "Neonatal_Diseases_and_Abnormalities",
        16: "Skin_and_Connective_Tissue_Diseases",
        17: "Nutritional_and_Metabolic_Diseases",
        18: "Endocrine_Diseases",
        19: "Immunologic_Diseases",
        20: "Disorders_of_Environmental_Origin",
        21: "Animal_Diseases",
        22: "Pathological_Conditions_Signs_and_Symptoms",
    }

    examples = []
    for split_name, fold_id in [("train", 0), ("test", 1)]:
        fpath = DATASETS_DIR / f"full_joao-luz_ohsumed-single_{split_name}.json"
        if not fpath.exists():
            log(f"  WARNING: {fpath.name} not found, skipping")
            continue
        data = load_json(fpath)
        log(f"  Loaded {split_name}: {len(data)} rows")
        for idx, row in enumerate(data):
            text = row.get("text", "").strip()
            label = row.get("label")
            if not text or label is None:
                continue
            label_str = label_names.get(int(label), f"Category_{label}")
            examples.append({
                "input": text,
                "output": label_str,
                "metadata_fold": fold_id,
                "metadata_task_type": "classification",
                "metadata_n_classes": 23,
                "metadata_row_index": idx,
                "metadata_source": "huggingface/joao-luz/ohsumed-single",
            })

    examples = stratified_subsample(examples, max_n=MAX_SAMPLES_PER_DATASET)
    log(f"  Final: {len(examples)} examples")
    return {"dataset": "ohsumed_single", "examples": examples}


def process_mental_health_conditions() -> dict:
    """Process sai1908/Mental_Health_Condition_Classification."""
    log("Processing sai1908/Mental_Health_Condition_Classification...")

    fpath = DATASETS_DIR / "full_sai1908_Mental_Health_Condition_Classification_train.json"
    if not fpath.exists():
        log(f"  WARNING: {fpath.name} not found")
        return {"dataset": "mental_health_conditions", "examples": []}

    data = load_json(fpath)
    log(f"  Loaded: {len(data)} rows")

    examples = []
    for idx, row in enumerate(data):
        text = row.get("text", "").strip()
        status = row.get("status", "").strip()
        if not text or not status:
            continue
        examples.append({
            "input": text,
            "output": status,
            "metadata_fold": 0,
            "metadata_task_type": "classification",
            "metadata_row_index": idx,
            "metadata_source": "huggingface/sai1908/Mental_Health_Condition_Classification",
        })

    examples = stratified_subsample(examples, max_n=MAX_SAMPLES_PER_DATASET)
    log(f"  Final: {len(examples)} examples")
    return {"dataset": "mental_health_conditions", "examples": examples}


def main() -> None:
    log("Starting dataset standardization pipeline")
    log(f"Workspace: {WORKSPACE}")
    log(f"Datasets dir: {DATASETS_DIR}")
    log(f"Output: {OUTPUT_FILE}")

    datasets = []

    # Process BEST 5 datasets selected for CRND pipeline:
    # 1. medical_abstracts - PRIMARY (14K, 5 disease classes, published paper)
    # 2. mimic_iv_ed_demo - Real ED triage data (207 rows, ESI acuity + chief complaints)
    # 3. clinical_patient_triage_nl - Clinical notes with severity (31 rows, exact schema match)
    # 4. ohsumed_single - Medical abstracts (7.4K, 23 disease categories, classic benchmark)
    # 5. mental_health_conditions - Health text (15K, 7 conditions, natural class overlap)
    processors = [
        process_medical_abstracts,
        process_mimic_iv_ed_demo,
        process_shoriful_triage,
        process_ohsumed_single,
        process_mental_health_conditions,
    ]

    for proc in processors:
        try:
            result = proc()
            if result["examples"]:
                datasets.append(result)
                log(f"  ✓ {result['dataset']}: {len(result['examples'])} examples")
            else:
                log(f"  ✗ {result['dataset']}: no valid examples, skipping")
        except Exception as e:
            log(f"  ✗ ERROR in {proc.__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Build output
    output = {"datasets": datasets}

    # Summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    total = 0
    for ds in datasets:
        n = len(ds["examples"])
        total += n
        # Count unique labels
        labels = set(ex["output"] for ex in ds["examples"])
        log(f"  {ds['dataset']:40s} | {n:6d} examples | {len(labels):3d} classes")
    log(f"  {'TOTAL':40s} | {total:6d} examples | {len(datasets):3d} datasets")

    # Write output
    log(f"Writing {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    log(f"  Written: {file_size:.1f} MB")

    elapsed = time.time() - start_time
    log(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
