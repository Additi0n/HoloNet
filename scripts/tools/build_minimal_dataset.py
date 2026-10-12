# -*- coding: utf-8 -*-
"""Assemble the minimal dataset MDPI asks for at submission.

MDPI defines it as "the minimal dataset that is required to support the central
findings", so this collects the numbers behind every table and figure in the
manuscript, not the RGBA corpus or the model weights (those are large and are
covered by the Data Availability Statement instead).

Output: article/minimal_dataset.zip
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "article" / "minimal_dataset.zip"

RUNS = ROOT / "all_results" / "runs"
VAR = ROOT / "runs" / "variance"
LOCAL = ROOT / "experiments_local" / "results"
TABLES = ROOT / "result_tables"
CURVES = ROOT / "article" / "data"

MAIN = {"CICMalDroid2020": "holonet_cicmaldroid2020",
        "BODMAS": "holonet_bodmas",
        "ELF2020": "holonet_elf2020"}
ABLATIONS = {
    "w/o Conv-KAN": "ablation_no_ckan_{}_no_ckan",
    "w/o Transformer Router": "ablation_no_transformer_{}_no_transformer",
    "w/o Margin Loss": "ablation_no_margin_{}_no_margin",
}
SLUG = {"CICMalDroid2020": "cicmaldroid2020", "BODMAS": "bodmas", "ELF2020": "elf2020"}


def metrics(history_path):
    h = json.loads(Path(history_path).read_text(encoding="utf-8"))
    return {
        "dataset": h["dataset"],
        "epochs": h["args"]["epochs"],
        "seed": h["args"]["seed"],
        "best_epoch": h["best_epoch"],
        "test": {k: round(v, 6) for k, v in h["test_metrics"].items()},
    }


# ---- Tables 1-3: HoloNet test metrics per platform -------------------------
main_results = {}
for ds, run in MAIN.items():
    p = RUNS / run / "history.json"
    if p.is_file():
        main_results[ds] = metrics(p)

# ---- Table 4: ablations ----------------------------------------------------
ablation = {}
for label, pattern in ABLATIONS.items():
    ablation[label] = {}
    for ds, slug in SLUG.items():
        p = RUNS / pattern.format(slug) / "history.json"
        if p.is_file():
            ablation[label][ds] = metrics(p)["test"]

# ---- seed variance (supports the reproducibility claim) --------------------
variance = {}
for d in sorted(VAR.iterdir()):
    if d.is_dir() and (d / "history.json").is_file():
        m = metrics(d / "history.json")
        variance.setdefault(m["dataset"], {})[d.name] = m["test"]

summary = {
    "manuscript": "Cross-Platform Malware Fingerprint Identification with a "
                  "Convolutional KAN-Transformer Hybrid Architecture",
    "note": "Values behind the tables and figures of the manuscript. Datasets "
            "(CICMalDroid2020, BODMAS, ELF2020) are public and obtained from their "
            "original providers; the RGBA conversion is produced by Bin2Img as "
            "described in Section 3.2.",
    "tables_1_to_3_holonet_test_metrics": main_results,
    "table_4_ablation_test_metrics": ablation,
    "seed_variance_test_metrics": variance,
}

# ---- assemble --------------------------------------------------------------
extras = [
    (LOCAL / "transfer_probe.json", "figure_4_transfer_probe.json"),
    (LOCAL / "confusion_metrics.json", "per_class_confusion_metrics.json"),
    (TABLES / "robustness_tables.md", "tables_6_to_8_robustness.md"),
    (TABLES / "robustness_severity_design.md", "table_5_perturbation_severities.md"),
    (TABLES / "explainability_tables.md", "table_9_interpretability.md"),
    (CURVES / "training_curves_cicmaldroid2020.dat", "figure_3_curves_cicmaldroid2020.dat"),
    (CURVES / "training_curves_bodmas.dat", "figure_3_curves_bodmas.dat"),
    (CURVES / "training_curves_elf2020.dat", "figure_3_curves_elf2020.dat"),
]

README = """Minimal dataset
===============

Manuscript: Cross-Platform Malware Fingerprint Identification with a
Convolutional KAN-Transformer Hybrid Architecture

This archive holds the values behind every table and figure in the manuscript.

  results_summary.json                  Tables 1-3 (per-platform test metrics),
                                        Table 4 (ablations), and the per-seed
                                        variance behind the reproducibility claim
  figure_3_curves_*.dat                 Per-epoch train/validation Accuracy and F1
  figure_4_transfer_probe.json          Cross-platform linear-probe Macro-F1 matrix
                                        and the class-prior random baselines
  table_5_perturbation_severities.md    Perturbation definitions
  tables_6_to_8_robustness.md           Robustness results by perturbation type
  table_9_interpretability.md           Confidence statistics per dataset and class
  per_class_confusion_metrics.json      Per-class precision/recall/F1

Not included, by size: the RGBA image corpus produced by Bin2Img and the trained
model checkpoints. Both are covered by the Data Availability Statement and are
available from the corresponding author on request. The three source datasets are
public and obtained from their original providers.
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("README.txt", README)
    z.writestr("results_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    for src, name in extras:
        if src.is_file():
            z.write(src, name)
        else:
            print(f"  missing, skipped: {src.name}")

print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
with zipfile.ZipFile(OUT) as z:
    for i in z.infolist():
        print(f"   {i.filename:42s} {i.file_size/1024:7.1f} KB")
