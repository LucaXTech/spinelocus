"""Read-only reconstruction from the recovered CSV/JSON files, without image uploads."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .data import patient_id
from .utils import load_config, provenance, write_json


def recover(directory, out):
    root = Path(directory)
    inputs = {"train": "index_gt.csv", "val": "index_gt(1).csv", "test": "test_e2e_predictions.csv"}
    frames, sets, inventory = {}, {}, []
    for split, name in inputs.items():
        frame = pd.read_csv(root / name)
        if frame.duplicated(["orig_filename", "vertebra"]).any():
            raise ValueError(f"Duplicate image-landmark keys in {name}")
        names = sorted(frame.orig_filename.unique())
        ids = {patient_id(n) for n in names}
        frames[split], sets[split] = frame, (set(names), ids)
        for n in names:
            inventory.append({"image_name": n, "patient_id": patient_id(n), "split": split})
    summary = {
        "status": "reconstructed_from_csv_only_not_image_directory_verification",
        "identity_source": "COPDGene subject ID; supersedes erroneous Series-based v0.1.0 audit",
        "splits": {},
        "overlaps": {},
        "historical_versions": "Original library versions not recorded; exact replay unproven.",
    }
    for split, frame in frames.items():
        summary["splits"][split] = {
            "images": len(sets[split][0]),
            "patients": len(sets[split][1]),
            "points": len(frame),
            "all_images_have_11_points": bool((frame.groupby("orig_filename").size() == 11).all()),
        }
        if "jitter_used" in frame:
            summary["splits"][split]["jitter_counts"] = {
                str(k): int(v) for k, v in frame.jitter_used.value_counts().items()
            }
    for a, b in combinations(sets, 2):
        summary["overlaps"][f"{a}_{b}"] = {
            "images": len(sets[a][0] & sets[b][0]),
            "patients": len(sets[a][1] & sets[b][1]),
        }
    summary["subject_leakage_detected"] = any(v["patients"] > 0 for v in summary["overlaps"].values())
    summary["interpretation"] = "Historical metrics are descriptive; these splits do not isolate unseen subjects."
    test = frames["test"]
    for model in ("yolo", "hrnet"):
        dx = test[f"x_{model}_abs"] - test.x_gt_abs
        dy = test[f"y_{model}_abs"] - test.y_gt_abs
        error = np.hypot(dx, dy)
        summary[f"test_{model}"] = {
            "mean_euclidean_px": float(error.mean()),
            "component_mae_px": float(((abs(dx) + abs(dy)) / 2).mean()),
            "median_px": float(error.median()),
            "pck5": float((error <= 5).mean()),
            "max_error_column_discrepancy": float(np.max(np.abs(error - test[f"err_{model}_px"]))),
        }
    compare_name = "compare_raw_20250913_000138.csv"
    comparison = pd.read_csv(root / compare_name)
    comparison_images = set(comparison.orig_filename)
    summary["comparison_population_matches"] = {
        split: comparison_images == values[0] for split, values in sets.items()
    }
    for mode, part in comparison.groupby("mode"):
        summary[f"comparison_{mode}"] = {
            "mean_euclidean_px": float(part.err_px.mean()),
            "pck5": float((part.err_px <= 5).mean()),
        }
    summary["yolo_configs"] = {}
    for name in ("args.yaml", "args(1).yaml"):
        cfg = load_config(root / name)
        summary["yolo_configs"][name] = {
            k: cfg[k] for k in ("model", "epochs", "imgsz", "optimizer", "lr0", "cos_lr", "seed")
        }
        summary["yolo_configs"][name]["model"] = Path(str(cfg["model"])).name
    summary["hrnet_crop_predictions"] = {
        "rows": len(pd.read_csv(root / "hrnet_val_pred.csv")),
        "small_summary_count": load_config(root / "hrnet_val_metrics.json")["count"],
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_csv(out / "recovered_inventory.csv", index=False)
    write_json(out / "audit.json", summary)
    write_json(
        out / "provenance.json",
        provenance(
            files=[
                root / n
                for n in [
                    *inputs.values(),
                    compare_name,
                    "args.yaml",
                    "args(1).yaml",
                    "hrnet_val_pred.csv",
                    "hrnet_val_metrics.json",
                ]
            ]
        ),
    )
    return summary
