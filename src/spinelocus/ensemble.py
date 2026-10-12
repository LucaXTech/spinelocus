"""Equal coordinate ensembles, with strict pairing and no ground-truth decisions."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import paired_patient_bootstrap, summarize, summarize_patients
from .refinement_audit import KEY, META, replay
from .utils import provenance, write_json

PROTOCOL_FIELDS = (
    "split", "images", "patients", "imgsz", "tta", "conf", "keypoint_conf",
    "alpha", "min_roi", "max_roi", "coordinate_convention",
)


def mean_coordinates(predictions):
    """Average global coordinates only where every member supplies a finite point.

    A missing member makes that landmark missing, rather than silently changing
    the ensemble's weights. Partial missing coordinates and infinities are invalid.
    """
    members = np.asarray(predictions, dtype=float)
    if members.ndim != 3 or members.shape[0] < 1 or members.shape[2] != 2:
        raise ValueError("Expected [members, landmarks, 2] prediction coordinates")
    if np.isinf(members).any() or np.any(np.isnan(members).any(2) != np.isnan(members).all(2)):
        raise ValueError("Missing predictions must have both coordinates missing")
    valid = np.isfinite(members).all(axis=(0, 2))
    output = np.full(members.shape[1:], np.nan)
    output[valid] = members[:, valid].mean(axis=0)
    return output


def ensemble_protocol(count):
    return {"aggregation": "equal_mean_global_coordinates", "members": count,
            "missing_policy": "require_all_members", "guard_order": "after_aggregation",
            "uses_ground_truth": False}


def _read_evaluation(path):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    source = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    protocol = {key: summary["protocol"][key] for key in PROTOCOL_FIELDS}
    protocol['yolo_flip_tta'] = summary['protocol'].get('yolo_flip_tta', False)
    if not isinstance(protocol['yolo_flip_tta'], bool):
        raise ValueError("Invalid YOLO flip protocol flag")
    if protocol["split"] != "val":
        raise ValueError("Ensemble exploration is validation only")
    frame, _, _ = replay(pd.read_csv(path / "predictions.csv", dtype={"patient_id": str}))
    yolo = frame[frame["mode"] == "yolo"].set_index(KEY).sort_index()
    hrnet = frame[frame["mode"] == "yolo_hrnet"].set_index(KEY).sort_index()
    if (yolo.index.get_level_values("image").nunique(),
        yolo.index.get_level_values("patient_id").nunique()) != (protocol["images"], protocol["patients"]):
        raise ValueError("Saved protocol counts disagree with coordinates")
    for mode, part in (("yolo", yolo), ("yolo_hrnet", hrnet)):
        for key, value in summarize(part.reset_index()).items():
            saved = summary[mode][key]
            equal = value is saved if value is None or saved is None else math.isclose(
                value, saved, rel_tol=1e-7, abs_tol=1e-6)
            if not equal:
                raise ValueError(f"Saved summary disagrees with coordinates: {mode}/{key}")
    # The single-model evaluator writes hashes in this documented input order.
    files = list(source["files_sha256"].items())
    if len(files) != 3 or not files[0][0].lower().endswith(".csv") or any(
        not name.lower().endswith(".pt") for name, _ in files[1:]
    ):
        raise ValueError("Expected single-model provenance: manifest, YOLO checkpoint, HRNet checkpoint")
    if any(source["config"].get(key, False if key == 'yolo_flip_tta' else None) != value
           for key, value in protocol.items()):
        raise ValueError("Saved provenance and evaluation protocol differ")
    return protocol, files, yolo, hrnet


def ensemble_evaluations(evaluations, out):
    """Replay one specified ensemble; never search weights or choose points using GT."""
    paths = [Path(path).resolve() for path in evaluations]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Specify at least two distinct single-model evaluations")
    out = Path(out).resolve()
    if any(out == path or path in out.parents for path in paths):
        raise ValueError("Ensemble output must be outside source evaluations")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Ensemble output must be empty")
    prepared = [_read_evaluation(path) for path in paths]
    protocol, first_files, yolo, _ = prepared[0]
    member_hashes = [files[2][1] for _, files, _, _ in prepared]
    if len(set(member_hashes)) != len(member_hashes):
        raise ValueError("Ensemble members must have distinct HRNet checkpoint hashes")
    for other_protocol, files, other_yolo, _ in prepared[1:]:
        if other_protocol != protocol or [h for _, h in files[:2]] != [h for _, h in first_files[:2]]:
            raise ValueError("Ensemble members must share the protocol, manifest and YOLO checkpoint")
        if not yolo.index.equals(other_yolo.index) or not yolo[META].equals(other_yolo[META]):
            raise ValueError("Ensemble members must use identical ground truth and landmark keys")
        if not np.array_equal(yolo[["pred_x", "pred_y"]].to_numpy(),
                              other_yolo[["pred_x", "pred_y"]].to_numpy(), equal_nan=True):
            raise ValueError("Ensemble members must have identical YOLO predictions")
    coordinates = mean_coordinates([part[["pred_x", "pred_y"]].to_numpy()
                                    for _, _, _, part in prepared])
    refined = yolo.reset_index().copy()
    refined["mode"] = "yolo_hrnet"
    refined[["pred_x", "pred_y"]] = coordinates
    delta = coordinates - refined[["gt_x", "gt_y"]].to_numpy()
    refined["error_px"] = np.linalg.norm(delta, axis=1)
    refined["component_mae_px"] = np.abs(delta).mean(1)
    combined, decisions, _ = replay(pd.concat([yolo.reset_index(), refined], ignore_index=True))
    summary, by_patient, by_vertebra, comparison = {}, [], [], []
    for mode, part in combined.groupby("mode"):
        patients, macro = summarize_patients(part)
        summary[mode] = {**summarize(part), "patient_macro": macro}
        by_patient.extend({"mode": mode, **record} for record in patients)
        comparison.append({"method": mode, "macro_patient_pck5": macro["pck_at_5px_all_visible"],
                           "point_pck5": summary[mode]["pck_at_5px_all_visible"],
                           "mean_px": summary[mode]["mean_euclidean_px_on_predictions"],
                           "p95_px": summary[mode]["p95_px_on_predictions"],
                           "failed_points": summary[mode]["failed_points"]})
        by_vertebra.extend({"mode": mode, "vertebra": int(vertex), **summarize(rows)}
                           for vertex, rows in part.groupby("vertebra"))
    summary["protocol"] = {**protocol, "identity_guard": "nearest-yolo",
                           "hrnet_ensemble": ensemble_protocol(len(paths)), "replayed_from_csv": True}
    summary["refinement_guard"] = {"rule": "nearest-yolo", "uses_ground_truth": False,
                                   "rejected_refinements": int(decisions.rejected.sum()),
                                   "decisions": len(decisions)}
    summary["paired_patient_bootstrap"] = paired_patient_bootstrap(combined)
    summary["guard_vs_raw_patient_bootstrap"] = paired_patient_bootstrap(
        combined, baseline="yolo_hrnet", candidate="yolo_hrnet_guarded")
    summary["members"] = []
    for number, (path, (_, files, _, part)) in enumerate(zip(paths, prepared), start=1):
        _, macro = summarize_patients(part.reset_index())
        member = part.reset_index().assign(mode="member")
        metrics = summarize(member)
        summary["members"].append({"evaluation": str(path), "hrnet_sha256": files[2][1],
                                   "metrics": {**metrics, "patient_macro": macro},
                                   "ensemble_vs_member_patient_bootstrap": paired_patient_bootstrap(
                                       pd.concat([member, refined], ignore_index=True), baseline="member")})
        comparison.append({"method": f"member-{number}", "macro_patient_pck5": macro["pck_at_5px_all_visible"],
                           "point_pck5": metrics["pck_at_5px_all_visible"],
                           "mean_px": metrics["mean_euclidean_px_on_predictions"],
                           "p95_px": metrics["p95_px_on_predictions"],
                           "failed_points": metrics["failed_points"]})
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "predictions.csv", index=False)
    decisions.to_csv(out / "refinement_guard.csv", index=False)
    pd.DataFrame(by_patient).to_csv(out / "by_patient.csv", index=False)
    pd.DataFrame(by_vertebra).to_csv(out / "by_vertebra.csv", index=False)
    pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)
    write_json(out / "summary.json", summary)
    receipt = provenance(summary["protocol"], [path / name for path in paths
                         for name in ("predictions.csv", "summary.json", "provenance.json")])
    receipt["exploratory_validation"] = True
    receipt["test_evaluated"] = False
    receipt["member_hrnet_sha256"] = member_hashes
    write_json(out / "provenance.json", receipt)
    lines = ["# Ensemble HRNet — validation", "",
             "Media a pesi uguali delle coordinate globali, prima del controllo identità. "
             "La TTA viene applicata separatamente a ogni modello. Nessuna scelta per punto usa GT.", "",
             "| Metodo | PCK5 soggetti | Media px | P95 px | Mancanti |",
             "|---|---:|---:|---:|---:|"]
    for row in comparison:
        mean = "n/d" if row["mean_px"] is None else f'{row["mean_px"]:.3f}'
        p95 = "n/d" if row["p95_px"] is None else f'{row["p95_px"]:.3f}'
        lines.append(f'| {row["method"]} | {row["macro_patient_pck5"]:.2%} | {mean} | {p95} | {row["failed_points"]} |')
    lines += ["", "Se manca un punto in un membro, manca anche nell'ensemble. "
              "Le predizioni raw e quelle corrette restano separate.", "",
              "Risultati esplorativi sui medesimi soggetti di validation già usati per confrontare "
              "i modelli. Il piccolo guadagno non dimostra superiorità su dati nuovi. "
              "Bootstrap descrittivo per soggetto nel JSON; il test non viene letto.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f'Ensemble completed: {out / "comparison.csv"}', flush=True)
    return combined, summary
