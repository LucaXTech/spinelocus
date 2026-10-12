"""Metrics include failed predictions and do not use GT to select detections."""

import numpy as np


def summarize(frame):
    if frame.empty:
        raise ValueError("No visible ground-truth keypoints to evaluate")
    error = frame.error_px.to_numpy(float)
    success = np.isfinite(error)
    finite = error[success]
    diagonal = np.hypot(frame.width.to_numpy(float), frame.height.to_numpy(float))
    n = len(error)
    return {
        "visible_points": n,
        "predicted_points": int(success.sum()),
        "failed_points": int((~success).sum()),
        "prediction_coverage": float(success.mean()),
        "mean_euclidean_px_on_predictions": float(finite.mean()) if len(finite) else None,
        "median_px_on_predictions": float(np.median(finite)) if len(finite) else None,
        "p95_px_on_predictions": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "component_mae_px_on_predictions": float(frame.loc[success, "component_mae_px"].mean())
        if len(finite)
        else None,
        "nme_diagonal_on_predictions": float(np.mean(finite / diagonal[success])) if len(finite) else None,
        **{f"pck_at_{t}px_all_visible": float(np.sum(error <= t) / n) for t in (2, 3, 5, 8, 10)},
        # Exact integral of the empirical CDF on [0,30], including failures in denominator.
        "auc_cdf_30px_all_visible": float(np.sum(np.maximum(30 - finite, 0)) / (30 * n)),
    }


def coordinate_rows(gt, pred, visibility, image, patient, split, mode, width, height):
    rows = []
    for i, xy in enumerate(np.asarray(gt)):
        if visibility[i] <= 0:
            continue
        predicted = np.asarray(pred[i], float)
        finite = np.isfinite(predicted).all()
        delta = predicted - xy
        rows.append(
            {
                "image": image,
                "patient_id": patient,
                "split": split,
                "vertebra": i + 1,
                "mode": mode,
                "width": width,
                "height": height,
                "gt_x": float(xy[0]),
                "gt_y": float(xy[1]),
                "pred_x": float(predicted[0]) if finite else None,
                "pred_y": float(predicted[1]) if finite else None,
                "error_px": float(np.linalg.norm(delta)) if finite else np.nan,
                "component_mae_px": float(np.abs(delta).mean()) if finite else np.nan,
            }
        )
    return rows


def summarize_patients(frame):
    """Equal weight per subject; missing predictions stay in PCK denominators."""
    records = [
        {"patient_id": str(patient), **summarize(part)}
        for patient, part in frame.groupby("patient_id", sort=True)
    ]
    if not records:
        raise ValueError("No patients to evaluate")
    means = [r["mean_euclidean_px_on_predictions"] for r in records
             if r["mean_euclidean_px_on_predictions"] is not None]
    macro = {
        "patients": len(records),
        "patients_with_predictions": len(means),
        "mean_euclidean_px_on_predictions": float(np.mean(means)) if means else None,
        **{key: float(np.mean([r[key] for r in records])) for key in (
            "prediction_coverage", "pck_at_2px_all_visible", "pck_at_3px_all_visible",
            "pck_at_5px_all_visible", "pck_at_8px_all_visible", "pck_at_10px_all_visible",
            "auc_cdf_30px_all_visible",
        )},
    }
    return records, macro


def paired_patient_bootstrap(frame, baseline="yolo", candidate="yolo_hrnet", repeats=2000, seed=42):
    """Patients are the independent units; points/images are correlated clusters.

    Failure-sensitive primary outcome is PCK@5. Mean Euclidean gains are conditional
    on both methods returning predictions and are explicitly labelled as such.
    """
    key = ["patient_id", "image", "vertebra"]
    a = frame[frame["mode"] == baseline].set_index(key)
    b = frame[frame["mode"] == candidate].set_index(key)
    if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(b.index):
        raise ValueError("Paired methods must cover exactly the same visible GT keys")
    joined = a[["error_px"]].join(b[["error_px"]], lsuffix="_a", rsuffix="_b")
    joined["pck_gain"] = (joined.error_px_b <= 5).astype(float) - (joined.error_px_a <= 5).astype(float)
    finite = np.isfinite(joined.error_px_a) & np.isfinite(joined.error_px_b)
    joined["error_gain"] = np.where(finite, joined.error_px_a - joined.error_px_b, np.nan)
    patients = joined.groupby(level="patient_id").agg(
        pck5_gain=("pck_gain", "mean"), mean_error_gain_on_paired_predictions=("error_gain", "mean")
    )
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    rng = np.random.default_rng(seed)
    result = {
        "independent_unit": "patient",
        "patients": len(patients),
        "bootstrap_repeats": repeats,
        "warning": "Few independent patients: intervals are descriptive and generalization is uncertain."
        if len(patients) < 10
        else None,
    }
    for col in patients:
        values = patients[col].dropna().to_numpy()
        if len(values) < 2:
            result[col] = {"mean": float(values.mean()) if len(values) else None, "ci95": None}
            continue
        samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
        result[col] = {"mean": float(values.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
    return result
