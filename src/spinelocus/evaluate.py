"""Full-image inference and evaluation with explicit split and complete denominators."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data import load_manifest, read_label, verify_manifest_files
from .ensemble import ensemble_protocol, mean_coordinates
from .geometry import crop, roi_size, to_global, window
from .metrics import coordinate_rows, paired_patient_bootstrap, summarize, summarize_patients
from .refinement import identity_guard
from .utils import provenance, sha256, write_json
from .yolo_views import flip_protocol, yolo_view_points


def yolo_points(result, min_confidence=0.0):
    points = np.full((11, 2), np.nan)
    if result.boxes is None or result.keypoints is None or not len(result.boxes):
        return points
    # Selection uses the network's confidence, never GT coordinates or GT bbox.
    index = int(result.boxes.conf.argmax().item())
    xy = result.keypoints.xy[index].cpu().numpy()
    if xy.shape != (11, 2):
        raise ValueError(f"Expected 11 spine keypoints, got {xy.shape}")
    good = np.isfinite(xy).all(1)
    if result.keypoints.conf is not None:
        good &= result.keypoints.conf[index].cpu().numpy() > min_confidence
    points[good] = xy[good]
    return points


def roi_diagnostics(gt, visibility, points, width, height, alpha=2.2, minimum=112, maximum=256):
    """GT is inspected only AFTER inference, never to choose or change a crop."""
    records = []
    for i, xy in enumerate(gt):
        if visibility[i] <= 0:
            continue
        available = bool(np.isfinite(points[i]).all())
        bound = window(width, height, *points[i], roi_size(points, i, alpha, minimum, maximum)) \
            if available else None
        inside = bool(bound and bound[0] <= xy[0] < bound[2] and bound[1] <= xy[1] < bound[3])
        records.append({
            "vertebra": i + 1, "crop_available": available, "gt_inside_crop": inside,
            **{name: bound[j] if bound else None for j, name in enumerate(("x1", "y1", "x2", "y2"))},
        })
    return records


def refine_points(image, points, model, cfg, device, alpha=2.2, minimum=112, maximum=256, tta=False):
    import torch

    from .crops import image_tensor
    from .model import decode

    output = np.full_like(points, np.nan)
    indices, patches, bounds = [], [], []
    for i in range(len(points)):
        if not np.isfinite(points[i]).all():
            continue
        size = roi_size(points, i, alpha, minimum, maximum)
        patch, bound = crop(image, points[i], size, cfg["crop_size"])
        indices.append(i)
        patches.append(image_tensor(patch))
        bounds.append(bound)
    if not patches:
        return output
    with torch.inference_mode():
        batch = torch.stack(patches).to(device)
        logits = model(batch)
        if tta:
            logits = (logits + model(batch.flip(-1)).flip(-1)) / 2
        local = (
            decode(
                logits,
                cfg["crop_size"],
                cfg.get("temperature_decode", 0.35),
                cfg.get("coordinate_convention", "pixel_centers"),
            )
            .cpu()
            .numpy()
        )
    for i, xy, bound in zip(indices, local, bounds):
        output[i] = to_global(xy, bound, cfg["crop_size"], cfg.get("coordinate_convention", "pixel_centers"))
    return output


def load_refiners(args, device):
    """Load fixed ensemble members, rejecting repeated weights and mixed decoding."""
    extra = getattr(args, "hrnet_extra", []) or []
    if extra and not args.hrnet:
        raise ValueError("--hrnet-extra requires --hrnet")
    paths = [args.hrnet, *extra] if args.hrnet else []
    if len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError("HRNet ensemble checkpoint paths must be distinct")
    if len(paths) > 1 and len({sha256(path) for path in paths}) != len(paths):
        raise ValueError("HRNet ensemble checkpoints must have distinct hashes")
    if not paths:
        return [], paths
    from .model import load_model

    refiners = [load_model(path, device) for path in paths]
    conventions = {cfg.get("coordinate_convention", "pixel_centers") for _, cfg in refiners}
    if len(conventions) != 1:
        raise ValueError("HRNet ensemble members must share the coordinate convention")
    return refiners, paths


def refine_ensemble(image, points, refiners, device, alpha=2.2, minimum=112, maximum=256, tta=False):
    """TTA/decode each HRNet first; average global points before any identity guard."""
    predictions = [refine_points(image, points, model, cfg, device, alpha, minimum, maximum, tta)
                   for model, cfg in refiners]
    return mean_coordinates(predictions)


def evaluate(args):
    from ultralytics import YOLO

    from .train import resolve_device

    root, out = Path(args.data), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be empty to avoid mixing experiments")
    manifest = load_manifest(args.manifest)
    verify_manifest_files(args.data, manifest)
    subset = manifest[manifest.split == args.split]
    if subset.empty:
        raise ValueError(f"No images in {args.split}")
    device = resolve_device(args.device)
    yolo = YOLO(args.yolo)
    refiners, hrnet_paths = load_refiners(args, device)
    cfg = refiners[0][1] if refiners else None
    rows, roi_rows, guard_rows, view_rows = [], [], [], []
    guard = getattr(args, "identity_guard", "none")
    yolo_device = str(device).replace("cuda:", "")
    for number, row in enumerate(subset.itertuples(), start=1):
        with Image.open(root / row.image) as im:
            image = im.convert("RGB")
        if image.size != (row.width, row.height):
            raise ValueError(f"Manifest image size changed: {row.image}")
        gt, vis = read_label(root / row.label, row.width, row.height)
        points, views = yolo_view_points(yolo, image, args, yolo_device, yolo_points)
        for mode, coordinates in views.items():
            view_rows.extend(coordinate_rows(gt, coordinates, vis, row.image, row.patient_id,
                                             args.split, mode, row.width, row.height))
        rows.extend(
            coordinate_rows(
                gt, points, vis, row.image, row.patient_id, args.split, "yolo", row.width, row.height
            )
        )
        if refiners:
            refined = refine_ensemble(
                image, points, refiners, device, args.alpha, args.min_roi, args.max_roi, args.tta
            )
            rows.extend(
                coordinate_rows(
                    gt,
                    refined,
                    vis,
                    row.image,
                    row.patient_id,
                    args.split,
                    "yolo_hrnet",
                    row.width,
                    row.height,
                )
            )
            if guard == "nearest-yolo":
                guarded, rejected, nearest = identity_guard(points, refined)
                rows.extend(coordinate_rows(
                    gt, guarded, vis, row.image, row.patient_id, args.split,
                    "yolo_hrnet_guarded", row.width, row.height,
                ))
                guard_rows.extend({"image": row.image, "patient_id": row.patient_id,
                                   "vertebra": i + 1, "rejected": bool(rejected[i]),
                                   "nearest_yolo_vertebra": int(nearest[i]) + 1 if nearest[i] >= 0 else None}
                                  for i in range(len(points)))
            roi_rows.extend({"image": row.image, "patient_id": row.patient_id, **record}
                            for record in roi_diagnostics(
                                gt, vis, points, row.width, row.height,
                                args.alpha, args.min_roi, args.max_roi,
                            ))
        if number % 50 == 0 or number == len(subset):
            print(f"Evaluated {number}/{len(subset)} {args.split} images", flush=True)
    frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "predictions.csv", index=False)
    if view_rows:
        pd.DataFrame(view_rows).to_csv(out / "yolo_views.csv", index=False)
    summary, patient_records = {}, []
    for mode, part in frame.groupby("mode"):
        patients, macro = summarize_patients(part)
        summary[mode] = {**summarize(part), "patient_macro": macro}
        patient_records.extend({"mode": mode, **record} for record in patients)
    pd.DataFrame(patient_records).to_csv(out / "by_patient.csv", index=False)
    summary["protocol"] = {
        "split": args.split,
        "images": len(subset),
        "patients": subset.patient_id.nunique(),
        "imgsz": args.imgsz,
        "tta": args.tta,
        "conf": args.conf,
        "keypoint_conf": args.keypoint_conf,
        "alpha": args.alpha,
        "min_roi": args.min_roi,
        "max_roi": args.max_roi,
        "coordinate_convention": cfg.get("coordinate_convention") if cfg else None,
        "identity_guard": guard,
        "yolo_flip_tta": bool(getattr(args, "yolo_flip_tta", False)),
    }
    if getattr(args, "yolo_flip_tta", False):
        summary["protocol"]["yolo_view_ensemble"] = flip_protocol()
    if len(refiners) > 1:
        summary["protocol"]["hrnet_ensemble"] = ensemble_protocol(len(refiners))
    if refiners:
        summary["paired_patient_bootstrap"] = paired_patient_bootstrap(frame)
        if guard_rows:
            guard_frame = pd.DataFrame(guard_rows)
            guard_frame.to_csv(out / "refinement_guard.csv", index=False)
            summary["refinement_guard"] = {
                "rule": guard, "rejected_refinements": int(guard_frame.rejected.sum()),
                "decisions": len(guard_frame), "uses_ground_truth": False,
            }
            summary["guard_vs_raw_patient_bootstrap"] = paired_patient_bootstrap(
                frame, baseline="yolo_hrnet", candidate="yolo_hrnet_guarded")
        roi_frame = pd.DataFrame(roi_rows)
        roi_frame.to_csv(out / "roi_diagnostics.csv", index=False)
        summary["roi_diagnostics"] = {
            "visible_points": len(roi_frame),
            "available_crops": int(roi_frame.crop_available.sum()),
            "gt_inside_crop": int(roi_frame.gt_inside_crop.sum()),
            "gt_outside_available_crop": int((roi_frame.crop_available & ~roi_frame.gt_inside_crop).sum()),
            "coverage_all_visible": float(roi_frame.gt_inside_crop.mean()),
            "purpose": "Post-inference diagnostic only; GT does not influence crop selection.",
        }
    write_json(out / "summary.json", summary)
    records = []
    for (mode, vertebra), part in frame.groupby(["mode", "vertebra"]):
        records.append({"mode": mode, "vertebra": int(vertebra), **summarize(part)})
    pd.DataFrame(records).to_csv(out / "by_vertebra.csv", index=False)
    weight_files = [args.yolo, *hrnet_paths]
    write_json(out / "provenance.json", provenance(summary["protocol"], [args.manifest, *weight_files]))
    print(summary)
    return frame, summary


def predict(args):
    """Unlabelled image inference. No dataset or GT is needed."""
    from ultralytics import YOLO

    from .train import resolve_device

    device = resolve_device(args.device)
    image = Image.open(args.image).convert("RGB")
    points, views = yolo_view_points(YOLO(args.yolo), image, args,
                                   str(device).replace("cuda:", ""), yolo_points)
    output = {"image_size": image.size, "yolo": points.tolist()}
    if views:
        output["yolo_view_ensemble"] = flip_protocol()
        output["yolo_views"] = {
            mode: [[v if np.isfinite(v) else None for v in xy] for xy in coordinates]
            for mode, coordinates in views.items()
        }
    if args.hrnet:
        refiners, paths = load_refiners(args, device)
        refined = refine_ensemble(
            image, points, refiners, device, args.alpha, args.min_roi, args.max_roi, args.tta
        )
        output["yolo_hrnet"] = refined.tolist()
        if len(refiners) > 1:
            output["hrnet_ensemble"] = {**ensemble_protocol(len(refiners)),
                                        "checkpoint_sha256": {str(path): sha256(path) for path in paths}}
        if getattr(args, "identity_guard", "none") == "nearest-yolo":
            guarded, rejected, _ = identity_guard(points, refined)
            output["yolo_hrnet_guarded"] = guarded.tolist()
            output["refinement_guard_rejected"] = rejected.tolist()
    elif getattr(args, "hrnet_extra", []):
        raise ValueError("--hrnet-extra requires --hrnet")
    # JSON uses null for missing predictions, never non-standard NaN literals.
    for mode in ("yolo", "yolo_hrnet", "yolo_hrnet_guarded"):
        if mode in output:
            output[mode] = [[v if np.isfinite(v) else None for v in xy] for xy in output[mode]]
    write_json(args.out, output)
