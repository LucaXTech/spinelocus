"""Strict labels and manifests. Split identity is explicit, never inferred from globals."""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from .utils import sha256

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
KEYPOINTS = 11


def patient_id(name):
    # Series numbers describe acquisitions, not subjects. Never fall back to them.
    matches = re.findall(r"COPDGene_([A-Za-z][A-Za-z0-9]*)_", str(name))
    if len(set(matches)) != 1:
        raise ValueError(f"Cannot resolve a unique patient ID from {name!r}; provide patient_id in manifest")
    return matches[0]


def read_label(path, width, height):
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one spine instance, got {len(lines)}")
    values = np.array([float(v) for v in lines[0].split()])
    if len(values) != 5 + KEYPOINTS * 3 or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected 38 finite tokens")
    if values[0] != 0:
        raise ValueError(f"{path}: class must be 0")
    bbox, points = values[1:5], values[5:].reshape(KEYPOINTS, 3)
    if np.any(bbox < 0) or np.any(bbox > 1) or np.any(bbox[2:] <= 0):
        raise ValueError(f"{path}: invalid normalized bounding box")
    if not np.isin(points[:, 2], [0, 1, 2]).all():
        raise ValueError(f"{path}: visibility must be 0, 1 or 2")
    visible = points[:, 2] > 0
    if np.any(points[visible, :2] < 0) or np.any(points[visible, :2] > 1):
        raise ValueError(f"{path}: visible keypoint outside image")
    return points[:, :2] * [width, height], points[:, 2]


def validate_manifest(frame, require_all=True, *, allow_source_overlap=False):
    required = {"image", "label", "split", "patient_id", "width", "height"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Manifest requires nonempty rows and columns {sorted(required)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing required values")
    for row in frame.itertuples():
        if "COPDGene_" in str(row.image) and str(row.patient_id) != patient_id(row.image):
            raise ValueError(
                "Manifest patient_id does not match COPDGene subject: "
                f"{row.image}. Old Series-based manifests are invalid; use split to create a new dataset."
            )
    if frame.image.duplicated().any():
        raise ValueError("Duplicate image paths in manifest")
    if not set(frame.split).issubset({"train", "val", "test"}):
        raise ValueError("Only train/val/test split names are supported")
    if require_all and set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Manifest must include nonempty train, val and test")
    if not allow_source_overlap and (frame.groupby("patient_id").split.nunique() > 1).any():
        raise ValueError("Patient leakage across splits")
    if (frame[["width", "height"]] <= 0).any().any():
        raise ValueError("Invalid image sizes")
    if not allow_source_overlap and "image_sha256" in frame and (frame.groupby("image_sha256").split.nunique() > 1).any():
        raise ValueError("Identical image bytes found across splits")
    return frame


def load_manifest(path):
    frame = pd.read_csv(path, dtype={"patient_id": str})
    return validate_manifest(frame)


def verify_manifest_files(root, frame):
    """Fail if registered labels/images were replaced after manifest generation."""
    root = Path(root).resolve()
    for number, row in enumerate(frame.itertuples(), 1):
        for relative in (row.image, row.label):
            if not (root / relative).resolve().is_relative_to(root):
                raise ValueError(f"Manifest path escapes dataset root: {relative}")
        with Image.open(root / row.image) as image:
            if image.size != (row.width, row.height):
                raise ValueError(f"Manifest dimensions changed: {row.image}")
        read_label(root / row.label, row.width, row.height)
        for column, relative in (("image_sha256", row.image), ("label_sha256", row.label)):
            if column in frame.columns and sha256(root / relative) != getattr(row, column):
                raise ValueError(f"Registered content changed: {relative}")
        if number % 100 == 0 or number == len(frame):
            print(f"Verified {number}/{len(frame)} image/label pairs", flush=True)


def resolve_subjects(frame):
    """Reinterpret a source inventory without authorizing its use for training."""
    frame = frame.copy()
    frame["patient_id"] = frame.image.map(patient_id)
    return validate_manifest(frame, allow_source_overlap=True)


def audit_manifest(frame):
    frame = resolve_subjects(frame)
    groups = {s: set(g.patient_id) for s, g in frame.groupby("split")}
    return {
        "identity_source": "COPDGene identifier in image filename; Series is not patient identity",
        "unique_subjects": int(frame.patient_id.nunique()),
        "splits": {
            s: {"images": len(g), "subjects": int(g.patient_id.nunique())}
            for s, g in frame.groupby("split")
        },
        "overlaps": {
            f"{a}_{b}": sorted(groups[a] & groups[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "subjects_in_multiple_splits": int(frame.groupby("patient_id").split.nunique().gt(1).sum()),
        "safe_for_unseen_subject_evaluation": all(
            not groups[a] & groups[b]
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
    }


def scan_dataset(root, hash_images=False):
    root, rows = Path(root).resolve(), []
    for split in ("train", "val", "test"):
        directory = root / "images" / split
        if not directory.is_dir():
            raise ValueError(f"Missing image directory: {directory}")
        images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        for image in images:
            relative = image.relative_to(directory)
            label = root / "labels" / split / relative.with_suffix(".txt")
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            read_label(label, width, height)
            row = {
                "image": image.relative_to(root).as_posix(),
                "label": label.relative_to(root).as_posix(),
                "split": split,
                "patient_id": patient_id(image.name),
                "width": width,
                "height": height,
                "label_sha256": sha256(label),
            }
            if hash_images:
                row["image_sha256"] = sha256(image)
            rows.append(row)
    return validate_manifest(pd.DataFrame(rows))


def export_grouped_dataset(manifest, source, destination, seed=42, train_fraction=0.6, val_fraction=0.2):
    """Explicitly creates a NEW split; never mutates the historical dataset."""
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions must leave three positive partitions")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Destination already exists; choose a new directory")
    if destination.is_relative_to(source):
        raise ValueError("Destination must be outside the original dataset")
    # Only this explicit repair path accepts overlap in the SOURCE inventory.
    frame = resolve_subjects(manifest)
    verify_manifest_files(source, frame)
    groups = np.array(sorted(frame.patient_id.unique()))
    np.random.default_rng(seed).shuffle(groups)
    nt, nv = round(len(groups) * train_fraction), round(len(groups) * val_fraction)
    if min(nt, nv, len(groups) - nt - nv) < 1:
        raise ValueError("Too few patients for requested split")
    mapping = {g: ("train" if i < nt else "val" if i < nt + nv else "test") for i, g in enumerate(groups)}
    targets = set()
    for row in frame.itertuples():
        key = (mapping[row.patient_id], Path(row.image).stem)
        if key in targets:
            raise ValueError("Image basename collision; preserve historical dataset or rename explicitly")
        targets.add(key)
    rows = []
    for row in frame.itertuples():
        split = mapping[row.patient_id]
        outputs = {}
        for kind, original in (("images", row.image), ("labels", row.label)):
            target = destination / kind / split / Path(original).name
            outputs["image" if kind == "images" else "label"] = target.relative_to(destination).as_posix()
        rows.append(
            {
                **{c: getattr(row, c) for c in ("image_sha256", "label_sha256") if c in frame},
                **outputs,
                "split": split,
                "patient_id": row.patient_id,
                "width": row.width,
                "height": row.height,
                "source_image": row.image,
                "source_label": row.label,
                "source_split": row.split,
            }
        )
    result = validate_manifest(pd.DataFrame(rows))
    # Validate all assignments and byte overlap BEFORE writing any destination files.
    for number, row in enumerate(result.itertuples(), 1):
        for original, relative in ((row.source_image, row.image), (row.source_label, row.label)):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / original, target)
            hash_column = "image_sha256" if relative == row.image else "label_sha256"
            if hash_column in result and sha256(target) != getattr(row, hash_column):
                raise ValueError(f"Copied content hash mismatch: {relative}")
        if number % 100 == 0 or number == len(result):
            print(f"Copied {number}/{len(result)} image/label pairs", flush=True)
    result.to_csv(destination / "manifest.csv", index=False)
    write_dataset_yaml(destination)
    from .utils import provenance, write_json

    write_json(destination / "split_provenance.json", {
        **provenance(), "source": str(source), "seed": seed,
        "train_fraction": train_fraction, "val_fraction": val_fraction,
        "subject_assignments": mapping, "source_audit": audit_manifest(frame),
        "destination_audit": audit_manifest(result),
        "manifest_sha256": sha256(destination / "manifest.csv"),
        "test_use": "Previously inspected dataset; new split is not a prospective untouched cohort",
    })
    return result


def write_dataset_yaml(root):
    root = Path(root).resolve()
    value = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "spine"},
        "kpt_shape": [11, 3],
        "flip_idx": list(range(11)),
    }
    (root / "data.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return root / "data.yaml"
