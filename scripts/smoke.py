"""Offline synthetic pipeline test; temporary images are NOT research data."""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from spinelocus.data import scan_dataset, write_dataset_yaml


def synthetic_dataset(root):
    root = Path(root)
    for number, split in enumerate(("train", "val", "test"), start=1):
        for index in range(2):
            name = f"Series_{number:03d}-123456-COPDGene_SYN{number:03d}_synthetic_{index}"
            image = root / "images" / split / f"{name}.png"
            label = root / "labels" / split / f"{name}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.zeros((96, 96, 3), np.uint8)
            pixels[10:86, 38:58] = 150
            Image.fromarray(pixels).save(image)
            points = [(0.5, float(y), 2) for y in np.linspace(0.15, 0.85, 11)]
            tokens = [0, 0.5, 0.5, 1, 1] + [v for point in points for v in point]
            label.write_text(" ".join(str(v) for v in tokens) + "\n")
    write_dataset_yaml(root)
    return scan_dataset(root)


def run(root, yolo=False):
    import torch

    from spinelocus.model import load_model
    from spinelocus.train import train_hrnet

    torch.set_num_threads(2)
    root = Path(root)
    frame = synthetic_dataset(root / "data")
    manifest = root / "manifest.csv"
    frame.to_csv(manifest, index=False)
    cfg = {
        "model_name": "smoke_cnn",
        "feature_index": 0,
        "pretrained": False,
        "crop_size": 32,
        "heatmap_size": 16,
        "sigma": 1.5,
        "coordinate_convention": "pixel_centers",
        "temperature_ce": 0.5,
        "temperature_decode": 0.35,
        "lambda_l1": 0.01,
        "seed": 42,
        "epochs": 2,
        "batch_size": 8,
        "accumulate": 2,
        "workers": 0,
        "lr": 0.0003,
        "translate_fraction": 0.1,
        "scale_range": [1.0, 1.0],
        "validation_jitter": 0.1,
        "horizontal_flip": 0.5,
    }
    config = root / "smoke.yaml"
    config.write_text(yaml.safe_dump(cfg))
    args = SimpleNamespace(
        config=str(config),
        manifest=str(manifest),
        data=str(root / "data"),
        device="cpu",
        batch=None,
        workers=0,
        out=str(root / "hrnet"),
        resume=None,
    )
    train_hrnet(args)
    model, loaded = load_model(root / "hrnet" / "best.pt")
    assert loaded["model_name"] == "smoke_cnn"
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 1, 16, 16)
    # Resume a completed run should be a no-op, preserving the valid checkpoint.
    args.resume = str(root / "hrnet" / "last.pt")
    train_hrnet(args)
    if yolo:
        from unittest.mock import patch

        from matplotlib import font_manager

        from spinelocus.evaluate import evaluate
        from spinelocus.train import train_yolo

        yc = {
            "model": "yolo11n-pose.yaml",
            "task": "pose",
            "epochs": 1,
            "imgsz": 64,
            "batch": 2,
            "workers": 0,
            "pretrained": False,
            "plots": False,
            "amp": False,
            "mosaic": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "fliplr": 0.0,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "cos_lr": True,
            "seed": 42,
        }
        config.write_text(yaml.safe_dump(yc))
        ya = SimpleNamespace(
            config=str(config),
            manifest=str(manifest),
            data=str(root / "data"),
            device="cpu",
            batch=2,
            workers=0,
            out=str(root / "yolo"),
            resume=None,
            weights=None,
        )
        # Synthetic checks do not need Ultralytics' optional Arial asset download.
        with patch("ultralytics.data.utils.check_font", return_value=font_manager.findfont("DejaVu Sans")):
            train_yolo(ya)
        ea = SimpleNamespace(
            data=str(root / "data"),
            manifest=str(manifest),
            out=str(root / "evaluation"),
            split="val",
            device="cpu",
            yolo=str(root / "yolo/weights/best.pt"),
            hrnet=str(root / "hrnet/best.pt"),
            imgsz=64,
            conf=0.00001,
            keypoint_conf=0.0,
            alpha=2.2,
            min_roi=32,
            max_roi=64,
            tta=False,
        )
        predictions, summary = evaluate(ea)
        assert len(predictions) == 44
        assert summary["yolo"]["visible_points"] == 22
    print(json.dumps({"smoke_passed": True, "synthetic_only": True, "real_training": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo", action="store_true", help="Also train YOLO from YAML for 1 synthetic CPU epoch"
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.out:
        run(args.out, args.yolo)
    else:
        with tempfile.TemporaryDirectory() as directory:
            run(directory, args.yolo)
