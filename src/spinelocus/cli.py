"""CLI imports deep-learning dependencies only for commands that need them."""

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path


def doctor():
    result = {}
    for package in ("torch", "torchvision", "ultralytics", "timm"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            result["gpu"] = prop.name
            result["vram_gib"] = round(prop.total_memory / 1024**3, 2)
    except ImportError:
        pass
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = "not available in this environment"
    print(json.dumps(result, indent=2))


def parser():
    p = argparse.ArgumentParser(description="YOLO-Pose + HRNet thoracic DRR research pipeline")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check installed stack, CUDA and GPU memory")
    campaign = commands.add_parser("campaign", help="Restartable HRNet seed campaign on validation only")
    for field in ("data", "manifest", "yolo640", "yolo960", "out"):
        campaign.add_argument(f"--{field}", required=True)
    campaign.add_argument("--config", default="configs/hrnet_v2.yaml")
    campaign.add_argument("--device", default="0")
    campaign.add_argument("--epochs", type=int, default=60)
    campaign.add_argument("--batch", type=int, default=64)
    campaign.add_argument("--workers", type=int, default=2)
    campaign.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    campaign.add_argument("--no-tta", action="store_true", help="Evaluate HRNet without flip TTA only")
    campaign.add_argument("--identity-guard", choices=["none", "nearest-yolo"], default="none")
    refinement = commands.add_parser("audit-refinement", help="Diagnose/replay saved validation refinements without GPU")
    refinement.add_argument("--evaluations", nargs="+", required=True, help="Evaluation directories with predictions.csv")
    refinement.add_argument("--out", required=True)
    refinement.add_argument("--data", help="Optional local dataset for validation-label and crop review")
    refinement.add_argument("--manifest", help="Required with --data; only CSV validation labels are read")
    refinement.add_argument("--overlays", type=int, default=6, help="Number of review PNGs when --data is given")
    ensemble = commands.add_parser("ensemble", help="Replay an equal HRNet coordinate ensemble on saved validation CSVs")
    ensemble.add_argument("--evaluations", nargs="+", required=True, help="At least two single-model evaluation directories")
    ensemble.add_argument("--out", required=True)
    compare = commands.add_parser("compare-yolo-flip", help="Validate a paired original/flip YOLO experiment")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out", required=True)
    scan = commands.add_parser("manifest", help="Validate existing image/label split and export manifest")
    scan.add_argument("--data", required=True)
    scan.add_argument("--out", required=True)
    scan.add_argument("--hash-images", action="store_true")
    identity = commands.add_parser("audit-manifest", help="Inspect COPDGene subject overlap, without images")
    identity.add_argument("--manifest", required=True)
    identity.add_argument("--out", required=True)
    export = commands.add_parser("split", help="Create a NEW patient-grouped dataset")
    export.add_argument("--data", required=True)
    export.add_argument("--manifest", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--train-fraction", type=float, default=0.6)
    export.add_argument("--val-fraction", type=float, default=0.2)
    audit = commands.add_parser(
        "audit-legacy", help="Verify recovered indices and predictions, no images needed"
    )
    audit.add_argument("--input", required=True)
    audit.add_argument("--out", required=True)
    plot = commands.add_parser("plot", help="Export PNG/SVG figures from evaluation predictions.csv")
    plot.add_argument("--predictions", required=True)
    plot.add_argument("--out", required=True)
    for name in ("train-yolo", "train-hrnet"):
        t = commands.add_parser(name)
        for field in ("data", "manifest", "config", "out"):
            t.add_argument(f"--{field}", required=True)
        t.add_argument("--device", default="auto")
        t.add_argument("--batch", type=int)
        t.add_argument("--workers", type=int)
        t.add_argument("--resume", help="last.pt from this experiment")
        if name == "train-yolo":
            t.add_argument("--weights", help="Override model/initial weights, e.g. best.pt for 960 FT")
    for name in ("evaluate", "predict"):
        e = commands.add_parser(name)
        e.add_argument("--yolo", required=True)
        e.add_argument("--hrnet")
        e.add_argument("--hrnet-extra", nargs="+", default=[],
                       help="Additional fixed HRNet checkpoints, averaged equally in original pixels")
        e.add_argument("--device", default="auto")
        e.add_argument("--imgsz", type=int, default=960)
        e.add_argument("--conf", type=float, default=0.01)
        e.add_argument("--keypoint-conf", type=float, default=0.0)
        e.add_argument("--alpha", type=float, default=2.2)
        e.add_argument("--min-roi", type=int, default=112)
        e.add_argument("--max-roi", type=int, default=256)
        e.add_argument("--tta", action="store_true")
        e.add_argument("--yolo-flip-tta", action="store_true",
                       help="Experimental original/flip YOLO coordinate mean before HRNet crops")
        e.add_argument("--identity-guard", choices=["none", "nearest-yolo"], default="none",
                       help="Also retain guarded HRNet predictions; raw predictions remain available")
        e.add_argument("--out", required=True)
        if name == "evaluate":
            e.add_argument("--data", required=True)
            e.add_argument("--manifest", required=True)
            e.add_argument("--split", choices=["val", "test"], required=True)
            e.add_argument(
                "--final-test", action="store_true", help="Confirm model/ROI choices were frozen on val"
            )
        else:
            e.add_argument("--image", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        doctor()
    elif args.command == "campaign":
        from .campaign import run_campaign

        run_campaign(args)
    elif args.command == "audit-refinement":
        from .refinement_audit import audit_refinement

        audit_refinement(args.evaluations, args.out, data=args.data, manifest=args.manifest,
                         overlays=args.overlays)
    elif args.command == "ensemble":
        from .ensemble import ensemble_evaluations

        ensemble_evaluations(args.evaluations, args.out)
    elif args.command == "compare-yolo-flip":
        from .yolo_flip_comparison import compare_yolo_flip

        compare_yolo_flip(args.baseline, args.candidate, args.out)
    elif args.command == "manifest":
        from .data import scan_dataset
        from .utils import write_json

        frame = scan_dataset(args.data, args.hash_images)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        summary = frame.groupby("split").agg(images=("image", "size"), patients=("patient_id", "nunique"))
        write_json(Path(args.out).with_suffix(".summary.json"), summary.to_dict(orient="index"))
        print(summary.to_string())
    elif args.command == "split":
        import pandas as pd

        from .data import export_grouped_dataset

        frame = export_grouped_dataset(
            pd.read_csv(args.manifest, dtype={"patient_id": str}),
            args.data,
            args.out,
            args.seed,
            args.train_fraction,
            args.val_fraction,
        )
        print(frame.groupby("split").agg(images=("image", "size"), subjects=("patient_id", "nunique")))
    elif args.command == "audit-manifest":
        import pandas as pd

        from .data import audit_manifest
        from .utils import write_json

        audit = audit_manifest(pd.read_csv(args.manifest, dtype={"patient_id": str}))
        write_json(args.out, audit)
        print(json.dumps(audit, indent=2))
    elif args.command == "audit-legacy":
        from .legacy import recover

        print(json.dumps(recover(args.input, args.out), indent=2))
    elif args.command == "train-yolo":
        from .train import train_yolo

        train_yolo(args)
    elif args.command == "plot":
        from .plots import plot_evaluation

        plot_evaluation(args.predictions, args.out)
    elif args.command == "train-hrnet":
        from .train import train_hrnet

        train_hrnet(args)
    elif args.command == "evaluate":
        if args.hrnet_extra and not args.hrnet:
            raise SystemExit("--hrnet-extra requires --hrnet")
        if args.identity_guard != "none" and not args.hrnet:
            raise SystemExit("--identity-guard requires --hrnet")
        if args.split == "test" and not args.final_test:
            raise SystemExit("Use validation for model selection. Add --final-test once protocol is frozen.")
        from .evaluate import evaluate

        evaluate(args)
    elif args.command == "predict":
        if args.hrnet_extra and not args.hrnet:
            raise SystemExit("--hrnet-extra requires --hrnet")
        if args.identity_guard != "none" and not args.hrnet:
            raise SystemExit("--identity-guard requires --hrnet")
        from .evaluate import predict

        predict(args)
