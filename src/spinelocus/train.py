"""Checkpointed HRNet and portable Ultralytics training entry points."""

import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from .data import load_manifest, verify_manifest_files, write_dataset_yaml
from .utils import load_config, provenance, write_json


def resolve_device(requested="auto"):
    import torch

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(requested).isdigit():
        requested = f"cuda:{requested}"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install a CUDA PyTorch wheel or choose --device cpu")
    return device


def atomic_checkpoint(value, path):
    import torch

    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_hrnet(args):
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from .crops import LandmarkCrops
    from .model import HRNetHeatmap, decode, loss

    cfg = load_config(args.config)
    manifest = load_manifest(args.manifest)
    verify_manifest_files(args.data, manifest)
    device = resolve_device(args.device)
    cfg["batch_size"] = args.batch or cfg.get("batch_size", 16)
    cfg["workers"] = args.workers if args.workers is not None else cfg.get("workers", 0)
    out = Path(args.out)
    if args.resume and Path(args.resume).resolve().parent != out.resolve():
        raise ValueError("Resume into the original run directory so history and best.pt stay together")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Run directory is not empty; resume explicitly or choose a new run")
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.get("seed", 42))
    torch.use_deterministic_algorithms(cfg.get("deterministic", True), warn_only=True)
    torch.backends.cudnn.benchmark = not cfg.get("deterministic", True)
    dataset_hash = provenance(files=[args.manifest])["files_sha256"][str(args.manifest)]
    model = HRNetHeatmap(cfg, pretrained=cfg.get("pretrained", True) and not args.resume).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg["lr"] * 0.3},
            {"params": model.head.parameters(), "lr": cfg["lr"]},
        ],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, stale, best_epoch = 0, math.inf, 0, None
    generator = torch.Generator().manual_seed(cfg.get("seed", 42))
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["manifest_sha256"] != dataset_hash or state["config"] != cfg:
            raise ValueError(
                "Resume requires the same manifest/config; use a new run for a changed experiment"
            )
        if state.get("device_type") != device.type:
            raise ValueError("Resume requires the same device type; use saved model weights for a new run")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale = state["epoch"], state["best"], state["stale"]
        best_epoch = state.get("best_epoch")
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        generator.set_state(state["rng_loader"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    train_set = LandmarkCrops(manifest, args.data, "train", cfg)
    val_set = LandmarkCrops(manifest, args.data, "val", cfg)
    options = dict(batch_size=cfg["batch_size"], num_workers=cfg["workers"], pin_memory=amp)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_set, shuffle=False, **options)
    write_json(out / "provenance.json", provenance(cfg, [args.manifest, args.config]))
    completed_epoch = start
    # A finished/early-stopped checkpoint is already complete; do not train an extra epoch.
    end = start if stale >= cfg.get("patience", 15) else cfg["epochs"]
    for epoch in range(start, end):
        train_set.epoch = epoch
        model.train()
        total, count = 0.0, 0
        tick = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        accumulation = max(1, int(cfg.get("accumulate", 1)))
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']} train", mininterval=2)
        for step, (images, xy) in enumerate(progress):
            images, xy = images.to(device), xy.to(device)
            with torch.amp.autocast(device.type, enabled=amp):
                value = loss(model(images), xy, cfg)
            # Normalize the final partial accumulation group correctly.
            group_size = min(accumulation, len(train_loader) - (step // accumulation) * accumulation)
            scaler.scale(value / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += value.item() * len(images)
            count += len(images)
            if step % 10 == 0:
                progress.set_postfix(loss=f"{total / count:.4f}", refresh=False)
        model.eval()
        errors = []
        with torch.inference_mode():
            for images, xy in tqdm(val_loader, desc=f"Epoch {epoch + 1} val", mininterval=2):
                prediction = decode(
                    model(images.to(device)),
                    cfg["crop_size"],
                    cfg.get("temperature_decode", 0.35),
                    cfg.get("coordinate_convention", "pixel_centers"),
                )
                errors.extend(torch.linalg.vector_norm(prediction - xy.to(device), dim=1).cpu().tolist())
        mean = float(np.mean(errors))
        if not np.isfinite(mean):
            raise RuntimeError("Non-finite validation metric")
        improved = mean < best
        best, stale = (mean, 0) if improved else (best, stale + 1)
        if improved:
            best_epoch = epoch + 1
        row = {
            "epoch": epoch + 1,
            "train_loss": total / count,
            "val_mean_euclidean_crop_px": mean,
            "val_pck5": float(np.mean(np.array(errors) <= 5)),
            "lr": opt.param_groups[-1]["lr"],
            "seconds": time.perf_counter() - tick,
        }
        print(row, flush=True)
        csv_path = out / "history.csv"
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        scheduler.step(mean)
        state = {
            "model": model.state_dict(),
            "config": cfg,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "best": best,
            "best_epoch": best_epoch,
            "stale": stale,
            "manifest_sha256": dataset_hash,
            "device_type": device.type,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_loader": generator.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        if improved:
            atomic_checkpoint(state, out / "best.pt")
        atomic_checkpoint(state, out / "last.pt")
        write_json(
            out / "status.json",
            {
                "epoch": epoch + 1,
                "best_val_mean_euclidean_crop_px": best,
                "train_points": len(train_set),
                "val_points": len(val_set),
                "completed": False,
            },
        )
        completed_epoch = epoch + 1
        if stale >= cfg.get("patience", 15):
            break
    write_json(out / "status.json", {
        "epoch": completed_epoch, "best_epoch": best_epoch,
        "best_val_mean_euclidean_crop_px": best,
        "train_points": len(train_set), "val_points": len(val_set), "completed": True,
        "stop_reason": "early_stopping" if stale >= cfg.get("patience", 15) else "epochs_completed",
    })


def train_yolo(args):
    from ultralytics import YOLO, settings

    # File-based experiment tracking; no external tracker accounts.
    settings.update({"wandb": False, "comet": False, "clearml": False, "mlflow": False})
    cfg = load_config(args.config)
    frame = load_manifest(args.manifest)
    verify_manifest_files(args.data, frame)
    from .data import scan_dataset

    actual = scan_dataset(args.data)
    expected = frame.set_index("image")["split"].to_dict()
    if actual.set_index("image")["split"].to_dict() != expected:
        raise ValueError("Manifest image/split assignments do not match dataset folders")
    # Regenerate portable paths; image and label content stays unchanged.
    data_yaml = write_dataset_yaml(args.data)
    cfg["data"] = str(data_yaml)
    cfg["device"] = str(resolve_device(args.device))
    if cfg["device"].startswith("cuda:"):
        cfg["device"] = cfg["device"].split(":")[-1]
    cfg["batch"] = args.batch or cfg.get("batch", 4)
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.weights:
        cfg["model"] = args.weights
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Choose an empty run directory or resume explicitly")
    out.mkdir(parents=True, exist_ok=True)
    cfg.update(project=str(out.parent), name=out.name, exist_ok=True)
    current = provenance(cfg, [args.manifest, args.config])
    if args.resume:
        if Path(args.resume).resolve().parent.parent != out:
            raise ValueError("Resume into the original YOLO run directory")
        original = json.loads((out / "provenance.json").read_text())
        if original["config"] != current["config"] or original["files_sha256"] != current["files_sha256"]:
            raise ValueError("YOLO resume requires the original config and manifest")
        model = YOLO(args.resume)
        # Resume state/epochs come from checkpoint, not current YAML.
        model.train(resume=True, data=str(data_yaml), device=cfg["device"], workers=cfg.get("workers", 0))
    else:
        write_json(out / "provenance.json", current)
        model = YOLO(cfg.pop("model"))
        model.train(**cfg)
