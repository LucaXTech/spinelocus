"""Sequential, restartable validation campaign. Never evaluates the test split."""

import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .data import load_manifest, verify_manifest_files
from .utils import load_config, provenance, sha256, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextlib.contextmanager
def keep_awake():
    """Prevent Windows idle sleep for this thread; do not override lid/manual sleep."""
    previous, set_state = None, None
    if os.name == "nt":
        import ctypes

        set_state = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
        set_state.argtypes, set_state.restype = [ctypes.c_uint], ctypes.c_uint
        previous = set_state(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if not previous:
            raise OSError("Windows idle sleep prevention failed; check power settings")
    try:
        yield
    finally:
        if set_state is not None:
            set_state(previous | 0x80000000)


@contextlib.contextmanager
def campaign_lock(path):
    """OS lock is released even after a crash; blocks concurrent GPU campaigns."""
    with Path(path).open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another process is already running this campaign") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def execute(command, log):
    """Stream native process output to console and a per-stage file."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as output:
        output.write(("\n" + now() + " " + subprocess.list2cmdline(command) + "\n").encode())
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env) as proc:
            try:
                while chunk := proc.stdout.read1(4096):
                    output.write(chunk)
                    output.flush()
                    if hasattr(sys.stdout, "buffer"):
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    else:
                        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
                code = proc.wait()
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def ensure_config(path, value):
    if path.exists():
        if load_config(path) != value:
            raise ValueError(f"Campaign configuration changed: {path}; choose a new campaign output")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def run_stage(state, state_file, stage_id, target, arguments, outputs, runner, *, training=False):
    """Skip verified successes; resume training or retain an incomplete attempt."""
    signature = fingerprint({"arguments": arguments, "training": training})
    previous = state["steps"].get(stage_id, {})
    if previous and previous["signature"] != signature:
        raise ValueError(f"Changed stage inputs: {stage_id}")
    if previous.get("status") == "completed":
        for path, digest in previous["outputs_sha256"].items():
            if not Path(path).is_file() or sha256(path) != digest:
                raise ValueError(f"Completed campaign output changed or disappeared: {path}")
        print(f"Already completed: {stage_id}", flush=True)
        return Path(previous["out"])
    target = Path(previous.get("out", target))
    resume = training and (target / "last.pt").is_file()
    if target.exists() and any(target.iterdir()) and not resume:
        # Never delete or mix an incomplete attempt, including interrupted evaluation.
        base = target.parent / stage_id
        attempt = 2
        while (target := base.with_name(f"{base.name}-attempt{attempt}")).exists():
            attempt += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "spinelocus", *arguments, "--out", str(target)]
    if resume:
        command += ["--resume", str(target / "last.pt")]
    log = state_file.parent / "logs" / f"{stage_id}.log"
    entry = {"signature": signature, "status": "running", "out": str(target),
             "command": command, "started_utc": now(), "log": str(log)}
    state["steps"][stage_id] = entry
    state.update(status="running", current_stage=stage_id, updated_utc=now())
    write_json(state_file, state)
    print(f"\nStarting {stage_id}\nOutput: {target}", flush=True)
    runner(command, log)
    files = [target / name for name in outputs]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Stage returned without required outputs: {missing}")
    if training and not json.loads((target / "status.json").read_text())["completed"]:
        raise RuntimeError("HRNet training did not report completion")
    entry.update(status="completed", finished_utc=now(),
                 outputs_sha256={str(path): sha256(path) for path in files})
    write_json(state_file, state)
    return target


def build_comparison(state, out):
    records, baselines = [], {}
    for stage, item in state["steps"].items():
        if not stage.startswith("eval-") or item["status"] != "completed":
            continue
        summary = json.loads((Path(item["out"]) / "summary.json").read_text())
        protocol = summary["protocol"]
        # The campaign has no test option, and refuses reports containing test results.
        if protocol["split"] != "val":
            raise ValueError("Campaign comparisons must use validation only")
        yolo = f"yolo{protocol['imgsz']}"
        baseline = summary["yolo"]
        if yolo in baselines and not same_metrics(baselines[yolo], baseline):
            raise ValueError("YOLO baseline changed between paired evaluations")
        baselines[yolo] = baseline
        for mode in ("yolo_hrnet", "yolo_hrnet_guarded"):
            if mode not in summary:
                continue
            value = summary[mode]
            records.append({
                "experiment": stage + ("-guarded" if mode.endswith("guarded") else ""),
                "mode": mode, "seed": item["seed"], "yolo": yolo,
                "tta": protocol["tta"], "images": protocol["images"], "patients": protocol["patients"],
                "macro_patient_pck5": value["patient_macro"]["pck_at_5px_all_visible"],
                "pck5_all_visible": value["pck_at_5px_all_visible"],
                "mean_px_on_predictions": value["mean_euclidean_px_on_predictions"],
                "macro_patient_mean_px_on_predictions": value["patient_macro"]["mean_euclidean_px_on_predictions"],
                "median_px": value["median_px_on_predictions"],
                "p95_px": value["p95_px_on_predictions"], "failed_points": value["failed_points"],
                "roi_gt_coverage": summary["roi_diagnostics"]["coverage_all_visible"],
                "out": item["out"],
            })
    if not records:
        return
    # Primary failure-sensitive outcome was fixed before running the HRNet campaign.
    records.sort(key=lambda r: (-r["macro_patient_pck5"], r["experiment"]))
    pd.DataFrame(records).to_csv(out / "comparison.csv", index=False)
    write_json(out / "comparison.json", {
        "selection_metric": "PCK@5 original-image pixels, averaged equally over validation subjects",
        "baselines": baselines, "pipelines": records,
        "test_evaluated": False,
        "interpretation": "Validation exploration, two initializations; not independent confirmatory evidence.",
    })
    lines = ["# Campagna HRNet — validation", "",
             "Ordine: PCK@5 px medio fra soggetti, includendo le predizioni mancanti. "
             "Media, mediana e P95 usano pixel originali e solo predizioni disponibili.", "",
             "| Esperimento | PCK5 soggetti | PCK5 punti | Media px | P95 px | Mancanti | GT nei crop |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return "n/d" if value is None else f"{value:.3f}"
    for r in records:
        lines.append(f"| {r['experiment']} | {r['macro_patient_pck5']:.2%} | {r['pck5_all_visible']:.2%} "
                     f"| {number(r['mean_px_on_predictions'])} | {number(r['p95_px'])} "
                     f"| {r['failed_points']} | {r['roi_gt_coverage']:.2%} |")
    lines += ["", "Confrontare i due seed per la stessa pipeline; non interpretarli come due coorti. "
              "TTA applica il flip orizzontale solo a HRNet. ROI e soglie restano fissi. "
              "La copertura GT nei crop e' un controllo dopo l'inferenza: non guida i crop.", "",
              "Il test non viene valutato. Congelare il protocollo prima del test. "
              "Il test raggruppato deriva comunque da dati gia' ispezionati, non da una coorte prospettica.", "",
              "Le metriche durante il training sono sui crop perturbati da GT e non sono direttamente "
              "confrontabili con queste metriche della pipeline completa.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def same_metrics(left, right):
    """Allow roundoff, while requiring equal counts and missing-data states."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_metrics(left[k], right[k]) for k in left)
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-6)
    return left == right


def run_campaign(args, runner=execute):
    root, manifest = Path(args.data).resolve(), Path(args.manifest).resolve()
    out, template = Path(args.out).resolve(), Path(args.config).resolve()
    weights = {640: Path(args.yolo640).resolve(), 960: Path(args.yolo960).resolve()}
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Choose distinct seeds")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        raise ValueError("Epochs/batch must be positive and workers nonnegative")
    for path in (manifest, template, *weights.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out == root or root in out.parents:
        raise ValueError("Campaign output must be outside the dataset")
    cfg = load_config(template)
    if cfg.get("model_name") != "hrnet_w18" or cfg.get("feature_index") not in (1, 3):
        raise ValueError("Campaign requires hrnet_w18 with feature_index 1 or 3")
    cfg.update(epochs=args.epochs, batch_size=args.batch, workers=args.workers,
               accumulate=1, pretrained=True)
    code_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    protocol = {
        "data": str(root), "manifest": str(manifest), "yolo_weights": {str(k): str(v) for k, v in weights.items()},
        "hrnet_config": cfg, "seeds": args.seeds, "device": args.device, "split": "val",
        "tta_modes": [False] if args.no_tta else [False, True],
        "identity_guard": getattr(args, "identity_guard", "none"),
        "alpha": 2.2, "min_roi": 112, "max_roi": 256, "conf": 0.01, "keypoint_conf": 0.0,
        "selection_metric": "macro_patient_pck_at_5px_all_visible",
        "provenance": provenance(files=[manifest, template, *weights.values()]),
        "source_sha256": code_hashes,
    }
    signature = fingerprint(protocol)
    state_file = out / "campaign.json"
    if out.exists() and any(out.iterdir()) and not state_file.exists():
        raise ValueError("Choose an empty campaign directory, or one containing campaign.json")
    out.mkdir(parents=True, exist_ok=True)
    with campaign_lock(out / "campaign.lock"), keep_awake():
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state["signature"] != signature:
                raise ValueError("Campaign inputs, software or configuration changed; choose a new output")
        else:
            state = {"signature": signature, "protocol": protocol, "steps": {}, "created_utc": now()}
            write_json(state_file, state)
        try:
            frame = load_manifest(manifest)
            verify_manifest_files(root, frame)
            common = ["--data", str(root), "--manifest", str(manifest), "--device", args.device]
            for seed in args.seeds:
                config_path = out / "configs" / f"hrnet-seed{seed}.yaml"
                ensure_config(config_path, {**cfg, "seed": seed})
                training = run_stage(
                    state, state_file, f"hrnet-seed{seed}", out / "train" / f"hrnet-seed{seed}",
                    ["train-hrnet", *common, "--config", str(config_path),
                     "--batch", str(args.batch), "--workers", str(args.workers)],
                    ["best.pt", "last.pt", "history.csv", "status.json", "provenance.json"],
                    runner, training=True,
                )
                for size, weight in weights.items():
                    for tta in protocol["tta_modes"]:
                        stage = f"eval-seed{seed}-yolo{size}-tta{'on' if tta else 'off'}"
                        arguments = ["evaluate", *common, "--split", "val", "--imgsz", str(size),
                                     "--yolo", str(weight), "--hrnet", str(training / "best.pt"),
                                     "--alpha", "2.2", "--min-roi", "112", "--max-roi", "256",
                                     "--conf", "0.01", "--keypoint-conf", "0.0"]
                        if tta:
                            arguments.append("--tta")
                        outputs = ["summary.json", "predictions.csv", "by_patient.csv",
                                   "by_vertebra.csv", "roi_diagnostics.csv", "provenance.json"]
                        if protocol["identity_guard"] != "none":
                            arguments += ["--identity-guard", protocol["identity_guard"]]
                            outputs.append("refinement_guard.csv")
                        run_stage(state, state_file, stage, out / "evaluations" / stage,
                                  arguments, outputs, runner)
                        state["steps"][stage]["seed"] = seed
                        write_json(state_file, state)
                        build_comparison(state, out)
            state.update(status="completed", current_stage=None, updated_utc=now())
            state.pop("error", None)
            write_json(state_file, state)
            print(f"\nCampaign completed. Comparison: {out / 'comparison.csv'}", flush=True)
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=f"{type(error).__name__}: {error}", updated_utc=now())
            write_json(state_file, state)
            raise
