# SpineLocus

**Research software only; not validated for clinical use.**

Research code for locating T1–T11 thoracic landmarks in digitally reconstructed radiographs (DRRs) with YOLO-Pose and optional HRNet refinement. It includes dataset checks, subject-level splitting, training, full-image evaluation, coordinate ensembles, and diagnostic audits. Windows and Linux with NVIDIA GPUs are supported; CPU-only tests cover the data and evaluation logic.

## Research status

The recovered historical dataset split reused COPDGene subjects across train, validation, and test. Those historical results cannot establish performance on unseen subjects. The current workflow creates a new subject-disjoint split and retrains models from generic pretrained weights. Dataset access and permissions remain with the dataset holder.

On a repeatedly inspected validation set of **750 images from 15 subjects**, the fixed YOLO960 + two HRNet feature1 models with HRNet flip TTA and an identity guard gave **82.31% mean subject PCK@5**, **6.424 px mean error**, and **36.452 px P95** across the 11 landmarks. An exploratory two-view YOLO flip variant gave 82.86%, 6.126 px, and 34.201 px respectively. Its paired subject bootstrap interval for the PCK@5 gain was **−0.29 to +1.23 percentage points**, and the gain in mean error was **−0.045 to +0.553 px**. The intervals include zero. The no-flip method remains the primary validation baseline; no outcome from a final test is claimed here. Neither model weights nor the image dataset are distributed.

## Pipeline overview

![SpineLocus two-stage pipeline](docs/figures/pipeline_overview.png)

*AI-generated conceptual illustration of the YOLO-Pose detection and HRNet refinement pipeline. The depicted anatomy, coordinates and heatmaps are illustrative, not actual model predictions or validation evidence. Optional ensembles, test-time augmentation and identity safeguards are omitted for clarity.*

## Install and check

Use a separate Python 3.12 environment. On Windows with a compatible NVIDIA driver:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux, install the compatible PyTorch/torchvision pair for your platform, then:

```bash
python -m pip install -e '.[train,dev]'
python -m unittest discover -s tests -v
python -m ruff check src tests scripts
```

`python -m spinelocus --help` lists commands. A dataset must be supplied locally; `data/`, `runs/`, `reports/`, weights, and private evidence are ignored by Git.

## Working with a local dataset

Create a manifest from images and labels, then make a new subject-grouped dataset if the original split overlaps:

```bash
python -m spinelocus manifest --data data/original --out data/original-manifest.csv --hash-images
python -m spinelocus split --data data/original --manifest data/original-manifest.csv --out data/subjectsplit --seed 42
```

Check the produced split and train **all** models on that split. The config examples in `configs/` separate reference settings from exploratory settings. Historical dataset checkpoints must not initialize a subject-disjoint experiment.

For full-image validation with local checkpoints:

```bash
python -m spinelocus evaluate --data data/subjectsplit --manifest data/subjectsplit/manifest.csv --split val --yolo runs/yolo960/weights/best.pt --hrnet runs/hrnet-seed42/best.pt --hrnet-extra runs/hrnet-seed43/best.pt --imgsz 960 --tta --identity-guard nearest-yolo --out runs/validation-baseline --device 0
```

The exploratory variant adds `--yolo-flip-tta` and writes to a **new** output directory. It averages the original and mirrored YOLO coordinates before forming HRNet crops. `compare-yolo-flip` validates paired results from saved validation runs. Evaluate the final test only after the analysis protocol has been fixed; the CLI requires `--final-test` for that step.

Results depend on the dataset, source manifest, checkpoint hashes, and configured protocol. The reported validation population was used during method development and cannot provide an independent generalization estimate.

## Distribution and attribution

This public candidate contains implementation, configurations, and synthetic tests. It omits DRRs, labels, prediction rows, individual subject identifiers, research reports, model weights, supplied source notebooks. Original research and source materials were supplied by project collaborators; confirm the appropriate author credits and permission to release the reconstructed code before publishing. See [LICENSE](LICENSE) (AGPL-3.0), [third-party notices](THIRD_PARTY_NOTICES.md), and [software citation metadata](CITATION.cff). Scientific-study attribution is distinct from software authorship; add collaborators to citation metadata if they also contributed to the released software. Dependency and dataset terms are separate.
