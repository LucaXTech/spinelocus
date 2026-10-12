"""A resumed synthetic CPU run must match uninterrupted training."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml


@unittest.skipUnless(importlib.util.find_spec("torch"), "Training extras required")
class ResumeTests(unittest.TestCase):
    def test_interrupted_epoch_resume_matches_uninterrupted(self):
        import torch

        from scripts.smoke import synthetic_dataset
        from spinelocus.train import atomic_checkpoint, train_hrnet

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            synthetic_dataset(root / "data").to_csv(manifest, index=False)
            cfg = {
                "model_name": "smoke_cnn",
                "pretrained": False,
                "crop_size": 32,
                "heatmap_size": 16,
                "sigma": 1.5,
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
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            args = SimpleNamespace(
                config=str(config),
                manifest=str(manifest),
                data=str(root / "data"),
                device="cpu",
                batch=None,
                workers=0,
                out=str(root / "continuous"),
                resume=None,
            )
            train_hrnet(args)
            args.out = str(root / "interrupted")

            def interrupt_after_checkpoint(state, path):
                atomic_checkpoint(state, path)
                if Path(path).name == "last.pt" and state["epoch"] == 1:
                    raise InterruptedError("Simulated interruption after saved epoch")

            with patch("spinelocus.train.atomic_checkpoint", interrupt_after_checkpoint):
                with self.assertRaises(InterruptedError):
                    train_hrnet(args)
            args.resume = str(root / "interrupted/last.pt")
            train_hrnet(args)
            expected = torch.load(root / "continuous/last.pt", weights_only=False)
            resumed = torch.load(args.resume, weights_only=False)
            self.assertEqual(resumed["epoch"], 2)
            for name, tensor in expected["model"].items():
                self.assertTrue(torch.equal(tensor, resumed["model"][name]), name)
            self.assertEqual(expected["best"], resumed["best"])
            status = json.loads((root / "interrupted/status.json").read_text())
            self.assertTrue(status["completed"])
            self.assertEqual(status["stop_reason"], "epochs_completed")
            history = (root / "interrupted/history.csv").read_bytes()
            train_hrnet(args)
            self.assertEqual((root / "interrupted/history.csv").read_bytes(), history)
