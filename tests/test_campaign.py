"""Campaign recovery is tested with processes simulated, without GPU dependencies."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from spinelocus.campaign import campaign_lock, run_campaign
from spinelocus.evaluate import roi_diagnostics
from spinelocus.metrics import summarize, summarize_patients


class EvaluationDiagnosticTests(unittest.TestCase):
    def test_patient_macro_does_not_weight_large_subject_more(self):
        frame = pd.DataFrame({"patient_id": ["A"] * 9 + ["B"],
                              "error_px": [1.] * 9 + [np.nan],
                              "component_mae_px": [1.] * 9 + [np.nan],
                              "width": [100] * 10, "height": [100] * 10})
        records, macro = summarize_patients(frame)
        self.assertEqual(summarize(frame)["pck_at_5px_all_visible"], .9)
        self.assertEqual(macro["pck_at_5px_all_visible"], .5)
        self.assertEqual(macro["patients_with_predictions"], 1)
        self.assertEqual(records[1]["failed_points"], 1)

    def test_roi_diagnostic_never_changes_prediction_crop(self):
        points = np.array([[50., 50.], [50., 100.], [np.nan, np.nan]])
        before = points.copy()
        near = roi_diagnostics(points, [2, 2, 2], points, 500, 500)
        far = roi_diagnostics(np.array([[450., 450.]] * 3), [2, 2, 2], points, 500, 500)
        self.assertEqual([(r["x1"], r["y1"]) for r in near], [(r["x1"], r["y1"]) for r in far])
        self.assertTrue(near[0]["gt_inside_crop"])
        self.assertFalse(far[0]["gt_inside_crop"])
        self.assertFalse(far[2]["crop_available"])
        np.testing.assert_equal(points, before)


class CampaignTests(unittest.TestCase):
    def setup_case(self, tmp):
        root = Path(tmp)
        data = root / "dataset"
        data.mkdir()
        manifest = data / "manifest.csv"
        pd.DataFrame([dict(image=f"{s}.jpg", label=f"{s}.txt", patient_id=str(i),
                           split=s, width=100, height=100)
                      for i, s in enumerate(("train", "val", "test"))]).to_csv(manifest, index=False)
        config = root / "config.yaml"
        config.write_text(yaml.safe_dump(dict(model_name="hrnet_w18", feature_index=1, lr=.0003)))
        for size in (640, 960):
            (root / f"yolo{size}.pt").write_bytes(str(size).encode())
        args = SimpleNamespace(data=str(data), manifest=str(manifest), config=str(config),
                               yolo640=str(root / "yolo640.pt"), yolo960=str(root / "yolo960.pt"),
                               out=str(root / "campaign"), seeds=[42, 43], epochs=2,
                               batch=64, workers=2, device="0", no_tta=False)
        return args

    def fake_runner(self, command, log):
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("simulated")
        (out / "provenance.json").write_text("{}")
        if "train-hrnet" in command:
            (out / "best.pt").write_bytes(b"best")
            (out / "last.pt").write_bytes(b"last")
            (out / "history.csv").write_text("epoch\n2\n")
            (out / "status.json").write_text('{"completed":true}')
        else:
            self.assertEqual(command[command.index("--split") + 1], "val")
            size = int(command[command.index("--imgsz") + 1])
            macro = {"pck_at_5px_all_visible": .8, "mean_euclidean_px_on_predictions": 3.}
            baseline = {"patient_macro": macro}
            value = {"patient_macro": macro, "pck_at_5px_all_visible": .8,
                     "mean_euclidean_px_on_predictions": 3., "median_px_on_predictions": 2.,
                     "p95_px_on_predictions": 8., "failed_points": 0}
            summary = {"yolo": baseline, "yolo_hrnet": value,
                       "protocol": dict(split="val", imgsz=size, tta="--tta" in command, images=1, patients=1),
                       "roi_diagnostics": {"coverage_all_visible": 1.}}
            if "--identity-guard" in command:
                summary['yolo_hrnet_guarded'] = value
                (out / 'refinement_guard.csv').write_text('rejected\nFalse\n')
            (out / "summary.json").write_text(json.dumps(summary))
            for name in ("predictions.csv", "by_patient.csv", "by_vertebra.csv", "roi_diagnostics.csv"):
                (out / name).write_text("simulated\n")

    @patch("spinelocus.campaign.verify_manifest_files")
    def test_sequence_restart_and_input_hash_guards(self, verify):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_case(tmp)
            calls = []
            def runner(command, log):
                calls.append(command)
                self.fake_runner(command, log)
            run_campaign(args, runner)
            self.assertEqual(len(calls), 10)  # two training processes, eight validations
            self.assertEqual(len(pd.read_csv(Path(args.out) / "comparison.csv")), 8)
            run_campaign(args, runner)
            self.assertEqual(len(calls), 10)
            Path(args.yolo960).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "inputs, software"):
                run_campaign(args, runner)

    @patch("spinelocus.campaign.verify_manifest_files")
    def test_interrupted_training_resumes_and_finished_work_is_retained(self, verify):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_case(tmp)
            def interrupt(command, log):
                target = Path(command[command.index("--out") + 1])
                target.mkdir(parents=True)
                (target / "last.pt").write_bytes(b"checkpoint")
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                run_campaign(args, interrupt)
            calls = []
            def runner(command, log):
                calls.append(command)
                self.fake_runner(command, log)
            run_campaign(args, runner)
            self.assertIn("--resume", calls[0])
            state = json.loads((Path(args.out) / "campaign.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertNotIn("error", state)
            best = Path(state["steps"]["hrnet-seed42"]["out"]) / "best.pt"
            best.write_bytes(b"externally edited")
            with self.assertRaisesRegex(ValueError, "output changed"):
                run_campaign(args, runner)

    @patch("spinelocus.campaign.verify_manifest_files")
    def test_partial_evaluation_retries_without_mixing_or_deleting(self, verify):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_case(tmp)
            args.seeds, args.no_tta = [42], True
            def interrupt(command, log):
                if "evaluate" in command:
                    target = Path(command[command.index("--out") + 1])
                    target.mkdir(parents=True)
                    (target / "partial.csv").write_text("keep")
                    raise RuntimeError("interrupted evaluation")
                self.fake_runner(command, log)
            with self.assertRaisesRegex(RuntimeError, "interrupted evaluation"):
                run_campaign(args, interrupt)
            partial = Path(args.out) / "evaluations/eval-seed42-yolo640-ttaoff/partial.csv"
            run_campaign(args, self.fake_runner)
            self.assertEqual(partial.read_text(), "keep")
            state = json.loads((Path(args.out) / "campaign.json").read_text())
            self.assertTrue(state["steps"]["eval-seed42-yolo640-ttaoff"]["out"].endswith("attempt2"))

    def test_concurrent_campaign_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            with campaign_lock(path):
                with self.assertRaisesRegex(RuntimeError, "Another process"):
                    with campaign_lock(path):
                        pass

    @patch("spinelocus.campaign.verify_manifest_files")
    def test_architecture3_campaign_keeps_raw_and_guarded_comparisons(self, verify):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_case(tmp)
            cfg = yaml.safe_load(Path(args.config).read_text())
            cfg['feature_index'] = 3
            Path(args.config).write_text(yaml.safe_dump(cfg))
            args.identity_guard = 'nearest-yolo'
            run_campaign(args, self.fake_runner)
            table = pd.read_csv(Path(args.out) / 'comparison.csv')
            self.assertEqual(len(table), 16)
            self.assertEqual(set(table['mode']), {'yolo_hrnet', 'yolo_hrnet_guarded'})
            generated = yaml.safe_load((Path(args.out) / 'configs/hrnet-seed42.yaml').read_text())
            self.assertEqual(generated['feature_index'], 3)
            self.assertTrue(generated['pretrained'])
            self.assertEqual(generated['batch_size'], 64)
            self.assertEqual(generated['accumulate'], 1)
