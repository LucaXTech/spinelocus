"""Coordinate pairing, failure handling and inference/replay agreement for ensembles."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from spinelocus.ensemble import PROTOCOL_FIELDS, ensemble_evaluations, mean_coordinates
from spinelocus.evaluate import evaluate, load_refiners, predict, refine_ensemble, refine_points
from spinelocus.metrics import coordinate_rows, summarize
from spinelocus.refinement import identity_guard
from spinelocus.utils import provenance, write_json


class EnsembleTests(unittest.TestCase):
    def test_replay_rejects_mixed_yolo_flip_protocols_with_identical_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = self.saved(root, 0), self.saved(root, 1)
            summary = json.loads((second/'summary.json').read_text())
            receipt = json.loads((second/'provenance.json').read_text())
            summary['protocol']['yolo_flip_tta'] = True
            receipt['config']['yolo_flip_tta'] = True
            write_json(second/'summary.json', summary)
            write_json(second/'provenance.json', receipt)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                ensemble_evaluations([first, second], root/'mixed')

    def fixture(self):
        gt = np.stack([np.ones(11) * 80, np.arange(11) * 20 + 40], axis=1)
        return gt, gt + [3., 2.], [gt + [-1., 0.], gt + [1., 0.]]

    def saved(self, root, number, hrnet=None, yolo=None, gt=None):
        default_gt, default_yolo, hrnets = self.fixture()
        gt = default_gt if gt is None else gt
        yolo = default_yolo if yolo is None else yolo
        hrnet = hrnets[number] if hrnet is None else hrnet
        path = root / f"member{number}"
        path.mkdir(exist_ok=True)
        weights = root / f"hrnet{number}.pt"
        weights.write_text(str(number))
        for name in ("manifest.csv", "yolo.pt"):
            (root / name).write_text("fixture")
        protocol = dict(split="val", images=1, patients=1, imgsz=960, tta=True,
                        conf=.01, keypoint_conf=0., alpha=2.2, min_roi=112, max_roi=256,
                        coordinate_convention="pixel_centers")
        rows = []
        for mode, xy in (("yolo", yolo), ("yolo_hrnet", hrnet)):
            rows.extend(coordinate_rows(gt, xy, np.ones(11), "image.jpg", "A", "val", mode, 300, 300))
        frame = pd.DataFrame(rows)
        frame.sample(frac=1, random_state=number).to_csv(path / "predictions.csv", index=False)
        summary = {mode: summarize(part) for mode, part in frame.groupby("mode")}
        summary["protocol"] = protocol
        write_json(path / "summary.json", summary)
        write_json(path / "provenance.json", provenance(protocol, [root / "manifest.csv", root / "yolo.pt", weights]))
        return path

    def test_mean_is_equal_in_global_coordinates_and_preserves_missingness(self):
        members = np.array([[[1., 4.], [np.nan, np.nan]], [[3., 8.], [5., 9.]]])
        result = mean_coordinates(members)
        np.testing.assert_equal(result, [[2., 6.], [np.nan, np.nan]])
        np.testing.assert_equal(result, mean_coordinates(members[::-1]))
        for invalid in ([], [[1., 2.]], [[[1., np.nan]]], [[[1., np.inf]]]):
            with self.assertRaises(ValueError):
                mean_coordinates(invalid)

    def test_guard_runs_after_averaging(self):
        yolo = np.array([[0., 0.], [0., 10.], [0., 20.]])
        a, b = yolo.copy(), yolo.copy()
        a[0] = [0., 8.]
        after, _, _ = identity_guard(yolo, mean_coordinates([a, b]))
        before = mean_coordinates([identity_guard(yolo, a)[0], identity_guard(yolo, b)[0]])
        np.testing.assert_equal(after[0], [0., 4.])
        np.testing.assert_equal(before[0], [0., 0.])

    def test_replay_aligns_shuffled_keys_and_keeps_missing_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt, _, hrnets = self.fixture()
            hrnets[1][0] = np.nan
            paths = [self.saved(root, i, hrnet=xy) for i, xy in enumerate(hrnets)]
            combined, summary = ensemble_evaluations(paths, root / "ensemble")
            raw = combined[combined["mode"] == "yolo_hrnet"]
            np.testing.assert_equal(raw[["pred_x", "pred_y"]].to_numpy()[1:], gt[1:])
            self.assertEqual(summary["yolo_hrnet"]["failed_points"], 1)
            self.assertEqual(summary["yolo_hrnet"]["pck_at_5px_all_visible"], 10 / 11)
            self.assertEqual(summary["yolo_hrnet_guarded"]["failed_points"], 1)
            with self.assertRaisesRegex(ValueError, "empty"):
                ensemble_evaluations(paths, root / "ensemble")
            with self.assertRaisesRegex(ValueError, "distinct"):
                ensemble_evaluations([paths[0], paths[0]], root / "duplicate")
            self.assertFalse((root / "duplicate").exists())

    def test_replay_rejects_incompatible_protocol_gt_yolo_and_duplicate_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt, yolo, _ = self.fixture()
            a, b = self.saved(root, 0), self.saved(root, 1)
            provenance_b = json.loads((b / "provenance.json").read_text())
            # Copy the first member's HRNet identity, even with a different source path.
            provenance_a = json.loads((a / "provenance.json").read_text())
            provenance_b["files_sha256"][str(root / "hrnet1.pt")] = list(provenance_a["files_sha256"].values())[2]
            write_json(b / "provenance.json", provenance_b)
            with self.assertRaisesRegex(ValueError, "distinct HRNet"):
                ensemble_evaluations([a, b], root / "duplicate")
            b = self.saved(root, 1, gt=gt + 1, yolo=yolo + 1)
            with self.assertRaisesRegex(ValueError, "ground truth"):
                ensemble_evaluations([a, b], root / "gt")
            b = self.saved(root, 1, yolo=yolo + 1)
            with self.assertRaisesRegex(ValueError, "YOLO predictions"):
                ensemble_evaluations([a, b], root / "yolo")
            b = self.saved(root, 1)
            for name, key in (("summary.json", "protocol"), ("provenance.json", "config")):
                value = json.loads((b / name).read_text())
                value[key]["tta"] = False
                write_json(b / name, value)
            with self.assertRaisesRegex(ValueError, "share the protocol"):
                ensemble_evaluations([a, b], root / "protocol")
            self.assertFalse((root / "protocol").exists())
            b = self.saved(root, 1)
            value = json.loads((b / "summary.json").read_text())
            value["protocol"]["split"] = "test"
            write_json(b / "summary.json", value)
            with self.assertRaisesRegex(ValueError, "validation only"):
                ensemble_evaluations([a, b], root / "test")

    def test_replay_does_not_choose_points_using_gt_and_rejects_corrupt_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self.saved(root, i) for i in range(2)]
            combined, _ = ensemble_evaluations(paths, root / "first")
            gt, yolo, hrnets = self.fixture()
            paths = [self.saved(root, i, gt=gt + 100, yolo=yolo, hrnet=hrnets[i]) for i in range(2)]
            altered, _ = ensemble_evaluations(paths, root / "altered")
            pd.testing.assert_frame_equal(combined[["mode", "pred_x", "pred_y"]],
                                          altered[["mode", "pred_x", "pred_y"]])
            value = json.loads((paths[0] / "summary.json").read_text())
            value["yolo_hrnet"]["mean_euclidean_px_on_predictions"] = 999
            write_json(paths[0] / "summary.json", value)
            with self.assertRaisesRegex(ValueError, "Saved summary"):
                ensemble_evaluations(paths, root / "corrupt")

    def test_live_evaluation_predict_and_replay_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self.saved(root, i) for i in range(2)]
            offline, expected = ensemble_evaluations(paths, root / "offline")
            gt, yolo, hrnets = self.fixture()
            Image.new("RGB", (300, 300)).save(root / "image.jpg")
            args = SimpleNamespace(data=str(root), manifest=str(root / "manifest.csv"), split="val",
                                   out=str(root / "live"), yolo=str(root / "yolo.pt"),
                                   hrnet=str(root / "hrnet0.pt"), hrnet_extra=[str(root / "hrnet1.pt")],
                                   device="0", imgsz=960, conf=.01, keypoint_conf=0., alpha=2.2,
                                   min_roi=112, max_roi=256, tta=True, identity_guard="nearest-yolo")
            cfg = {"coordinate_convention": "pixel_centers"}
            models = {args.hrnet: (0, cfg), args.hrnet_extra[0]: (1, cfg)}
            modules = {"ultralytics": SimpleNamespace(YOLO=lambda _: SimpleNamespace(predict=lambda *a, **k: [None])),
                       "spinelocus.train": SimpleNamespace(resolve_device=lambda _: "cuda:0"),
                       "spinelocus.model": SimpleNamespace(load_model=lambda path, *a: models[path])}
            manifest = pd.DataFrame([dict(image="image.jpg", label="label.txt", patient_id="A",
                                          split="val", width=300, height=300)])
            with patch.dict("sys.modules", modules), \
                 patch("spinelocus.evaluate.load_manifest", return_value=manifest), \
                 patch("spinelocus.evaluate.verify_manifest_files"), \
                 patch("spinelocus.evaluate.read_label", return_value=(gt, np.ones(11))), \
                 patch("spinelocus.evaluate.yolo_points", return_value=yolo), \
                 patch("spinelocus.evaluate.refine_points", side_effect=lambda im, p, model, *a: hrnets[model]) as refiner:
                live, actual = evaluate(args)
                pd.testing.assert_frame_equal(live.sort_values(["mode", "vertebra"]).reset_index(drop=True),
                                              offline.sort_values(["mode", "vertebra"]).reset_index(drop=True))
                self.assertEqual(refiner.call_count, 2)
                args.image, args.out = str(root / "image.jpg"), str(root / "prediction.json")
                predict(args)
                prediction = json.loads(Path(args.out).read_text())
                np.testing.assert_equal(prediction["yolo_hrnet"], gt)
                self.assertEqual(prediction["hrnet_ensemble"]["members"], 2)
            for mode in ("yolo", "yolo_hrnet", "yolo_hrnet_guarded"):
                self.assertEqual(actual[mode], expected[mode])
            for field in PROTOCOL_FIELDS:
                self.assertEqual(actual["protocol"][field], expected["protocol"][field])

    def test_live_rejects_repeated_weights_and_mixed_coordinate_conventions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"{i}.pt" for i in range(2)]
            paths[0].write_text("0")
            paths[1].write_text("0")
            args = SimpleNamespace(hrnet=str(paths[0]), hrnet_extra=[str(paths[1])])
            with self.assertRaisesRegex(ValueError, "distinct hashes"):
                load_refiners(args, "cpu")
            paths[1].write_text("1")
            fake = SimpleNamespace(load_model=lambda path, *a: (object(), {
                "coordinate_convention": "pixel_centers" if path == str(paths[0]) else "legacy_scale"}))
            with patch.dict("sys.modules", {"spinelocus.model": fake}):
                with self.assertRaisesRegex(ValueError, "coordinate convention"):
                    load_refiners(args, "cpu")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Install torch for real decoding")
    def test_real_torch_crop_tta_and_global_decode_match_member_average(self):
        import torch

        from spinelocus.model import HRNetHeatmap

        torch.set_num_threads(2)
        image = Image.fromarray(np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3))
        _, points, _ = self.fixture()
        points /= 3
        cfg = {"model_name": "smoke_cnn", "crop_size": 32, "heatmap_size": 16,
               "coordinate_convention": "pixel_centers", "temperature_decode": .35}
        refiners = []
        for seed in (42, 43):
            torch.manual_seed(seed)
            refiners.append((HRNetHeatmap(cfg).eval(), cfg.copy()))
        singles = [refine_points(image, points, model, config, "cpu", minimum=24, maximum=40, tta=True)
                   for model, config in refiners]
        result = refine_ensemble(image, points, refiners, "cpu", minimum=24, maximum=40, tta=True)
        np.testing.assert_allclose(result, mean_coordinates(singles), rtol=0, atol=1e-10)
        self.assertTrue(np.isfinite(result).all())
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{i}.pt" for i in range(2)]
            for path, (model, config) in zip(paths, refiners):
                torch.save({"config": config, "model": model.state_dict()}, path)
            loaded, _ = load_refiners(SimpleNamespace(hrnet=str(paths[0]), hrnet_extra=[str(paths[1])]), "cpu")
            decoded = refine_ensemble(image, points, loaded, "cpu", minimum=24, maximum=40, tta=True)
            np.testing.assert_allclose(decoded, result, rtol=0, atol=1e-10)
