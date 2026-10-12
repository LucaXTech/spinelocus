import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from spinelocus.data import (
    audit_manifest,
    export_grouped_dataset,
    patient_id,
    read_label,
    scan_dataset,
    validate_manifest,
)
from spinelocus.geometry import roi_size, to_global, to_local, window
from spinelocus.metrics import coordinate_rows, paired_patient_bootstrap, summarize
from spinelocus.utils import sha256


class GeometryTests(unittest.TestCase):
    def test_crop_bounds_and_roundtrip(self):
        for center in [(0, 0), (399, 199), (150.7, 99.1), (-15, 800)]:
            bounds = window(400, 200, *center, 160)
            x1, y1, x2, y2 = bounds
            self.assertTrue(0 <= x1 < x2 <= 400 and 0 <= y1 < y2 <= 200)
            for convention in ("pixel_centers", "legacy_scale"):
                xy = np.array([x1 + 17.25, y1 + 88.5])
                np.testing.assert_allclose(
                    to_global(to_local(xy, bounds, 160, convention), bounds, 160, convention), xy
                )

    def test_small_image_keeps_actual_window(self):
        self.assertEqual(window(40, 60, 20, 20, 160), (0, 0, 40, 60))

    def test_roi_uses_prediction_neighbors_and_clamps(self):
        points = np.array([[50.0, 10.0], [50.0, 80.0], [50.0, 150.0]])
        self.assertEqual(roi_size(points, 1), 154)
        self.assertEqual(roi_size(points * 10, 1), 256)


class DataTests(unittest.TestCase):
    def test_patient_extraction_is_strict(self):
        self.assertEqual(patient_id("Series_003-121943-COPDGene_SYNTH001_INSP_STD.jpg"), "SYNTH001")
        self.assertEqual(patient_id("Series_004-121943-COPDGene_SYNTH001_EXP_STD.jpg"), "SYNTH001")
        with self.assertRaises(ValueError):
            patient_id("Series_003-121943_DRR_512.jpg")
        with self.assertRaises(ValueError):
            patient_id("no_patient_here.jpg")

    def test_label_exact_tokens_and_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "label.txt"
            tokens = [0, 0.5, 0.5, 1, 1] + [v for _ in range(11) for v in (0.25, 0.5, 2)]
            p.write_text(" ".join(map(str, tokens)))
            xy, vis = read_label(p, 200, 100)
            np.testing.assert_allclose(xy, np.tile([50, 50], (11, 1)))
            self.assertTrue((vis == 2).all())
            p.write_text(p.read_text() + " 0.9")
            with self.assertRaises(ValueError):
                read_label(p, 200, 100)

    def test_patient_leakage_fails(self):
        d = pd.DataFrame(
            {
                "image": ["a", "b", "c"],
                "label": ["x", "y", "z"],
                "split": ["train", "val", "test"],
                "patient_id": ["001", "001", "003"],
                "width": [10] * 3,
                "height": [10] * 3,
            }
        )
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            validate_manifest(d)

    def test_duplicate_bytes_across_patients_fail(self):
        d = pd.DataFrame(
            {
                "image": ["a", "b", "c"],
                "label": ["x", "y", "z"],
                "split": ["train", "val", "test"],
                "patient_id": ["001", "002", "003"],
                "width": [10] * 3,
                "height": [10] * 3,
                "image_sha256": ["same"] * 3,
            }
        )
        with self.assertRaisesRegex(ValueError, "Identical image bytes"):
            validate_manifest(d)

    def test_scan_checks_missing_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(root / "images/train/Series_001-1234.png")
            with self.assertRaises(FileNotFoundError):
                scan_dataset(root)

    def test_old_series_manifest_is_rejected_and_grouped_export_repairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source", Path(tmp) / "grouped"
            rows = []
            for i, (split, subject) in enumerate([
                ("train", "A1"), ("val", "A1"), ("test", "B2"),
                ("train", "C3"), ("val", "D4"), ("test", "E5"),
            ]):
                name = f"Series_{i:03d}-123456-COPDGene_{subject}_view{i}"
                image = source / "images" / split / (name + ".png")
                label = source / "labels" / split / (name + ".txt")
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (10, 10), (i, 0, 0)).save(image)
                label.write_text(" ".join(map(str, [0, .5, .5, 1, 1] + [.5, .5, 2] * 11)))
                rows.append(dict(image=image.relative_to(source).as_posix(),
                                 label=label.relative_to(source).as_posix(), split=split,
                                 patient_id=f"{i:03d}", width=10, height=10,
                                 image_sha256=sha256(image), label_sha256=sha256(label)))
            old = pd.DataFrame(rows)
            with self.assertRaisesRegex(ValueError, "does not match COPDGene"):
                validate_manifest(old)
            self.assertEqual(audit_manifest(old)["overlaps"]["train_val"], ["A1"])
            corrected = old.copy()
            corrected["patient_id"] = corrected.image.map(patient_id)
            with self.assertRaisesRegex(ValueError, "Patient leakage"):
                validate_manifest(corrected)
            result = export_grouped_dataset(old, source, dest)
            self.assertEqual(len(result), len(old))
            self.assertEqual(result.groupby("patient_id").split.nunique().max(), 1)
            self.assertTrue(audit_manifest(result)["safe_for_unseen_subject_evaluation"])
            for row in result.itertuples():
                self.assertEqual((dest / row.image).read_bytes(), (source / row.source_image).read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_grouped_dataset(old, source, dest)
            (source / old.iloc[0].image).write_bytes(b"changed after manifest")
            with self.assertRaises((ValueError, OSError)):
                export_grouped_dataset(old, source, Path(tmp) / "corrupted-copy")
            self.assertFalse((Path(tmp) / "corrupted-copy").exists())


class MetricTests(unittest.TestCase):
    def test_failed_prediction_stays_in_denominator(self):
        d = pd.DataFrame(
            {
                "error_px": [3.0, np.nan],
                "component_mae_px": [2.0, np.nan],
                "width": [100, 100],
                "height": [100, 100],
            }
        )
        value = summarize(d)
        self.assertEqual(value["failed_points"], 1)
        self.assertEqual(value["pck_at_5px_all_visible"], 0.5)
        self.assertAlmostEqual(value["auc_cdf_30px_all_visible"], 0.45)

    def test_euclidean_and_component_mae_are_different(self):
        rows = coordinate_rows(
            np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]), [2], "a", "001", "val", "yolo", 100, 100
        )
        self.assertEqual(rows[0]["error_px"], 5.0)
        self.assertEqual(rows[0]["component_mae_px"], 3.5)

    def test_all_missing_predictions_has_no_fabricated_mean(self):
        d = pd.DataFrame(
            {"error_px": [np.nan], "component_mae_px": [np.nan], "width": [100], "height": [100]}
        )
        self.assertIsNone(summarize(d)["mean_euclidean_px_on_predictions"])
        self.assertEqual(summarize(d)["auc_cdf_30px_all_visible"], 0.0)

    def test_bootstrap_independent_unit_is_patient(self):
        rows = []
        for patient in ("001", "002", "003"):
            for mode, error in (("yolo", 10.0), ("yolo_hrnet", 2.0)):
                rows.append(
                    {"patient_id": patient, "image": patient, "vertebra": 1, "mode": mode, "error_px": error}
                )
        result = paired_patient_bootstrap(pd.DataFrame(rows), repeats=100)
        self.assertEqual(result["patients"], 3)
        self.assertEqual(result["mean_error_gain_on_paired_predictions"]["ci95"], [8.0, 8.0])

    def test_unpaired_methods_fail(self):
        d = pd.DataFrame(
            {
                "patient_id": ["001", "001"],
                "image": ["a", "b"],
                "vertebra": [1, 1],
                "mode": ["yolo", "yolo_hrnet"],
                "error_px": [1, 2],
            }
        )
        with self.assertRaises(ValueError):
            paired_patient_bootstrap(d)


if __name__ == "__main__":
    unittest.main()
