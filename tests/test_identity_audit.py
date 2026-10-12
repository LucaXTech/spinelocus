"""Coherent identity shifts, missing predictions and source/crop verification."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from spinelocus.cli import parser
from spinelocus.geometry import roi_size, window
from spinelocus.identity_audit import (
    export_review_overlays,
    identity_diagnostics,
    validate_visual_sources,
    write_identity_audit,
)
from spinelocus.metrics import coordinate_rows, summarize
from spinelocus.refinement_audit import audit_refinement, replay
from spinelocus.utils import write_json

PROTOCOL = {'split': 'val', 'images': 1, 'patients': 1, 'alpha': 2.2, 'min_roi': 112, 'max_roi': 256}


class IdentityAuditTests(unittest.TestCase):
    def fixture(self, yolo_shift=26, hrnet_shift=40):
        gt = np.stack([np.ones(11)*300, np.arange(11)*40+100], axis=1)
        rows = []
        for mode, shift in [('yolo', yolo_shift), ('yolo_hrnet', hrnet_shift)]:
            rows.extend(coordinate_rows(gt, gt+[0, shift], np.ones(11), 'images/val/image.png',
                                        'A', 'val', mode, 640, 640))
        return pd.DataFrame(rows)

    def test_shared_shift_is_coherent_before_and_after_guard(self):
        original = self.fixture()
        frame, decisions, _ = replay(original.sample(frac=1, random_state=7))
        before = frame.copy(deep=True)
        points, images, patients, summary = identity_diagnostics(frame, PROTOCOL)
        pd.testing.assert_frame_equal(before, frame)
        self.assertEqual(decisions.rejected.sum(), 0)
        self.assertTrue(images.coherent_adjacent_shift.all())
        self.assertEqual(images.plus_one_nearest_gt.tolist(), [10, 10, 10])
        self.assertTrue((patients.coherent_images == 1).all())
        self.assertEqual(summary['guarded_adjacent_over_30_same_direction_in_yolo'], 10)
        self.assertEqual(summary['methods']['yolo_hrnet_guarded']['errors_over_30_in_coherent_images'], 11)
        self.assertTrue(summary['diagnostic_uses_ground_truth'])
        self.assertFalse(summary['changes_predictions'])
        self.assertTrue(points.gt_landmarks_in_yolo_crop.gt(1).any())

    def test_negative_shift_and_tie_preserve_identity(self):
        negative, _, _ = replay(self.fixture(-26, -40))
        _, images, _, _ = identity_diagnostics(negative, PROTOCOL)
        self.assertTrue((images.coherent_shift_direction == -1).all())
        tied, _, _ = replay(self.fixture(20, 20))
        points, images, _, _ = identity_diagnostics(tied, PROTOCOL)
        self.assertTrue(points.nearest_gt_offset.eq(0).all())
        self.assertFalse(images.coherent_adjacent_shift.any())

    def test_missing_predictions_are_not_nearest_gt_or_resolved_errors(self):
        frame = self.fixture()
        missing = (frame['mode'] == 'yolo_hrnet') & (frame.vertebra <= 4)
        frame.loc[missing, ['pred_x', 'pred_y', 'error_px', 'component_mae_px']] = np.nan
        combined, _, _ = replay(frame)
        points, images, _, summary = identity_diagnostics(combined, PROTOCOL)
        raw = points[points['mode'] == 'yolo_hrnet']
        self.assertEqual(raw.nearest_gt_offset.isna().sum(), 4)
        self.assertEqual(raw.nearest_gt_distance_px.isna().sum(), 4)
        self.assertFalse(images[images['mode'] == 'yolo_hrnet'].coherent_adjacent_shift.iloc[0])
        self.assertEqual(summary['methods']['yolo_hrnet']['visible_points'], 11)
        self.assertEqual(summary['paired_threshold_transitions'][0]['unpaired_predictions'], 4)

    def test_roi_is_exactly_inference_window_and_independent_of_gt(self):
        frame, _, _ = replay(self.fixture())
        points, _, _, _ = identity_diagnostics(frame, PROTOCOL)
        yolo = frame[frame['mode'] == 'yolo'][['pred_x', 'pred_y']].to_numpy()
        bound = window(640, 640, *yolo[3], roi_size(yolo, 3, 2.2, 112, 256))
        actual = points[(points['mode'] == 'yolo') & (points.vertebra == 4)].iloc[0]
        self.assertEqual(tuple(actual[k] for k in ['x1', 'y1', 'x2', 'y2']), bound)
        changed = frame.copy()
        changed[['gt_x', 'gt_y']] += 300
        other, _, _, _ = identity_diagnostics(changed, PROTOCOL)
        pd.testing.assert_frame_equal(points[['x1', 'y1', 'x2', 'y2']], other[['x1', 'y1', 'x2', 'y2']])
        without, _, _, summary = identity_diagnostics(frame, {})
        self.assertTrue(without.x1.isna().all())
        self.assertFalse(summary['roi_parameters_verified_from_summary'])
        with self.assertRaisesRegex(ValueError, 'ROI'):
            identity_diagnostics(frame, {**PROTOCOL, 'alpha': -1})

    def source_dataset(self, root, frame):
        val = frame[frame['mode'] == 'yolo'].sort_values('vertebra')
        (root / 'images/val').mkdir(parents=True)
        (root / 'labels/val').mkdir(parents=True)
        Image.new('RGB', (640, 640), (65, 65, 65)).save(root / 'images/val/image.png')
        values = [0, .5, .5, .9, .9]
        for row in val.itertuples():
            values.extend([row.gt_x/640, row.gt_y/640, 2])
        (root / 'labels/val/image.txt').write_text(' '.join(map(str, values)))
        manifest = pd.DataFrame([
            dict(image='images/val/image.png', label='labels/val/image.txt', split='val',
                 patient_id='A', width=640, height=640),
            # Train/test deliberately do not exist: review must not read them.
            dict(image='absent-train.png', label='absent-train.txt', split='train',
                 patient_id='B', width=640, height=640),
            dict(image='absent-test.png', label='absent-test.txt', split='test',
                 patient_id='C', width=640, height=640),
        ])
        path = root / 'manifest.csv'
        manifest.to_csv(path, index=False)
        return path

    def test_label_check_and_overlay_never_read_test_and_record_hashes(self):
        frame, _, _ = replay(self.fixture())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.source_dataset(root, frame)
            labels = validate_visual_sources(frame, root, manifest)
            points, images, _, _ = identity_diagnostics(frame, PROTOCOL)
            records = export_review_overlays(frame, points, images, labels, root, root/'overlays', count=1)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0]['image_sha256']), 64)
            with Image.open(root/'overlays'/records[0]['overlay']) as image:
                self.assertEqual(image.size, (1536, 1200))
                self.assertGreater(len(image.getcolors(maxcolors=10**6)), 5)
                focus = records[0]['focus_vertebra']
                row = frame[(frame['mode'] == 'yolo_hrnet') & (frame.vertebra == focus)].iloc[0]
                diag = points[(points['mode'] == 'yolo') & (points.vertebra == focus)].iloc[0]
                px = 512 + (row.pred_x-diag.x1)*512/(diag.x2-diag.x1)
                py = 640 + (row.pred_y-diag.y1)*512/(diag.y2-diag.y1)
                self.assertEqual(image.getpixel((int(px+8), int(py))), (255, 0, 0))
            changed = frame.copy()
            changed['gt_y'] += 1
            with self.assertRaisesRegex(ValueError, 'Label GT'):
                validate_visual_sources(changed, root, manifest)
            manifest_frame = pd.read_csv(manifest)
            manifest_frame.loc[0, 'label'] = '../outside.txt'
            manifest_frame.to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                validate_visual_sources(frame, root, manifest)

    def test_saved_guard_must_match_replay_and_outputs_are_written(self):
        frame, _, _ = replay(self.fixture())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'evaluation'
            source.mkdir()
            frame.to_csv(source/'predictions.csv', index=False)
            summary = {mode: summarize(part) for mode, part in frame.groupby('mode')}
            summary['protocol'] = PROTOCOL
            write_json(source/'summary.json', summary)
            audit_refinement([source], root/'audit')
            saved = json.loads((root/'audit/evaluation/identity_summary.json').read_text())
            self.assertEqual(saved['methods']['yolo']['coherent_images'], 1)
            self.assertFalse(saved['local_label_verification']['performed'])
            damaged = frame.copy()
            damaged.loc[damaged['mode'] == 'yolo_hrnet_guarded', 'pred_y'] += 1
            damaged.to_csv(source/'predictions.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'guarded predictions'):
                audit_refinement([source], root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_report_with_local_labels_runs_end_to_end(self):
        frame, _, _ = replay(self.fixture())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.source_dataset(root, frame)
            labels = validate_visual_sources(frame, root, manifest)
            target = root/'report'
            target.mkdir()
            summary = write_identity_audit(frame, target, PROTOCOL, labels=labels, data=root, overlays=1)
            self.assertTrue(summary['local_label_verification']['performed'])
            self.assertEqual(summary['local_label_verification']['validation_images'], 1)
            self.assertTrue((target/'identity_report.md').is_file())
            self.assertTrue((target/'overlays/index.csv').is_file())

    def test_cli_visual_options_and_invalid_arguments(self):
        args = parser().parse_args(['audit-refinement', '--evaluations', 'eval', '--out', 'new',
                                   '--data', 'dataset', '--manifest', 'manifest.csv', '--overlays', '2'])
        self.assertEqual(args.overlays, 2)
        with self.assertRaisesRegex(ValueError, 'both'):
            audit_refinement([], 'unused', data='dataset')
        with self.assertRaisesRegex(ValueError, 'positive'):
            audit_refinement([], 'unused', overlays=0)
        with self.assertRaisesRegex(ValueError, 'coherent_min'):
            identity_diagnostics(pd.DataFrame(), {}, coherent_min=12)
