"""YOLO view reversal, failure denominators, live inference and paired controls."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from spinelocus.ensemble import ensemble_protocol
from spinelocus.evaluate import evaluate, predict
from spinelocus.metrics import coordinate_rows, summarize
from spinelocus.refinement_audit import replay
from spinelocus.utils import provenance, write_json
from spinelocus.yolo_flip_comparison import compare_yolo_flip
from spinelocus.yolo_views import flip_protocol, yolo_view_points


class YoloFlipTests(unittest.TestCase):
    def args(self, flip=True):
        return SimpleNamespace(imgsz=960, conf=.01, keypoint_conf=0., yolo_flip_tta=flip)

    def test_disabled_matches_single_view_and_enabled_preserves_spine_indices(self):
        pixels = np.arange(24*32*3, dtype=np.uint8).reshape(24, 32, 3)
        image = Image.fromarray(pixels)
        original = np.stack([np.arange(11)+3., np.arange(11)+4.], axis=1)
        mirrored = original.copy()
        mirrored[:, 0] = image.width-original[:, 0]
        calls = []
        def forward(values, **options):
            calls.append((values.copy(), options))
            return [original if len(calls) == 1 else mirrored]
        model = SimpleNamespace(predict=forward)
        points, views = yolo_view_points(model, image, self.args(False), '0', lambda r, c: r)
        np.testing.assert_array_equal(points, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual(views, {})
        calls.clear()
        points, views = yolo_view_points(model, image, self.args(), '0', lambda r, c: r)
        np.testing.assert_array_equal(points, original)
        np.testing.assert_array_equal(views['yolo_horizontal_flip'], original)
        np.testing.assert_array_equal(calls[0][0], pixels[:, :, ::-1])
        np.testing.assert_array_equal(calls[1][0], pixels[:, ::-1, ::-1])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertTrue(calls[1][0].flags.c_contiguous)
        self.assertEqual(flip_protocol()['identity_permutation'], list(range(1, 12)))

    def test_view_disagreement_is_averaged_and_missing_view_remains_missing(self):
        a = np.stack([np.arange(11)+3., np.arange(11)+4.], axis=1)
        b = a+[2, 6]
        b[4] = np.nan
        reflected = b.copy()
        reflected[:, 0] = 100-reflected[:, 0]
        results = iter([a, reflected])
        model = SimpleNamespace(predict=lambda *args, **kwargs: [next(results)])
        points, views = yolo_view_points(model, Image.new('RGB', (100, 80)), self.args(), 'cpu', lambda r, c: r)
        expected = a+[1, 3]
        expected[4] = np.nan
        np.testing.assert_equal(points, expected)
        np.testing.assert_equal(views['yolo_original'], a)
        np.testing.assert_equal(reflected[0], [95., 10.])

    def test_live_and_predict_use_mean_before_hrnet_and_save_original_view(self):
        gt = np.stack([np.ones(11)*80, np.arange(11)*20+40], axis=1)
        a, b = gt+[2, 4], gt+[6, 8]
        reflected = b.copy()
        reflected[:, 0] = 300-reflected[:, 0]
        mean = gt+[4, 6]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (300, 300)).save(root/'image.png')
            for name in ['manifest.csv', 'yolo.pt', 'hrnet.pt']:
                (root/name).write_text(name)
            args = SimpleNamespace(**vars(self.args()), data=str(root), manifest=str(root/'manifest.csv'),
                                   split='val', out=str(root/'live'), yolo=str(root/'yolo.pt'),
                                   hrnet=str(root/'hrnet.pt'), hrnet_extra=[], device='0',
                                   alpha=2.2, min_roi=112, max_roi=256, tta=True, identity_guard='nearest-yolo')
            manifest = pd.DataFrame([dict(image='image.png', label='label.txt', patient_id='A',
                                          split='val', width=300, height=300)])
            result_arrays = iter([a, reflected, a, reflected])
            fake_yolo = SimpleNamespace(YOLO=lambda _: SimpleNamespace(
                predict=lambda *args, **kwargs: [next(result_arrays)]))
            modules = {'ultralytics': fake_yolo,
                       'spinelocus.train': SimpleNamespace(resolve_device=lambda _: 'cuda:0'),
                       'spinelocus.model': SimpleNamespace(load_model=lambda *args: (
                           object(), {'coordinate_convention': 'pixel_centers'}))}
            centers = []
            def refine(image, points, *args, **kwargs):
                centers.append(points.copy())
                return points.copy()
            with patch.dict('sys.modules', modules), \
                 patch('spinelocus.evaluate.load_manifest', return_value=manifest), \
                 patch('spinelocus.evaluate.verify_manifest_files'), \
                 patch('spinelocus.evaluate.read_label', return_value=(gt, np.ones(11))), \
                 patch('spinelocus.evaluate.yolo_points', side_effect=lambda r, c: r), \
                 patch('spinelocus.evaluate.refine_points', side_effect=refine):
                frame, summary = evaluate(args)
                args.image, args.out = str(root/'image.png'), str(root/'prediction.json')
                predict(args)
            for center in centers:
                np.testing.assert_equal(center, mean)
            self.assertTrue(summary['protocol']['yolo_flip_tta'])
            self.assertEqual(summary['protocol']['yolo_view_ensemble'], flip_protocol())
            self.assertEqual(len(pd.read_csv(root/'live/yolo_views.csv')), 22)
            output = json.loads((root/'prediction.json').read_text())
            np.testing.assert_equal(output['yolo'], mean)
            np.testing.assert_equal(output['yolo_views']['yolo_original'], a)
            self.assertEqual(len(frame), 33)

    def saved_pair(self, root, missing=False):
        gt = np.stack([np.ones(11)*300, np.arange(11)*40+100], axis=1)
        original, mirrored = gt+[0, 26], gt+[0, -10]
        if missing:
            mirrored[0] = np.nan
        mean = (original+mirrored)/2
        protocol = dict(split='val', images=1, patients=1, imgsz=960, tta=True, conf=.01,
                        keypoint_conf=0., alpha=2.2, min_roi=112, max_roi=256,
                        coordinate_convention='pixel_centers', identity_guard='nearest-yolo',
                        hrnet_ensemble=ensemble_protocol(2))
        weights = [root/'manifest.csv', root/'yolo.pt', root/'hrnet42.pt', root/'hrnet43.pt']
        for path in weights:
            path.write_text(path.name)
        def rows(mode, xy):
            return coordinate_rows(gt, xy, np.ones(11), 'image.png', 'A', 'val', mode, 640, 640)
        for role, yy, hh, flip in [('baseline', original, gt+[0, 40], False),
                                   ('candidate', mean, np.where(np.isfinite(mean), gt, np.nan), True)]:
            directory = root/role
            directory.mkdir()
            frame, _, _ = replay(pd.DataFrame(rows('yolo', yy)+rows('yolo_hrnet', hh)))
            frame.to_csv(directory/'predictions.csv', index=False)
            proto = {**protocol, 'yolo_flip_tta': flip}
            if flip:
                proto['yolo_view_ensemble'] = flip_protocol()
                pd.DataFrame(rows('yolo_original', original)+rows('yolo_horizontal_flip', mirrored)).to_csv(
                    directory/'yolo_views.csv', index=False)
            summary = {mode: summarize(part) for mode, part in frame.groupby('mode')}
            summary['protocol'] = proto
            write_json(directory/'summary.json', summary)
            write_json(directory/'provenance.json', provenance(proto, files=weights))
        return root/'baseline', root/'candidate'

    def test_paired_comparison_checks_controls_and_keeps_failure_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.saved_pair(root, missing=True)
            records = compare_yolo_flip(a, b, root/'comparison')
            guarded = records[(records['run'] == 'candidate') & (records['mode'] == 'yolo_hrnet_guarded')].iloc[0]
            self.assertEqual(guarded.failed_points, 1)
            self.assertAlmostEqual(guarded.point_pck5, 10/11)
            summary = json.loads((root/'comparison/summary.json').read_text())
            self.assertTrue(summary['original_yolo_baseline_verified'])
            self.assertFalse(summary['automatically_adopts_candidate'])
            self.assertEqual(len(records), 6)
            with self.assertRaisesRegex(ValueError, 'empty'):
                compare_yolo_flip(a, b, root/'comparison')

    def test_changed_weights_or_original_view_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.saved_pair(root)
            receipt = json.loads((b/'provenance.json').read_text())
            saved_receipt = json.loads(json.dumps(receipt))
            receipt['files_sha256'][str(root/'yolo.pt')] = 'different'
            write_json(b/'provenance.json', receipt)
            with self.assertRaisesRegex(ValueError, 'Only YOLO flip'):
                compare_yolo_flip(a, b, root/'bad-weights')
            write_json(b/'provenance.json', saved_receipt)
            views = pd.read_csv(b/'yolo_views.csv')
            original = views['mode'] == 'yolo_original'
            views.loc[original, 'pred_y'] += 2
            delta = views[['pred_x', 'pred_y']].to_numpy()-views[['gt_x', 'gt_y']].to_numpy()
            views['error_px'], views['component_mae_px'] = np.linalg.norm(delta, axis=1), np.abs(delta).mean(1)
            views.to_csv(b/'yolo_views.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Original YOLO view'):
                compare_yolo_flip(a, b, root/'bad-original')
            self.assertFalse((root/'bad-original').exists())

    def test_test_split_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.saved_pair(root)
            summary = json.loads((b/'summary.json').read_text())
            summary['protocol']['split'] = 'test'
            write_json(b/'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'validation only'):
                compare_yolo_flip(a, b, root/'forbidden')
            self.assertFalse((root/'forbidden').exists())

    def test_candidate_must_be_the_mean_even_if_saved_metrics_are_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.saved_pair(root)
            frame = pd.read_csv(b/'predictions.csv')
            frame.loc[frame['mode'] == 'yolo', 'pred_y'] += 1
            delta = frame[['pred_x', 'pred_y']].to_numpy()-frame[['gt_x', 'gt_y']].to_numpy()
            frame['error_px'], frame['component_mae_px'] = np.linalg.norm(delta, axis=1), np.abs(delta).mean(1)
            frame.to_csv(b/'predictions.csv', index=False)
            summary = json.loads((b/'summary.json').read_text())
            for mode, part in frame.groupby('mode'):
                summary[mode] = summarize(part)
            write_json(b/'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'specified view mean'):
                compare_yolo_flip(a, b, root/'bad-mean')
            self.assertFalse((root/'bad-mean').exists())
