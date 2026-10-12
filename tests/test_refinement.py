"""Identity decisions and CSV replay must agree with the inference entry point."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from spinelocus.evaluate import evaluate
from spinelocus.metrics import coordinate_rows
from spinelocus.refinement import identity_guard
from spinelocus.refinement_audit import audit_refinement, replay


class RefinementTests(unittest.TestCase):
    def test_guard_rejects_other_identity_retains_tie_and_does_not_mutate(self):
        y = np.array([[0., 0.], [0., 10.], [0., 20.]])
        h = np.array([[0., 6.], [0., 15.], [1., 21.]])
        before = h.copy()
        out, reject, nearest = identity_guard(y, h)
        np.testing.assert_equal(reject, [True, False, False])
        np.testing.assert_equal(out[0], y[0])
        np.testing.assert_equal(out[1:], h[1:])
        np.testing.assert_equal(h, before)
        self.assertEqual(nearest[0], 1)

    def test_missing_neighbors_and_refinements_do_not_fabricate_predictions(self):
        y = np.array([[0., 0.], [np.nan, np.nan], [0., 20.]])
        h = np.array([[0., 9.], [np.nan, np.nan], [np.nan, np.nan]])
        out, reject, nearest = identity_guard(y, h)
        np.testing.assert_equal(out, h)
        self.assertFalse(reject.any())
        self.assertEqual(nearest[1], -1)
        empty, _, _ = identity_guard(np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(empty.shape, (0, 2))
        with self.assertRaises(ValueError):
            identity_guard(np.zeros((3, 2)), np.zeros((2, 2)))

    def frame(self):
        gt = np.stack([np.ones(11) * 80, np.arange(11) * 20 + 40], axis=1)
        y = gt + [3., 2.]
        h = gt.copy()
        h[2] = gt[3]
        rows = []
        for mode, pred in [('yolo', y), ('yolo_hrnet', h)]:
            rows.extend(coordinate_rows(gt, pred, np.ones(11), 'image.jpg', 'A', 'val', mode, 300, 300))
        return pd.DataFrame(rows)

    def test_replay_uses_only_predictions_for_decisions(self):
        frame = self.frame()
        out, decisions, _ = replay(frame.sample(frac=1, random_state=3))
        self.assertEqual(len(out), 33)
        self.assertEqual(decisions.rejected.sum(), 1)
        altered = frame.copy()
        altered[['gt_x', 'gt_y']] += 100
        delta = altered[['pred_x', 'pred_y']].to_numpy() - altered[['gt_x', 'gt_y']].to_numpy()
        altered['error_px'] = np.linalg.norm(delta, axis=1)
        altered['component_mae_px'] = np.abs(delta).mean(1)
        other, other_decisions, _ = replay(altered)
        pd.testing.assert_frame_equal(decisions, other_decisions)
        np.testing.assert_equal(out.loc[out['mode'] == 'yolo_hrnet_guarded', ['pred_x', 'pred_y']].to_numpy(),
                                other.loc[other['mode'] == 'yolo_hrnet_guarded', ['pred_x', 'pred_y']].to_numpy())

    def test_replay_rejects_test_unpaired_and_corrupted_metrics(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, 'validation'):
            replay(frame.assign(split='test'))
        with self.assertRaisesRegex(ValueError, 'keys differ'):
            replay(frame.iloc[:-1])
        wrong_gt = frame.copy()
        wrong_gt.loc[0, 'gt_x'] += 1
        with self.assertRaisesRegex(ValueError, 'ground truth'):
            replay(wrong_gt)
        damaged = frame.copy()
        damaged.loc[0, 'error_px'] = 1000
        with self.assertRaisesRegex(ValueError, 'Saved error'):
            replay(damaged)
        with self.assertRaisesRegex(ValueError, 'all 11'):
            replay(frame[frame.vertebra != 11])

    def test_live_evaluation_and_offline_replay_agree(self):
        frame = self.frame()
        y = frame[frame['mode'] == 'yolo'][['pred_x', 'pred_y']].to_numpy()
        h = frame[frame['mode'] == 'yolo_hrnet'][['pred_x', 'pred_y']].to_numpy()
        gt = frame[frame['mode'] == 'yolo'][['gt_x', 'gt_y']].to_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (300, 300)).save(root / 'image.jpg')
            for name in ('manifest.csv', 'yolo.pt', 'hrnet.pt'):
                (root / name).write_text('fixture')
            args = SimpleNamespace(data=str(root), manifest=str(root / 'manifest.csv'), split='val',
                                   out=str(root / 'live'), yolo=str(root / 'yolo.pt'),
                                   hrnet=str(root / 'hrnet.pt'), device='0', imgsz=960,
                                   conf=.01, keypoint_conf=0., alpha=2.2, min_roi=112, max_roi=256,
                                   tta=True, identity_guard='nearest-yolo')
            manifest = pd.DataFrame([dict(image='image.jpg', label='label.txt', patient_id='A',
                                          split='val', width=300, height=300)])
            fake_yolo = SimpleNamespace(YOLO=lambda _: SimpleNamespace(predict=lambda *a, **k: [None]))
            modules = {'ultralytics': fake_yolo,
                       'spinelocus.train': SimpleNamespace(resolve_device=lambda _: 'cuda:0'),
                       'spinelocus.model': SimpleNamespace(load_model=lambda *a: (object(), {'coordinate_convention': 'pixel_centers'}))}
            with patch.dict('sys.modules', modules), \
                 patch('spinelocus.evaluate.load_manifest', return_value=manifest), \
                 patch('spinelocus.evaluate.verify_manifest_files'), \
                 patch('spinelocus.evaluate.read_label', return_value=(gt, np.ones(11))), \
                 patch('spinelocus.evaluate.yolo_points', return_value=y), \
                 patch('spinelocus.evaluate.refine_points', return_value=h):
                live, summary = evaluate(args)
            offline, _, _ = replay(live)
            pd.testing.assert_frame_equal(live.sort_values(['mode', 'vertebra']).reset_index(drop=True),
                                          offline.sort_values(['mode', 'vertebra']).reset_index(drop=True))
            self.assertEqual(summary['refinement_guard']['rejected_refinements'], 1)
            compared = audit_refinement([root / 'live'], root / 'audit')
            self.assertEqual(len(compared), 3)
            saved = json.loads((root / 'audit/live/summary.json').read_text())
            self.assertEqual(saved['yolo_hrnet_guarded'], summary['yolo_hrnet_guarded'])
            with self.assertRaisesRegex(ValueError, 'empty'):
                audit_refinement([root / 'live'], root / 'audit')
            summary['yolo']['mean_euclidean_px_on_predictions'] = 999
            (root / 'live/summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'Saved summary'):
                audit_refinement([root / 'live'], root / 'corrupt-audit')
            self.assertFalse((root / 'corrupt-audit').exists())
