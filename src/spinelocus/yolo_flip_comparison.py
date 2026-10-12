"""A paired validation comparison that verifies the unchanged original YOLO view."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .ensemble import PROTOCOL_FIELDS, mean_coordinates
from .identity_audit import identity_diagnostics
from .metrics import paired_patient_bootstrap, summarize, summarize_patients
from .refinement_audit import KEY, META, replay
from .utils import provenance, write_json
from .yolo_views import flip_protocol

MODES = ('yolo', 'yolo_hrnet', 'yolo_hrnet_guarded')


def _read_run(path):
    summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
    receipt = json.loads((path/'provenance.json').read_text(encoding='utf-8'))
    protocol = summary['protocol']
    if protocol['split'] != 'val':
        raise ValueError("YOLO flip exploration is validation only")
    saved = pd.read_csv(path/'predictions.csv', dtype={'patient_id': str})
    if set(saved['mode']) != set(MODES):
        raise ValueError("Comparison requires YOLO, raw HRNet and guarded HRNet")
    frame, _, _ = replay(saved)
    actual = saved.set_index([*KEY, 'mode']).sort_index()
    expected = frame.set_index([*KEY, 'mode']).sort_index()
    columns = [*META, 'pred_x', 'pred_y', 'error_px', 'component_mae_px']
    if actual.index.has_duplicates or not actual.index.equals(expected.index) or not np.allclose(
        actual[columns], expected[columns], rtol=1e-7, atol=1e-6, equal_nan=True
    ):
        raise ValueError("Saved guarded predictions differ from replay")
    if (frame.image.nunique(), frame.patient_id.nunique()) != (protocol['images'], protocol['patients']):
        raise ValueError("Saved protocol counts differ from coordinates")
    for mode, part in frame.groupby('mode'):
        for key, value in summarize(part).items():
            stored = summary[mode][key]
            equal = value is stored if value is None or stored is None else math.isclose(
                value, stored, rel_tol=1e-7, abs_tol=1e-6)
            if not equal:
                raise ValueError(f"Saved summary differs from coordinates: {mode}/{key}")
    for key in (*PROTOCOL_FIELDS, 'identity_guard', 'hrnet_ensemble'):
        if key not in protocol or receipt['config'].get(key) != protocol[key]:
            raise ValueError(f"Saved provenance/protocol differ: {key}")
    flip = protocol.get('yolo_flip_tta', False)
    if not isinstance(flip, bool) or receipt['config'].get('yolo_flip_tta', False) != flip:
        raise ValueError("Invalid YOLO flip flag or provenance")
    if flip and receipt['config'].get('yolo_view_ensemble') != protocol.get('yolo_view_ensemble'):
        raise ValueError("Saved YOLO view provenance/protocol differ")
    files = list(receipt['files_sha256'].items())
    if len(files) < 3 or not files[0][0].lower().endswith('.csv') or any(
        not name.lower().endswith('.pt') for name, _ in files[1:]
    ):
        raise ValueError("Expected manifest and fixed model checkpoints in provenance")
    return frame, protocol, [digest for _, digest in files]


def compare_yolo_flip(baseline, candidate, out):
    baseline, candidate, out = Path(baseline).resolve(), Path(candidate).resolve(), Path(out).resolve()
    if baseline == candidate or any(out == p or p in out.parents for p in (baseline, candidate)):
        raise ValueError("Comparison output must be outside distinct input runs")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Comparison output must be empty")
    a, pa, ha = _read_run(baseline)
    b, pb, hb = _read_run(candidate)
    if pa.get('yolo_flip_tta', False) or not pb.get('yolo_flip_tta', False):
        raise ValueError("Baseline must be original-only and candidate must use YOLO flip")
    if pb.get('yolo_view_ensemble') != flip_protocol():
        raise ValueError("Candidate does not use the specified two-view protocol")
    if ha != hb or any(pa[key] != pb[key] for key in (*PROTOCOL_FIELDS, 'identity_guard', 'hrnet_ensemble')):
        raise ValueError("Only YOLO flip may differ; manifest, weights and other protocol fields must match")
    aa = a[a['mode'] == 'yolo'].set_index(KEY).sort_index()
    bb = b[b['mode'] == 'yolo'].set_index(KEY).sort_index()
    if not aa.index.equals(bb.index) or not np.array_equal(aa[META], bb[META]):
        raise ValueError("Evaluation ground truth and landmark keys must be identical")
    views = pd.read_csv(candidate/'yolo_views.csv', dtype={'patient_id': str})
    if set(views['mode']) != {'yolo_original', 'yolo_horizontal_flip'}:
        raise ValueError("Candidate must save exactly the two YOLO views")
    views['mode'] = views['mode'].map({'yolo_original': 'yolo', 'yolo_horizontal_flip': 'yolo_hrnet'})
    validated, _, _ = replay(views)
    original = validated[validated['mode'] == 'yolo'].set_index(KEY).sort_index()
    mirrored = validated[validated['mode'] == 'yolo_hrnet'].set_index(KEY).sort_index()
    if not aa.index.equals(original.index) or not np.array_equal(aa[META], original[META]):
        raise ValueError("YOLO view ground truth differs from evaluation")
    xy = ['pred_x', 'pred_y']
    if not np.allclose(aa[xy], original[xy], rtol=0, atol=1e-3, equal_nan=True):
        raise ValueError("Original YOLO view does not reproduce the frozen baseline within 0.001px")
    average = mean_coordinates([original[xy].to_numpy(), mirrored[xy].to_numpy()])
    if not np.allclose(average, bb[xy], rtol=0, atol=1e-6, equal_nan=True):
        raise ValueError("Candidate YOLO coordinates differ from the specified view mean")
    records, patients, bootstraps, diagnostics = [], [], {}, {}
    for role, frame, protocol in [('baseline', a, pa), ('candidate', b, pb)]:
        _, _, _, diagnostics[role] = identity_diagnostics(frame, protocol)
        for mode, part in frame.groupby('mode'):
            by_patient, macro = summarize_patients(part)
            metric = summarize(part)
            records.append({'run': role, 'mode': mode,
                            'macro_patient_pck5': macro['pck_at_5px_all_visible'],
                            'point_pck5': metric['pck_at_5px_all_visible'],
                            'mean_px': metric['mean_euclidean_px_on_predictions'],
                            'p95_px': metric['p95_px_on_predictions'],
                            'failed_points': metric['failed_points'],
                            'coherent_images': diagnostics[role]['methods'][mode]['coherent_images']})
            patients.extend({'run': role, 'mode': mode, **row} for row in by_patient)
    for mode in MODES:
        paired = pd.concat([a[a['mode'] == mode].assign(mode='baseline'),
                            b[b['mode'] == mode].assign(mode='candidate')], ignore_index=True)
        bootstraps[mode] = paired_patient_bootstrap(paired, baseline='baseline', candidate='candidate')
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out/'comparison.csv', index=False)
    pd.DataFrame(patients).to_csv(out/'by_patient.csv', index=False)
    write_json(out/'summary.json', {'original_yolo_baseline_verified': True,
                                  'original_yolo_tolerance_px': 1e-3,
                                  'fixed_input_sha256': ha, 'metrics': records,
                                  'paired_patient_bootstrap': bootstraps, 'identity_diagnostics': diagnostics,
                                  'automatically_adopts_candidate': False})
    write_json(out/'provenance.json', provenance(
        {'experiment': 'yolo-horizontal-flip', 'split': 'val', 'test_evaluated': False},
        files=[p/f for p in (baseline, candidate) for f in ('predictions.csv', 'summary.json', 'provenance.json')]
        + [candidate/'yolo_views.csv']))
    lines = ['# YOLO flip comparison - validation', '',
             'Same manifest, model checkpoints, HRNet TTA and ROI parameters. '
             'The saved original YOLO view reproduces the baseline within 0.001px.', '',
             '| Run | Method | PCK5 subjects | Mean px | P95 px | Missing | Coherent images |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in records:
        mean = 'n/a' if row['mean_px'] is None else f"{row['mean_px']:.3f}"
        p95 = 'n/a' if row['p95_px'] is None else f"{row['p95_px']:.3f}"
        lines.append(f"| {row['run']} | {row['mode']} | {row['macro_patient_pck5']:.2%} | "
                     f"{mean} | {p95} | {row['failed_points']} | {row['coherent_images']} |")
    lines += ['', 'Missing points remain in PCK denominators. The candidate needs both YOLO views.',
              'Patient bootstrap is descriptive on repeatedly inspected validation subjects.',
              'This experiment does not establish superiority on new subjects or change the baseline.', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return pd.DataFrame(records)
