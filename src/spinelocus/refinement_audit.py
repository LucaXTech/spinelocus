"""Replay the identity guard and diagnose saved validation predictions, without GPU."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .identity_audit import validate_visual_sources, write_identity_audit
from .metrics import paired_patient_bootstrap, summarize, summarize_patients
from .refinement import identity_guard
from .utils import provenance, write_json

KEY = ["image", "patient_id", "split", "vertebra"]
META = ["width", "height", "gt_x", "gt_y"]


def replay(frame):
    """Require complete paired rows: otherwise unlabelled YOLO neighbors are lost."""
    columns = {*KEY, *META, "mode", "pred_x", "pred_y", "error_px", "component_mae_px"}
    if not columns.issubset(frame.columns) or frame.empty:
        raise ValueError("Expected a nonempty evaluation predictions.csv")
    if frame[KEY].isna().any().any() or set(frame.split) != {"val"}:
        raise ValueError("Refinement exploration requires validation rows with complete keys")
    parts = []
    for mode in ("yolo", "yolo_hrnet"):
        part = frame[frame['mode'] == mode].copy().set_index(KEY).sort_index()
        if part.empty or part.index.has_duplicates:
            raise ValueError("Expected unique YOLO and HRNet rows")
        parts.append(part)
    yolo, hrnet = parts
    if not yolo.index.equals(hrnet.index) or not np.array_equal(yolo[META], hrnet[META]):
        raise ValueError("YOLO and HRNet ground truth/keys differ")
    for part in parts:
        if not np.isfinite(part[META].to_numpy(float)).all() or (part[['width', 'height']] <= 0).any().any():
            raise ValueError("Invalid ground truth/image dimensions")
        xy = part[['pred_x', 'pred_y']].to_numpy(float)
        if np.isinf(xy).any() or np.any(np.isnan(xy).any(1) != np.isnan(xy).all(1)):
            raise ValueError("Missing predictions must have both coordinates missing")
        delta = xy - part[['gt_x', 'gt_y']].to_numpy(float)
        if not np.allclose(np.linalg.norm(delta, axis=1), part.error_px, equal_nan=True):
            raise ValueError("Saved error disagrees with prediction coordinates")
        if not np.allclose(np.abs(delta).mean(1), part.component_mae_px, equal_nan=True):
            raise ValueError("Saved component MAE disagrees with prediction coordinates")
    source = hrnet.reset_index()
    baseline = yolo.reset_index()
    corrected, decisions, diagnostics = [], [], []
    for image, part in source.groupby('image', sort=True):
        ids = part.index.to_numpy()
        if part.patient_id.nunique() != 1 or sorted(part.vertebra.tolist()) != list(range(1, 12)):
            raise ValueError("Offline replay requires all 11 landmarks per image; evaluate with --identity-guard instead")
        # set_index sorting makes YOLO/HRNet arrays identical in image/vertebra order.
        yy = baseline.loc[ids, ['pred_x', 'pred_y']].to_numpy()
        hh = part[['pred_x', 'pred_y']].to_numpy()
        gt = part[['gt_x', 'gt_y']].to_numpy()
        out, rejected, nearest_yolo = identity_guard(yy, hh)
        guarded = part.copy()
        guarded['mode'] = 'yolo_hrnet_guarded'
        guarded[['pred_x', 'pred_y']] = out
        guarded['error_px'] = np.linalg.norm(out - gt, axis=1)
        guarded['component_mae_px'] = np.abs(out - gt).mean(1)
        corrected.append(guarded)
        near_gt = np.linalg.norm(hh[:, None] - gt[None, :], axis=-1)
        near_yolo_gt = np.linalg.norm(yy[:, None] - gt[None, :], axis=-1)
        finite_h = np.isfinite(hh).all(1)
        finite_y = np.isfinite(yy).all(1)
        vertices = part.vertebra.to_numpy()
        for i, row in enumerate(part.itertuples()):
            nearest_gt = int(vertices[near_gt[i].argmin()]) if finite_h[i] else None
            wrong_yolo = bool(vertices[near_yolo_gt[i].argmin()] != row.vertebra) if finite_y[i] else None
            decisions.append(dict(image=image, patient_id=row.patient_id, vertebra=row.vertebra,
                                  rejected=bool(rejected[i]),
                                  nearest_yolo_vertebra=int(vertices[nearest_yolo[i]]) if nearest_yolo[i] >= 0 else None))
            diagnostics.append(dict(image=image, patient_id=row.patient_id, vertebra=row.vertebra,
                                    error_px=row.error_px, nearest_gt_vertebra=nearest_gt,
                                    wrong_nearest_gt=(nearest_gt != row.vertebra) if nearest_gt else None,
                                    yolo_wrong_nearest_gt=wrong_yolo))
    combined = pd.concat([baseline, source, *corrected], ignore_index=True)
    return combined, pd.DataFrame(decisions), pd.DataFrame(diagnostics)


def audit_refinement(evaluations, out, *, data=None, manifest=None, overlays=6):
    if (data is None) != (manifest is None):
        raise ValueError("Visual review requires both data and manifest")
    if not isinstance(overlays, int) or overlays < 1:
        raise ValueError("Overlay count must be a positive integer")
    paths = [Path(p).resolve() for p in evaluations]
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Evaluation directory names must be distinct")
    out = Path(out).resolve()
    if any(out == p or p in out.parents for p in paths):
        raise ValueError("Audit output must be outside source evaluations")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Audit output must be empty")
    prepared, shared = [], None
    for path in paths:
        source = path / 'predictions.csv'
        summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
        if summary['protocol']['split'] != 'val':
            raise ValueError("Refinement exploration is validation only")
        saved_frame = pd.read_csv(source, dtype={'patient_id': str})
        if 'mode' not in saved_frame or not set(saved_frame['mode']).issubset(
            {'yolo', 'yolo_hrnet', 'yolo_hrnet_guarded'}
        ):
            raise ValueError("Unsupported saved evaluation modes")
        frame, decisions, diagnostic = replay(saved_frame)
        saved_guard = saved_frame[saved_frame['mode'] == 'yolo_hrnet_guarded'].set_index(KEY).sort_index()
        if not saved_guard.empty:
            expected = frame[frame['mode'] == 'yolo_hrnet_guarded'].set_index(KEY).sort_index()
            columns = [*META, 'pred_x', 'pred_y', 'error_px', 'component_mae_px']
            if (saved_guard.index.has_duplicates or not saved_guard.index.equals(expected.index)
                    or not np.allclose(saved_guard[columns].to_numpy(float), expected[columns].to_numpy(float),
                                       rtol=1e-7, atol=1e-6, equal_nan=True)):
                raise ValueError("Saved guarded predictions disagree with identity-guard replay")
        raw = frame[frame['mode'] == 'yolo']
        if (raw.image.nunique(), raw.patient_id.nunique()) != (
            summary['protocol']['images'], summary['protocol']['patients']
        ):
            raise ValueError("Saved protocol counts disagree with prediction rows")
        checked_modes = ('yolo', 'yolo_hrnet', 'yolo_hrnet_guarded') if not saved_guard.empty \
            else ('yolo', 'yolo_hrnet')
        for mode in checked_modes:
            if mode not in summary:
                raise ValueError(f"Missing saved summary for {mode}")
            computed = summarize(frame[frame['mode'] == mode])
            for key, value in computed.items():
                saved = summary[mode][key]
                equal = value is saved if value is None or saved is None else math.isclose(
                    value, saved, rel_tol=1e-7, abs_tol=1e-6)
                if not equal:
                    raise ValueError(f"Saved summary disagrees with coordinates: {mode}/{key}")
        gt = frame[frame['mode'] == 'yolo'].set_index(KEY)[META].sort_index()
        if shared is not None and not gt.equals(shared):
            raise ValueError("Evaluations must use exactly the same validation ground truth")
        shared = gt
        if data is not None and not all(k in summary['protocol'] for k in ('alpha', 'min_roi', 'max_roi')):
            raise ValueError("Visual crop review requires saved ROI parameters")
        labels = validate_visual_sources(frame, data, manifest) if data is not None else None
        prepared.append((path, summary['protocol'], frame, decisions, diagnostic, labels))
    if not prepared:
        raise ValueError("At least one evaluation is required")
    out.mkdir(parents=True, exist_ok=True)
    records, diagnosis = [], []
    for path, protocol, frame, decisions, diagnostic, labels in prepared:
        target = out / path.name
        target.mkdir()
        frame.to_csv(target / 'predictions.csv', index=False)
        decisions.to_csv(target / 'refinement_guard.csv', index=False)
        diagnostic.to_csv(target / 'neighbor_diagnostics.csv', index=False)
        write_identity_audit(frame, target, protocol, labels=labels, data=data, overlays=overlays)
        summaries = {}
        patient_rows = []
        for mode, part in frame.groupby('mode'):
            patients, macro = summarize_patients(part)
            metric = {**summarize(part), 'patient_macro': macro}
            summaries[mode] = metric
            patient_rows.extend({'mode': mode, **row} for row in patients)
            records.append(dict(experiment=path.name, mode=mode,
                                macro_patient_pck5=macro['pck_at_5px_all_visible'],
                                pck5_all_visible=metric['pck_at_5px_all_visible'],
                                mean_px=metric['mean_euclidean_px_on_predictions'],
                                p95_px=metric['p95_px_on_predictions'],
                                failed_points=metric['failed_points'],
                                rejected_refinements=int(decisions.rejected.sum())))
        tail = diagnostic[diagnostic.error_px > 30]
        diagnosis.append(dict(experiment=path.name, errors_over_30=len(tail),
                              wrong_neighbor_over_30=int(tail.wrong_nearest_gt.fillna(False).sum()),
                              newly_wrong_over_30=int((tail.wrong_nearest_gt.fillna(False)
                                                      & ~tail.yolo_wrong_nearest_gt.fillna(True)).sum())))
        summaries['protocol'] = {**protocol, 'identity_guard': 'nearest-yolo', 'replayed_from_csv': True}
        summaries['guard_vs_raw_patient_bootstrap'] = paired_patient_bootstrap(
            frame, baseline='yolo_hrnet', candidate='yolo_hrnet_guarded')
        write_json(target / 'summary.json', summaries)
        pd.DataFrame(patient_rows).to_csv(target / 'by_patient.csv', index=False)
    pd.DataFrame(records).to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame(diagnosis).to_csv(out / 'diagnosis.csv', index=False)
    write_json(out / 'provenance.json', provenance(
        {'rule': 'nearest-yolo', 'exploratory_validation': True, 'test_evaluated': False},
        files=[p / f for p in paths for f in ('predictions.csv', 'summary.json')]
        + ([Path(manifest)] if manifest is not None else [])))
    lines = ['# Diagnosi HRNet e controllo identità', '',
             'Analisi esplorativa sulla validation. Nessun riaddestramento, nessuna selezione con GT.', '',
             '| Esperimento | Metodo | PCK5 soggetti | Media px | P95 px |',
             '|---|---|---:|---:|---:|']
    for r in records:
        mean = 'n/d' if r['mean_px'] is None else f"{r['mean_px']:.3f}"
        p95 = 'n/d' if r['p95_px'] is None else f"{r['p95_px']:.3f}"
        lines.append(f"| {r['experiment']} | {r['mode']} | {r['macro_patient_pck5']:.2%} | {mean} | {p95} |")
    lines += ['', 'Regola: se HRNet è più vicino a un altro landmark YOLO che a quello richiesto, '
              'conservare il punto YOLO originale. Le parità conservano HRNet. '
              'Le predizioni HRNet mancanti restano mancanti.', '',
              'La regola è stata proposta dopo aver visto questi errori: i risultati non sono una '
              'verifica indipendente. Non corregge un landmark già assegnato alla vertebra sbagliata da YOLO. '
              'Le colonne neighbor_diagnostics usano GT solo per spiegare gli errori, dopo le decisioni.', '']
    lines += ['diagnosis.csv and neighbor_diagnostics.csv describe RAW HRNet only. '
              'Each experiment now also has identity_report.md, identity_summary.json and '
              'identity_points/images/by_patient CSVs for all three methods.', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Audit completed: {out / "comparison.csv"}', flush=True)
    return pd.DataFrame(records)
