"""Post-inference GT diagnostics and visual review; never correct predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .data import load_manifest, read_label
from .geometry import roi_size, window
from .utils import sha256, write_json


def identity_diagnostics(frame, protocol, coherent_min=8):
    """Require validated, paired T1-T11 rows from refinement_audit.replay.

    Nearest GT is an explanatory label, not an anatomical adjudication. Endpoints
    are censored by the available T1-T11 GT; ties prefer the requested identity.
    """
    if not isinstance(coherent_min, int) or not 2 <= coherent_min <= 11:
        raise ValueError("coherent_min must be an integer from 2 to 11")
    roi_available = all(k in protocol for k in ('alpha', 'min_roi', 'max_roi'))
    if roi_available:
        alpha, minimum, maximum = (protocol[k] for k in ('alpha', 'min_roi', 'max_roi'))
        if not np.isfinite([alpha, minimum, maximum]).all() or alpha <= 0 or not 0 < minimum <= maximum:
            raise ValueError("Invalid saved ROI parameters")
    point_rows, image_rows = [], []
    for image, group in frame.groupby('image', sort=True):
        parts = {mode: part.sort_values('vertebra') for mode, part in group.groupby('mode')}
        base = parts['yolo']
        vertices = base.vertebra.to_numpy()
        if not np.array_equal(vertices, np.arange(1, 12)) or base.patient_id.nunique() != 1:
            raise ValueError("Identity diagnostics require all 11 landmarks per image")
        gt = base[['gt_x', 'gt_y']].to_numpy(float)
        yolo = base[['pred_x', 'pred_y']].to_numpy(float)
        bounds = []
        for i in range(11):
            bound = window(int(base.width.iloc[0]), int(base.height.iloc[0]), *yolo[i],
                           roi_size(yolo, i, alpha, minimum, maximum)) \
                if roi_available and np.isfinite(yolo[i]).all() else None
            bounds.append(bound)
        for mode, part in parts.items():
            if not np.array_equal(part.vertebra, vertices):
                raise ValueError("Paired landmark identities differ")
            pred = part[['pred_x', 'pred_y']].to_numpy(float)
            finite = np.isfinite(pred).all(1)
            distances = np.linalg.norm(pred[:, None] - gt[None, :], axis=-1)
            offsets = np.full(11, np.nan)
            for i, row in enumerate(part.itertuples()):
                nearest = None
                if finite[i]:
                    j = int(distances[i].argmin())
                    if np.isclose(distances[i, i], distances[i, j], rtol=0, atol=1e-8):
                        j = i
                    nearest = int(vertices[j])
                    offsets[i] = nearest - row.vertebra
                bound = bounds[i]
                inside = None if bound is None else bool(
                    bound[0] <= gt[i, 0] < bound[2] and bound[1] <= gt[i, 1] < bound[3])
                contained = None if bound is None else int(np.sum(
                    (gt[:, 0] >= bound[0]) & (gt[:, 0] < bound[2]) &
                    (gt[:, 1] >= bound[1]) & (gt[:, 1] < bound[3])))
                point_rows.append({
                    'image': image, 'patient_id': row.patient_id, 'mode': mode, 'vertebra': row.vertebra,
                    'prediction_available': bool(finite[i]), 'error_px': row.error_px,
                    'over_30px': bool(row.error_px > 30),
                    'nearest_gt_vertebra': nearest,
                    'nearest_gt_offset': int(offsets[i]) if finite[i] else None,
                    'nearest_gt_distance_px': float(distances[i, nearest - 1]) if nearest is not None else None,
                    'requested_gt_inside_yolo_crop': inside, 'gt_landmarks_in_yolo_crop': contained,
                    **{k: bound[n] if bound else None for n, k in enumerate(('x1', 'y1', 'x2', 'y2'))},
                })
            plus, minus = int(np.sum(offsets == 1)), int(np.sum(offsets == -1))
            errors = part.error_px.to_numpy(float)
            valid_errors = errors[np.isfinite(errors)]
            image_rows.append({
                'image': image, 'patient_id': base.patient_id.iloc[0], 'mode': mode,
                'visible_points': 11, 'predicted_points': int(finite.sum()),
                'failed_points': int((~finite).sum()), 'errors_over_30': int(np.sum(errors > 30)),
                'mean_px_on_predictions': float(valid_errors.mean()) if len(valid_errors) else None,
                'plus_one_nearest_gt': plus, 'minus_one_nearest_gt': minus,
                'coherent_adjacent_shift': bool(max(plus, minus) >= coherent_min),
                'coherent_shift_direction': (1 if plus >= coherent_min else -1 if minus >= coherent_min else 0),
            })
    points, images = pd.DataFrame(point_rows), pd.DataFrame(image_rows)
    for column in ('nearest_gt_vertebra', 'nearest_gt_offset', 'gt_landmarks_in_yolo_crop',
                   'x1', 'y1', 'x2', 'y2'):
        points[column] = points[column].astype('Int64')
    patients = images.groupby(['patient_id', 'mode'], as_index=False).agg(
        images=('image', 'size'), visible_points=('visible_points', 'sum'),
        predicted_points=('predicted_points', 'sum'), failed_points=('failed_points', 'sum'),
        errors_over_30=('errors_over_30', 'sum'), coherent_images=('coherent_adjacent_shift', 'sum'))
    patients['errors_over_30_fraction_all_visible'] = patients.errors_over_30 / patients.visible_points
    summaries = {}
    for mode, part in points.groupby('mode'):
        selected = images[images['mode'] == mode]
        coherent = selected[selected.coherent_adjacent_shift]
        tail = part[part.over_30px]
        summaries[mode] = {
            'images': len(selected), 'patients': int(selected.patient_id.nunique()),
            'visible_points': len(part), 'predicted_points': int(part.prediction_available.sum()),
            'errors_over_30': len(tail),
            'plus_one_over_30': int((tail.nearest_gt_offset == 1).sum()),
            'minus_one_over_30': int((tail.nearest_gt_offset == -1).sum()),
            'coherent_images': len(coherent),
            'coherent_image_fraction': len(coherent) / len(selected),
            'errors_over_30_in_coherent_images': int(coherent.errors_over_30.sum()),
            'available_yolo_crops': int(part.x1.notna().sum()) if roi_available else None,
            'requested_gt_inside_yolo_crop': int(part.requested_gt_inside_yolo_crop.eq(True).sum())
            if roi_available else None,
            'errors_over_30_gt_inside_yolo_crop': int(tail.requested_gt_inside_yolo_crop.eq(True).sum())
            if roi_available else None,
            'errors_over_30_crop_contains_multiple_gt': int((tail.gt_landmarks_in_yolo_crop > 1).sum())
            if roi_available else None,
        }
    key = ['image', 'patient_id', 'vertebra']
    errors = points.pivot(index=key, columns='mode', values='error_px')
    offset = points.pivot(index=key, columns='mode', values='nearest_gt_offset')
    transitions = []
    for before, after in [('yolo', 'yolo_hrnet'), ('yolo_hrnet', 'yolo_hrnet_guarded')]:
        a, b = errors[before], errors[after]
        finite = np.isfinite(a) & np.isfinite(b)
        transitions.append({
            'baseline': before, 'candidate': after, 'paired_predictions': int(finite.sum()),
            'unpaired_predictions': int((~finite).sum()),
            'resolved_over_30': int((finite & (a > 30) & (b <= 30)).sum()),
            'introduced_over_30': int((finite & (a <= 30) & (b > 30)).sum()),
        })
    guarded_bad = errors['yolo_hrnet_guarded'] > 30
    adjacent = offset['yolo_hrnet_guarded'].abs() == 1
    summary = {
        'diagnostic_uses_ground_truth': True, 'changes_predictions': False, 'split': 'val',
        'coherent_min_points': coherent_min, 'coherent_denominator': 11,
        'roi_parameters_verified_from_summary': roi_available,
        'methods': summaries, 'paired_threshold_transitions': transitions,
        'guarded_adjacent_over_30': int((guarded_bad & adjacent).sum()),
        'guarded_adjacent_over_30_same_direction_in_yolo': int(
            (guarded_bad & adjacent & (offset['yolo'] == offset['yolo_hrnet_guarded'])).sum()),
        'guarded_adjacent_over_30_yolo_nearest_own_gt': int(
            (guarded_bad & adjacent & (offset['yolo'] == 0)).sum()),
        'limitations': [
            'Nearest GT does not establish anatomical label correctness.',
            'T1/T11 endpoints have no GT beyond the selected interval.',
            'Images and reconstructions from one subject are correlated.',
            'These exploratory validation subsets cannot replace whole-cohort metrics.',
            'Crop contents do not prove which visual features caused a model decision.',
        ],
    }
    return points, images, patients, summary


def validate_visual_sources(frame, data, manifest_path):
    """Check only the validation labels used by this CSV; never open test files."""
    root = Path(data).resolve()
    manifest = load_manifest(manifest_path).set_index('image')
    labels = {}
    for image, part in frame[frame['mode'] == 'yolo'].groupby('image', sort=True):
        if image not in manifest.index:
            raise ValueError(f"Image not in manifest: {image}")
        row = manifest.loc[image]
        if row.split != 'val' or row.patient_id != part.patient_id.iloc[0]:
            raise ValueError(f"Manifest split/patient differs from CSV: {image}")
        if not ((part.width == row.width) & (part.height == row.height)).all():
            raise ValueError(f"Manifest dimensions differ from CSV: {image}")
        for relative in (image, row.label):
            if not (root / relative).resolve().is_relative_to(root):
                raise ValueError(f"Manifest path escapes dataset root: {relative}")
        label = root / row.label
        gt, vis = read_label(label, row.width, row.height)
        part = part.sort_values('vertebra')
        if not (vis > 0).all() or not np.allclose(
            gt, part[['gt_x', 'gt_y']].to_numpy(), rtol=0, atol=1e-4
        ):
            raise ValueError(f"Label GT differs from saved CSV: {image}")
        digest = sha256(label)
        if 'label_sha256' in manifest.columns and digest != row.label_sha256:
            raise ValueError(f"Registered label content changed: {image}")
        registered = row.get('image_sha256')
        labels[image] = {'label': row.label, 'label_sha256': digest,
                         'image_sha256_registered': registered if pd.notna(registered) else None}
    return labels


def _font(size=16):
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def export_review_overlays(frame, points, images, labels, data, target, count=6):
    """Show original frames and the exact YOLO windows, without another inference."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("Overlay count must be a positive integer")
    ranked = images[images['mode'] == 'yolo_hrnet_guarded'].sort_values(
        ['coherent_adjacent_shift', 'errors_over_30', 'mean_px_on_predictions', 'image'],
        ascending=[False, False, False, True], na_position='last')
    # First cover different subjects, then take the remaining worst images.
    order = pd.concat([ranked.drop_duplicates('patient_id'), ranked]).drop_duplicates('image').head(count)
    root, target = Path(data).resolve(), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    font, records = _font(), []
    modes = ['yolo', 'yolo_hrnet', 'yolo_hrnet_guarded']
    for number, selected in enumerate(order.itertuples(), 1):
        image = selected.image
        part = frame[frame.image == image]
        base = part[part['mode'] == 'yolo'].sort_values('vertebra')
        source = root / image
        digest = sha256(source)
        registered = labels[image]['image_sha256_registered']
        if registered is not None and digest != registered:
            raise ValueError(f"Registered image content changed: {image}")
        with Image.open(source) as opened:
            original = opened.convert('RGB')
        if original.size != (int(base.width.iloc[0]), int(base.height.iloc[0])):
            raise ValueError(f"Image dimensions changed: {image}")
        candidates = part[part['mode'] == 'yolo_hrnet_guarded']
        adjacent = points[(points.image == image) & (points['mode'] == 'yolo_hrnet_guarded') &
                          (points.nearest_gt_offset.abs() == 1)].vertebra
        if len(adjacent):
            candidates = candidates[candidates.vertebra.isin(adjacent)]
        worst = candidates.sort_values(
            ['error_px', 'vertebra'], ascending=[False, True], na_position='last').iloc[0]
        focus = int(worst.vertebra)
        diagnostic = points[(points.image == image) & (points['mode'] == 'yolo') &
                            (points.vertebra == focus)].iloc[0]
        bound = tuple(int(diagnostic[k]) for k in ('x1', 'y1', 'x2', 'y2')) \
            if pd.notna(diagnostic.x1) else None
        canvas = Image.new('RGB', (1536, 1200), 'white')
        draw = ImageDraw.Draw(canvas)
        for panel, mode in enumerate(modes):
            rows = part[part['mode'] == mode].sort_values('vertebra')
            draw.text((panel*512+8, 8), f'{mode}: mean {rows.error_px.mean():.2f}px', fill='black', font=font)
            draw.text((panel*512+8, 30), 'GT green | prediction red | ROI cyan', fill='black', font=font)
            display = ImageOps.contain(original, (512, 512))
            canvas.paste(display, (panel*512, 60))
            sx, sy = display.width/original.width, display.height/original.height
            if bound is not None:
                draw.rectangle((panel*512+bound[0]*sx, 60+bound[1]*sy,
                                panel*512+(bound[2]-1)*sx, 60+(bound[3]-1)*sy), outline='cyan', width=2)
            for color, fields, prefix in [('lime', ('gt_x', 'gt_y'), 'T'), ('red', ('pred_x', 'pred_y'), 'P')]:
                for row in rows.itertuples():
                    x, y = (getattr(row, field) for field in fields)
                    if np.isfinite([x, y]).all():
                        x, y = panel*512+x*sx, 60+y*sy
                        radius = 3 if prefix == 'T' else 5
                        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
                        draw.text((x+6, y-14 if prefix == 'T' else y+3), f'{prefix}{row.vertebra}',
                                  fill=color, font=font)
            draw.text((panel*512+8, 590), f'T{focus} YOLO window; original pixels {bound}',
                      fill='black', font=font)
            if bound is None:
                draw.text((panel*512+8, 630), 'Crop unavailable: missing YOLO point', fill='black', font=font)
                continue
            patch = original.crop(bound).resize((512, 512), Image.Resampling.BILINEAR)
            canvas.paste(patch, (panel*512, 640))
            draw.text((panel*512+8, 615), 'Requested GT green | other GT cyan | prediction red',
                      fill='black', font=font)
            marks = [(r.gt_x, r.gt_y, 'lime' if r.vertebra == focus else 'cyan', f'T{r.vertebra}')
                     for r in rows.itertuples()]
            focused = rows[rows.vertebra == focus].iloc[0]
            # Paint the prediction last so a neighboring GT cannot hide an exact wrong-identity hit.
            marks.append((focused.pred_x, focused.pred_y, 'red', f'P{focus}'))
            for x, y, color, label in marks:
                if np.isfinite([x, y]).all() and bound[0] <= x < bound[2] and bound[1] <= y < bound[3]:
                    x = panel*512+(x-bound[0])*512/(bound[2]-bound[0])
                    y = 640+(y-bound[1])*512/(bound[3]-bound[1])
                    radius = 8 if color == 'red' else 5
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)
                    draw.text((x+12, y+10 if color == 'red' else y-14), label, fill=color, font=font)
        draw.text((8, 1170), f'{selected.patient_id} | diagnostic GT only | focus T{focus}',
                  fill='black', font=font)
        name = f'{number:02d}-{selected.patient_id}.png'
        canvas.save(target / name)
        records.append({'image': image, 'patient_id': selected.patient_id, 'overlay': name,
                        'focus_vertebra': focus, 'image_sha256': digest,
                        'label': labels[image]['label'], 'label_sha256': labels[image]['label_sha256']})
    pd.DataFrame(records).to_csv(target / 'index.csv', index=False)
    return records


def write_identity_audit(frame, target, protocol, *, labels=None, data=None, overlays=6):
    target = Path(target)
    points, images, patients, summary = identity_diagnostics(frame, protocol)
    points.to_csv(target / 'identity_points.csv', index=False)
    images.to_csv(target / 'identity_images.csv', index=False)
    patients.to_csv(target / 'identity_by_patient.csv', index=False)
    summary['local_label_verification'] = {
        'performed': labels is not None, 'validation_images': len(labels) if labels is not None else 0,
        'absolute_coordinate_tolerance_px': 1e-4,
        'establishes_anatomical_label_correctness': False,
    }
    if labels is not None:
        if not summary['roi_parameters_verified_from_summary']:
            raise ValueError("Visual crop review requires saved alpha/min_roi/max_roi in summary.json")
        summary['overlays'] = export_review_overlays(frame, points, images, labels, data,
                                                    target / 'overlays', overlays)
    write_json(target / 'identity_summary.json', summary)
    lines = ['# Identity shift diagnostic', '',
             'Exploratory validation only. GT explains saved predictions; no prediction is changed.', '',
             'Coherent shift: at least 8 of 11 predictions are nearest to the GT with the same adjacent offset.',
             'GT ties retain the requested identity. Missing predictions remain missing.', '',
             '| Method | Errors >30px | Nearest next GT | Nearest previous GT | Coherent images | Errors in them |',
             '|---|---:|---:|---:|---:|---:|']
    for mode, record in summary['methods'].items():
        lines.append(f"| {mode} | {record['errors_over_30']} | {record['plus_one_over_30']} | "
                     f"{record['minus_one_over_30']} | {record['coherent_images']} | "
                     f"{record['errors_over_30_in_coherent_images']} |")
    lines += ['', 'The two nearest-GT columns refer only to errors >30px.', '',
              'A consistently shifted ordered sequence can satisfy the nearest-YOLO guard. '
              'Order or spacing alone cannot establish absolute vertebral identity.', '',
              'Local validation labels were compared with the CSV.' if labels is not None else
              'Local labels and images were unavailable: anatomical/source correctness remains unverified.', '',
              *[f'- {item}' for item in summary['limitations']], '']
    if summary['roi_parameters_verified_from_summary']:
        lines += ['## Reconstructed YOLO windows', '',
                  '| Method | Available crops | Requested GT inside | Errors >30px with GT inside |',
                  '|---|---:|---:|---:|']
        for mode, record in summary['methods'].items():
            lines.append(f"| {mode} | {record['available_yolo_crops']} | "
                         f"{record['requested_gt_inside_yolo_crop']} | "
                         f"{record['errors_over_30_gt_inside_yolo_crop']} |")
        lines.append('')
    (target / 'identity_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary
