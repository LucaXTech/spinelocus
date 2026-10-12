"""An optional two-view YOLO coordinate mean before any HRNet crop."""

import numpy as np

from .ensemble import mean_coordinates


def flip_protocol():
    return {
        'views': ['original', 'horizontal_flip'],
        'aggregation': 'equal_mean_global_coordinates',
        'missing_policy': 'require_all_views',
        'reflection': 'x_original = image_width - x_flipped',
        'identity_permutation': list(range(1, 12)),
        'order': 'before_roi_and_hrnet',
        'uses_ground_truth': False,
    }


def yolo_view_points(model, image, args, device, extract):
    pixels = np.array(image)[:, :, ::-1].copy()
    options = dict(imgsz=args.imgsz, conf=args.conf, max_det=1, device=device, verbose=False)
    original = extract(model.predict(pixels, **options)[0], args.keypoint_conf)
    if not getattr(args, 'yolo_flip_tta', False):
        return original, {}
    flipped = extract(model.predict(pixels[:, ::-1].copy(), **options)[0], args.keypoint_conf).copy()
    # Match keypoint augmentation in the pinned Ultralytics v8.3.199 (Instances.fliplr).
    # Spine identities T1-T11 retain their order, unlike contralateral human-pose joints.
    flipped[:, 0] = image.width - flipped[:, 0]
    return mean_coordinates([original, flipped]), {'yolo_original': original, 'yolo_horizontal_flip': flipped}
