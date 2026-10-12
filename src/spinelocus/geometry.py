"""One coordinate convention for crops, training and inference.

Windows are half-open [x1,x2) x [y1,y2). New experiments use pixel-center
coordinates consistent with resizing. Legacy checkpoints retain scale-only mapping.
"""

import numpy as np
from PIL import Image


def window(width, height, cx, cy, size):
    if min(width, height, size) <= 0 or not np.isfinite([cx, cy]).all():
        raise ValueError("Invalid image/window dimensions or center")
    sx, sy = min(int(size), width), min(int(size), height)
    x1 = int(np.clip(round(cx) - sx // 2, 0, width - sx))
    y1 = int(np.clip(round(cy) - sy // 2, 0, height - sy))
    return x1, y1, x1 + sx, y1 + sy


def to_local(xy, bounds, crop_size, convention="pixel_centers"):
    x1, y1, x2, y2 = bounds
    offset = 0.5 if convention == "pixel_centers" else 0.0
    return (np.asarray(xy, dtype=float) - [x1, y1] + offset) * [
        crop_size / (x2 - x1),
        crop_size / (y2 - y1),
    ] - offset


def to_global(xy, bounds, crop_size, convention="pixel_centers"):
    x1, y1, x2, y2 = bounds
    offset = 0.5 if convention == "pixel_centers" else 0.0
    return (
        (np.asarray(xy, dtype=float) + offset) * [(x2 - x1) / crop_size, (y2 - y1) / crop_size]
        + [x1, y1]
        - offset
    )


def crop(image, center, size, output_size=160):
    bounds = window(*image.size, *center, size)
    return image.crop(bounds).resize((output_size, output_size), Image.Resampling.BILINEAR), bounds


def roi_size(points, index, alpha=2.2, minimum=112, maximum=256):
    points = np.asarray(points)
    distances = [
        np.linalg.norm(points[index] - points[j]) for j in (index - 1, index + 1) if 0 <= j < len(points)
    ]
    distances = [d for d in distances if np.isfinite(d) and d > 0]
    spacing = np.median(distances) if distances else 80.0
    return int(round(np.clip(alpha * spacing, minimum, maximum)))
