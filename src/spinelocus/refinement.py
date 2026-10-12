"""Optional identity check using predictions only; no ground truth or fitted threshold."""

import numpy as np


def identity_guard(yolo, refined):
    """Keep YOLO if a refinement lies closer to another available YOLO landmark.

    Missing refinements remain missing. Equal distances retain the refinement.
    The rule can reject a valid correction when YOLO itself mislocates a landmark;
    raw HRNet predictions must therefore remain available for comparison.
    """
    yolo, refined = np.asarray(yolo, float), np.asarray(refined, float)
    if yolo.shape != refined.shape or yolo.ndim != 2 or yolo.shape[1] != 2:
        raise ValueError("YOLO and HRNet must have matching [landmarks, 2] shapes")
    valid_yolo = np.isfinite(yolo).all(1)
    valid_refined = np.isfinite(refined).all(1)
    distance = np.sum((refined[:, None, :] - yolo[None, :, :]) ** 2, axis=-1)
    distance[:, ~valid_yolo] = np.inf
    distance[~valid_refined] = np.inf
    own = np.diag(distance).copy()
    nearest = distance.argmin(1) if len(yolo) else np.array([], dtype=int)
    best = distance.min(1) if len(yolo) else np.array([])
    rejected = valid_yolo & valid_refined & (best < own)
    nearest[~(valid_refined & valid_yolo.any())] = -1
    output = refined.copy()
    output[rejected] = yolo[rejected]
    return output, rejected, nearest
