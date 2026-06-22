"""Non-maximum suppression utilities."""

from __future__ import annotations

import numpy as np


def bbox_iou(box_a, box_b) -> float:
    """Compute IoU between two top-left/height/width bounding boxes."""

    ua1, va1, wa, ha = float(box_a.u), float(box_a.v), float(box_a.w), float(box_a.h)
    ub1, vb1, wb, hb = float(box_b.u), float(box_b.v), float(box_b.w), float(box_b.h)

    ua2, va2 = ua1 + wa, va1 + ha
    ub2, vb2 = ub1 + wb, vb1 + hb

    inter_u1, inter_v1 = max(ua1, ub1), max(va1, vb1)
    inter_u2, inter_v2 = min(ua2, ub2), min(va2, vb2)
    inter_w = max(0.0, inter_u2 - inter_u1)
    inter_h = max(0.0, inter_v2 - inter_v1)
    inter_area = inter_w * inter_h

    union_area = wa * ha + wb * hb - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def non_max_suppression_2d(
    boxes: list,
    scores: np.ndarray,
    confidence_threshold: float = 0.05,
    iou_threshold: float = 0.4,
) -> list[int]:
    """Return indices of kept boxes after confidence filtering and 2D NMS."""

    scores = np.asarray(scores).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")

    valid_indices = np.where(scores >= confidence_threshold)[0]
    sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]

    keep: list[int] = []
    suppressed = np.zeros(len(sorted_indices), dtype=bool)

    for i, current_idx in enumerate(sorted_indices):
        if suppressed[i]:
            continue
        keep.append(int(current_idx))

        for j in range(i + 1, len(sorted_indices)):
            if suppressed[j]:
                continue
            next_idx = sorted_indices[j]
            if bbox_iou(boxes[current_idx], boxes[next_idx]) > iou_threshold:
                suppressed[j] = True

    return keep


def bev_distance_nms(
    pedestrian_dicts: list[dict],
    T_cam_lidar: np.ndarray,
    distance_threshold: float = 0.8,
) -> list[dict]:
    """Suppress duplicate 3D detections that are too close in BEV."""

    if len(pedestrian_dicts) <= 1:
        return pedestrian_dicts

    detections = sorted(pedestrian_dicts, key=lambda det: det["score"], reverse=True)
    T_lidar_cam = np.linalg.inv(T_cam_lidar)

    keep: list[dict] = []
    suppressed = np.zeros(len(detections), dtype=bool)

    for i, det_i in enumerate(detections):
        if suppressed[i]:
            continue
        keep.append(det_i)

        p_cam_i = np.array([*det_i["T_cam_object"][:3, 3], 1.0])
        pos_i = (T_lidar_cam @ p_cam_i)[:2]

        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            p_cam_j = np.array([*detections[j]["T_cam_object"][:3, 3], 1.0])
            pos_j = (T_lidar_cam @ p_cam_j)[:2]
            if np.linalg.norm(pos_i - pos_j) < distance_threshold:
                suppressed[j] = True

    return keep
