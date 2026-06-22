"""Sequence-level execution, evaluation, and export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc

from .bounding_boxes import get_bounding_box_from_object
from .visualization import get_positions_and_scores


def run_detector_on_sequence(sequence, pedestrian_detector) -> dict[int, list[dict]]:
    """Run a detector over an iterable of synchronized sensor frames."""

    frame_pedestrian_dicts: dict[int, list[dict]] = {}
    for frame in sequence:
        pedestrian_detector.set_frame(frame)
        frame_pedestrian_dicts[frame.index] = pedestrian_detector.get_pedestrian_dicts()
    return frame_pedestrian_dicts


def build_sequence_proposals(frame_pedestrian_dicts: dict, sequence) -> list:
    """Project frame detections into image-plane bounding boxes."""

    sequence_proposals = []
    for frame in sequence:
        detections = frame_pedestrian_dicts[frame.index]
        projection_matrix = frame.camera_projection_matrix
        sequence_proposals.append(
            [get_bounding_box_from_object(det, projection_matrix) for det in detections]
        )
    return sequence_proposals


def average_precision_from_metrics(metrics_dict: dict, threshold: float) -> float:
    """Compute AP from an array of ``[precision, recall]`` rows."""

    precisions, recalls = np.asarray(metrics_dict[threshold]).T
    return float(auc(recalls, precisions))


def build_bev_proposals(frame_pedestrian_dicts: dict) -> dict[int, list[dict]]:
    """Convert detections into frame-indexed BEV center/score records."""

    frame_positions, frame_scores = get_positions_and_scores(frame_pedestrian_dicts)
    return {
        frame_index: [
            {"center": position.tolist(), "score": float(score)}
            for position, score in zip(frame_positions[frame_index], frame_scores[frame_index])
        ]
        for frame_index in frame_positions
    }


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def save_detections(frame_pedestrian_dicts: dict, output_path: str | Path) -> Path:
    """Serialize frame-indexed detections as readable JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_compatible(frame_pedestrian_dicts), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
