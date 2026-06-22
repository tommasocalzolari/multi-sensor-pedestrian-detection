"""Visualization helpers for camera and bird's-eye views."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .bounding_boxes import BoundingBox, get_bounding_box_from_object


def draw_bbox_to_image(
    image: np.ndarray,
    bbox: BoundingBox,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> None:
    """Draw one bounding box in-place on a BGR image."""

    top_left = (int(bbox.u), int(bbox.v))
    bottom_right = (int(bbox.u + bbox.w), int(bbox.v + bbox.h))
    cv2.rectangle(image, top_left, bottom_right, color, thickness)


def draw_pedestrian_bounding_boxes(frame_pedestrian_dicts: dict, sequence) -> list[np.ndarray]:
    """Draw projected detections for each synchronized frame."""

    images_draw: list[np.ndarray] = []
    for frame in sequence:
        frame_index = frame.index
        image_draw = frame.camera_image.copy()
        projection_matrix = frame.camera_projection_matrix
        for detection in frame_pedestrian_dicts[frame_index]:
            bbox = get_bounding_box_from_object(detection, projection_matrix)
            draw_bbox_to_image(image_draw, bbox)
        images_draw.append(image_draw)
    return images_draw


def get_positions_and_scores(frame_pedestrian_dicts: dict) -> tuple[dict, dict]:
    """Extract camera-frame lateral/depth positions and confidence scores."""

    frame_positions: dict[int, np.ndarray] = {}
    frame_scores: dict[int, np.ndarray] = {}
    for frame_index, detections in frame_pedestrian_dicts.items():
        frame_positions[frame_index] = np.asarray(
            [detection["T_cam_object"][[0, 2], 3] for detection in detections],
            dtype=np.float32,
        ).reshape(-1, 2)
        frame_scores[frame_index] = np.asarray(
            [detection["score"] for detection in detections],
            dtype=np.float32,
        ).reshape(-1)
    return frame_positions, frame_scores


def create_bev_animation(frame_positions: dict[int, np.ndarray]) -> FuncAnimation:
    """Create an animation of predicted positions in bird's-eye view."""

    non_empty = [positions for positions in frame_positions.values() if positions.size > 0]
    if non_empty:
        all_points = np.vstack(non_empty)
        lateral_max, depth_max = np.max(np.abs(all_points), axis=0)
    else:
        lateral_max, depth_max = 10.0, 20.0

    figure, axis = plt.subplots(figsize=(10, 8))

    def plot_frame(frame_index: int) -> None:
        axis.clear()
        axis.set_xlim(-lateral_max - 2.0, lateral_max + 2.0)
        axis.set_ylim(0.0, depth_max + 2.0)
        axis.set_aspect("equal")
        axis.set_xlabel("lateral position")
        axis.set_ylabel("forward position")
        axis.set_title(f"frame: {frame_index}")
        axis.grid(True, alpha=0.5)
        axis.scatter(0.0, 0.0, color="red")
        positions = frame_positions[frame_index]
        if positions.size > 0:
            axis.scatter(positions[:, 0], positions[:, 1])

    animation = FuncAnimation(figure, plot_frame, frames=list(frame_positions))
    plt.close(figure)
    return animation
