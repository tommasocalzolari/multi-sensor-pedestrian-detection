"""Geometric transformation and projection utilities."""

from __future__ import annotations

import numpy as np


def transform_plane(plane_model_source: np.ndarray, T_source_target: np.ndarray) -> np.ndarray:
    """Transform plane coefficients from a source frame to a target frame.

    The plane is represented by homogeneous coefficients ``[a, b, c, d]`` such
    that ``plane @ point = 0``. If ``T_source_target`` maps target-frame points
    into the source frame, then the target-frame plane is obtained by
    ``plane_source @ T_source_target``.
    """

    plane_model_source = np.asarray(plane_model_source, dtype=np.float32).reshape(1, 4)
    T_source_target = np.asarray(T_source_target, dtype=np.float32)
    return plane_model_source.dot(T_source_target)[0]


def project_points(projection_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project homogeneous 3D camera-frame points into image coordinates.

    Parameters
    ----------
    projection_matrix:
        Camera projection matrix with shape ``(3, 4)``.
    points:
        Homogeneous 3D points with shape ``(N, 4)``.

    Returns
    -------
    np.ndarray
        Integer pixel coordinates with shape ``(N, 2)`` in ``[u, v]`` order.
    """

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points must have shape (N, 4), got {points.shape}")

    projected = (projection_matrix @ points.T).T
    depth = projected[:, 2]
    valid_depth = np.abs(depth) > 1e-8
    if not np.all(valid_depth):
        projected = projected[valid_depth]
        depth = depth[valid_depth]

    u = projected[:, 0] / depth
    v = projected[:, 1] / depth
    return np.round(np.stack([u, v], axis=1)).astype(int)


def transform_points(T_target_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    """Transform homogeneous points from source frame to target frame."""

    points_source = np.asarray(points_source)
    if points_source.ndim != 2 or points_source.shape[1] != 4:
        raise ValueError(f"points_source must have shape (N, 4), got {points_source.shape}")
    return (T_target_source @ points_source.T).T
