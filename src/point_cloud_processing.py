"""LiDAR, radar, and point-cloud preprocessing utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import LidarFilterConfig, RadarFilterConfig


def subtract_3d_points(points: np.ndarray, points_to_remove: np.ndarray) -> np.ndarray:
    """Return rows from ``points`` that are not present in ``points_to_remove``.

    This is mainly used for debug visualizations, where it is useful to show
    which points were removed by a filtering stage.
    """

    points = np.asarray(points)
    points_to_remove = np.asarray(points_to_remove)
    if points.size == 0 or points_to_remove.size == 0:
        return points

    points_view = np.ascontiguousarray(points).view(
        np.dtype((np.void, points.dtype.itemsize * points.shape[1]))
    )
    remove_view = np.ascontiguousarray(points_to_remove).view(
        np.dtype((np.void, points_to_remove.dtype.itemsize * points_to_remove.shape[1]))
    )
    mask = ~np.isin(points_view, remove_view)
    return points[mask.ravel()]


def filter_lidar_roi(
    lidar_points: np.ndarray,
    config: LidarFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter LiDAR points to a forward road-area region of interest."""

    lidar_xyz = np.asarray(lidar_points)[:, :3]
    mask = (
        (lidar_xyz[:, 0] > config.x_min)
        & (lidar_xyz[:, 0] < config.x_max)
        & (lidar_xyz[:, 1] > config.y_min)
        & (lidar_xyz[:, 1] < config.y_max)
    )
    return lidar_xyz[mask], mask


def remove_ground_points(
    lidar_xyz_roi: np.ndarray,
    ground_plane_lidar: np.ndarray,
    distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove points close to the estimated ground plane.

    Returns non-ground points, ground points, and signed point-to-plane distances.
    """

    a, b, c, d = ground_plane_lidar
    distances = (
        a * lidar_xyz_roi[:, 0]
        + b * lidar_xyz_roi[:, 1]
        + c * lidar_xyz_roi[:, 2]
        + d
    ) / np.sqrt(a**2 + b**2 + c**2)

    non_ground_mask = np.abs(distances) > distance_threshold
    return lidar_xyz_roi[non_ground_mask], lidar_xyz_roi[~non_ground_mask], distances


def filter_radar_human_points(
    radar_points_homogeneous: np.ndarray,
    radar_measurements: np.ndarray,
    radar_compensated_radial_velocities: np.ndarray,
    T_lidar_radar: np.ndarray,
    config: RadarFilterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform radar points into LiDAR frame and keep human-like returns.

    Moving points are accepted with a broader RCS range, while nearly static
    points use a stricter RCS range to retain plausible pedestrian returns while
    removing noise and large reflective structures.
    """

    radar_xyz_lidar = (T_lidar_radar @ radar_points_homogeneous.T).T[:, :3]
    rcs_values = radar_measurements[:, 3]
    speeds = radar_compensated_radial_velocities

    moving_human_mask = (
        (np.abs(speeds) > config.moving_speed_threshold)
        & (rcs_values > config.moving_rcs_min)
        & (rcs_values < config.moving_rcs_max)
    )
    static_human_mask = (
        (np.abs(speeds) <= config.moving_speed_threshold)
        & (rcs_values > config.static_rcs_min)
        & (rcs_values < config.static_rcs_max)
    )
    keep_mask = moving_human_mask | static_human_mask

    return (
        radar_xyz_lidar[keep_mask],
        radar_xyz_lidar[~keep_mask],
        radar_measurements[keep_mask],
        speeds[keep_mask],
    )


def pillarize_xy_centroids_with_mapping(
    xyz: np.ndarray,
    pillar_size: float = 0.25,
    min_points_per_pillar: int = 2,
    use_xy_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Regularize a point cloud into XY pillars and return pillar centroids.

    The returned mapping makes it possible to reconstruct the original 3D points
    belonging to each DBSCAN cluster after clustering the lower-dimensional
    pillar centroids.
    """

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")

    if xyz.shape[0] == 0:
        out_dim = 2 if use_xy_only else 3
        return np.zeros((0, out_dim), dtype=np.float32), np.zeros((0,), dtype=np.int32), []

    pillar_ids_xy = np.floor(xyz[:, :2] / float(pillar_size)).astype(np.int32)
    _, inverse = np.unique(pillar_ids_xy, axis=0, return_inverse=True)
    n_pillars = int(inverse.max()) + 1

    counts_all = np.bincount(inverse, minlength=n_pillars).astype(np.int32)
    keep_ids = np.nonzero(counts_all >= int(min_points_per_pillar))[0]
    if keep_ids.size == 0:
        out_dim = 2 if use_xy_only else 3
        return np.zeros((0, out_dim), dtype=np.float32), np.zeros((0,), dtype=np.int32), []

    sum_x = np.bincount(inverse, weights=xyz[:, 0], minlength=n_pillars)
    sum_y = np.bincount(inverse, weights=xyz[:, 1], minlength=n_pillars)
    sum_z = np.bincount(inverse, weights=xyz[:, 2], minlength=n_pillars)

    cx = sum_x / counts_all
    cy = sum_y / counts_all
    cz = sum_z / counts_all

    if use_xy_only:
        centroids_all = np.stack([cx, cy], axis=1).astype(np.float32)
    else:
        centroids_all = np.stack([cx, cy, cz], axis=1).astype(np.float32)

    pillar_to_point_indices: List[np.ndarray] = []
    for pillar_id in keep_ids:
        pillar_to_point_indices.append(np.nonzero(inverse == pillar_id)[0].astype(np.int32))

    return centroids_all[keep_ids], counts_all[keep_ids], pillar_to_point_indices
