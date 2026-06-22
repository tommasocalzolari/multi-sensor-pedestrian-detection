"""3D and 2D bounding-box utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ClusteringConfig
from .transforms import project_points


@dataclass(frozen=True)
class BoundingBox:
    """Image bounding box stored as top, left, height, and width."""

    v: int
    u: int
    h: int
    w: int


def create_3d_box_corners(extent: np.ndarray) -> np.ndarray:
    """Return the eight homogeneous corners of a bottom-centered 3D box."""

    length, width, height = np.asarray(extent, dtype=np.float32)
    half_length = length / 2.0
    half_width = width / 2.0
    return np.array(
        [
            [-half_length, -half_width, 0.0, 1.0],
            [-half_length, half_width, 0.0, 1.0],
            [half_length, half_width, 0.0, 1.0],
            [half_length, -half_width, 0.0, 1.0],
            [-half_length, -half_width, height, 1.0],
            [-half_length, half_width, height, 1.0],
            [half_length, half_width, height, 1.0],
            [half_length, -half_width, height, 1.0],
        ],
        dtype=np.float32,
    )


def aabb_from_points(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an axis-aligned 3D bounding box from a set of points.

    Returns the bottom-center point and the box extent ``[dx, dy, dz]`` in the
    same frame as the input points.
    """

    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    extent = (maxs - mins).astype(np.float32)
    bottom_center = np.array(
        [
            (mins[0] + maxs[0]) / 2.0,
            (mins[1] + maxs[1]) / 2.0,
            mins[2],
        ],
        dtype=np.float32,
    )
    return bottom_center, extent


def attach_box_to_ground_plane(
    bottom_center_lidar: np.ndarray,
    points_xyz: np.ndarray,
    ground_plane_lidar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap the bottom of a candidate box to the ground plane."""

    a, b, c, d = ground_plane_lidar
    cx, cy = float(bottom_center_lidar[0]), float(bottom_center_lidar[1])
    z_ground = -(a * cx + b * cy + d) / c
    z_max = float(np.max(points_xyz[:, 2]))

    corrected_bottom_center = bottom_center_lidar.copy().astype(np.float32)
    corrected_bottom_center[2] = z_ground

    raw_extent = points_xyz.max(axis=0) - points_xyz.min(axis=0)
    corrected_extent = np.array(
        [raw_extent[0], raw_extent[1], z_max - z_ground],
        dtype=np.float32,
    )
    return corrected_bottom_center, corrected_extent


def camera_object_transform_from_lidar_bottom_center(
    bottom_center_lidar: np.ndarray,
    T_cam_lidar: np.ndarray,
) -> np.ndarray:
    """Create a camera-frame object transform for a box.

    The object origin is placed at the bottom-center of the box. Object z is
    aligned with LiDAR z, while the translation is represented in camera frame.
    """

    p_lidar = np.array([*bottom_center_lidar, 1.0], dtype=np.float32)
    bottom_center_cam = (T_cam_lidar @ p_lidar)[:3]

    return np.array(
        [
            [0.0, -1.0, 0.0, float(bottom_center_cam[0])],
            [0.0, 0.0, -1.0, float(bottom_center_cam[1])],
            [1.0, 0.0, 0.0, float(bottom_center_cam[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def label_from_aabb_lidar(
    bottom_center_lidar: np.ndarray,
    extent_lidar: np.ndarray,
    label_class: str = "Pedestrian",
) -> dict:
    """Create a label-like dictionary in LiDAR frame for debugging plots."""

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = bottom_center_lidar.astype(np.float32)
    return {
        "label_class": label_class,
        "extent_object": extent_lidar.astype(np.float32),
        "T_cam_object": T,
    }


def get_bounding_box_from_object(object_dict: dict, projection_matrix: np.ndarray):
    """Project a bottom-centered 3D object into a 2D camera bounding box."""

    T_cam_object = object_dict["T_cam_object"]
    corners_object = create_3d_box_corners(object_dict["extent_object"])
    corners_camera = (T_cam_object @ corners_object.T).T
    uvs = project_points(projection_matrix, corners_camera)

    u_min = int(np.min(uvs[:, 0]))
    v_min = int(np.min(uvs[:, 1]))
    u_max = int(np.max(uvs[:, 0]))
    v_max = int(np.max(uvs[:, 1]))

    height = (v_max - v_min) + 2
    width = (u_max - u_min) + 2
    return BoundingBox(v_min, u_min, height, width)


def radar_points_in_candidate_box(
    filtered_radar_lidar: np.ndarray,
    bottom_center_lidar: np.ndarray,
    extent: np.ndarray,
    config: ClusteringConfig,
) -> bool:
    """Check whether a candidate 3D box is supported by nearby radar points."""

    if filtered_radar_lidar.size == 0:
        return False

    cx, cy, cz = bottom_center_lidar
    dx, dy, dz = extent
    x_min, x_max = cx - dx / 2.0, cx + dx / 2.0
    y_min, y_max = cy - dy / 2.0, cy + dy / 2.0
    z_min, z_max = cz, cz + dz

    in_box = (
        (filtered_radar_lidar[:, 0] >= x_min - config.radar_margin_x)
        & (filtered_radar_lidar[:, 0] <= x_max + config.radar_margin_x)
        & (filtered_radar_lidar[:, 1] >= y_min - config.radar_margin_y)
        & (filtered_radar_lidar[:, 1] <= y_max + config.radar_margin_y)
        & (filtered_radar_lidar[:, 2] >= z_min - config.radar_margin_z)
        & (filtered_radar_lidar[:, 2] <= z_max + config.radar_margin_z)
    )
    return bool(np.any(in_box))


def clusters_to_candidate_boxes(
    lidar_xyz_noground: np.ndarray,
    cluster_labels: np.ndarray,
    pillar_to_point_indices: list[np.ndarray],
    ground_plane_lidar: np.ndarray,
    T_cam_lidar: np.ndarray,
    filtered_radar_lidar: np.ndarray | None,
    config: ClusteringConfig,
) -> list[dict]:
    """Convert DBSCAN clusters into filtered 3D pedestrian candidates."""

    boxes: list[dict] = []
    cluster_ids = [cluster_id for cluster_id in np.unique(cluster_labels) if cluster_id != -1]

    for cluster_id in cluster_ids:
        pillar_ids = np.nonzero(cluster_labels == cluster_id)[0]
        if pillar_ids.size == 0:
            continue

        point_ids = np.concatenate([pillar_to_point_indices[i] for i in pillar_ids], axis=0)
        points_cluster = lidar_xyz_noground[point_ids]
        if points_cluster.size == 0:
            continue

        bottom_center_raw, _ = aabb_from_points(points_cluster)
        bottom_center, extent = attach_box_to_ground_plane(
            bottom_center_raw,
            points_cluster,
            ground_plane_lidar,
        )
        dx, dy, dz = extent

        # Pedestrian-shape constraints.
        if dz < config.min_height or dz > config.max_height:
            continue
        if dx > config.max_horizontal_extent or dy > config.max_horizontal_extent:
            continue
        if dz < dx or dz < dy:
            continue

        # For close-range boxes, require radar support and enough LiDAR points.
        if bottom_center[0] + dx / 2.0 < config.radar_required_until_x:
            if filtered_radar_lidar is not None and not radar_points_in_candidate_box(
                filtered_radar_lidar,
                bottom_center,
                extent,
                config,
            ):
                continue
            if points_cluster.shape[0] < config.min_points_close_range:
                continue

        boxes.append(
            {
                "cluster_id": int(cluster_id),
                "bottom_center_lidar": bottom_center,
                "extent_object": extent.astype(np.float32),
                "T_cam_object": camera_object_transform_from_lidar_bottom_center(
                    bottom_center,
                    T_cam_lidar,
                ),
            }
        )

    return boxes
