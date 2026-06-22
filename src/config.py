"""Configuration constants for the pedestrian detection pipeline.

The values in this file capture the tuned operating point while keeping the
pipeline explicit and easy to adjust.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LidarFilterConfig:
    """LiDAR region-of-interest and ground-removal settings."""

    x_min: float = 3.0
    x_max: float = 41.0
    y_min: float = -15.0
    y_max: float = 15.0
    ground_distance_threshold: float = 0.15


@dataclass(frozen=True)
class RadarFilterConfig:
    """Radar filtering settings used to keep human-like radar returns."""

    moving_speed_threshold: float = 0.1
    moving_rcs_min: float = -20.0
    moving_rcs_max: float = 5.0
    static_rcs_min: float = -10.0
    static_rcs_max: float = 2.0


@dataclass(frozen=True)
class ClusteringConfig:
    """Pillarization, DBSCAN, and 3D box filtering parameters."""

    pillar_size: float = 0.1
    min_points_per_pillar: int = 2
    dbscan_eps: float = 0.2
    dbscan_min_samples: int = 6
    min_height: float = 0.5
    max_height: float = 2.5
    max_horizontal_extent: float = 1.8
    radar_required_until_x: float = 25.0
    radar_margin_x: float = 1.0
    radar_margin_y: float = 0.8
    radar_margin_z: float = 1.0
    min_points_close_range: int = 8


@dataclass(frozen=True)
class PatchClassificationConfig:
    """Image-patch classification and non-maximum suppression settings."""

    confidence_threshold: float = 0.05
    iou_threshold: float = 0.4
    bev_distance_threshold: float = 0.8
    min_patch_height_px: int = 10
    min_patch_width_px: int = 5


LIDAR_FILTER_CONFIG = LidarFilterConfig()
RADAR_FILTER_CONFIG = RadarFilterConfig()
CLUSTERING_CONFIG = ClusteringConfig()
PATCH_CLASSIFICATION_CONFIG = PatchClassificationConfig()
