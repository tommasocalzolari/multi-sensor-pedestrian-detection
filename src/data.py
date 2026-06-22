"""Public data structures used by the detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorFrame:
    """One synchronized camera, LiDAR, and radar observation."""

    index: int
    camera_image: np.ndarray
    lidar_points: np.ndarray
    radar_points: np.ndarray
    radar_measurements: np.ndarray
    radar_compensated_radial_velocities: np.ndarray
    T_camera_lidar: np.ndarray
    T_camera_radar: np.ndarray
    camera_projection_matrix: np.ndarray
    ground_plane_camera: np.ndarray
