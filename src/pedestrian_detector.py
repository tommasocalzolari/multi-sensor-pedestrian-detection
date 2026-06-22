"""Multi-sensor pedestrian detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from .bounding_boxes import clusters_to_candidate_boxes, get_bounding_box_from_object
from .classifier import ImagePatchClassifier, extract_preprocessed_patches, load_patch_classifier, score_patches
from .config import (
    CLUSTERING_CONFIG,
    LIDAR_FILTER_CONFIG,
    PATCH_CLASSIFICATION_CONFIG,
    RADAR_FILTER_CONFIG,
    ClusteringConfig,
    LidarFilterConfig,
    PatchClassificationConfig,
    RadarFilterConfig,
)
from .data import SensorFrame
from .nms import bev_distance_nms, non_max_suppression_2d
from .point_cloud_processing import (
    filter_lidar_roi,
    filter_radar_human_points,
    pillarize_xy_centroids_with_mapping,
    remove_ground_points,
)
from .transforms import transform_plane


class MultiSensorPedestrianDetector:
    """LiDAR/radar proposal generator with camera-patch classification."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        patch_classifier: ImagePatchClassifier | None = None,
        device: torch.device | None = None,
        lidar_config: LidarFilterConfig = LIDAR_FILTER_CONFIG,
        radar_config: RadarFilterConfig = RADAR_FILTER_CONFIG,
        clustering_config: ClusteringConfig = CLUSTERING_CONFIG,
        patch_config: PatchClassificationConfig = PATCH_CLASSIFICATION_CONFIG,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if patch_classifier is None:
            if checkpoint_path is None:
                raise ValueError("Provide checkpoint_path or an initialized patch_classifier")
            patch_classifier = load_patch_classifier(checkpoint_path, self.device)
        self.patch_classifier = patch_classifier.to(self.device).eval()
        self.lidar_config = lidar_config
        self.radar_config = radar_config
        self.clustering_config = clustering_config
        self.patch_config = patch_config
        self.frame: SensorFrame | None = None

    def set_frame(self, frame: SensorFrame) -> None:
        """Set the synchronized sensor frame processed by the next call."""

        self.frame = frame

    def get_pedestrian_dicts(self) -> list[dict]:
        """Return pedestrian detections for the current synchronized frame."""

        if self.frame is None:
            raise RuntimeError("Call set_frame() before detection")

        frame = self.frame
        image = frame.camera_image
        lidar_points = frame.lidar_points
        radar_points = frame.radar_points
        radar_measurements = frame.radar_measurements
        radar_velocities = frame.radar_compensated_radial_velocities

        T_cam_lidar = frame.T_camera_lidar
        T_cam_radar = frame.T_camera_radar
        T_lidar_radar = np.linalg.inv(T_cam_lidar) @ T_cam_radar

        filtered_radar_lidar, _, _, _ = filter_radar_human_points(
            radar_points,
            radar_measurements,
            radar_velocities,
            T_lidar_radar,
            self.radar_config,
        )

        lidar_xyz_roi, _ = filter_lidar_roi(lidar_points, self.lidar_config)
        ground_plane_lidar = transform_plane(frame.ground_plane_camera, T_cam_lidar)
        lidar_xyz_noground, _, _ = remove_ground_points(
            lidar_xyz_roi,
            ground_plane_lidar,
            self.lidar_config.ground_distance_threshold,
        )
        if lidar_xyz_noground.shape[0] == 0:
            return []

        centroids_xy, _, pillar_to_points = pillarize_xy_centroids_with_mapping(
            lidar_xyz_noground,
            pillar_size=self.clustering_config.pillar_size,
            min_points_per_pillar=self.clustering_config.min_points_per_pillar,
            use_xy_only=True,
        )
        if centroids_xy.shape[0] == 0:
            return []

        cluster_labels = DBSCAN(
            eps=self.clustering_config.dbscan_eps,
            min_samples=self.clustering_config.dbscan_min_samples,
        ).fit_predict(centroids_xy)

        candidate_boxes = clusters_to_candidate_boxes(
            lidar_xyz_noground=lidar_xyz_noground,
            cluster_labels=cluster_labels,
            pillar_to_point_indices=pillar_to_points,
            ground_plane_lidar=ground_plane_lidar,
            T_cam_lidar=T_cam_lidar,
            filtered_radar_lidar=filtered_radar_lidar,
            config=self.clustering_config,
        )
        if not candidate_boxes:
            return []

        projection_matrix = frame.camera_projection_matrix
        proposal_bboxes = [get_bounding_box_from_object(box, projection_matrix) for box in candidate_boxes]
        patch_tensors, bbox_indices, _ = extract_preprocessed_patches(
            image,
            proposal_bboxes,
            min_patch_height_px=self.patch_config.min_patch_height_px,
            min_patch_width_px=self.patch_config.min_patch_width_px,
        )
        if not patch_tensors:
            return []

        scores = score_patches(self.patch_classifier, patch_tensors, self.device)
        scored_bboxes = [proposal_bboxes[index] for index in bbox_indices]
        keep_patch_indices = non_max_suppression_2d(
            scored_bboxes,
            scores,
            confidence_threshold=self.patch_config.confidence_threshold,
            iou_threshold=self.patch_config.iou_threshold,
        )

        detections: list[dict] = []
        for patch_index in keep_patch_indices:
            candidate = candidate_boxes[bbox_indices[patch_index]]
            detections.append(
                {
                    "label_class": "Pedestrian",
                    "extent_object": candidate["extent_object"],
                    "T_cam_object": candidate["T_cam_object"],
                    "score": float(scores[patch_index]),
                }
            )

        return bev_distance_nms(
            detections,
            T_cam_lidar=T_cam_lidar,
            distance_threshold=self.patch_config.bev_distance_threshold,
        )
