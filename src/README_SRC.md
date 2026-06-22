# Source Package

The package implements a dataset-agnostic multi-sensor pedestrian-detection pipeline:

1. LiDAR and radar filtering
2. Ground-plane removal
3. Pillarization and DBSCAN clustering
4. 3D candidate-box generation
5. Camera projection
6. Binary CNN patch scoring
7. Image-plane and BEV suppression
8. Sequence execution, export, and visualization

## Modules

| File | Purpose |
|------|---------|
| `data.py` | Public `SensorFrame` input structure. |
| `config.py` | Tunable filtering, clustering, scoring, and suppression settings. |
| `transforms.py` | Plane, point, and projection transformations. |
| `point_cloud_processing.py` | LiDAR ROI filtering, ground removal, radar filtering, and pillarization. |
| `bounding_boxes.py` | Local bounding-box types, 3D candidate generation, and 2D projection. |
| `classifier.py` | Binary MobileNetV2 loading, preprocessing, and scoring. |
| `nms.py` | Image IoU and BEV distance suppression. |
| `pedestrian_detector.py` | Main `MultiSensorPedestrianDetector` pipeline. |
| `visualization.py` | Camera and bird's-eye-view visualization. |
| `evaluation.py` | Sequence execution, proposal conversion, metrics, and JSON export. |
| `run_sequence.py` | Convenience function for processing an adapted sequence. |

## Integration

Convert synchronized sensor data into `SensorFrame` instances. The detector requires a binary MobileNetV2 checkpoint trained with the preprocessing implemented in `classifier.py`.

```python
from src import MultiSensorPedestrianDetector, SensorFrame, run_detector_on_sequence

frames: list[SensorFrame] = load_and_adapt_frames()
detector = MultiSensorPedestrianDetector(checkpoint_path="weights/pedestrian_mobilenet_v2.pt")
frame_detections = run_detector_on_sequence(frames, detector)
```
