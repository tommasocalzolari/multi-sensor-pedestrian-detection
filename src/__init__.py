"""Multi-sensor pedestrian detection package."""

from __future__ import annotations

from .bounding_boxes import BoundingBox
from .data import SensorFrame

__all__ = [
    "BoundingBox",
    "SensorFrame",
    "MultiSensorPedestrianDetector",
    "run_detector_on_sequence",
]


def __getattr__(name: str):
    """Load model-dependent objects only when they are requested."""

    if name == "MultiSensorPedestrianDetector":
        from .pedestrian_detector import MultiSensorPedestrianDetector

        return MultiSensorPedestrianDetector
    if name == "run_detector_on_sequence":
        from .evaluation import run_detector_on_sequence

        return run_detector_on_sequence
    raise AttributeError(name)
