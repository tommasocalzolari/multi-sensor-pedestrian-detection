"""Convenience entry point for processing an adapted sensor sequence."""

from __future__ import annotations

from pathlib import Path

from .evaluation import run_detector_on_sequence
from .pedestrian_detector import MultiSensorPedestrianDetector


def process_sequence(sequence, checkpoint_path: str | Path) -> dict[int, list[dict]]:
    """Build the detector and process a caller-supplied sequence adapter."""

    detector = MultiSensorPedestrianDetector(checkpoint_path=checkpoint_path)
    return run_detector_on_sequence(sequence, detector)
