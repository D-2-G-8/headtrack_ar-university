"""
headtrack_ar - Python package for human head tracking and AR marker overlay.

A library for real-time head tracking in video streams with AR marker overlay
capabilities. Designed for AR interfaces, camera auto-framing, posture training,
streaming, and other peaceful applications.
"""

from headtrack_ar.config import TrackerConfig
from headtrack_ar.tracker import HeadTracker
from headtrack_ar.types import FrameInfo, HeadInfo

__version__ = "0.1.0"
__all__ = [
    "HeadTracker",
    "TrackerConfig",
    "HeadInfo",
    "FrameInfo",
]

