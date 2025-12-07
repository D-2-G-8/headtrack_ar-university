"""
Type definitions and data structures for headtrack_ar package.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HeadInfo:
    """Information about a detected head in a frame.
    
    Attributes:
        bbox: Bounding box as (x, y, width, height) in frame coordinates.
        forehead_point: Pixel coordinates (x, y) of the point on the forehead.
        landmarks: Optional list of facial landmark points as (x, y) tuples.
        confidence: Optional confidence score for the detection (0.0 to 1.0).
    """
    bbox: tuple[int, int, int, int]
    forehead_point: tuple[int, int]
    landmarks: Optional[list[tuple[int, int]]] = None
    confidence: Optional[float] = None


@dataclass
class FrameInfo:
    """Information about a processed video frame.
    
    Attributes:
        frame: Processed frame (numpy array in BGR format) with overlay if enabled.
        raw_frame: Original frame before processing (None if not preserved).
        heads: List of detected head information.
    """
    frame: np.ndarray
    raw_frame: Optional[np.ndarray] = None
    heads: list[HeadInfo] = None
    
    def __post_init__(self):
        """Initialize heads list if not provided."""
        if self.heads is None:
            self.heads = []

