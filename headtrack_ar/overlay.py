"""
Overlay rendering functions for drawing AR markers on frames.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from headtrack_ar.config import OverlayConfig

logger = logging.getLogger(__name__)


def draw_crosshair(
    frame: np.ndarray,
    point: tuple[int, int],
    size: int = 20,
    thickness: int = 2,
    color: tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw a crosshair marker at the specified point.
    
    Args:
        frame: Input frame in BGR format (will be modified in-place).
        point: Center point (x, y) for the crosshair.
        size: Size of the crosshair in pixels (half-length of each line).
        thickness: Thickness of the lines in pixels.
        color: Color in BGR format (blue, green, red).
        
    Returns:
        Frame with crosshair drawn (same reference as input).
    """
    if frame is None or frame.size == 0:
        logger.warning("Cannot draw crosshair on empty frame")
        return frame
    
    x, y = point
    height, width = frame.shape[:2]
    
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    x1 = max(0, x - size)
    x2 = min(width - 1, x + size)
    cv2.line(frame, (x1, y), (x2, y), color, thickness)
    
    y1 = max(0, y - size)
    y2 = min(height - 1, y + size)
    cv2.line(frame, (x, y1), (x, y2), color, thickness)
    
    return frame


def draw_circle_marker(
    frame: np.ndarray,
    point: tuple[int, int],
    radius: int = 10,
    thickness: int = 2,
    color: tuple[int, int, int] = (0, 255, 0),
    filled: bool = False
) -> np.ndarray:
    """Draw a circle marker at the specified point.
    
    Args:
        frame: Input frame in BGR format (will be modified in-place).
        point: Center point (x, y) for the circle.
        radius: Radius of the circle in pixels.
        thickness: Thickness of the circle outline (-1 for filled circle).
        color: Color in BGR format.
        filled: Whether to fill the circle (thickness=-1).
        
    Returns:
        Frame with circle drawn (same reference as input).
    """
    if frame is None or frame.size == 0:
        logger.warning("Cannot draw circle on empty frame")
        return frame
    
    x, y = point
    height, width = frame.shape[:2]
    
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    draw_thickness = -1 if filled else thickness
    cv2.circle(frame, (x, y), radius, color, draw_thickness)
    
    return frame


def draw_overlay(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    config: Optional[OverlayConfig] = None
) -> np.ndarray:
    """Draw overlay markers for all provided points.
    
    Args:
        frame: Input frame in BGR format (will be modified in-place).
        points: List of points (x, y) to draw markers at.
        config: Overlay configuration. If None, uses default config.
        
    Returns:
        Frame with overlays drawn (same reference as input).
    """
    if frame is None or frame.size == 0:
        logger.warning("Cannot draw overlay on empty frame")
        return frame
    
    if config is None:
        config = OverlayConfig()
    
    for point in points:
        draw_crosshair(
            frame,
            point,
            size=config.size,
            thickness=config.thickness,
            color=config.color
        )
    
    return frame

