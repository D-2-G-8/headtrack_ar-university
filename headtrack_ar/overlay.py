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
    try:
        if frame is None or frame.size == 0:
            logger.warning("Cannot draw crosshair on empty frame")
            return frame
        
        if point is None or len(point) != 2:
            logger.warning(f"Invalid point provided: {point}")
            return frame
        
        try:
            x, y = point
            x = int(x) if isinstance(x, (int, float)) else 0
            y = int(y) if isinstance(y, (int, float)) else 0
        except (TypeError, ValueError) as e:
            logger.warning(f"Error parsing point coordinates: {e}")
            return frame
        
        if len(frame.shape) < 2:
            logger.warning("Invalid frame shape")
            return frame
        
        height, width = frame.shape[:2]
        
        # Validate and clamp coordinates
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        # Validate size and thickness
        size = max(1, min(size, min(width, height) // 2))
        thickness = max(1, thickness)
        
        # Draw horizontal line
        x1 = max(0, x - size)
        x2 = min(width - 1, x + size)
        cv2.line(frame, (x1, y), (x2, y), color, thickness)
        
        # Draw vertical line
        y1 = max(0, y - size)
        y2 = min(height - 1, y + size)
        cv2.line(frame, (x, y1), (x, y2), color, thickness)
        
    except Exception as e:
        logger.error(f"Error drawing crosshair: {e}", exc_info=True)
    
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
    try:
        if frame is None or frame.size == 0:
            logger.warning("Cannot draw overlay on empty frame")
            return frame
        
        if points is None:
            return frame
        
        if config is None:
            config = OverlayConfig()
        
        # Filter out invalid points
        valid_points = []
        for point in points:
            if point is not None and isinstance(point, (tuple, list)) and len(point) >= 2:
                try:
                    # Validate point coordinates are numeric
                    x, y = point[0], point[1]
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        valid_points.append((int(x), int(y)))
                except (TypeError, ValueError, IndexError):
                    logger.warning(f"Skipping invalid point: {point}")
                    continue
        
        # Draw crosshair for each valid point
        for point in valid_points:
            try:
                draw_crosshair(
                    frame,
                    point,
                    size=config.size,
                    thickness=config.thickness,
                    color=config.color
                )
            except Exception as e:
                logger.warning(f"Error drawing crosshair for point {point}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Error in draw_overlay: {e}", exc_info=True)
    
    return frame

