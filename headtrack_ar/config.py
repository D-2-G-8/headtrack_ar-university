"""
Configuration classes for headtrack_ar package.
"""

from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class OverlayConfig:
    """Configuration for overlay rendering.
    
    Attributes:
        color: Color in BGR format (blue, green, red) as tuple.
        size: Size of the crosshair marker in pixels.
        thickness: Thickness of the marker lines in pixels.
    """
    color: tuple[int, int, int] = (0, 255, 0)  # Green in BGR
    size: int = 20
    thickness: int = 2


@dataclass
class TrackerConfig:
    """Main configuration for HeadTracker.
    
    Attributes:
        source: Video source - int for camera index or str for video file path.
        target_resolution: Target resolution as (width, height) or None to keep original.
        draw_overlay: Whether to draw AR markers on frames.
        overlay_config: Configuration for overlay rendering.
        smoothing_alpha: Alpha parameter for exponential moving average smoothing (0.0 to 1.0).
                        Higher values mean less smoothing. Set to None to disable smoothing.
        min_detection_confidence: Minimum confidence for face detection (0.0 to 1.0).
                                 Lower values = more sensitive but may have false positives.
        model_selection: MediaPipe face detection model selection (0=short-range up to 2m, 
                        1=full-range up to 5m). Use 1 for better detection at distance.
    """
    source: Union[int, str] = 0
    target_resolution: Optional[tuple[int, int]] = (640, 480)
    draw_overlay: bool = True
    overlay_config: Optional[OverlayConfig] = None
    smoothing_alpha: Optional[float] = 0.7
    min_detection_confidence: float = 0.4  # Lowered for better sensitivity
    model_selection: int = 1  # Use full-range model by default (better for distance)
    
    def __post_init__(self):
        """Initialize overlay config if not provided."""
        if self.overlay_config is None:
            self.overlay_config = OverlayConfig()

