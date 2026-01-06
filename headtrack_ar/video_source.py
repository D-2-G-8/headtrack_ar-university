"""
Video source handling for capturing frames from camera or video file.
"""

import logging
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_RESOLUTION_CANDIDATES = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1600, 1200),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
]


class VideoSource:
    """Manages video input from camera or video file.
    
    Attributes:
        source: Video source identifier (int for camera index, str for file path).
        target_resolution: Optional target resolution as (width, height).
        capture_width: Optional camera capture width.
        capture_height: Optional camera capture height.
        force_capture_resolution: Force camera capture resolution.
        cap: OpenCV VideoCapture object.
    """
    
    def __init__(
        self,
        source: Union[int, str],
        target_resolution: Optional[tuple[int, int]] = None,
        capture_width: Optional[int] = None,
        capture_height: Optional[int] = None,
        force_capture_resolution: bool = True
    ):
        """Initialize video source.
        
        Args:
            source: Camera index (int) or video file path (str).
            target_resolution: Optional target resolution (width, height).
            capture_width: Optional camera capture width.
            capture_height: Optional camera capture height.
            force_capture_resolution: Force camera capture resolution.
            
        Raises:
            RuntimeError: If video source cannot be opened.
        """
        self.source = source
        self.target_resolution = target_resolution
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.force_capture_resolution = force_capture_resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()
    
    def _open(self) -> None:
        """Open the video source."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")
            
            if self.force_capture_resolution:
                if self.capture_width is None or self.capture_height is None:
                    logger.warning("force_capture_resolution enabled but capture_width/height missing; using auto resolution")
                    self._apply_auto_resolution(allow_capture_request=False)
                else:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))
            else:
                self._apply_auto_resolution(allow_capture_request=False)
            
            logger.info(f"Video source opened: {self.source}")
            self._log_actual_resolution()
        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            raise RuntimeError(f"Failed to open video source: {self.source}") from e

    def _set_max_camera_resolution(self) -> None:
        """Try to set the highest available camera resolution."""
        if self.cap is None:
            return
        best_area = 0
        best_res = None
        for width, height in MAX_RESOLUTION_CANDIDATES:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width <= 0 or actual_height <= 0:
                continue
            area = actual_width * actual_height
            if area > best_area:
                best_area = area
                best_res = (actual_width, actual_height)
        if best_res is not None:
            best_width, best_height = best_res
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)
            logger.info(f"Auto-selected camera resolution: {best_width}x{best_height}")

    def _apply_auto_resolution(self, allow_capture_request: bool = True) -> None:
        """Apply auto/target resolution selection."""
        if self.cap is None:
            return
        if allow_capture_request and (self.capture_width is not None or self.capture_height is not None):
            if self.capture_width is None or self.capture_height is None:
                logger.warning("Both capture_width and capture_height must be set; using auto resolution")
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))
            return
        if self.target_resolution:
            width, height = self.target_resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return
        if isinstance(self.source, int):
            self._set_max_camera_resolution()

    def _log_actual_resolution(self) -> None:
        """Log actual capture resolution after initialization."""
        if self.cap is None:
            return
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width > 0 and actual_height > 0:
            logger.info(f"Actual capture resolution: {actual_width}x{actual_height}")
    
    def read(self) -> Optional[np.ndarray]:
        """Read next frame from video source.
        
        Returns:
            Frame as numpy array in BGR format, or None if no frame available.
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame from video source")
            return None
        
        if self.target_resolution:
            current_height, current_width = frame.shape[:2]
            target_width, target_height = self.target_resolution
            if current_width != target_width or current_height != target_height:
                frame = cv2.resize(frame, (target_width, target_height))
        
        return frame
    
    def release(self) -> None:
        """Release video source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video source released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
