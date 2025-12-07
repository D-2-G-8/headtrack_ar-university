"""
Video source handling for capturing frames from camera or video file.
"""

import logging
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSource:
    """Manages video input from camera or video file.
    
    Attributes:
        source: Video source identifier (int for camera index, str for file path).
        target_resolution: Optional target resolution as (width, height).
        cap: OpenCV VideoCapture object.
    """
    
    def __init__(
        self,
        source: Union[int, str],
        target_resolution: Optional[tuple[int, int]] = None
    ):
        """Initialize video source.
        
        Args:
            source: Camera index (int) or video file path (str).
            target_resolution: Optional target resolution (width, height).
            
        Raises:
            RuntimeError: If video source cannot be opened.
        """
        self.source = source
        self.target_resolution = target_resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()
    
    def _open(self) -> None:
        """Open the video source."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")
            
            if self.target_resolution:
                width, height = self.target_resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            logger.info(f"Video source opened: {self.source}")
        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            raise RuntimeError(f"Failed to open video source: {self.source}") from e
    
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

