"""
Main head tracking logic with smoothing and frame processing.
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from headtrack_ar.config import TrackerConfig
from headtrack_ar.detector import FaceDetector
from headtrack_ar.overlay import draw_overlay
from headtrack_ar.types import FrameInfo, HeadInfo
from headtrack_ar.video_source import VideoSource

logger = logging.getLogger(__name__)


class HeadTracker:
    """Main class for tracking heads in video streams.
    
    This class coordinates video capture, face detection, smoothing,
    and overlay rendering to provide a complete head tracking solution.
    """
    
    def __init__(self, config: TrackerConfig):
        """Initialize head tracker.
        
        Args:
            config: Tracker configuration.
            
        Raises:
            RuntimeError: If initialization fails.
        """
        self.config = config
        self.video_source: Optional[VideoSource] = None
        self.detector: Optional[FaceDetector] = None
        self.smoothing_states: dict[int, dict[str, float]] = defaultdict(
            lambda: {'x': None, 'y': None}
        )
        self.head_id_counter = 0
        
        try:
            self.video_source = VideoSource(
                source=config.source,
                target_resolution=config.target_resolution
            )
            
            self.detector = FaceDetector(
                min_detection_confidence=config.min_detection_confidence
            )
            
            logger.info("HeadTracker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HeadTracker: {e}")
            self.release()
            raise RuntimeError("Failed to initialize HeadTracker") from e
    
    def run(self):
        """Generator that yields FrameInfo for each processed frame.
        
        Yields:
            FrameInfo objects containing processed frame and head information.
        """
        if self.video_source is None or self.detector is None:
            raise RuntimeError("Tracker not properly initialized")
        
        frame_count = 0
        
        while True:
            frame = self.video_source.read()
            if frame is None:
                logger.warning("No more frames available")
                break
            
            frame_info = self.process_frame(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.debug(f"Processed {frame_count} frames")
            
            yield frame_info
    
    def process_frame(self, frame: np.ndarray) -> FrameInfo:
        """Process a single frame: detect heads, smooth coordinates, draw overlay.
        
        Args:
            frame: Input frame in BGR format.
            
        Returns:
            FrameInfo containing processed frame and head information.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided")
            return FrameInfo(frame=frame, raw_frame=frame, heads=[])
        
        raw_frame = frame.copy() if self.config.draw_overlay else None
        
        heads = self.detector.detect(frame)
        
        if self.config.smoothing_alpha is not None:
            heads = self._apply_smoothing(heads)
        
        if self.config.draw_overlay:
            frame = draw_overlay(
                frame,
                [head.forehead_point for head in heads],
                config=self.config.overlay_config
            )
        
        return FrameInfo(
            frame=frame,
            raw_frame=raw_frame,
            heads=heads
        )
    
    def _apply_smoothing(self, heads: list[HeadInfo]) -> list[HeadInfo]:
        """Apply exponential moving average smoothing to forehead points.
        
        Args:
            heads: List of detected heads.
            
        Returns:
            List of heads with smoothed forehead points.
        """
        if not heads:
            return heads
        
        smoothed_heads = []
        alpha = self.config.smoothing_alpha
        
        for head in heads:
            fx, fy = head.forehead_point
            
            head_id = self._assign_head_id(head, fx, fy)
            
            state = self.smoothing_states[head_id]
            
            if state['x'] is None or state['y'] is None:
                state['x'] = float(fx)
                state['y'] = float(fy)
            else:
                state['x'] = alpha * fx + (1 - alpha) * state['x']
                state['y'] = alpha * fy + (1 - alpha) * state['y']
            
            smoothed_point = (int(state['x']), int(state['y']))
            
            smoothed_head = HeadInfo(
                bbox=head.bbox,
                forehead_point=smoothed_point,
                landmarks=head.landmarks,
                confidence=head.confidence
            )
            smoothed_heads.append(smoothed_head)
        
        active_ids = set(self._assign_head_id(h, *h.forehead_point) for h in heads)
        self.smoothing_states = {
            k: v for k, v in self.smoothing_states.items() if k in active_ids
        }
        
        return smoothed_heads
    
    def _assign_head_id(self, head: HeadInfo, fx: int, fy: int) -> int:
        """Assign or find ID for a head based on proximity to existing states.
        
        Args:
            head: Head information.
            fx, fy: Forehead point coordinates.
            
        Returns:
            Head ID.
        """
        threshold = 50  # pixels
        
        for head_id, state in self.smoothing_states.items():
            if state['x'] is not None and state['y'] is not None:
                dx = abs(fx - state['x'])
                dy = abs(fy - state['y'])
                distance = (dx**2 + dy**2)**0.5
                
                if distance < threshold:
                    return head_id
        
        new_id = self.head_id_counter
        self.head_id_counter += 1
        return new_id
    
    def release(self) -> None:
        """Release all resources."""
        if self.detector is not None:
            self.detector.release()
            self.detector = None
        
        if self.video_source is not None:
            self.video_source.release()
            self.video_source = None
        
        self.smoothing_states.clear()
        logger.info("HeadTracker released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

