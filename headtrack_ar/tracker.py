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
            try:
                frame = self.video_source.read()
                if frame is None:
                    logger.warning("No more frames available")
                    break
                
                frame_info = self.process_frame(frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.debug(f"Processed {frame_count} frames")
                
                yield frame_info
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}", exc_info=True)
                # Continue processing instead of crashing
                frame_count += 1
                # Yield an empty frame info to keep the loop going
                try:
                    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    yield FrameInfo(frame=empty_frame, raw_frame=empty_frame, heads=[])
                except Exception:
                    # If even this fails, break the loop
                    logger.error("Fatal error, breaking loop")
                    break
    
    def process_frame(self, frame: np.ndarray) -> FrameInfo:
        """Process a single frame: detect heads, smooth coordinates, draw overlay.
        
        Args:
            frame: Input frame in BGR format.
            
        Returns:
            FrameInfo containing processed frame and head information.
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame provided")
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame
                return FrameInfo(frame=empty_frame, raw_frame=empty_frame, heads=[])
            
            raw_frame = frame.copy() if self.config.draw_overlay else None
            
            # Safely detect heads - this should always return a list, never None
            if self.detector is None:
                logger.error("Detector not initialized")
                return FrameInfo(frame=frame, raw_frame=raw_frame, heads=[])
            
            heads = self.detector.detect(frame)
            
            # Ensure heads is always a list
            if heads is None:
                logger.warning("Detector returned None, using empty list")
                heads = []
            
            # Apply smoothing if enabled
            if self.config.smoothing_alpha is not None and heads:
                try:
                    heads = self._apply_smoothing(heads)
                except Exception as e:
                    logger.warning(f"Error applying smoothing: {e}, using unsmoothed heads")
                    # Continue with unsmoothed heads
            
            # Draw overlay if enabled
            if self.config.draw_overlay:
                try:
                    forehead_points = [head.forehead_point for head in heads if head.forehead_point]
                    frame = draw_overlay(
                        frame,
                        forehead_points,
                        config=self.config.overlay_config
                    )
                except Exception as e:
                    logger.warning(f"Error drawing overlay: {e}")
                    # Continue without overlay
            
            return FrameInfo(
                frame=frame,
                raw_frame=raw_frame,
                heads=heads
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in process_frame: {e}", exc_info=True)
            # Return a safe fallback frame info
            try:
                fallback_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                return FrameInfo(frame=fallback_frame, raw_frame=fallback_frame, heads=[])
            except Exception:
                # Ultimate fallback
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return FrameInfo(frame=empty_frame, raw_frame=empty_frame, heads=[])
    
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
        
        if alpha is None:
            return heads
        
        try:
            for head in heads:
                if head is None:
                    continue
                
                # Safely get forehead point
                if not hasattr(head, 'forehead_point') or head.forehead_point is None:
                    logger.warning("Head missing forehead_point, skipping smoothing")
                    smoothed_heads.append(head)
                    continue
                
                try:
                    fx, fy = head.forehead_point
                    
                    # Validate coordinates
                    if not isinstance(fx, (int, float)) or not isinstance(fy, (int, float)):
                        logger.warning(f"Invalid forehead point coordinates: ({fx}, {fy})")
                        smoothed_heads.append(head)
                        continue
                    
                    head_id = self._assign_head_id(head, int(fx), int(fy))
                    
                    # Safely get or create state (defaultdict should handle this, but be explicit)
                    if head_id not in self.smoothing_states:
                        self.smoothing_states[head_id] = {'x': None, 'y': None}
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
                    
                except (TypeError, ValueError, AttributeError) as e:
                    logger.warning(f"Error smoothing head: {e}, using original")
                    smoothed_heads.append(head)
            
            # Clean up old smoothing states (only remove states that are not in current active heads)
            # Don't remove states we just created in this iteration
            try:
                active_ids = set()
                for h in heads:
                    if h and hasattr(h, 'forehead_point') and h.forehead_point:
                        try:
                            fx, fy = h.forehead_point
                            if isinstance(fx, (int, float)) and isinstance(fy, (int, float)):
                                hid = self._assign_head_id(h, int(fx), int(fy))
                                active_ids.add(hid)
                        except Exception:
                            continue
                
                # Only remove states that are not in active_ids
                # Keep states that were just created in this iteration
                keys_to_remove = [k for k in self.smoothing_states.keys() if k not in active_ids]
                for key in keys_to_remove:
                    del self.smoothing_states[key]
                    
            except Exception as e:
                logger.warning(f"Error cleaning up smoothing states: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in _apply_smoothing: {e}", exc_info=True)
            return heads  # Return original heads on error
        
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

