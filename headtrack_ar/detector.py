"""
Face detection and landmark extraction using MediaPipe.
"""

import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from headtrack_ar.types import HeadInfo

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection and landmark extraction using MediaPipe Face Detection.
    
    This class uses MediaPipe's face detection and face mesh models to detect
    faces and extract facial landmarks for forehead point calculation.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 1):
        """Initialize face detector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold for face detection.
            model_selection: Model selection (0=short-range up to 2m, 1=full-range up to 5m).
            
        Raises:
            RuntimeError: If MediaPipe models cannot be initialized.
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        
        try:
            # Initialize MediaPipe face detection
            # model_selection: 0 for short-range (0.5-2m), 1 for full-range (0.5-5m)
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
            
            # Initialize MediaPipe face mesh for landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
            
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise RuntimeError("Failed to initialize MediaPipe face detection") from e
    
    def detect(self, frame: np.ndarray) -> list[HeadInfo]:
        """Detect faces and extract head information from frame.
        
        Args:
            frame: Input frame in BGR format.
            
        Returns:
            List of HeadInfo objects for each detected face.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided to detector")
            return []
        
        # Validate frame shape (should be 2D or 3D array with valid dimensions)
        try:
            if len(frame.shape) < 2 or len(frame.shape) > 3:
                logger.warning(f"Invalid frame shape: {frame.shape}")
                return []
            if len(frame.shape) == 3 and frame.shape[2] != 3:
                logger.warning(f"Invalid frame shape (expected 3 channels): {frame.shape}")
                return []
        except (AttributeError, TypeError):
            logger.warning("Invalid frame type or shape")
            return []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
        except cv2.error as e:
            logger.warning(f"Error converting frame color space: {e}")
            return []
        
        heads = []
        
        try:
            detection_results = self.face_detection.process(rgb_frame)
            
            # Safely check if detection results are valid
            if detection_results is None:
                return heads
            
            mesh_results = self.face_mesh.process(rgb_frame)
            
            # Check if detections exist and is not empty
            if detection_results.detections:
                landmarks_map = {}
                # Safely extract landmarks from mesh results
                if mesh_results is not None and mesh_results.multi_face_landmarks:
                    try:
                        for idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                            if face_landmarks is None:
                                continue
                            landmarks = []
                            if hasattr(face_landmarks, 'landmark') and face_landmarks.landmark:
                                for landmark in face_landmarks.landmark:
                                    if landmark is not None:
                                        x = int(landmark.x * width)
                                        y = int(landmark.y * height)
                                        landmarks.append((x, y))
                                if landmarks:
                                    landmarks_map[idx] = landmarks
                    except (AttributeError, IndexError, TypeError) as e:
                        logger.warning(f"Error extracting landmarks: {e}")
                        landmarks_map = {}
                
                # Process each detection
                for idx, detection in enumerate(detection_results.detections):
                    try:
                        if detection is None:
                            continue
                        
                        # Safely get bounding box
                        if not hasattr(detection, 'location_data') or detection.location_data is None:
                            continue
                        
                        bbox = detection.location_data.relative_bounding_box
                        if bbox is None:
                            continue
                        
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        
                        # Validate bounding box dimensions
                        if w <= 0 or h <= 0:
                            logger.warning(f"Invalid bounding box dimensions: {w}x{h}")
                            continue
                        
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        landmarks = landmarks_map.get(idx, None)
                        
                        forehead_point = self._calculate_forehead_point(
                            x, y, w, h, landmarks, width, height
                        )
                        
                        # Safely get confidence score
                        confidence = None
                        if hasattr(detection, 'score') and detection.score:
                            try:
                                confidence = float(detection.score[0])
                            except (IndexError, TypeError, ValueError):
                                confidence = None
                        
                        head_info = HeadInfo(
                            bbox=(x, y, w, h),
                            forehead_point=forehead_point,
                            landmarks=landmarks,
                            confidence=confidence
                        )
                        heads.append(head_info)
                        
                    except (AttributeError, IndexError, TypeError, ValueError) as e:
                        logger.warning(f"Error processing detection {idx}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error during face detection: {e}", exc_info=True)
            # Return empty list instead of crashing
            return []
        
        return heads
    
    def _calculate_forehead_point(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        landmarks: Optional[list[tuple[int, int]]],
        frame_width: int,
        frame_height: int
    ) -> tuple[int, int]:
        """Calculate forehead point from bounding box or landmarks.
        
        Args:
            x, y, w, h: Bounding box coordinates.
            landmarks: Optional list of facial landmark points.
            frame_width: Frame width.
            frame_height: Frame height.
            
        Returns:
            Forehead point coordinates (x, y).
        """
        try:
            if landmarks is not None and len(landmarks) > 0:
                # MediaPipe face mesh has specific landmark indices
                # For forehead, we use landmarks around the forehead area
                # Landmark 10 is typically the top of the head, but we need to find forehead
                
                # Use approximate positions: between eyes area and above
                # MediaPipe landmarks: 10 (forehead top), 151 (chin), etc.
                # We'll use a simple heuristic: center of top part of face
                
                # Find top-most landmark points (forehead area)
                # Filter out invalid points first
                valid_landmarks = [
                    (int(px), int(py))
                    for px, py in landmarks
                    if isinstance(px, (int, float)) and isinstance(py, (int, float))
                ]
                
                if valid_landmarks:
                    top_points = sorted(valid_landmarks, key=lambda p: p[1])[:5]  # Top 5 points
                    
                    if top_points:
                        avg_x = sum(p[0] for p in top_points) // len(top_points)
                        avg_y = sum(p[1] for p in top_points) // len(top_points)
                        avg_y = int(avg_y + h * 0.15)
                        
                        avg_x = max(0, min(avg_x, frame_width - 1))
                        avg_y = max(0, min(avg_y, frame_height - 1))
                        
                        return (avg_x, avg_y)
        except (TypeError, ValueError, IndexError) as e:
            logger.warning(f"Error calculating forehead point from landmarks: {e}")
        
        # Fallback to bounding box calculation
        try:
            forehead_x = x + w // 2
            forehead_y = y + int(h * 0.2)
            
            forehead_x = max(0, min(forehead_x, frame_width - 1))
            forehead_y = max(0, min(forehead_y, frame_height - 1))
            
            return (forehead_x, forehead_y)
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating forehead point from bbox: {e}")
            # Ultimate fallback - center of frame
            return (frame_width // 2, frame_height // 2)
    
    def release(self) -> None:
        """Release detector resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("Face detector released")

