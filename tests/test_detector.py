"""
Tests for face detector module.
"""

import unittest
import numpy as np
import cv2

from headtrack_ar.detector import FaceDetector
from headtrack_ar.types import HeadInfo
from tests.test_utils import (
    create_synthetic_frame_with_face,
    create_empty_frame,
    adjust_brightness,
    create_frame_with_rotation
)


class TestFaceDetector(unittest.TestCase):
    """Test FaceDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(
            min_detection_confidence=0.3,
            model_selection=1
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'detector'):
            self.detector.release()
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector.face_detection)
        self.assertIsNotNone(self.detector.face_mesh)
        self.assertEqual(self.detector.min_detection_confidence, 0.3)
        self.assertEqual(self.detector.model_selection, 1)
    
    def test_detect_empty_frame(self):
        """Test detection on empty frame."""
        empty_frame = create_empty_frame()
        heads = self.detector.detect(empty_frame)
        
        self.assertIsInstance(heads, list)
        self.assertEqual(len(heads), 0)
    
    def test_detect_none_frame(self):
        """Test detection with None frame."""
        heads = self.detector.detect(None)
        
        self.assertIsInstance(heads, list)
        self.assertEqual(len(heads), 0)
    
    def test_detect_zero_size_frame(self):
        """Test detection with zero-size frame."""
        zero_frame = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        heads = self.detector.detect(zero_frame)
        
        self.assertIsInstance(heads, list)
        self.assertEqual(len(heads), 0)
    
    def test_detect_synthetic_frame(self):
        """Test detection on synthetic frame with face-like pattern."""
        # Note: MediaPipe may not detect synthetic patterns well,
        # but this tests the error handling
        frame = create_synthetic_frame_with_face()
        heads = self.detector.detect(frame)
        
        self.assertIsInstance(heads, list)
        # May or may not detect synthetic face, but should not crash
    
    def test_detect_multiple_calls(self):
        """Test multiple detection calls."""
        frame = create_synthetic_frame_with_face()
        
        for _ in range(5):
            heads = self.detector.detect(frame)
            self.assertIsInstance(heads, list)
    
    def test_detect_different_resolutions(self):
        """Test detection on different frame resolutions."""
        resolutions = [
            (320, 240),
            (640, 480),
            (1280, 720)
        ]
        
        for width, height in resolutions:
            frame = create_synthetic_frame_with_face(width=width, height=height)
            heads = self.detector.detect(frame)
            self.assertIsInstance(heads, list)
    
    def test_detect_brightness_variations(self):
        """Test detection with different brightness levels."""
        base_frame = create_synthetic_frame_with_face()
        
        brightness_levels = [0.3, 0.5, 1.0, 1.5, 2.0]
        
        for brightness in brightness_levels:
            frame = adjust_brightness(base_frame, brightness)
            heads = self.detector.detect(frame)
            self.assertIsInstance(heads, list)
    
    def test_detect_rotation_variations(self):
        """Test detection with rotated frames."""
        base_frame = create_synthetic_frame_with_face()
        
        angles = [-45, -30, 0, 30, 45]
        
        for angle in angles:
            rotated_frame = create_frame_with_rotation(base_frame, angle)
            heads = self.detector.detect(rotated_frame)
            self.assertIsInstance(heads, list)
    
    def test_forehead_point_calculation(self):
        """Test forehead point calculation logic."""
        # Test with bounding box only (no landmarks)
        x, y, w, h = 100, 100, 200, 250
        frame_width, frame_height = 640, 480
        
        forehead_point = self.detector._calculate_forehead_point(
            x, y, w, h, None, frame_width, frame_height
        )
        
        self.assertIsInstance(forehead_point, tuple)
        self.assertEqual(len(forehead_point), 2)
        self.assertIsInstance(forehead_point[0], int)
        self.assertIsInstance(forehead_point[1], int)
        
        # Check that point is within frame bounds
        self.assertGreaterEqual(forehead_point[0], 0)
        self.assertLess(forehead_point[0], frame_width)
        self.assertGreaterEqual(forehead_point[1], 0)
        self.assertLess(forehead_point[1], frame_height)
    
    def test_forehead_point_with_landmarks(self):
        """Test forehead point calculation with landmarks."""
        x, y, w, h = 100, 100, 200, 250
        frame_width, frame_height = 640, 480
        
        # Create mock landmarks (top points)
        landmarks = [
            (150, 80),  # Top point
            (160, 85),  # Top point
            (140, 90),  # Top point
            (170, 88),  # Top point
            (155, 92),  # Top point
            (200, 300),  # Bottom point (should be ignored)
        ]
        
        forehead_point = self.detector._calculate_forehead_point(
            x, y, w, h, landmarks, frame_width, frame_height
        )
        
        self.assertIsInstance(forehead_point, tuple)
        self.assertEqual(len(forehead_point), 2)
        # Should be in upper part of face
        self.assertLess(forehead_point[1], y + h // 2)
    
    def test_detector_release(self):
        """Test detector resource release."""
        detector = FaceDetector()
        detector.release()
        
        # After release, detector should still exist but resources freed
        self.assertIsNotNone(detector)


class TestFaceDetectorConfigurations(unittest.TestCase):
    """Test different detector configurations."""
    
    def test_short_range_model(self):
        """Test short-range model selection."""
        detector = FaceDetector(model_selection=0, min_detection_confidence=0.5)
        self.assertEqual(detector.model_selection, 0)
        detector.release()
    
    def test_full_range_model(self):
        """Test full-range model selection."""
        detector = FaceDetector(model_selection=1, min_detection_confidence=0.5)
        self.assertEqual(detector.model_selection, 1)
        detector.release()
    
    def test_different_confidence_thresholds(self):
        """Test different confidence thresholds."""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            detector = FaceDetector(min_detection_confidence=threshold)
            self.assertEqual(detector.min_detection_confidence, threshold)
            detector.release()


if __name__ == '__main__':
    unittest.main()
