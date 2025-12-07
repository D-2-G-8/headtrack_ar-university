"""
Tests for various conditions: lighting, rotation angles, etc.
"""

import unittest
import numpy as np

from headtrack_ar import HeadTracker, TrackerConfig
from headtrack_ar.detector import FaceDetector
from tests.test_utils import (
    create_synthetic_frame_with_face,
    adjust_brightness,
    create_frame_with_rotation
)


class TestLightingConditions(unittest.TestCase):
    """Test detection under various lighting conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(min_detection_confidence=0.3)
        self.base_frame = create_synthetic_frame_with_face()
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'detector'):
            self.detector.release()
    
    def test_bright_lighting(self):
        """Test detection in bright lighting conditions."""
        bright_frame = adjust_brightness(self.base_frame, 1.5)
        heads = self.detector.detect(bright_frame)
        
        self.assertIsInstance(heads, list)
        # May or may not detect, but should not crash
    
    def test_dim_lighting(self):
        """Test detection in dim lighting conditions."""
        dim_frame = adjust_brightness(self.base_frame, 0.4)
        heads = self.detector.detect(dim_frame)
        
        self.assertIsInstance(heads, list)
        # Detection may fail in very dim light, but should not crash
    
    def test_very_dim_lighting(self):
        """Test detection in very dim lighting."""
        very_dim_frame = adjust_brightness(self.base_frame, 0.2)
        heads = self.detector.detect(very_dim_frame)
        
        self.assertIsInstance(heads, list)
        # Very dim - detection likely to fail, but should handle gracefully
    
    def test_normal_lighting(self):
        """Test detection in normal lighting conditions."""
        heads = self.detector.detect(self.base_frame)
        
        self.assertIsInstance(heads, list)
    
    def test_brightness_range(self):
        """Test detection across a range of brightness levels."""
        brightness_levels = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
        
        detection_rates = []
        
        for brightness in brightness_levels:
            frame = adjust_brightness(self.base_frame, brightness)
            heads = self.detector.detect(frame)
            
            detection_rates.append(len(heads))
            self.assertIsInstance(heads, list)
        
        # Should handle all brightness levels without crashing
        # Detection rate may vary, but that's expected
        print(f"Detection rates across brightness: {detection_rates}")
    
    def test_room_lighting_simulation(self):
        """Simulate typical room lighting conditions."""
        # Room lighting typically ranges from 0.6 to 1.2 brightness factor
        room_lighting_levels = [0.6, 0.8, 1.0, 1.2]
        
        for brightness in room_lighting_levels:
            frame = adjust_brightness(self.base_frame, brightness)
            heads = self.detector.detect(frame)
            
            self.assertIsInstance(heads, list)
            # In room lighting, should work reasonably well


class TestRotationAngles(unittest.TestCase):
    """Test detection with various head rotation angles."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(min_detection_confidence=0.3)
        self.base_frame = create_synthetic_frame_with_face()
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'detector'):
            self.detector.release()
    
    def test_frontal_face(self):
        """Test detection with frontal face (0 degrees)."""
        heads = self.detector.detect(self.base_frame)
        
        self.assertIsInstance(heads, list)
        # Frontal face should work best
    
    def test_slight_rotation(self):
        """Test detection with slight rotation (±15 degrees)."""
        angles = [-15, 15]
        
        for angle in angles:
            rotated_frame = create_frame_with_rotation(self.base_frame, angle)
            heads = self.detector.detect(rotated_frame)
            
            self.assertIsInstance(heads, list)
            # Slight rotation should still work well
    
    def test_moderate_rotation(self):
        """Test detection with moderate rotation (±30-45 degrees)."""
        angles = [-30, -45, 30, 45]
        
        for angle in angles:
            rotated_frame = create_frame_with_rotation(self.base_frame, angle)
            heads = self.detector.detect(rotated_frame)
            
            self.assertIsInstance(heads, list)
            # Moderate rotation may reduce detection quality
    
    def test_strong_rotation(self):
        """Test detection with strong rotation (±60-70 degrees)."""
        angles = [-60, -70, 60, 70]
        
        for angle in angles:
            rotated_frame = create_frame_with_rotation(self.base_frame, angle)
            heads = self.detector.detect(rotated_frame)
            
            self.assertIsInstance(heads, list)
            # Strong rotation - detection may fail, but should not crash
    
    def test_rotation_range(self):
        """Test detection across a range of rotation angles."""
        angles = [-90, -70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70, 90]
        
        detection_rates = []
        
        for angle in angles:
            rotated_frame = create_frame_with_rotation(self.base_frame, angle)
            heads = self.detector.detect(rotated_frame)
            
            detection_rates.append(len(heads))
            self.assertIsInstance(heads, list)
        
        # Should handle all angles without crashing
        print(f"Detection rates across rotation angles: {detection_rates}")
        
        # Frontal angles (around 0) should generally work better
        frontal_indices = [5, 6, 7]  # -15, 0, 15
        frontal_detections = sum(detection_rates[i] for i in frontal_indices)
        
        # Extreme angles may have lower detection rates
        extreme_indices = [0, 12]  # -90, 90
        extreme_detections = sum(detection_rates[i] for i in extreme_indices)
        
        # Frontal should generally be better (but not always true with synthetic data)
        print(f"Frontal detections: {frontal_detections}, Extreme detections: {extreme_detections}")


class TestCombinedConditions(unittest.TestCase):
    """Test detection under combined challenging conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(min_detection_confidence=0.3)
        self.base_frame = create_synthetic_frame_with_face()
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'detector'):
            self.detector.release()
    
    def test_dim_lighting_with_rotation(self):
        """Test detection with dim lighting and rotation."""
        dim_frame = adjust_brightness(self.base_frame, 0.5)
        rotated_frame = create_frame_with_rotation(dim_frame, 30)
        
        heads = self.detector.detect(rotated_frame)
        
        self.assertIsInstance(heads, list)
        # Challenging conditions, but should not crash
    
    def test_bright_lighting_with_rotation(self):
        """Test detection with bright lighting and rotation."""
        bright_frame = adjust_brightness(self.base_frame, 1.5)
        rotated_frame = create_frame_with_rotation(bright_frame, -45)
        
        heads = self.detector.detect(rotated_frame)
        
        self.assertIsInstance(heads, list)
    
    def test_extreme_conditions(self):
        """Test detection under extreme conditions."""
        # Very dim + strong rotation
        very_dim_frame = adjust_brightness(self.base_frame, 0.3)
        extreme_frame = create_frame_with_rotation(very_dim_frame, 70)
        
        heads = self.detector.detect(extreme_frame)
        
        self.assertIsInstance(heads, list)
        # May not detect, but should handle gracefully


class TestRealWorldScenarios(unittest.TestCase):
    """Test scenarios that simulate real-world usage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TrackerConfig(
            source=0,  # Will be mocked
            target_resolution=(640, 480),
            draw_overlay=True,
            smoothing_alpha=0.7,
            min_detection_confidence=0.4
        )
    
    def test_head_movement_simulation(self):
        """Simulate head movement during tracking."""
        from unittest.mock import patch, MagicMock
        from headtrack_ar.types import HeadInfo
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(self.config)
            
            # Simulate head moving across frame
            positions = [
                (150, 200),  # Left side
                (250, 200),  # Center-left
                (320, 200),  # Center
                (400, 200),  # Center-right
                (500, 200),  # Right side
            ]
            
            for x, y in positions:
                test_frame = create_synthetic_frame_with_face(
                    face_center=(x, y)
                )
                
                mock_head = HeadInfo(
                    bbox=(x-75, y-100, 150, 200),
                    forehead_point=(x, y-50)
                )
                mock_detector_instance.detect.return_value = [mock_head]
                
                frame_info = tracker.process_frame(test_frame)
                
                self.assertIsNotNone(frame_info)
                if frame_info.heads:
                    self.assertIsNotNone(frame_info.heads[0].forehead_point)
            
            tracker.release()
    
    def test_temporary_face_loss(self):
        """Test handling of temporary face loss (face leaves frame)."""
        from unittest.mock import patch, MagicMock
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(self.config)
            
            test_frame = create_synthetic_frame_with_face()
            
            # Simulate: face present -> face lost -> face present again
            scenarios = [
                [True],   # Face present
                [False],  # Face lost
                [False],  # Still lost
                [True],   # Face returns
            ]
            
            for has_face in scenarios:
                if has_face[0]:
                    from headtrack_ar.types import HeadInfo
                    mock_head = HeadInfo(
                        bbox=(100, 100, 200, 250),
                        forehead_point=(200, 150)
                    )
                    mock_detector_instance.detect.return_value = [mock_head]
                else:
                    mock_detector_instance.detect.return_value = []
                
                frame_info = tracker.process_frame(test_frame)
                
                self.assertIsNotNone(frame_info)
                if has_face[0]:
                    self.assertGreater(len(frame_info.heads), 0)
                else:
                    self.assertEqual(len(frame_info.heads), 0)
            
            tracker.release()
    
    def test_multiple_faces(self):
        """Test tracking multiple faces simultaneously."""
        from unittest.mock import patch, MagicMock
        from headtrack_ar.types import HeadInfo
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(self.config)
            
            # Simulate two faces
            heads = [
                HeadInfo(bbox=(50, 50, 150, 200), forehead_point=(125, 100)),
                HeadInfo(bbox=(400, 100, 150, 200), forehead_point=(475, 150)),
            ]
            mock_detector_instance.detect.return_value = heads
            
            test_frame = create_synthetic_frame_with_face()
            frame_info = tracker.process_frame(test_frame)
            
            self.assertIsNotNone(frame_info)
            self.assertEqual(len(frame_info.heads), 2)
            
            # Both should have forehead points
            forehead_points = [
                h.forehead_point for h in frame_info.heads if h.forehead_point
            ]
            self.assertEqual(len(forehead_points), 2)
            
            tracker.release()


if __name__ == '__main__':
    unittest.main()
