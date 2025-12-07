"""
Tests for error handling and edge cases.
"""

import unittest
import numpy as np

from headtrack_ar import HeadTracker, TrackerConfig
from headtrack_ar.detector import FaceDetector
from headtrack_ar.video_source import VideoSource
from headtrack_ar.overlay import draw_overlay, draw_crosshair
from tests.test_utils import create_empty_frame, create_synthetic_frame_with_face


class TestDetectorErrorHandling(unittest.TestCase):
    """Test error handling in detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector()
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'detector'):
            self.detector.release()
    
    def test_detector_none_input(self):
        """Test detector with None input."""
        heads = self.detector.detect(None)
        self.assertIsInstance(heads, list)
        self.assertEqual(len(heads), 0)
    
    def test_detector_empty_array(self):
        """Test detector with empty array."""
        empty_frame = create_empty_frame()
        heads = self.detector.detect(empty_frame)
        self.assertIsInstance(heads, list)
        self.assertEqual(len(heads), 0)
    
    def test_detector_invalid_shape(self):
        """Test detector with invalid array shape."""
        invalid_frame = np.array([1, 2, 3])  # Wrong shape (1D array)
        heads = self.detector.detect(invalid_frame)
        # Should handle gracefully and return empty list
        self.assertIsInstance(heads, list)
        self.assertEqual(len(heads), 0)
    
    def test_detector_very_large_frame(self):
        """Test detector with very large frame."""
        large_frame = create_synthetic_frame_with_face(width=4000, height=3000)
        heads = self.detector.detect(large_frame)
        self.assertIsInstance(heads, list)
    
    def test_detector_very_small_frame(self):
        """Test detector with very small frame."""
        small_frame = create_synthetic_frame_with_face(width=64, height=48)
        heads = self.detector.detect(small_frame)
        self.assertIsInstance(heads, list)
    
    def test_detector_repeated_calls(self):
        """Test detector stability with repeated calls."""
        frame = create_synthetic_frame_with_face()
        
        for _ in range(100):
            heads = self.detector.detect(frame)
            self.assertIsInstance(heads, list)
            # Should not crash or leak memory


class TestVideoSourceErrorHandling(unittest.TestCase):
    """Test error handling in video source."""
    
    def test_video_source_invalid_path(self):
        """Test video source with invalid file path."""
        invalid_path = "/nonexistent/path/to/video.mp4"
        
        with self.assertRaises(RuntimeError):
            VideoSource(source=invalid_path)
    
    def test_video_source_read_after_release(self):
        """Test reading after release."""
        # This will fail if no camera, but that's okay
        try:
            source = VideoSource(source=0)
            source.release()
            
            frame = source.read()
            self.assertIsNone(frame)
        except RuntimeError:
            # No camera available, skip
            self.skipTest("No camera available")
    
    def test_video_source_double_release(self):
        """Test releasing video source twice."""
        try:
            source = VideoSource(source=0)
            source.release()
            source.release()  # Should not crash
        except RuntimeError:
            self.skipTest("No camera available")


class TestTrackerErrorHandling(unittest.TestCase):
    """Test error handling in tracker."""
    
    def test_tracker_none_frame(self):
        """Test tracker with None frame."""
        config = TrackerConfig(source=0, draw_overlay=False)
        
        from unittest.mock import patch, MagicMock
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(config)
            
            frame_info = tracker.process_frame(None)
            
            self.assertIsNotNone(frame_info)
            self.assertEqual(len(frame_info.heads), 0)
            
            tracker.release()
    
    def test_tracker_empty_frame(self):
        """Test tracker with empty frame."""
        config = TrackerConfig(source=0, draw_overlay=False)
        
        from unittest.mock import patch, MagicMock
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(config)
            
            empty_frame = create_empty_frame()
            frame_info = tracker.process_frame(empty_frame)
            
            self.assertIsNotNone(frame_info)
            self.assertEqual(len(frame_info.heads), 0)
            
            tracker.release()
    
    def test_tracker_detector_returns_none(self):
        """Test tracker when detector returns None."""
        config = TrackerConfig(source=0, draw_overlay=False)
        
        from unittest.mock import patch, MagicMock
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            mock_detector_instance.detect.return_value = None  # Simulate error
            
            tracker = HeadTracker(config)
            
            test_frame = create_synthetic_frame_with_face()
            frame_info = tracker.process_frame(test_frame)
            
            self.assertIsNotNone(frame_info)
            self.assertEqual(len(frame_info.heads), 0)  # Should handle None gracefully
            
            tracker.release()
    
    def test_tracker_detector_exception(self):
        """Test tracker when detector raises exception."""
        config = TrackerConfig(source=0, draw_overlay=False)
        
        from unittest.mock import patch, MagicMock
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            mock_detector_instance.detect.side_effect = Exception("Test exception")
            
            tracker = HeadTracker(config)
            
            test_frame = create_synthetic_frame_with_face()
            # Should handle exception gracefully
            frame_info = tracker.process_frame(test_frame)
            
            self.assertIsNotNone(frame_info)
            # Should return empty heads list on error
            self.assertEqual(len(frame_info.heads), 0)
            
            tracker.release()
    
    def test_tracker_invalid_head_data(self):
        """Test tracker with invalid head data."""
        config = TrackerConfig(source=0, draw_overlay=True)
        
        from unittest.mock import patch, MagicMock
        from headtrack_ar.types import HeadInfo
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            # Create head with invalid forehead point
            invalid_head = HeadInfo(
                bbox=(100, 100, 200, 250),
                forehead_point=None  # Invalid
            )
            mock_detector_instance.detect.return_value = [invalid_head]
            
            tracker = HeadTracker(config)
            
            test_frame = create_synthetic_frame_with_face()
            frame_info = tracker.process_frame(test_frame)
            
            self.assertIsNotNone(frame_info)
            # Should handle invalid data gracefully
            self.assertIsInstance(frame_info.heads, list)
            
            tracker.release()
    
    def test_tracker_double_release(self):
        """Test releasing tracker twice."""
        config = TrackerConfig(source=0)
        
        from unittest.mock import patch, MagicMock
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(config)
            tracker.release()
            tracker.release()  # Should not crash


class TestOverlayErrorHandling(unittest.TestCase):
    """Test error handling in overlay functions."""
    
    def test_draw_overlay_none_frame(self):
        """Test draw_overlay with None frame."""
        result = draw_overlay(None, [(100, 100)])
        self.assertIsNone(result)
    
    def test_draw_overlay_empty_frame(self):
        """Test draw_overlay with empty frame."""
        empty_frame = create_empty_frame()
        result = draw_overlay(empty_frame, [(100, 100)])
        self.assertIsNotNone(result)
    
    def test_draw_overlay_none_points(self):
        """Test draw_overlay with None points."""
        frame = create_synthetic_frame_with_face()
        result = draw_overlay(frame, None)
        self.assertIsNotNone(result)
    
    def test_draw_overlay_empty_points(self):
        """Test draw_overlay with empty points list."""
        frame = create_synthetic_frame_with_face()
        result = draw_overlay(frame, [])
        self.assertIsNotNone(result)
    
    def test_draw_overlay_invalid_points(self):
        """Test draw_overlay with invalid points."""
        frame = create_synthetic_frame_with_face()
        
        invalid_points = [
            None,
            (100,),  # Too few coordinates
            (100, 200, 300),  # Too many coordinates
            ("invalid", "data"),  # Wrong type
            (1000, 1000),  # Out of bounds (should be clamped)
        ]
        
        result = draw_overlay(frame, invalid_points)
        self.assertIsNotNone(result)
    
    def test_draw_crosshair_none_frame(self):
        """Test draw_crosshair with None frame."""
        result = draw_crosshair(None, (100, 100))
        self.assertIsNone(result)
    
    def test_draw_crosshair_none_point(self):
        """Test draw_crosshair with None point."""
        frame = create_synthetic_frame_with_face()
        result = draw_crosshair(frame, None)
        self.assertIsNotNone(result)
    
    def test_draw_crosshair_invalid_point(self):
        """Test draw_crosshair with invalid point."""
        frame = create_synthetic_frame_with_face()
        
        invalid_points = [
            (100,),  # Too few coordinates
            (100, 200, 300),  # Too many
            ("invalid", "data"),  # Wrong type
        ]
        
        for point in invalid_points:
            result = draw_crosshair(frame, point)
            self.assertIsNotNone(result)
    
    def test_draw_crosshair_out_of_bounds(self):
        """Test draw_crosshair with out-of-bounds point."""
        frame = create_synthetic_frame_with_face()
        
        # Points far outside frame bounds
        out_of_bounds_points = [
            (-100, 100),
            (100, -100),
            (10000, 100),
            (100, 10000),
        ]
        
        for point in out_of_bounds_points:
            result = draw_crosshair(frame, point)
            self.assertIsNotNone(result)  # Should clamp, not crash


class TestConfigurationErrorHandling(unittest.TestCase):
    """Test error handling with various configurations."""
    
    def test_tracker_invalid_source(self):
        """Test tracker with invalid video source."""
        config = TrackerConfig(source="/nonexistent/video.mp4")
        
        with self.assertRaises(RuntimeError):
            tracker = HeadTracker(config)
            # Should fail during initialization
    
    def test_tracker_extreme_resolution(self):
        """Test tracker with extreme resolution."""
        # Very small resolution
        config = TrackerConfig(
            source=0,
            target_resolution=(32, 24)
        )
        
        from unittest.mock import patch, MagicMock
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(config)
            
            # Should initialize successfully
            self.assertIsNotNone(tracker)
            
            tracker.release()
    
    def test_tracker_zero_confidence(self):
        """Test tracker with zero confidence threshold."""
        config = TrackerConfig(
            source=0,
            min_detection_confidence=0.0
        )
        
        from unittest.mock import patch, MagicMock
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(config)
            
            # Should work (may have more false positives)
            self.assertIsNotNone(tracker)
            
            tracker.release()


if __name__ == '__main__':
    unittest.main()
