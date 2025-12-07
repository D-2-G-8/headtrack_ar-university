"""Basic tests for headtrack_ar package."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from headtrack_ar import HeadTracker, TrackerConfig, HeadInfo, FrameInfo
from headtrack_ar.overlay import draw_crosshair, draw_circle_marker, draw_overlay
from headtrack_ar.config import OverlayConfig


class TestOverlayFunctions(unittest.TestCase):
    """Test overlay rendering functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_point = (320, 240)
    
    def test_draw_crosshair_basic(self):
        """Test basic crosshair drawing."""
        frame_copy = self.test_frame.copy()
        result = draw_crosshair(frame_copy, self.test_point, size=20, thickness=2)
        
        self.assertIs(result, frame_copy)
        
        self.assertFalse(np.all(frame_copy == 0))
    
    def test_draw_crosshair_out_of_bounds(self):
        """Test crosshair drawing with out-of-bounds point."""
        frame_copy = self.test_frame.copy()
        out_point = (1000, 1000)
        result = draw_crosshair(frame_copy, out_point, size=20, thickness=2)
        
        self.assertIsNotNone(result)
    
    def test_draw_circle_marker_basic(self):
        """Test basic circle marker drawing."""
        frame_copy = self.test_frame.copy()
        result = draw_circle_marker(frame_copy, self.test_point, radius=10, thickness=2)
        
        self.assertIs(result, frame_copy)
        
        self.assertFalse(np.all(frame_copy == 0))
    
    def test_draw_overlay_empty_points(self):
        """Test overlay drawing with empty points list."""
        frame_copy = self.test_frame.copy()
        result = draw_overlay(frame_copy, [])
        
        self.assertIsNotNone(result)
    
    def test_draw_overlay_multiple_points(self):
        """Test overlay drawing with multiple points."""
        frame_copy = self.test_frame.copy()
        points = [(100, 100), (200, 200), (300, 300)]
        result = draw_overlay(frame_copy, points, config=OverlayConfig())
        
        self.assertIsNotNone(result)
        self.assertFalse(np.all(frame_copy == 0))


class TestDataStructures(unittest.TestCase):
    """Test data structure classes."""
    
    def test_head_info_creation(self):
        """Test HeadInfo creation."""
        head = HeadInfo(
            bbox=(10, 20, 100, 150),
            forehead_point=(60, 50),
            landmarks=[(10, 10), (20, 20)],
            confidence=0.95
        )
        
        self.assertEqual(head.bbox, (10, 20, 100, 150))
        self.assertEqual(head.forehead_point, (60, 50))
        self.assertEqual(len(head.landmarks), 2)
        self.assertEqual(head.confidence, 0.95)
    
    def test_frame_info_creation(self):
        """Test FrameInfo creation."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        heads = [
            HeadInfo(bbox=(0, 0, 100, 100), forehead_point=(50, 50))
        ]
        
        frame_info = FrameInfo(frame=frame, heads=heads)
        
        self.assertIs(frame_info.frame, frame)
        self.assertEqual(len(frame_info.heads), 1)
        self.assertIsNone(frame_info.raw_frame)
    
    def test_frame_info_default_heads(self):
        """Test FrameInfo with default heads list."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_info = FrameInfo(frame=frame)
        
        self.assertEqual(len(frame_info.heads), 0)
        self.assertIsInstance(frame_info.heads, list)


class TestTrackerConfig(unittest.TestCase):
    """Test TrackerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrackerConfig()
        
        self.assertEqual(config.source, 0)
        self.assertEqual(config.target_resolution, (640, 480))
        self.assertTrue(config.draw_overlay)
        self.assertIsNotNone(config.overlay_config)
        self.assertIsNotNone(config.smoothing_alpha)
    
    def test_custom_config(self):
        """Test custom configuration."""
        overlay_config = OverlayConfig(color=(255, 0, 0), size=30, thickness=3)
        config = TrackerConfig(
            source="test_video.mp4",
            target_resolution=(1280, 720),
            draw_overlay=False,
            overlay_config=overlay_config,
            smoothing_alpha=None
        )
        
        self.assertEqual(config.source, "test_video.mp4")
        self.assertEqual(config.target_resolution, (1280, 720))
        self.assertFalse(config.draw_overlay)
        self.assertEqual(config.overlay_config.color, (255, 0, 0))
        self.assertIsNone(config.smoothing_alpha)


class TestTrackerInitialization(unittest.TestCase):
    """Test HeadTracker initialization."""
    
    @patch('headtrack_ar.tracker.VideoSource')
    @patch('headtrack_ar.tracker.FaceDetector')
    def test_tracker_initialization_mock(self, mock_detector, mock_video_source):
        """Test tracker initialization with mocked dependencies."""
        mock_video_source_instance = MagicMock()
        mock_video_source.return_value = mock_video_source_instance
        
        mock_detector_instance = MagicMock()
        mock_detector.return_value = mock_detector_instance
        
        config = TrackerConfig(source=0)
        
        tracker = HeadTracker(config)
        
        self.assertIsNotNone(tracker.video_source)
        self.assertIsNotNone(tracker.detector)
        
        tracker.release()
    
    def test_tracker_process_frame_mock(self):
        """Test process_frame method with mocked detector."""
        from headtrack_ar.tracker import HeadTracker
        from headtrack_ar.detector import FaceDetector
        
        config = TrackerConfig(source=0, draw_overlay=False)
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source_class, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector_class:
            
            mock_video_source_instance = MagicMock()
            mock_video_source_class.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector_class.return_value = mock_detector_instance
            
            test_head = HeadInfo(
                bbox=(10, 10, 100, 100),
                forehead_point=(60, 30)
            )
            mock_detector_instance.detect.return_value = [test_head]
            
            tracker = HeadTracker(config)
            
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            frame_info = tracker.process_frame(test_frame)
            
            self.assertIsInstance(frame_info, FrameInfo)
            self.assertIsNotNone(frame_info.frame)
            self.assertEqual(len(frame_info.heads), 1)
            self.assertEqual(frame_info.heads[0].forehead_point, (60, 30))
            
            tracker.release()


class TestOverlayConfig(unittest.TestCase):
    """Test OverlayConfig."""
    
    def test_default_overlay_config(self):
        """Test default overlay configuration."""
        config = OverlayConfig()
        
        self.assertEqual(config.color, (0, 255, 0))
        self.assertEqual(config.size, 20)
        self.assertEqual(config.thickness, 2)
    
    def test_custom_overlay_config(self):
        """Test custom overlay configuration."""
        config = OverlayConfig(
            color=(255, 0, 0),
            size=30,
            thickness=3
        )
        
        self.assertEqual(config.color, (255, 0, 0))
        self.assertEqual(config.size, 30)
        self.assertEqual(config.thickness, 3)


if __name__ == '__main__':
    unittest.main()

