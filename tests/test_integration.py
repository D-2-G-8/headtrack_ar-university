"""
Integration tests for the complete head tracking pipeline.
"""

import unittest
import numpy as np
import time

from headtrack_ar import HeadTracker, TrackerConfig
from headtrack_ar.config import OverlayConfig
from tests.test_utils import (
    create_test_video_file,
    cleanup_test_video,
    create_synthetic_frame_with_face,
    create_empty_frame
)


class TestHeadTrackerIntegration(unittest.TestCase):
    """Integration tests for HeadTracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_video_path = None
    
    def tearDown(self):
        """Clean up after tests."""
        if self.test_video_path:
            cleanup_test_video(self.test_video_path)
    
    def test_tracker_full_pipeline(self):
        """Test complete tracking pipeline with video file."""
        self.test_video_path = create_test_video_file(num_frames=20)
        
        config = TrackerConfig(
            source=self.test_video_path,
            target_resolution=(640, 480),
            draw_overlay=True,
            smoothing_alpha=0.7
        )
        
        tracker = HeadTracker(config)
        
        frame_count = 0
        head_detections = 0
        
        try:
            for frame_info in tracker.run():
                self.assertIsNotNone(frame_info)
                self.assertIsNotNone(frame_info.frame)
                self.assertIsInstance(frame_info.heads, list)
                
                frame_count += 1
                if frame_info.heads:
                    head_detections += len(frame_info.heads)
                
                # Limit to prevent infinite loops
                if frame_count >= 20:
                    break
        finally:
            tracker.release()
        
        self.assertGreater(frame_count, 0)
    
    def test_tracker_without_overlay(self):
        """Test tracker without overlay rendering."""
        self.test_video_path = create_test_video_file(num_frames=10)
        
        config = TrackerConfig(
            source=self.test_video_path,
            draw_overlay=False
        )
        
        tracker = HeadTracker(config)
        
        frame_count = 0
        try:
            for frame_info in tracker.run():
                self.assertIsNotNone(frame_info)
                self.assertIsNone(frame_info.raw_frame)  # Should be None when overlay disabled
                frame_count += 1
                if frame_count >= 10:
                    break
        finally:
            tracker.release()
    
    def test_tracker_with_custom_overlay(self):
        """Test tracker with custom overlay configuration."""
        self.test_video_path = create_test_video_file(num_frames=10)
        
        overlay_config = OverlayConfig(
            color=(255, 0, 0),  # Red
            size=30,
            thickness=3
        )
        
        config = TrackerConfig(
            source=self.test_video_path,
            overlay_config=overlay_config
        )
        
        tracker = HeadTracker(config)
        
        frame_count = 0
        try:
            for frame_info in tracker.run():
                self.assertIsNotNone(frame_info)
                frame_count += 1
                if frame_count >= 10:
                    break
        finally:
            tracker.release()
    
    def test_tracker_smoothing(self):
        """Test tracker with smoothing enabled."""
        self.test_video_path = create_test_video_file(num_frames=15)
        
        config = TrackerConfig(
            source=self.test_video_path,
            smoothing_alpha=0.5
        )
        
        tracker = HeadTracker(config)
        
        forehead_points = []
        
        try:
            for frame_info in tracker.run():
                if frame_info.heads:
                    for head in frame_info.heads:
                        if head.forehead_point:
                            forehead_points.append(head.forehead_point)
                
                if len(forehead_points) >= 10:
                    break
        finally:
            tracker.release()
        
        # If we got points, smoothing should make them more stable
        if len(forehead_points) > 1:
            # Check that points are reasonable (not all over the place)
            x_coords = [p[0] for p in forehead_points]
            y_coords = [p[1] for p in forehead_points]
            
            # Points should be within frame bounds
            self.assertTrue(all(0 <= x < 640 for x in x_coords))
            self.assertTrue(all(0 <= y < 480 for y in y_coords))
    
    def test_tracker_no_smoothing(self):
        """Test tracker with smoothing disabled."""
        self.test_video_path = create_test_video_file(num_frames=10)
        
        config = TrackerConfig(
            source=self.test_video_path,
            smoothing_alpha=None
        )
        
        tracker = HeadTracker(config)
        
        frame_count = 0
        try:
            for frame_info in tracker.run():
                frame_count += 1
                if frame_count >= 10:
                    break
        finally:
            tracker.release()
    
    def test_tracker_process_frame_directly(self):
        """Test process_frame method directly."""
        config = TrackerConfig(
            source=0,  # Will be mocked
            draw_overlay=True
        )
        
        # Create tracker with mocked video source
        from unittest.mock import patch, MagicMock
        from headtrack_ar.tracker import HeadTracker
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(config)
            
            # Create test frame
            test_frame = create_synthetic_frame_with_face()
            
            # Mock detector to return a head
            from headtrack_ar.types import HeadInfo
            mock_head = HeadInfo(
                bbox=(100, 100, 200, 250),
                forehead_point=(200, 150)
            )
            mock_detector_instance.detect.return_value = [mock_head]
            
            # Process frame
            frame_info = tracker.process_frame(test_frame)
            
            self.assertIsNotNone(frame_info)
            self.assertIsNotNone(frame_info.frame)
            self.assertEqual(len(frame_info.heads), 1)
            self.assertEqual(frame_info.heads[0].forehead_point, (200, 150))
            
            tracker.release()
    
    def test_tracker_empty_frames(self):
        """Test tracker handling of empty frames."""
        config = TrackerConfig(
            source=0,
            draw_overlay=False
        )
        
        from unittest.mock import patch, MagicMock
        from headtrack_ar.tracker import HeadTracker
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(config)
            
            # Test with empty frame
            empty_frame = create_empty_frame()
            mock_detector_instance.detect.return_value = []
            
            frame_info = tracker.process_frame(empty_frame)
            
            self.assertIsNotNone(frame_info)
            self.assertEqual(len(frame_info.heads), 0)
            
            tracker.release()
    
    def test_tracker_multiple_heads(self):
        """Test tracker with multiple detected heads."""
        config = TrackerConfig(
            source=0,
            draw_overlay=True
        )
        
        from unittest.mock import patch, MagicMock
        from headtrack_ar.tracker import HeadTracker
        from headtrack_ar.types import HeadInfo
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = HeadTracker(config)
            
            # Create multiple heads
            heads = [
                HeadInfo(bbox=(50, 50, 150, 200), forehead_point=(125, 100)),
                HeadInfo(bbox=(400, 100, 150, 200), forehead_point=(475, 150)),
            ]
            mock_detector_instance.detect.return_value = heads
            
            test_frame = create_synthetic_frame_with_face()
            frame_info = tracker.process_frame(test_frame)
            
            self.assertEqual(len(frame_info.heads), 2)
            self.assertEqual(len([h for h in frame_info.heads if h.forehead_point]), 2)
            
            tracker.release()


class TestTrackerContextManager(unittest.TestCase):
    """Test tracker as context manager."""
    
    def test_tracker_context_manager(self):
        """Test tracker used as context manager."""
        test_video_path = create_test_video_file(num_frames=5)
        
        try:
            config = TrackerConfig(source=test_video_path)
            
            with HeadTracker(config) as tracker:
                self.assertIsNotNone(tracker.video_source)
                self.assertIsNotNone(tracker.detector)
                
                frame_count = 0
                for frame_info in tracker.run():
                    frame_count += 1
                    if frame_count >= 5:
                        break
            
            # After context exit, resources should be released
            self.assertIsNone(tracker.video_source)
            self.assertIsNone(tracker.detector)
        finally:
            cleanup_test_video(test_video_path)


if __name__ == '__main__':
    unittest.main()
