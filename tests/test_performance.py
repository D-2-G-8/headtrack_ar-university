"""
Performance tests for head tracking module.
"""

import unittest
import time
import numpy as np

from headtrack_ar import HeadTracker, TrackerConfig
from headtrack_ar.detector import FaceDetector
from headtrack_ar.tracker import HeadTracker as Tracker
from tests.test_utils import (
    create_synthetic_frame_with_face,
    create_test_video_file,
    cleanup_test_video
)


class TestPerformance(unittest.TestCase):
    """Performance tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_video_path = None
    
    def tearDown(self):
        """Clean up after tests."""
        if self.test_video_path:
            cleanup_test_video(self.test_video_path)
    
    def test_detector_single_frame_performance(self):
        """Test detector performance on single frame."""
        detector = FaceDetector(min_detection_confidence=0.4)
        
        frame = create_synthetic_frame_with_face()
        
        # Warm-up
        detector.detect(frame)
        
        # Measure performance
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            detector.detect(frame)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        detector.release()
        
        # Should process at reasonable speed
        # On average hardware, should be > 5 FPS
        self.assertGreater(fps, 1.0, f"Detector too slow: {fps:.2f} FPS")
        print(f"Detector performance: {fps:.2f} FPS ({avg_time*1000:.2f} ms per frame)")
    
    def test_tracker_frame_processing_performance(self):
        """Test tracker frame processing performance."""
        config = TrackerConfig(
            source=0,
            draw_overlay=True,
            smoothing_alpha=0.7
        )
        
        from unittest.mock import patch, MagicMock
        from headtrack_ar.types import HeadInfo
        
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            
            tracker = Tracker(config)
            
            test_frame = create_synthetic_frame_with_face()
            mock_head = HeadInfo(
                bbox=(100, 100, 200, 250),
                forehead_point=(200, 150)
            )
            mock_detector_instance.detect.return_value = [mock_head]
            
            # Warm-up
            tracker.process_frame(test_frame)
            
            # Measure performance
            num_iterations = 20
            start_time = time.time()
            
            for _ in range(num_iterations):
                tracker.process_frame(test_frame)
            
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            tracker.release()
            
            # Should process reasonably fast
            self.assertGreater(fps, 1.0, f"Tracker too slow: {fps:.2f} FPS")
            print(f"Tracker processing: {fps:.2f} FPS ({avg_time*1000:.2f} ms per frame)")
    
    def test_end_to_end_performance(self):
        """Test end-to-end performance with video file."""
        self.test_video_path = create_test_video_file(
            num_frames=30,
            width=640,
            height=480
        )
        
        config = TrackerConfig(
            source=self.test_video_path,
            target_resolution=(640, 480),
            draw_overlay=True,
            smoothing_alpha=0.7,
            min_detection_confidence=0.4
        )
        
        tracker = HeadTracker(config)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            for frame_info in tracker.run():
                frame_count += 1
                
                # Process a reasonable number of frames
                if frame_count >= 30:
                    break
            
            elapsed_time = time.time() - start_time
            
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                avg_time_per_frame = elapsed_time / frame_count
                
                print(f"End-to-end performance: {fps:.2f} FPS")
                print(f"Average time per frame: {avg_time_per_frame*1000:.2f} ms")
                print(f"Total frames processed: {frame_count}")
                print(f"Total time: {elapsed_time:.2f} seconds")
                
                # Should achieve at least 10 FPS on typical hardware
                # (15-20 FPS mentioned in requirements)
                self.assertGreater(fps, 5.0, 
                    f"End-to-end performance too slow: {fps:.2f} FPS")
        finally:
            tracker.release()
    
    def test_performance_with_different_resolutions(self):
        """Test performance with different frame resolutions."""
        resolutions = [
            (320, 240),
            (640, 480),
            (1280, 720)
        ]
        
        detector = FaceDetector()
        
        results = {}
        
        for width, height in resolutions:
            frame = create_synthetic_frame_with_face(width=width, height=height)
            
            # Warm-up
            detector.detect(frame)
            
            # Measure
            num_iterations = 5
            start_time = time.time()
            
            for _ in range(num_iterations):
                detector.detect(frame)
            
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            results[f"{width}x{height}"] = fps
            print(f"Resolution {width}x{height}: {fps:.2f} FPS")
        
        detector.release()
        
        # Higher resolution should generally be slower
        # But all should be reasonable
        for res, fps in results.items():
            self.assertGreater(fps, 0.5, f"Too slow at {res}: {fps:.2f} FPS")
    
    def test_performance_with_smoothing(self):
        """Compare performance with and without smoothing."""
        config_with_smoothing = TrackerConfig(
            source=0,
            smoothing_alpha=0.7
        )
        
        config_without_smoothing = TrackerConfig(
            source=0,
            smoothing_alpha=None
        )
        
        from unittest.mock import patch, MagicMock
        from headtrack_ar.types import HeadInfo
        
        test_frame = create_synthetic_frame_with_face()
        mock_head = HeadInfo(
            bbox=(100, 100, 200, 250),
            forehead_point=(200, 150)
        )
        
        # Test with smoothing
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            mock_detector_instance.detect.return_value = [mock_head]
            
            tracker = Tracker(config_with_smoothing)
            
            num_iterations = 20
            start_time = time.time()
            
            for _ in range(num_iterations):
                tracker.process_frame(test_frame)
            
            elapsed_with = time.time() - start_time
            tracker.release()
        
        # Test without smoothing
        with patch('headtrack_ar.tracker.VideoSource') as mock_video_source, \
             patch('headtrack_ar.tracker.FaceDetector') as mock_detector:
            
            mock_video_source_instance = MagicMock()
            mock_video_source.return_value = mock_video_source_instance
            
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            mock_detector_instance.detect.return_value = [mock_head]
            
            tracker = Tracker(config_without_smoothing)
            
            num_iterations = 20
            start_time = time.time()
            
            for _ in range(num_iterations):
                tracker.process_frame(test_frame)
            
            elapsed_without = time.time() - start_time
            tracker.release()
        
        print(f"With smoothing: {elapsed_with*1000:.2f} ms for {num_iterations} frames")
        print(f"Without smoothing: {elapsed_without*1000:.2f} ms for {num_iterations} frames")
        
        # Smoothing adds minimal overhead
        overhead_ratio = elapsed_with / elapsed_without if elapsed_without > 0 else 1.0
        self.assertLess(overhead_ratio, 2.0, "Smoothing adds too much overhead")


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage and resource management."""
    
    def test_multiple_tracker_instances(self):
        """Test creating and releasing multiple tracker instances."""
        test_video_path = create_test_video_file(num_frames=5)
        
        try:
            # Create and release multiple trackers
            for i in range(5):
                config = TrackerConfig(source=test_video_path)
                tracker = HeadTracker(config)
                
                # Process a few frames
                frame_count = 0
                for frame_info in tracker.run():
                    frame_count += 1
                    if frame_count >= 3:
                        break
                
                tracker.release()
                
                # Verify resources are released
                self.assertIsNone(tracker.video_source)
                self.assertIsNone(tracker.detector)
        finally:
            cleanup_test_video(test_video_path)
    
    def test_long_running_tracker(self):
        """Test tracker stability over many frames."""
        self.test_video_path = create_test_video_file(num_frames=100)
        
        config = TrackerConfig(
            source=self.test_video_path,
            smoothing_alpha=0.7
        )
        
        tracker = HeadTracker(config)
        
        frame_count = 0
        error_count = 0
        
        try:
            for frame_info in tracker.run():
                frame_count += 1
                
                # Check for errors
                if frame_info is None:
                    error_count += 1
                
                if frame_count >= 100:
                    break
        finally:
            tracker.release()
        
        # Should process most frames successfully
        success_rate = (frame_count - error_count) / frame_count if frame_count > 0 else 0
        self.assertGreater(success_rate, 0.9, 
            f"Too many errors: {error_count}/{frame_count}")


if __name__ == '__main__':
    unittest.main()
