"""
Tests for video source module.
"""

import unittest
import numpy as np
import cv2
import os
import tempfile

from headtrack_ar.video_source import VideoSource
from tests.test_utils import (
    create_test_video_file,
    cleanup_test_video,
    create_synthetic_frame_with_face
)


class TestVideoSource(unittest.TestCase):
    """Test VideoSource class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_video_path = None
    
    def tearDown(self):
        """Clean up after tests."""
        if self.test_video_path and os.path.exists(self.test_video_path):
            cleanup_test_video(self.test_video_path)
    
    def test_video_source_initialization_camera(self):
        """Test video source initialization with camera index."""
        # This test may fail if no camera is available
        # In that case, we'll skip it
        try:
            source = VideoSource(source=0, target_resolution=(640, 480))
            self.assertIsNotNone(source.cap)
            self.assertTrue(source.cap.isOpened())
            source.release()
        except RuntimeError:
            # No camera available, skip test
            self.skipTest("No camera available for testing")
    
    def test_video_source_initialization_file(self):
        """Test video source initialization with video file."""
        # Create test video
        self.test_video_path = create_test_video_file(num_frames=10)
        
        source = VideoSource(source=self.test_video_path)
        self.assertIsNotNone(source.cap)
        self.assertTrue(source.cap.isOpened())
        source.release()
    
    def test_video_source_read_frame(self):
        """Test reading frames from video source."""
        self.test_video_path = create_test_video_file(num_frames=10)
        
        source = VideoSource(source=self.test_video_path)
        
        frame_count = 0
        while True:
            frame = source.read()
            if frame is None:
                break
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)  # Should be BGR image
            self.assertEqual(frame.shape[2], 3)  # 3 channels
            frame_count += 1
        
        self.assertGreater(frame_count, 0)
        source.release()
    
    def test_video_source_resolution(self):
        """Test video source with target resolution."""
        self.test_video_path = create_test_video_file(
            num_frames=5,
            width=320,
            height=240
        )
        
        source = VideoSource(
            source=self.test_video_path,
            target_resolution=(640, 480)
        )
        
        frame = source.read()
        if frame is not None:
            height, width = frame.shape[:2]
            # Resolution should be resized to target
            self.assertEqual(width, 640)
            self.assertEqual(height, 480)
        
        source.release()
    
    def test_video_source_context_manager(self):
        """Test video source as context manager."""
        self.test_video_path = create_test_video_file(num_frames=5)
        
        with VideoSource(source=self.test_video_path) as source:
            self.assertIsNotNone(source.cap)
            frame = source.read()
            self.assertIsNotNone(frame)
        
        # After context exit, cap should be released
        self.assertIsNone(source.cap)
    
    def test_video_source_release(self):
        """Test video source release."""
        self.test_video_path = create_test_video_file(num_frames=5)
        
        source = VideoSource(source=self.test_video_path)
        self.assertIsNotNone(source.cap)
        
        source.release()
        self.assertIsNone(source.cap)
    
    def test_video_source_invalid_file(self):
        """Test video source with invalid file path."""
        invalid_path = "/nonexistent/path/video.mp4"
        
        with self.assertRaises(RuntimeError):
            VideoSource(source=invalid_path)
    
    def test_video_source_read_after_release(self):
        """Test reading after release returns None."""
        self.test_video_path = create_test_video_file(num_frames=5)
        
        source = VideoSource(source=self.test_video_path)
        source.release()
        
        frame = source.read()
        self.assertIsNone(frame)
    
    def test_video_source_multiple_reads(self):
        """Test reading multiple frames."""
        self.test_video_path = create_test_video_file(num_frames=10)
        
        source = VideoSource(source=self.test_video_path)
        
        frames = []
        for _ in range(10):
            frame = source.read()
            if frame is not None:
                frames.append(frame)
        
        self.assertGreater(len(frames), 0)
        source.release()
    
    def test_video_source_end_of_file(self):
        """Test behavior at end of video file."""
        self.test_video_path = create_test_video_file(num_frames=3)
        
        source = VideoSource(source=self.test_video_path)
        
        # Read all frames
        frame_count = 0
        while True:
            frame = source.read()
            if frame is None:
                break
            frame_count += 1
        
        # Try reading again (should return None)
        frame = source.read()
        self.assertIsNone(frame)
        
        self.assertEqual(frame_count, 3)
        source.release()


class TestVideoSourceEdgeCases(unittest.TestCase):
    """Test edge cases for VideoSource."""
    
    def test_video_source_none_resolution(self):
        """Test video source without target resolution."""
        test_video_path = create_test_video_file(num_frames=5)
        
        try:
            source = VideoSource(source=test_video_path, target_resolution=None)
            frame = source.read()
            if frame is not None:
                # Frame should have original resolution
                self.assertIsInstance(frame, np.ndarray)
            source.release()
        finally:
            cleanup_test_video(test_video_path)
    
    def test_video_source_different_resolutions(self):
        """Test video source with various resolutions."""
        resolutions = [
            (320, 240),
            (640, 480),
            (1280, 720)
        ]
        
        for width, height in resolutions:
            test_video_path = create_test_video_file(
                num_frames=3,
                width=width,
                height=height
            )
            
            try:
                source = VideoSource(
                    source=test_video_path,
                    target_resolution=(640, 480)
                )
                frame = source.read()
                if frame is not None:
                    h, w = frame.shape[:2]
                    self.assertEqual(w, 640)
                    self.assertEqual(h, 480)
                source.release()
            finally:
                cleanup_test_video(test_video_path)


if __name__ == '__main__':
    unittest.main()
