"""
Utility functions for creating test data and synthetic video frames.
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
import os


def create_synthetic_frame_with_face(
    width: int = 640,
    height: int = 480,
    face_center: tuple[int, int] = None,
    face_size: int = 150,
    brightness: float = 1.0
) -> np.ndarray:
    """Create a synthetic frame with a simple face-like pattern.
    
    Args:
        width: Frame width.
        height: Frame height.
        face_center: Center of face (x, y). If None, uses center of frame.
        face_size: Approximate size of face region.
        brightness: Brightness multiplier (0.0 to 2.0).
        
    Returns:
        Synthetic frame in BGR format.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    if face_center is None:
        face_center = (width // 2, height // 2)
    
    cx, cy = face_center
    
    # Create a simple face-like pattern (oval shape)
    # This is a simplified representation - real face detection will work better
    # with actual face images, but this helps test the pipeline
    
    # Face oval
    face_color = (180, 150, 120)  # Skin tone in BGR
    cv2.ellipse(
        frame,
        face_center,
        (face_size // 2, int(face_size * 1.2)),
        0, 0, 360,
        face_color,
        -1
    )
    
    # Eyes (two circles)
    eye_y = cy - face_size // 4
    eye_size = face_size // 8
    cv2.circle(frame, (cx - face_size // 3, eye_y), eye_size, (50, 50, 50), -1)
    cv2.circle(frame, (cx + face_size // 3, eye_y), eye_size, (50, 50, 50), -1)
    
    # Nose
    cv2.circle(frame, (cx, cy), eye_size // 2, (100, 100, 100), -1)
    
    # Mouth
    cv2.ellipse(
        frame,
        (cx, cy + face_size // 3),
        (face_size // 4, face_size // 8),
        0, 0, 180,
        (50, 50, 50),
        2
    )
    
    # Apply brightness adjustment
    if brightness != 1.0:
        frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
    return frame


def create_test_video_file(
    num_frames: int = 30,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    face_movement: bool = True
) -> str:
    """Create a temporary video file with synthetic frames.
    
    Args:
        num_frames: Number of frames to generate.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        face_movement: Whether to simulate face movement.
        
    Returns:
        Path to temporary video file.
    """
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"test_headtrack_{os.getpid()}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    try:
        for i in range(num_frames):
            if face_movement:
                # Simulate face moving in a circle
                angle = (i / num_frames) * 2 * np.pi
                offset_x = int(50 * np.cos(angle))
                offset_y = int(30 * np.sin(angle))
                face_center = (width // 2 + offset_x, height // 2 + offset_y)
            else:
                face_center = (width // 2, height // 2)
            
            frame = create_synthetic_frame_with_face(
                width=width,
                height=height,
                face_center=face_center
            )
            out.write(frame)
    finally:
        out.release()
    
    return video_path


def create_frame_with_rotation(
    base_frame: np.ndarray,
    angle: float
) -> np.ndarray:
    """Rotate a frame by specified angle.
    
    Args:
        base_frame: Input frame.
        angle: Rotation angle in degrees (positive = clockwise).
        
    Returns:
        Rotated frame.
    """
    height, width = base_frame.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    rotated = cv2.warpAffine(
        base_frame,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Crop back to original size (centered)
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    return rotated[start_y:start_y + height, start_x:start_x + width]


def adjust_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    """Adjust frame brightness.
    
    Args:
        frame: Input frame.
        factor: Brightness factor (0.0 = black, 1.0 = original, >1.0 = brighter).
        
    Returns:
        Brightness-adjusted frame.
    """
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def create_empty_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create an empty (black) frame.
    
    Args:
        width: Frame width.
        height: Frame height.
        
    Returns:
        Empty frame.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def cleanup_test_video(video_path: str) -> None:
    """Remove a test video file.
    
    Args:
        video_path: Path to video file to remove.
    """
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception:
        pass  # Ignore cleanup errors
