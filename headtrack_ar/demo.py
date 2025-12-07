#!/usr/bin/env python3
"""
Command-line demo script for headtrack_ar package.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from headtrack_ar import HeadTracker, TrackerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Head tracking AR demo - Track human heads and display AR markers'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: camera index (e.g., 0) or path to video file'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Target frame width (default: 640)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Target frame height (default: 480)'
    )
    
    parser.add_argument(
        '--no-overlay',
        action='store_true',
        help='Disable overlay rendering'
    )
    
    parser.add_argument(
        '--color',
        type=str,
        default='green',
        help='Marker color: green, red, blue, yellow (default: green)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=20,
        help='Marker size in pixels (default: 20)'
    )
    
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Marker thickness in pixels (default: 2)'
    )
    
    parser.add_argument(
        '--model',
        type=int,
        default=1,
        choices=[0, 1],
        help='Face detection model: 0=short-range (0.5-2m), 1=full-range (0.5-5m) (default: 1)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.4,
        help='Minimum detection confidence (0.0-1.0). Lower = more sensitive (default: 0.4)'
    )
    
    return parser.parse_args()


def parse_source(source_str: str):
    """Parse source string to int or str."""
    try:
        return int(source_str)
    except ValueError:
        return source_str


def parse_color(color_str: str) -> tuple[int, int, int]:
    """Parse color string to BGR tuple."""
    color_map = {
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
    }
    
    color_lower = color_str.lower()
    if color_lower not in color_map:
        logger.warning(f"Unknown color '{color_str}', using green")
        return color_map['green']
    
    return color_map[color_lower]


def main():
    """Main demo function."""
    args = parse_args()
    
    source = parse_source(args.source)
    
    if isinstance(source, str):
        if not Path(source).exists():
            logger.error(f"Video file not found: {source}")
            sys.exit(1)
    
    color = parse_color(args.color)
    
    from headtrack_ar.config import OverlayConfig
    overlay_config = OverlayConfig(
        color=color,
        size=args.size,
        thickness=args.thickness
    )
    
    config = TrackerConfig(
        source=source,
        target_resolution=(args.width, args.height),
        draw_overlay=not args.no_overlay,
        overlay_config=overlay_config,
        model_selection=args.model,
        min_detection_confidence=args.confidence
    )
    
    logger.info(f"Starting head tracking demo")
    logger.info(f"Source: {source}")
    logger.info(f"Resolution: {args.width}x{args.height}")
    logger.info(f"Overlay: {config.draw_overlay}")
    logger.info(f"Detection model: {'full-range (0.5-5m)' if args.model == 1 else 'short-range (0.5-2m)'}")
    logger.info(f"Min confidence: {args.confidence}")
    
    try:
        tracker = HeadTracker(config)
        
        frame_count = 0
        
        print("\n" + "="*50)
        print("Controls:")
        print("  - Press 'q' or ESC to quit")
        print("  - Press 's' to save current frame")
        print("  - IMPORTANT: Click on the video window first to give it focus!")
        print("  - Keyboard only works when the video window is active")
        print("="*50 + "\n")
        
        # Create window first to ensure it's ready
        cv2.namedWindow("HeadTrack AR Demo", cv2.WINDOW_NORMAL)
        
        for frame_info in tracker.run():
            try:
                if frame_info is None:
                    logger.warning("Received None frame_info, skipping")
                    continue
                
                frame = frame_info.frame
                if frame is None:
                    logger.warning("Frame is None, skipping")
                    continue
                
                head_count = len(frame_info.heads) if frame_info.heads else 0
                info_text = f"Heads detected: {head_count}"
                
                try:
                    cv2.putText(
                        frame,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    fps_text = f"Frame: {frame_count}"
                    cv2.putText(
                        frame,
                        fps_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                except Exception as e:
                    logger.warning(f"Error drawing text on frame: {e}")
                
                try:
                    cv2.imshow("HeadTrack AR Demo", frame)
                except Exception as e:
                    logger.error(f"Error displaying frame: {e}")
                    break
                
                frame_count += 1
                
                # Use waitKey with longer delay (30ms) for better keyboard response
                # Also check for window close event
                # Note: On macOS, window must be in focus for keys to work
                try:
                    # Check if window still exists and is visible
                    window_prop = cv2.getWindowProperty("HeadTrack AR Demo", cv2.WND_PROP_VISIBLE)
                    if window_prop < 1:
                        logger.info("Window closed by user")
                        break
                except cv2.error:
                    logger.info("Window was closed")
                    break
                
                key = cv2.waitKey(30) & 0xFF
                
                # Process keyboard input
                # Check for 'q', 'Q', or ESC (27)
                if key == ord('q') or key == ord('Q') or key == 27:
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s') or key == ord('S'):
                    try:
                        filename = f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(filename, frame)
                        logger.info(f"Saved frame to {filename}")
                        # Show confirmation on frame
                        cv2.putText(
                            frame,
                            f"Saved: {filename}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
                        cv2.imshow("HeadTrack AR Demo", frame)
                        cv2.waitKey(500)  # Show message for 500ms
                    except Exception as e:
                        logger.error(f"Error saving frame: {e}")
                        
            except KeyboardInterrupt:
                logger.info("Interrupted by user during frame processing")
                break
            except Exception as e:
                logger.error(f"Error processing frame in demo: {e}", exc_info=True)
                # Continue processing instead of crashing
                frame_count += 1
                continue
        
        tracker.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Demo completed. Processed {frame_count} frames.")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

