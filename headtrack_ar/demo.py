"""
Command-line demo script for headtrack_ar package.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2

from headtrack_ar import HeadTracker, TrackerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_file_logging() -> str:
    """Add a file handler with datetime-based filename."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"headtrack_demo_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    return log_path


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
        default=None,
        help='Target frame width (omit for native)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Target frame height (omit for native)'
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
        default=0,
        choices=[0, 1],
        help='Face detection model: 0=short-range (0.5-2m), 1=full-range (0.5-5m) (default: 0)'
    )

    parser.add_argument(
        '--power-mode',
        type=str,
        choices=['eco', 'balanced', 'quality'],
        default='balanced',
        help='Power mode preset: eco, balanced, quality (default: balanced)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.6,
        help='Minimum detection confidence (0.0-1.0). Lower = more sensitive (default: 0.6)'
    )

    parser.add_argument(
        '--keep-conf-base',
        '--keep-conf-threshold',
        dest='keep_conf_base',
        type=float,
        default=0.50,
        help='Keep-tracking confidence threshold (base, default: 0.50)'
    )

    parser.add_argument(
        '--acquire-conf-base',
        '--acquire-conf-threshold',
        dest='acquire_conf_base',
        type=float,
        default=0.65,
        help='Reacquire confidence threshold (base, default: 0.65)'
    )

    parser.add_argument(
        '--tiny-face-px',
        type=int,
        default=80,
        help='Tiny-face height threshold in pixels (default: 80)'
    )

    parser.add_argument(
        '--tiny-keep-conf',
        type=float,
        default=0.45,
        help='Keep threshold for tiny faces (default: 0.45)'
    )

    parser.add_argument(
        '--tiny-acquire-conf',
        type=float,
        default=0.60,
        help='Acquire threshold for tiny faces (default: 0.60)'
    )

    parser.add_argument(
        '--lost-after-misses',
        type=int,
        default=6,
        help='Declare LOST after N consecutive misses (default: 6)'
    )

    parser.add_argument(
        '--reacquire-after-hits',
        type=int,
        default=3,
        help='Declare FOUND after N consecutive hits (default: 3)'
    )

    parser.add_argument(
        '--lost-timeout-sec',
        type=float,
        default=1.2,
        help='Declare LOST after timeout in seconds (default: 1.2)'
    )

    parser.add_argument(
        '--freeze-point-on-miss',
        '--freeze-output-when-missed',
        dest='freeze_point_on_miss',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Freeze last output point during misses (default: enabled)'
    )

    parser.add_argument(
        '--log-state-transitions-only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Log only state transitions (default: enabled)'
    )

    parser.add_argument(
        '--no-roi-zoom',
        action='store_true',
        help='Disable ROI zoom pipeline'
    )
    
    parser.add_argument(
        '--roi-zoom-scale',
        type=float,
        default=3.0,
        help='ROI zoom scale factor (default: 3.0)'
    )

    parser.add_argument(
        '--roi-margin',
        type=float,
        default=0.25,
        help='ROI margin as fraction of bbox size (default: 0.25)'
    )

    parser.add_argument(
        '--roi-zoom-scales',
        type=float,
        nargs='+',
        default=None,
        help='ROI pyramid zoom scales (space-separated)'
    )

    parser.add_argument(
        '--roi-margins',
        type=float,
        nargs='+',
        default=None,
        help='ROI pyramid margins (space-separated)'
    )

    parser.add_argument(
        '--roi-attempts-max',
        type=int,
        default=None,
        help='Max ROI attempts per processed frame (default: by power mode)'
    )

    parser.add_argument(
        '--roi-expand-on-miss',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Expand ROI margin when misses accumulate (default: enabled)'
    )
    
    parser.add_argument(
        '--tracking-process-every-n-frames',
        '--process-every-n-frames',
        dest='tracking_process_every_n_frames',
        type=int,
        default=None,
        help='Run detection every N frames while tracking (default: by power mode)'
    )

    parser.add_argument(
        '--lost-process-every-n-frames',
        type=int,
        default=1,
        help='Run detection every N frames while lost (default: 1)'
    )

    parser.add_argument(
        '--boost-processing-when-lost',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Boost processing cadence while lost (default: enabled)'
    )

    parser.add_argument(
        '--full-frame-redetect-every-n',
        '--full-redetect-every-n',
        dest='full_redetect_every_n',
        type=int,
        default=None,
        help='Full-frame re-detect interval in frames (default: by power mode)'
    )

    parser.add_argument(
        '--full-redetect-on-lost-every-n',
        type=int,
        default=6,
        help='Full-frame re-detect interval while lost (default: 6)'
    )
    
    parser.add_argument(
        '--min-face-size-px',
        type=int,
        default=40,
        help='Minimum face size in pixels before switching to full-range (default: 40)'
    )
    
    parser.add_argument(
        '--no-full-range-for-small-faces',
        action='store_true',
        help='Disable full-range model for small/unknown faces'
    )
    
    parser.add_argument(
        '--no-smoothing',
        action='store_true',
        help='Disable smoothing for forehead point'
    )
    
    parser.add_argument(
        '--smoothing-alpha',
        type=float,
        default=0.3,
        help='Smoothing alpha (default: 0.3)'
    )

    parser.add_argument(
        '--max-process-fps',
        type=float,
        default=30.0,
        help='Max processing FPS (default: 30.0)'
    )

    parser.add_argument(
        '--max-num-faces',
        type=int,
        default=1,
        help='Maximum number of faces to track (default: 1)'
    )

    parser.add_argument(
        '--capture-width',
        type=int,
        default=1280,
        help='Camera capture width (default: 1280)'
    )

    parser.add_argument(
        '--capture-height',
        type=int,
        default=720,
        help='Camera capture height (default: 720)'
    )

    parser.add_argument(
        '--force-capture-resolution',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Force camera capture resolution (default: enabled)'
    )

    parser.add_argument(
        '--enable-distance-estimation',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable distance estimation logs (default: enabled)'
    )

    parser.add_argument(
        '--effective-focal-px',
        type=float,
        default=None,
        help='Use calibrated focal length in pixels (optional)'
    )

    parser.add_argument(
        '--calibration-file',
        type=str,
        default=None,
        help='Load calibration file (json or plain text)'
    )

    parser.add_argument(
        '--calibrate-distance',
        action='store_true',
        help='Calibrate distance and save effective_focal_px'
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


def load_effective_focal_px(path: Path) -> float:
    """Load effective focal length from json or plain text file."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = handle.read().strip()
        if not content:
            raise ValueError("Empty calibration file")
        if content.lstrip().startswith("{"):
            data = json.loads(content)
            value = data.get("effective_focal_px")
            if value is None:
                raise ValueError("Missing effective_focal_px in calibration file")
            return float(value)
        return float(content)
    except Exception as exc:
        raise ValueError(f"Failed to load calibration from {path}: {exc}") from exc


def save_effective_focal_px(
    path: Path,
    effective_focal_px: float,
    distance_m: float,
    face_height_px: float
) -> None:
    """Save effective focal length to json file."""
    payload = {
        "effective_focal_px": float(effective_focal_px),
        "distance_m": float(distance_m),
        "face_height_px": float(face_height_px),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def main():
    """Main demo function."""
    args = parse_args()
    log_path = setup_file_logging()
    logger.info(f"Log file: {log_path}")
    
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
    
    target_resolution = None
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            logger.warning("Both --width and --height must be set; using native resolution")
        else:
            target_resolution = (args.width, args.height)
    
    effective_focal_px = args.effective_focal_px
    if args.calibration_file:
        try:
            effective_focal_px = load_effective_focal_px(Path(args.calibration_file))
            logger.info(f"Loaded calibration from {args.calibration_file}")
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)

    config = TrackerConfig(
        source=source,
        target_resolution=target_resolution,
        draw_overlay=not args.no_overlay,
        overlay_config=overlay_config,
        model_selection=args.model,
        min_detection_confidence=args.confidence,
        enable_roi_zoom=not args.no_roi_zoom,
        roi_zoom_scale=args.roi_zoom_scale,
        roi_margin=args.roi_margin,
        roi_zoom_scales=args.roi_zoom_scales if args.roi_zoom_scales else None,
        roi_margins=args.roi_margins if args.roi_margins else None,
        roi_attempts_max=args.roi_attempts_max,
        roi_expand_on_miss=args.roi_expand_on_miss,
        tracking_process_every_n_frames=args.tracking_process_every_n_frames,
        lost_process_every_n_frames=args.lost_process_every_n_frames,
        boost_processing_when_lost=args.boost_processing_when_lost,
        full_redetect_every_n=args.full_redetect_every_n,
        full_redetect_on_lost_every_n=args.full_redetect_on_lost_every_n,
        min_face_size_px=args.min_face_size_px,
        use_full_range_for_small_faces=not args.no_full_range_for_small_faces,
        enable_smoothing=not args.no_smoothing,
        smoothing_alpha=args.smoothing_alpha,
        max_num_faces=args.max_num_faces,
        power_mode=args.power_mode,
        lost_after_misses=args.lost_after_misses,
        reacquire_after_hits=args.reacquire_after_hits,
        keep_conf_base=args.keep_conf_base,
        acquire_conf_base=args.acquire_conf_base,
        tiny_face_px=args.tiny_face_px,
        tiny_keep_conf=args.tiny_keep_conf,
        tiny_acquire_conf=args.tiny_acquire_conf,
        lost_timeout_sec=args.lost_timeout_sec,
        freeze_point_on_miss=args.freeze_point_on_miss,
        log_state_transitions_only=args.log_state_transitions_only,
        max_process_fps=args.max_process_fps,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        force_capture_resolution=args.force_capture_resolution,
        enable_distance_estimation=args.enable_distance_estimation,
        effective_focal_px=effective_focal_px
    )
    
    logger.info(f"Starting head tracking demo")
    logger.info(f"Source: {source}")
    if target_resolution is None:
        logger.info("Resolution: native (auto)")
    else:
        logger.info(f"Resolution: {target_resolution[0]}x{target_resolution[1]}")
    if args.force_capture_resolution:
        logger.info(f"Capture request (forced): {args.capture_width}x{args.capture_height}")
    else:
        logger.info("Capture request: auto (force disabled)")
    logger.info(f"Overlay: {config.draw_overlay}")
    logger.info(f"Detection model: {'full-range (0.5-5m)' if args.model == 1 else 'short-range (0.5-2m)'}")
    logger.info(f"Min confidence: {args.confidence}")
    logger.info(f"Power mode: {config.power_mode}")
    logger.info(f"ROI zoom: {'enabled' if config.enable_roi_zoom else 'disabled'}")
    logger.info(f"Tracking process every N frames: {config.tracking_process_every_n_frames}")
    logger.info(
        f"Lost process every N frames: {config.lost_process_every_n_frames} "
        f"(boosted: {config.boost_processing_when_lost})"
    )
    logger.info(f"Full-frame redetect every N: {config.full_redetect_every_n}")
    logger.info(f"Full-frame redetect on lost every N: {config.full_redetect_on_lost_every_n}")
    logger.info(f"Max process FPS: {config.max_process_fps}")
    logger.info(f"Distance estimation: {'enabled' if config.enable_distance_estimation else 'disabled'}")
    logger.info(f"Smoothing: {'enabled' if config.enable_smoothing else 'disabled'}")
    
    try:
        tracker = HeadTracker(config)
        calibration_distance_m = None
        calibration_saved = False
        calibration_path = None
        if args.calibrate_distance:
            while calibration_distance_m is None:
                try:
                    user_input = input("Enter real distance in meters (e.g., 1.0): ").strip()
                    calibration_distance_m = float(user_input)
                    if calibration_distance_m <= 0:
                        raise ValueError("Distance must be positive")
                except ValueError as exc:
                    logger.warning(f"Invalid distance: {exc}")
                    calibration_distance_m = None
            calibration_path = Path(log_path).with_suffix(".calibration.json")
        
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
                
                head_count = (
                    int(frame_info.detected_count)
                    if getattr(frame_info, "detected_count", None) is not None
                    else (len(frame_info.heads) if frame_info.heads else 0)
                )
                info_text = f"Heads detected: {head_count}"
                
                if not config.disable_heavy_overlays:
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

                if args.calibrate_distance and not calibration_saved and frame_info.heads:
                    try:
                        from headtrack_ar.tracker import DEFAULT_FACE_HEIGHT_M
                        bbox = frame_info.heads[0].bbox
                        face_height_px = float(bbox[3])
                        if face_height_px > 0 and calibration_distance_m is not None:
                            effective_focal_px = (
                                calibration_distance_m * face_height_px / DEFAULT_FACE_HEIGHT_M
                            )
                            save_effective_focal_px(
                                calibration_path,
                                effective_focal_px,
                                calibration_distance_m,
                                face_height_px
                            )
                            tracker.config.effective_focal_px = effective_focal_px
                            calibration_saved = True
                            logger.info(
                                "Calibration saved to %s (effective_focal_px=%.2f)",
                                calibration_path,
                                effective_focal_px
                            )
                    except Exception as e:
                        logger.error(f"Failed to save calibration: {e}")
                
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
