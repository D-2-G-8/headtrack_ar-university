"""
Main head tracking logic with smoothing and frame processing.
"""

import logging
import math
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from headtrack_ar.config import TrackerConfig
from headtrack_ar.detector import FaceDetector
from headtrack_ar.kalman import KalmanFace2DScale
from headtrack_ar.overlay import draw_overlay
from headtrack_ar.types import FrameInfo, HeadInfo
from headtrack_ar.video_source import VideoSource

logger = logging.getLogger(__name__)

DEFAULT_FACE_HEIGHT_M = 0.22
DEFAULT_CAMERA_FOV_DEG = 60.0


def clip_roi(
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    margin: float
) -> Optional[tuple[int, int, int, int]]:
    """Clip ROI around bbox with margin to frame bounds."""
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    x0 = int(x - w * margin)
    y0 = int(y - h * margin)
    x1 = int(x + w * (1.0 + margin))
    y1 = int(y + h * (1.0 + margin))
    x0 = max(0, min(x0, frame_width - 1))
    y0 = max(0, min(y0, frame_height - 1))
    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return None
    return (x0, y0, x1, y1)


def clip_roi_centered(
    center_x: float,
    center_y: float,
    half_w: float,
    half_h: float,
    frame_width: int,
    frame_height: int
) -> Optional[tuple[int, int, int, int]]:
    """Clip ROI around center with given half sizes to frame bounds."""
    if half_w <= 0 or half_h <= 0:
        return None
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        return None
    x0 = int(center_x - half_w)
    y0 = int(center_y - half_h)
    x1 = int(center_x + half_w)
    y1 = int(center_y + half_h)
    x0 = max(0, min(x0, frame_width - 1))
    y0 = max(0, min(y0, frame_height - 1))
    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return None
    return (x0, y0, x1, y1)


def mp_rel_bbox_to_px(
    rel_bbox,
    frame_width: int,
    frame_height: int
) -> Optional[tuple[int, int, int, int]]:
    """Convert MediaPipe relative bbox to pixel bbox."""
    if rel_bbox is None:
        return None
    try:
        x = int(rel_bbox.xmin * frame_width)
        y = int(rel_bbox.ymin * frame_height)
        w = int(rel_bbox.width * frame_width)
        h = int(rel_bbox.height * frame_height)
    except (AttributeError, TypeError, ValueError):
        return None
    return (x, y, w, h)


def remap_bbox_from_zoom_to_frame(
    bbox: tuple[int, int, int, int],
    roi_x0: int,
    roi_y0: int,
    zoom_scale: float
) -> tuple[float, float, float, float]:
    """Remap bbox from zoom-ROI coordinates back to frame coordinates."""
    x, y, w, h = bbox
    return (
        roi_x0 + x / zoom_scale,
        roi_y0 + y / zoom_scale,
        w / zoom_scale,
        h / zoom_scale
    )


def compute_forehead_point(
    bbox: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Compute forehead point from bbox."""
    x, y, w, h = bbox
    return (x + w * 0.5, y + h * 0.2)


def bbox_from_center_scale(
    center_x: float,
    center_y: float,
    face_h_px: float,
    aspect_ratio: float
) -> tuple[float, float, float, float]:
    """Build bbox from center, height and aspect ratio."""
    h = max(1.0, float(face_h_px))
    w = max(1.0, float(aspect_ratio) * h)
    x = float(center_x) - w * 0.5
    y = float(center_y) - h * 0.5
    return (x, y, w, h)


def ema_smooth(
    previous: Optional[tuple[float, float]],
    current: tuple[float, float],
    alpha: float
) -> tuple[float, float]:
    """Apply EMA smoothing to a point."""
    if previous is None:
        return current
    px, py = previous
    cx, cy = current
    return (alpha * cx + (1.0 - alpha) * px, alpha * cy + (1.0 - alpha) * py)


def estimate_distance_m(
    face_height_px: float,
    frame_height: int,
    face_height_m: float = DEFAULT_FACE_HEIGHT_M,
    camera_fov_deg: float = DEFAULT_CAMERA_FOV_DEG,
    effective_focal_px: Optional[float] = None
) -> Optional[float]:
    """Estimate distance to face using a simple pinhole camera model (approx)."""
    if face_height_px <= 0 or frame_height <= 0:
        return None
    try:
        if effective_focal_px is not None and effective_focal_px > 0:
            focal_length_px = float(effective_focal_px)
        else:
            fov_rad = math.radians(camera_fov_deg)
            focal_length_px = 0.5 * frame_height / math.tan(fov_rad / 2.0)
            if focal_length_px <= 0:
                return None
    except (ValueError, ZeroDivisionError):
        return None
    return face_height_m * focal_length_px / face_height_px


def _select_best_detection(
    detections,
    frame_width: int,
    frame_height: int
) -> Optional[tuple[tuple[int, int, int, int], Optional[float]]]:
    """Select best detection by area, then score."""
    if not detections:
        return None
    best = None
    best_area = -1
    best_score = -1.0
    for detection in detections:
        if detection is None or not hasattr(detection, "location_data"):
            continue
        rel_bbox = detection.location_data.relative_bounding_box
        bbox = mp_rel_bbox_to_px(rel_bbox, frame_width, frame_height)
        if bbox is None:
            continue
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)
        if w <= 0 or h <= 0:
            continue
        area = w * h
        score = 0.0
        if hasattr(detection, "score") and detection.score:
            try:
                score = float(detection.score[0])
            except (TypeError, ValueError, IndexError):
                score = 0.0
        if area > best_area or (area == best_area and score > best_score):
            best_area = area
            best_score = score
            best = ((x, y, w, h), score)
    return best


class HeadTracker:
    """Main class for tracking heads in video streams.
    
    This class coordinates video capture, face detection, smoothing,
    and overlay rendering to provide a complete head tracking solution.
    """
    
    def __init__(self, config: TrackerConfig):
        """Initialize head tracker.
        
        Args:
            config: Tracker configuration.
            
        Raises:
            RuntimeError: If initialization fails.
        """
        self.config = config
        self.video_source: Optional[VideoSource] = None
        self.detector: Optional[FaceDetector] = None
        self.smoothing_states: dict[int, dict[str, float]] = defaultdict(
            lambda: {'x': None, 'y': None}
        )
        self.head_id_counter = 0
        self.state = "lost"
        self.miss_streak = 0
        self.hit_streak = 0
        self.last_good_bbox: Optional[tuple[float, float, float, float]] = None
        self.last_good_point: Optional[tuple[float, float]] = None
        self.last_good_conf: Optional[float] = None
        self.last_good_time: Optional[float] = None
        self.last_good_source: Optional[str] = None
        self.last_good_model: Optional[str] = None
        self.last_output_point: Optional[tuple[float, float]] = None
        self.last_output_bbox: Optional[tuple[float, float, float, float]] = None
        self.last_output_conf: Optional[float] = None
        self.last_output_source: Optional[str] = None
        self.last_output_model: Optional[str] = None
        self._last_output_predicted = False
        self.kalman: Optional[KalmanFace2DScale] = None
        self._kalman_last_time: Optional[float] = None
        self._kalman_last_state: Optional[dict[str, float]] = None
        self._kalman_last_d2: Optional[float] = None
        self._kalman_last_aspect: Optional[float] = None
        self._kalman_reject_streak = 0
        self._kalman_reacquire_cooldown = 0
        self._kalman_soft_reinit_cooldown = 0
        self._kalman_reject_reinit_cooldown = 0
        self._frame_index = 0
        self._process_index = 0
        self._last_point_log_time = 0.0
        self._last_process_time = 0.0
        self._last_summary_time = time.monotonic()
        self._last_kalman_dt: Optional[float] = None
        self._last_kf_reject_log_time = 0.0
        self._lost_process_counter = 0
        self._roi_center_override: Optional[tuple[float, float]] = None
        self._last_detected_count = 0
        self._stats = {
            "processed": 0,
            "attempted": 0,
            "found": 0,
            "detect_ms_sum": 0.0,
            "detect_count": 0,
            "roi_ms_sum": 0.0,
            "roi_count": 0,
            "full_ms_sum": 0.0,
            "full_count": 0,
            "kalman_accepts": 0,
            "kalman_rejects": 0,
            "kalman_soft_updates": 0,
            "kalman_predicted": 0,
            "kalman_reinits": 0,
            "kalman_d2_sum": 0.0,
            "kalman_d2_count": 0,
            "kalman_dt_sum": 0.0,
            "kalman_dt_count": 0,
            "kalman_residual_sum": 0.0,
            "kalman_residual_count": 0,
            "detected_ok": 0,
        }
        self._distance_assumption_logged = False
        self._focal_length_logged = False
        
        try:
            self.video_source = VideoSource(
                source=config.source,
                target_resolution=config.target_resolution,
                capture_width=config.capture_width,
                capture_height=config.capture_height,
                force_capture_resolution=config.force_capture_resolution
            )
            
            self.detector = FaceDetector(
                min_detection_confidence=config.min_detection_confidence,
                model_selection=config.model_selection,
                max_num_faces=config.max_num_faces
            )
            
            logger.info("HeadTracker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HeadTracker: {e}")
            self.release()
            raise RuntimeError("Failed to initialize HeadTracker") from e
    
    def run(self):
        """Generator that yields FrameInfo for each processed frame.
        
        Yields:
            FrameInfo objects containing processed frame and head information.
        """
        if self.video_source is None or self.detector is None:
            raise RuntimeError("Tracker not properly initialized")
        
        frame_count = 0
        
        while True:
            try:
                frame = self.video_source.read()
                if frame is None:
                    logger.warning("No more frames available")
                    break
                
                frame_info = self.process_frame(frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.debug(f"Processed {frame_count} frames")
                
                yield frame_info
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}", exc_info=True)
                # Continue processing instead of crashing
                frame_count += 1
                # Yield an empty frame info to keep the loop going
                try:
                    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    yield FrameInfo(frame=empty_frame, raw_frame=empty_frame, heads=[], detected_count=0)
                except Exception:
                    # If even this fails, break the loop
                    logger.error("Fatal error, breaking loop")
                    break
    
    def process_frame(self, frame: np.ndarray) -> FrameInfo:
        """Process a single frame: detect heads, smooth coordinates, draw overlay.
        
        Args:
            frame: Input frame in BGR format.
            
        Returns:
            FrameInfo containing processed frame and head information.
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame provided")
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame
                return FrameInfo(frame=empty_frame, raw_frame=empty_frame, heads=[], detected_count=0)
            
            raw_frame = frame.copy() if self.config.draw_overlay else None
            
            # Safely detect heads - this should always return a list, never None
            if self.detector is None:
                logger.error("Detector not initialized")
                return FrameInfo(frame=frame, raw_frame=raw_frame, heads=[], detected_count=0)

            use_roi_pipeline = (
                isinstance(self.detector, FaceDetector)
                and hasattr(self.detector, "fd_short")
                and hasattr(self.detector, "fd_full")
            )
            if not use_roi_pipeline:
                heads = self.detector.detect(frame)
                
                if heads is None:
                    logger.warning("Detector returned None, using empty list")
                    heads = []
                self._last_detected_count = len(heads)
                
                if self.config.enable_smoothing and self.config.smoothing_alpha is not None and heads:
                    try:
                        heads = self._apply_smoothing(heads)
                    except Exception as e:
                        logger.warning(f"Error applying smoothing: {e}, using unsmoothed heads")
                
                if self.config.draw_overlay:
                    try:
                        forehead_points = [head.forehead_point for head in heads if head.forehead_point]
                        frame = draw_overlay(
                            frame,
                            forehead_points,
                            config=self.config.overlay_config
                        )
                    except Exception as e:
                        logger.warning(f"Error drawing overlay: {e}")
                
                return FrameInfo(
                    frame=frame,
                    raw_frame=raw_frame,
                    heads=heads,
                    detected_count=self._last_detected_count
                )
            
            self._frame_index += 1
            frame_index = self._frame_index
            frame_height, frame_width = frame.shape[:2]
            now = time.monotonic()

            max_process_fps = float(self.config.max_process_fps or 0.0)
            if max_process_fps > 0:
                min_interval = 1.0 / max_process_fps
                if now - self._last_process_time < min_interval:
                    heads = self._build_heads_from_last_state(frame_width, frame_height)
                    if self.config.draw_overlay and heads:
                        frame = draw_overlay(
                            frame,
                            [head.forehead_point for head in heads],
                            config=self.config.overlay_config
                        )
                    if heads:
                        self._throttled_log_point(heads[0].forehead_point)
                    return FrameInfo(
                        frame=frame,
                        raw_frame=raw_frame,
                        heads=heads,
                        detected_count=self._last_detected_count
                    )

            starting_state = self.state
            tracking_every_n = (
                self.config.tracking_process_every_n_frames
                if self.config.tracking_process_every_n_frames is not None
                else self.config.process_every_n_frames
            )
            tracking_every_n = tracking_every_n if tracking_every_n is not None else 1
            lost_every_n = self.config.lost_process_every_n_frames or 1
            if starting_state == "lost" and self.config.boost_processing_when_lost:
                process_every_n = max(1, int(lost_every_n))
            else:
                process_every_n = max(1, int(tracking_every_n))
            if frame_index % process_every_n != 0:
                heads = self._build_heads_from_last_state(frame_width, frame_height)
                if self.config.draw_overlay and heads:
                    frame = draw_overlay(
                        frame,
                        [head.forehead_point for head in heads],
                        config=self.config.overlay_config
                    )
                if heads:
                    self._throttled_log_point(heads[0].forehead_point)
                return FrameInfo(
                    frame=frame,
                    raw_frame=raw_frame,
                    heads=heads,
                    detected_count=self._last_detected_count
                )

            self._process_index += 1
            self._last_process_time = now
            self._stats["processed"] += 1
            if self._kalman_soft_reinit_cooldown > 0:
                self._kalman_soft_reinit_cooldown -= 1
            if self._kalman_reject_reinit_cooldown > 0:
                self._kalman_reject_reinit_cooldown -= 1
            kalman_pred_state = self._predict_kalman(now, frame_width, frame_height)

            full_redetect_every_n = max(1, int(self.config.full_redetect_every_n))
            full_redetect_on_lost_every_n = max(1, int(self.config.full_redetect_on_lost_every_n))
            lost_process_counter = self._lost_process_counter if starting_state == "lost" else 0
            last_good_face_h = None
            distance_far = False
            small_face_roi = False
            if self.last_good_bbox is not None:
                _, _, bw, bh = self.last_good_bbox
                last_good_face_h = float(bh)
            pred_face_h = kalman_pred_state["h"] if kalman_pred_state is not None else None
            face_h_for_distance = pred_face_h if pred_face_h is not None else last_good_face_h
            if face_h_for_distance is not None:
                distance_m = estimate_distance_m(
                    face_h_for_distance,
                    frame_height,
                    effective_focal_px=self.config.effective_focal_px
                )
                distance_far = distance_m is not None and distance_m > 2.0
                small_face_roi = face_h_for_distance < self.config.min_face_size_px
            face_h_for_thr = pred_face_h if pred_face_h is not None else last_good_face_h
            used_keep_thr, used_acquire_thr, used_face_h_ref = self._get_conf_thresholds(face_h_for_thr)

            candidate_bbox = None
            candidate_confidence = None
            candidate_source = None
            candidate_model = None
            candidate_roi = None
            detected_ok = False
            kf_update_ok = False
            kf_reinit_done = False
            kf_d2 = None
            attempted_detection = False
            roi_time_total = 0.0
            full_time_total = 0.0
            roi_attempts_log = []

            if kalman_pred_state is not None and self.config.enable_roi_zoom:
                roi_miss_streak = self.miss_streak
                base_h = float(max(1.0, kalman_pred_state["h"]))
                roi_unc = self.config.roi_k_unc
                if starting_state == "lost":
                    roi_unc *= self.config.roi_k_unc_lost_mult
                half_w = self.config.roi_k_size * base_h + roi_unc * kalman_pred_state["sigma_x"]
                half_h = self.config.roi_k_size * base_h + roi_unc * kalman_pred_state["sigma_y"]
                if self.config.roi_expand_on_miss:
                    factor = 1.0 + self.config.roi_miss_expand_factor * min(
                        roi_miss_streak,
                        self.config.roi_miss_expand_cap
                    )
                    half_w *= factor
                    half_h *= factor
                if base_h <= self.config.very_tiny_face_px:
                    half_w *= self.config.roi_very_tiny_boost
                    half_h *= self.config.roi_very_tiny_boost
                roi_center = self._roi_center_override
                if roi_center is None:
                    roi_center = (kalman_pred_state["cx"], kalman_pred_state["cy"])
                roi_center_x, roi_center_y = roi_center
                scales = self._select_roi_scales(base_h)
                margins = self.config.roi_margins or [self.config.roi_margin]
                for zoom_scale, margin in self._build_roi_attempts(
                    roi_miss_streak,
                    scales=scales,
                    margins=margins
                ):
                    half_w_attempt = half_w * (1.0 + float(margin))
                    half_h_attempt = half_h * (1.0 + float(margin))
                    roi = clip_roi_centered(
                        roi_center_x,
                        roi_center_y,
                        half_w_attempt,
                        half_h_attempt,
                        frame_width,
                        frame_height
                    )
                    if roi is None:
                        continue
                    roi_x0, roi_y0, roi_x1, roi_y1 = roi
                    roi_frame = frame[roi_y0:roi_y1, roi_x0:roi_x1]
                    if roi_frame.size == 0 or roi_frame.shape[0] < 2 or roi_frame.shape[1] < 2:
                        continue
                    zoom_scale = max(1.0, float(zoom_scale))
                    zoom_roi = cv2.resize(
                        roi_frame,
                        None,
                        fx=zoom_scale,
                        fy=zoom_scale,
                        interpolation=cv2.INTER_LINEAR
                    )
                    rgb_zoom = cv2.cvtColor(zoom_roi, cv2.COLOR_BGR2RGB)
                    zoom_h, zoom_w = zoom_roi.shape[:2]
                    use_full_range_roi = (
                        distance_far
                        or (self.config.use_full_range_for_small_faces and small_face_roi)
                    )
                    roi_detector = self.detector.fd_full if use_full_range_roi else self.detector.fd_short
                    t0 = time.perf_counter()
                    zoom_results = roi_detector.process(rgb_zoom)
                    t_ms = (time.perf_counter() - t0) * 1000.0
                    roi_time_total += t_ms
                    self._stats["roi_ms_sum"] += t_ms
                    self._stats["roi_count"] += 1
                    attempted_detection = True
                    attempt_info = {"m": float(margin), "s": float(zoom_scale), "ok": 0}
                    if zoom_results is not None and zoom_results.detections:
                        best = _select_best_detection(zoom_results.detections, zoom_w, zoom_h)
                        if best is not None:
                            zoom_bbox, candidate_conf = best
                            candidate_conf = float(candidate_conf) if candidate_conf is not None else 0.0
                            candidate_face_h = float(zoom_bbox[3]) / zoom_scale if zoom_bbox is not None else None
                            face_h_ref = candidate_face_h if candidate_face_h else last_good_face_h
                            keep_thr, acquire_thr, face_h_used = self._get_conf_thresholds(face_h_ref)
                            used_keep_thr = keep_thr
                            used_acquire_thr = acquire_thr
                            used_face_h_ref = face_h_used
                            threshold = keep_thr if starting_state == "tracking" else acquire_thr
                            attempt_info["conf"] = candidate_conf
                            if candidate_face_h is not None:
                                attempt_info["face_h"] = candidate_face_h
                            if candidate_conf >= threshold:
                                candidate_bbox = remap_bbox_from_zoom_to_frame(
                                    zoom_bbox,
                                    roi_x0,
                                    roi_y0,
                                    zoom_scale
                                )
                                candidate_confidence = candidate_conf
                                candidate_source = "roi"
                                candidate_model = "full" if use_full_range_roi else "short"
                                candidate_roi = {
                                    "x0": int(roi_x0),
                                    "y0": int(roi_y0),
                                    "w": int(roi_x1 - roi_x0),
                                    "h": int(roi_y1 - roi_y0),
                                    "scale": float(zoom_scale),
                                }
                                attempt_info["ok"] = 1
                                roi_attempts_log.append(attempt_info)
                                break
                    roi_attempts_log.append(attempt_info)

            full_due = self._process_index % full_redetect_every_n == 0
            full_due_lost = (
                lost_process_counter % full_redetect_on_lost_every_n == 0
                if starting_state == "lost"
                else False
            )
            should_full = False
            roi_enabled = kalman_pred_state is not None and self.config.enable_roi_zoom
            roi_attempted = len(roi_attempts_log) > 0
            if candidate_bbox is None:
                if not roi_enabled or not roi_attempted:
                    if self.state == "lost":
                        if self.last_good_bbox is None or full_due_lost:
                            should_full = True
                    else:
                        if self.last_good_bbox is None or full_due:
                            should_full = True
                else:
                    if starting_state == "lost" and full_due_lost:
                        should_full = True

            if should_full:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_face_expected = self.last_good_bbox is None or small_face_roi
                use_full_range = (
                    distance_far
                    or (self.config.use_full_range_for_small_faces and small_face_expected)
                )
                if not use_full_range:
                    use_full_range = self.config.model_selection == 1
                detector = self.detector.fd_full if use_full_range else self.detector.fd_short
                t0 = time.perf_counter()
                full_results = detector.process(rgb_frame)
                t_ms = (time.perf_counter() - t0) * 1000.0
                full_time_total += t_ms
                self._stats["full_ms_sum"] += t_ms
                self._stats["full_count"] += 1
                attempted_detection = True
                if full_results is not None and full_results.detections:
                    best = _select_best_detection(full_results.detections, frame_width, frame_height)
                    if best is not None:
                        best_bbox, candidate_conf = best
                        candidate_conf = float(candidate_conf) if candidate_conf is not None else 0.0
                        candidate_face_h = float(best_bbox[3]) if best_bbox is not None else None
                        face_h_ref = candidate_face_h if candidate_face_h else last_good_face_h
                        keep_thr, acquire_thr, face_h_used = self._get_conf_thresholds(face_h_ref)
                        used_keep_thr = keep_thr
                        used_acquire_thr = acquire_thr
                        used_face_h_ref = face_h_used
                        threshold = keep_thr if starting_state == "tracking" else acquire_thr
                        if candidate_conf >= threshold:
                            candidate_bbox = best_bbox
                            candidate_confidence = candidate_conf
                            candidate_source = "full"
                            candidate_model = "full" if use_full_range else "short"

            detected_ok = candidate_bbox is not None
            if attempted_detection:
                self._last_detected_count = 1 if detected_ok else 0
            kf_update_ok, kf_reinit_done, kf_d2 = self._apply_kalman_measurement(
                candidate_bbox,
                candidate_confidence,
                now,
                in_lost=starting_state == "lost",
                source=candidate_source,
                roi_info=candidate_roi
            )
            if detected_ok:
                cx_meas = candidate_bbox[0] + candidate_bbox[2] * 0.5
                cy_meas = candidate_bbox[1] + candidate_bbox[3] * 0.5
                self._roi_center_override = (float(cx_meas), float(cy_meas))
            elif attempted_detection:
                self._roi_center_override = None

            if attempted_detection:
                self._stats["attempted"] += 1
                detect_ms_total = roi_time_total + full_time_total
                self._stats["detect_ms_sum"] += detect_ms_total
                self._stats["detect_count"] += 1
                if detected_ok:
                    self._stats["found"] += 1
                    self._stats["detected_ok"] += 1
                if not self.config.log_state_transitions_only:
                    d2_str = f"{kf_d2:.2f}" if kf_d2 is not None else "n/a"
                    logger.info(
                        "Frame status: detected_ok=%s kf_update_ok=%s kf_reinit_done=%s d2=%s gate_mode=%s "
                        "keep_thr=%.2f acquire_thr=%.2f",
                        int(detected_ok),
                        int(kf_update_ok),
                        int(kf_reinit_done),
                        d2_str,
                        self.config.kalman_gate_mode,
                        used_keep_thr,
                        used_acquire_thr
                    )

            kalman_state = self._get_kalman_state(frame_width, frame_height)
            kalman_bbox = None
            if kalman_state is not None:
                self._kalman_last_state = kalman_state
                aspect = self._kalman_last_aspect or self.config.face_aspect
                kalman_bbox = bbox_from_center_scale(
                    kalman_state["cx"],
                    kalman_state["cy"],
                    kalman_state["h"],
                    aspect
                )

            if detected_ok:
                x, y, w, h = candidate_bbox
                x = max(0, min(int(x), frame_width - 1))
                y = max(0, min(int(y), frame_height - 1))
                w = max(1, min(int(w), frame_width - x))
                h = max(1, min(int(h), frame_height - y))
                self.last_good_bbox = (float(x), float(y), float(w), float(h))
                self.last_good_conf = candidate_confidence
                self.last_good_source = candidate_source
                self.last_good_model = candidate_model
                self.last_good_time = now
                point = compute_forehead_point(self.last_good_bbox)
                if self.config.enable_smoothing and self.config.smoothing_alpha is not None:
                    point = ema_smooth(self.last_good_point, point, self.config.smoothing_alpha)
                self.last_good_point = point
                self.hit_streak += 1
                self.miss_streak = 0

                high_conf = (
                    candidate_confidence is not None
                    and candidate_confidence >= self.config.kf_soft_reinit_conf
                )
                output_bbox = None
                output_predicted = False
                if kf_update_ok and kalman_bbox is not None:
                    output_bbox = kalman_bbox
                elif kf_reinit_done:
                    output_bbox = self.last_good_bbox
                elif high_conf or kalman_bbox is None:
                    output_bbox = self.last_good_bbox
                else:
                    output_bbox = kalman_bbox
                    output_predicted = True

                if output_bbox is not None:
                    if output_predicted:
                        self.last_output_bbox = output_bbox
                        self.last_output_point = compute_forehead_point(output_bbox)
                        self._stats["kalman_predicted"] += 1
                    else:
                        self.last_output_bbox = self.last_good_bbox
                        self.last_output_point = self.last_good_point
                    self.last_output_conf = self.last_good_conf
                    self.last_output_source = self.last_good_source
                    self.last_output_model = self.last_good_model
                    self._last_output_predicted = output_predicted

                if self.state == "lost":
                    if self.hit_streak >= self.config.reacquire_after_hits:
                        self.state = "tracking"
                        self._kalman_reacquire_cooldown = int(
                            max(0, self.config.kalman_reacquire_cooldown_frames)
                        )
                        self._log_state_transition_to_tracking(
                            candidate_confidence,
                            self.last_good_bbox[3],
                            candidate_source,
                            detected_ok,
                            kf_update_ok,
                            kf_reinit_done,
                            kf_d2,
                            self.config.kalman_gate_mode,
                            used_keep_thr,
                            used_acquire_thr,
                            used_face_h_ref,
                            roi_attempts_log
                        )
                        if not self.config.log_state_transitions_only:
                            self._log_tracking_event(
                                event="found",
                                bbox=self.last_output_bbox,
                                point=self.last_output_point,
                                frame_height=frame_height,
                                confidence=self.last_output_conf,
                                source=self.last_output_source,
                                model=self.last_output_model
                            )
            elif attempted_detection:
                self.hit_streak = 0
                self.miss_streak += 1
                if self.state == "tracking":
                    dt_since_good = None
                    if self.last_good_time is not None:
                        dt_since_good = now - self.last_good_time
                    if (
                        self.miss_streak >= self.config.lost_after_misses
                        or (dt_since_good is not None and dt_since_good >= self.config.lost_timeout_sec)
                    ):
                        self.state = "lost"
                        self._log_state_transition_to_lost(
                            dt_since_good,
                            detected_ok,
                            kf_update_ok,
                            kf_reinit_done,
                            kf_d2,
                            self.config.kalman_gate_mode,
                            used_keep_thr,
                            used_acquire_thr,
                            used_face_h_ref,
                            roi_attempts_log
                        )
                if self.kalman is not None:
                    if self.state == "lost":
                        self._damp_kalman_velocity(0.0)
                    else:
                        self._damp_kalman_velocity(0.7)
                if kalman_bbox is not None:
                    self.last_output_bbox = kalman_bbox
                    self.last_output_point = compute_forehead_point(kalman_bbox)
                    self.last_output_conf = self.last_good_conf
                    self.last_output_source = self.last_good_source
                    self.last_output_model = self.last_good_model
                    self._last_output_predicted = True
                    self._stats["kalman_predicted"] += 1
                elif not self.config.freeze_point_on_miss:
                    self.last_output_bbox = None
                    self.last_output_point = None
                    self.last_output_conf = None
                    self.last_output_source = None
                    self.last_output_model = None
                    self._last_output_predicted = False

            if self.state == "lost":
                if starting_state == "lost":
                    self._lost_process_counter = lost_process_counter + 1
                else:
                    self._lost_process_counter = 0
            else:
                self._lost_process_counter = 0

            self._log_summary(now)
            
            heads = self._build_heads_from_last_state(frame_width, frame_height)
            if self.config.draw_overlay and heads:
                frame = draw_overlay(
                    frame,
                    [head.forehead_point for head in heads],
                    config=self.config.overlay_config
                )
            if heads:
                self._throttled_log_point(heads[0].forehead_point)
            return FrameInfo(
                frame=frame,
                raw_frame=raw_frame,
                heads=heads,
                detected_count=self._last_detected_count
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in process_frame: {e}", exc_info=True)
            # Return a safe fallback frame info
            try:
                fallback_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                return FrameInfo(frame=fallback_frame, raw_frame=fallback_frame, heads=[], detected_count=0)
            except Exception:
                # Ultimate fallback
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return FrameInfo(frame=empty_frame, raw_frame=empty_frame, heads=[], detected_count=0)

    def _build_heads_from_last_state(
        self,
        frame_width: int,
        frame_height: int
    ) -> list[HeadInfo]:
        """Build HeadInfo list from last known bbox/point."""
        if self.last_output_bbox is None:
            return []
        if self.last_output_point is None:
            self.last_output_point = compute_forehead_point(self.last_output_bbox)
        px, py = self.last_output_point
        px = int(max(0, min(px, frame_width - 1)))
        py = int(max(0, min(py, frame_height - 1)))
        x, y, w, h = self.last_output_bbox
        x = int(max(0, min(x, frame_width - 1)))
        y = int(max(0, min(y, frame_height - 1)))
        w = int(max(1, min(w, frame_width - x)))
        h = int(max(1, min(h, frame_height - y)))
        return [
            HeadInfo(
                bbox=(x, y, w, h),
                forehead_point=(px, py),
                landmarks=None,
                confidence=self.last_output_conf
            )
        ]

    def _predict_kalman(
        self,
        now: float,
        frame_width: int,
        frame_height: int
    ) -> Optional[dict[str, float]]:
        """Predict Kalman state and return derived values."""
        if self.kalman is None:
            return None
        dt = 0.0 if self._kalman_last_time is None else max(0.0, now - self._kalman_last_time)
        self.kalman.predict(dt)
        self._kalman_last_time = now
        self._last_kalman_dt = dt
        self._cap_kalman_state()
        if dt > 0.0:
            self._stats["kalman_dt_sum"] += dt
            self._stats["kalman_dt_count"] += 1
        return self._get_kalman_state(frame_width, frame_height)

    def _get_kalman_state(
        self,
        frame_width: int,
        frame_height: int
    ) -> Optional[dict[str, float]]:
        """Return current Kalman state with derived metrics."""
        if self.kalman is None:
            return None
        cx, cy, s, _, _, _ = self.kalman.get_state()
        try:
            h = math.exp(s)
        except OverflowError:
            h = float(max(frame_width, frame_height))
        max_dim = float(max(frame_width, frame_height))
        h = float(max(1.0, min(h, max_dim)))
        sigma_x, sigma_y, sigma_s = self.kalman.get_uncertainty()
        return {
            "cx": float(cx),
            "cy": float(cy),
            "s": float(s),
            "h": h,
            "sigma_x": float(sigma_x),
            "sigma_y": float(sigma_y),
            "sigma_s": float(sigma_s),
        }

    def _init_kalman_from_bbox(
        self,
        bbox: tuple[float, float, float, float],
        now: float
    ) -> None:
        """Initialize Kalman filter from a bbox measurement."""
        x, y, w, h = bbox
        cx = x + w * 0.5
        cy = y + h * 0.5
        s = math.log(max(h, 1.0))
        x0 = [cx, cy, s, 0.0, 0.0, 0.0]
        P0 = np.diag(
            [
                self.config.kalman_init_pos_var,
                self.config.kalman_init_pos_var,
                self.config.kalman_init_scale_var,
                self.config.kalman_init_vel_var,
                self.config.kalman_init_vel_var,
                self.config.kalman_init_scale_vel_var,
            ]
        )
        self.kalman = KalmanFace2DScale(
            x0,
            P0,
            q_pos=self.config.kalman_q_pos,
            q_vel=self.config.kalman_q_vel,
            q_scale=self.config.kalman_q_scale,
            q_scale_vel=self.config.kalman_q_scale_vel,
            r_pos_base=self.config.kalman_r_pos_base,
            r_scale_base=self.config.kalman_r_scale_base,
            r_conf_floor=self.config.kalman_r_conf_floor,
            gate_threshold=self.config.kalman_gate_threshold
        )
        self._kalman_last_time = now
        self._kalman_last_d2 = None
        self._last_kalman_dt = None
        self._kalman_soft_reinit_cooldown = 0
        self._kalman_reject_reinit_cooldown = 0
        aspect = self.config.face_aspect
        if h > 0:
            try:
                aspect = float(w) / float(h)
            except (TypeError, ValueError, ZeroDivisionError):
                aspect = self.config.face_aspect
        if not math.isfinite(aspect) or aspect <= 0:
            aspect = self.config.face_aspect
        self._kalman_last_aspect = min(max(aspect, 0.5), 1.5)

    def _soft_reinit_kalman(self, cx: float, cy: float, s: float) -> None:
        """Soft reinit Kalman state to a measurement without resetting the filter."""
        if self.kalman is None:
            return
        self.kalman.x[0, 0] = float(cx)
        self.kalman.x[1, 0] = float(cy)
        self.kalman.x[2, 0] = float(s)
        max_speed = float(self.config.kf_max_speed_px_per_sec or 0.0)
        if max_speed > 0.0:
            self.kalman.x[3, 0] = float(max(-max_speed, min(self.kalman.x[3, 0], max_speed)))
            self.kalman.x[4, 0] = float(max(-max_speed, min(self.kalman.x[4, 0], max_speed)))
        else:
            self.kalman.x[3, 0] = 0.0
            self.kalman.x[4, 0] = 0.0
        self.kalman.x[5, 0] = 0.0
        inflate = float(max(1.0, self.config.kf_soft_reinit_inflate_P))
        self.kalman.P *= inflate
        self._cap_kalman_state()

    def _reinit_kalman_to_measurement(self, cx: float, cy: float, s: float) -> None:
        """Reinitialize Kalman state to measurement with inflated covariance."""
        if self.kalman is None:
            return
        self.kalman.x[0, 0] = float(cx)
        self.kalman.x[1, 0] = float(cy)
        self.kalman.x[2, 0] = float(s)
        self.kalman.x[3, 0] = 0.0
        self.kalman.x[4, 0] = 0.0
        self.kalman.x[5, 0] = 0.0
        inflate = float(max(1.0, self.config.kf_reinit_inflate_P))
        self.kalman.P *= inflate
        self._cap_kalman_state()

    def _cap_kalman_state(self) -> None:
        """Cap Kalman uncertainty and velocities to avoid runaway growth."""
        if self.kalman is None:
            return
        cap_sigma = float(self.config.kf_cap_sigma_xy or 0.0)
        if cap_sigma > 0.0:
            cap_var = cap_sigma * cap_sigma
            for idx in (0, 1):
                if self.kalman.P[idx, idx] > cap_var:
                    self.kalman.P[idx, idx] = cap_var
        max_speed = float(self.config.kf_cap_speed_px_per_sec or 0.0)
        if max_speed > 0.0:
            self.kalman.x[3, 0] = float(max(-max_speed, min(self.kalman.x[3, 0], max_speed)))
            self.kalman.x[4, 0] = float(max(-max_speed, min(self.kalman.x[4, 0], max_speed)))

    def _compute_kalman_d2(
        self,
        z: tuple[float, float, float],
        conf: Optional[float],
        *,
        gate_mode: Optional[str],
        r_pos_base: float,
        r_scale_base: float,
        r_conf_floor: float
    ) -> float:
        """Compute Mahalanobis distance^2 for a measurement without updating."""
        if self.kalman is None:
            return 0.0
        z_vec = np.asarray(list(z), dtype=float).reshape(3, 1)
        conf_value = float(conf) if conf is not None else 0.0
        conf_value = max(conf_value, float(r_conf_floor))
        r_pos = float(r_pos_base) / conf_value
        r_scale = float(r_scale_base) / conf_value
        mode = (gate_mode or "xys").lower()
        if mode not in ("xy", "xys"):
            mode = "xys"
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float
        )
        R = np.diag([r_pos, r_pos, r_scale])
        y = z_vec - H @ self.kalman.x
        S = H @ self.kalman.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        if mode == "xy":
            y_xy = y[:2, :]
            S_xy = S[:2, :2]
            try:
                S_xy_inv = np.linalg.inv(S_xy)
            except np.linalg.LinAlgError:
                S_xy_inv = np.linalg.pinv(S_xy)
            return float((y_xy.T @ S_xy_inv @ y_xy)[0, 0])
        return float((y.T @ S_inv @ y)[0, 0])

    def _log_kalman_reject_debug(
        self,
        now: float,
        *,
        cx_pred: float,
        cy_pred: float,
        cx_meas: float,
        cy_meas: float,
        dx: float,
        dy: float,
        residual: float,
        dt: Optional[float],
        d2: float,
        gate_threshold: float,
        gate_mode: str,
        source: Optional[str],
        roi_info: Optional[dict]
    ) -> None:
        """Log a throttled Kalman reject debug line."""
        if now - self._last_kf_reject_log_time < 1.0:
            return
        self._last_kf_reject_log_time = now
        dt_str = f"{dt:.3f}" if dt is not None else "n/a"
        roi_str = "n/a"
        if source == "roi" and roi_info:
            roi_scale = roi_info.get("scale")
            roi_scale_str = f"{roi_scale:.2f}" if roi_scale is not None else "n/a"
            roi_str = (
                f"x0={roi_info.get('x0')} y0={roi_info.get('y0')} "
                f"w={roi_info.get('w')} h={roi_info.get('h')} "
                f"scale={roi_scale_str}"
            )
        logger.info(
            "KF gate reject: mode=%s d2=%.2f gate=%.2f dt=%s pred=(%.1f,%.1f) "
            "meas=(%.1f,%.1f) res=(%.1f,%.1f |%.1f|) source=%s roi=%s",
            gate_mode,
            d2,
            gate_threshold,
            dt_str,
            cx_pred,
            cy_pred,
            cx_meas,
            cy_meas,
            dx,
            dy,
            residual,
            source or "n/a",
            roi_str
        )

    def _apply_kalman_measurement(
        self,
        bbox: Optional[tuple[float, float, float, float]],
        confidence: Optional[float],
        now: float,
        *,
        in_lost: bool,
        source: Optional[str] = None,
        roi_info: Optional[dict] = None
    ) -> tuple[bool, bool, Optional[float]]:
        """Apply candidate measurement to Kalman and return (update_ok, reinit_done, d2)."""
        if bbox is None:
            return False, False, None
        if self.kalman is None:
            self._init_kalman_from_bbox(bbox, now)
            self._stats["kalman_accepts"] += 1
            self._kalman_last_d2 = 0.0
            self._kalman_reject_streak = 0
            self._stats["kalman_d2_sum"] += 0.0
            self._stats["kalman_d2_count"] += 1
            return True, False, 0.0
        cx_meas = bbox[0] + bbox[2] * 0.5
        cy_meas = bbox[1] + bbox[3] * 0.5
        s_meas = math.log(max(bbox[3], 1.0))
        conf_value = float(confidence) if confidence is not None else 0.0
        force_accept = (
            in_lost
            and self.config.kalman_accept_high_conf_in_lost
            and conf_value >= self.config.kalman_accept_conf_threshold
        )
        gate_mode = self.config.kalman_gate_mode
        if in_lost:
            if gate_mode == "xys":
                base_gate = self.config.kalman_gate_threshold_xys_lost
            else:
                base_gate = self.config.kalman_gate_threshold_lost
        else:
            if gate_mode == "xys":
                base_gate = self.config.kalman_gate_threshold_xys
            else:
                base_gate = self.config.kalman_gate_threshold
        sigma_x, sigma_y, _ = self.kalman.get_uncertainty()
        adaptive_gate = float(self.config.gate_thr_base) + float(self.config.gate_thr_k) * (sigma_x + sigma_y)
        gate_threshold = float(max(base_gate, adaptive_gate))
        r_pos_base = (
            self.config.kalman_r_pos_base_lost
            if in_lost
            else self.config.kalman_r_pos_base
        )
        r_scale_base = (
            self.config.kalman_r_scale_base_lost
            if in_lost
            else self.config.kalman_r_scale_base
        )
        if not in_lost and self._kalman_reacquire_cooldown > 0:
            scale = float(max(1.0, self.config.kalman_reacquire_r_scale))
            r_pos_base *= scale
            r_scale_base *= scale
        cx_pred, cy_pred, _, _, _, _ = self.kalman.get_state()
        dx = float(cx_meas - cx_pred)
        dy = float(cy_meas - cy_pred)
        residual = float(math.hypot(dx, dy))
        self._stats["kalman_residual_sum"] += residual
        self._stats["kalman_residual_count"] += 1
        d2 = self._compute_kalman_d2(
            (cx_meas, cy_meas, s_meas),
            confidence,
            gate_mode=gate_mode,
            r_pos_base=r_pos_base,
            r_scale_base=r_scale_base,
            r_conf_floor=self.config.kalman_r_conf_floor
        )
        self._kalman_last_d2 = d2
        self._stats["kalman_d2_sum"] += d2
        self._stats["kalman_d2_count"] += 1
        normal_accept = force_accept or d2 <= gate_threshold
        kf_update_ok = False
        kf_reinit_done = False
        if normal_accept:
            self.kalman.update(
                (cx_meas, cy_meas, s_meas),
                confidence,
                gate_mode=gate_mode,
                gate_threshold=gate_threshold,
                r_pos_base=r_pos_base,
                r_scale_base=r_scale_base,
                r_conf_floor=self.config.kalman_r_conf_floor,
                force_accept=True
            )
            self._cap_kalman_state()
            self._stats["kalman_accepts"] += 1
            self._kalman_reject_streak = 0
            if not in_lost and self._kalman_reacquire_cooldown > 0:
                self._kalman_reacquire_cooldown -= 1
            kf_update_ok = True
        else:
            self._stats["kalman_rejects"] += 1
            self._kalman_reject_streak += 1
            self._log_kalman_reject_debug(
                now,
                cx_pred=cx_pred,
                cy_pred=cy_pred,
                cx_meas=cx_meas,
                cy_meas=cy_meas,
                dx=dx,
                dy=dy,
                residual=residual,
                dt=self._last_kalman_dt,
                d2=d2,
                gate_threshold=gate_threshold,
                gate_mode=gate_mode,
                source=source,
                roi_info=roi_info
            )
            if (
                conf_value >= self.config.kf_soft_reinit_conf
                and d2 >= self.config.kf_soft_reinit_d2
                and self._kalman_soft_reinit_cooldown <= 0
            ):
                self._soft_reinit_kalman(cx_meas, cy_meas, s_meas)
                self._stats["kalman_reinits"] += 1
                self._kalman_reject_streak = 0
                self._kalman_soft_reinit_cooldown = int(
                    max(0, self.config.kf_soft_reinit_cooldown_frames)
                )
                kf_reinit_done = True
            elif (
                self.config.kf_reject_streak_reinit
                and self._kalman_reject_streak >= self.config.kf_reject_streak_reinit
                and conf_value >= self.config.kf_reinit_conf_min
                and self._kalman_reject_reinit_cooldown <= 0
            ):
                self._reinit_kalman_to_measurement(cx_meas, cy_meas, s_meas)
                self._stats["kalman_reinits"] += 1
                self._kalman_reject_streak = 0
                self._kalman_reject_reinit_cooldown = int(
                    max(0, self.config.kf_reinit_cooldown_frames)
                )
                kf_reinit_done = True
            elif (
                in_lost
                and self.config.kalman_reinit_after_rejects
                and self._kalman_reject_streak >= self.config.kalman_reinit_after_rejects
                and conf_value >= self.config.kalman_reinit_conf_threshold
            ):
                self._init_kalman_from_bbox(bbox, now)
                self._stats["kalman_reinits"] += 1
                self._kalman_reject_streak = 0
                kf_reinit_done = True
            elif self.config.kf_use_adaptive_update:
                clip_px = float(self.config.kf_residual_clip_px or 0.0)
                cx_update = cx_meas
                cy_update = cy_meas
                if clip_px > 0.0 and d2 >= self.config.kf_d2_soft:
                    dx_clip = max(-clip_px, min(dx, clip_px))
                    dy_clip = max(-clip_px, min(dy, clip_px))
                    cx_update = cx_pred + dx_clip
                    cy_update = cy_pred + dy_clip
                if d2 <= self.config.kf_d2_hard:
                    inflate = float(max(1.0, self.config.kf_R_inflate_soft))
                else:
                    inflate = float(max(1.0, self.config.kf_R_inflate_hard))
                self.kalman.update(
                    (cx_update, cy_update, s_meas),
                    confidence,
                    gate_mode=gate_mode,
                    gate_threshold=gate_threshold,
                    r_pos_base=r_pos_base * inflate,
                    r_scale_base=r_scale_base * inflate,
                    r_conf_floor=self.config.kalman_r_conf_floor,
                    force_accept=True
                )
                self._cap_kalman_state()
                self._stats["kalman_soft_updates"] += 1
                kf_update_ok = True
        w = bbox[2]
        h = bbox[3]
        if h and h > 0:
            try:
                aspect = float(w) / float(h)
            except (TypeError, ValueError, ZeroDivisionError):
                aspect = self.config.face_aspect
            if math.isfinite(aspect) and aspect > 0:
                self._kalman_last_aspect = min(max(aspect, 0.5), 1.5)
        return kf_update_ok, kf_reinit_done, d2

    def _damp_kalman_velocity(self, factor: float) -> None:
        """Dampen Kalman velocities to avoid drift during long misses."""
        if self.kalman is None:
            return
        factor = float(max(0.0, min(1.0, factor)))
        self.kalman.x[3, 0] *= factor
        self.kalman.x[4, 0] *= factor
        self.kalman.x[5, 0] *= factor

    def _select_roi_scales(self, face_h_pred: Optional[float]) -> list[float]:
        """Select ROI zoom scales based on predicted face height."""
        scales = self.config.roi_zoom_scales or [self.config.roi_zoom_scale]
        scales = [float(scale) for scale in scales] if scales else []
        if not scales:
            return []
        if face_h_pred is None:
            return scales
        if face_h_pred <= self.config.tiny_face_px:
            if face_h_pred <= self.config.very_tiny_face_px and 4.0 not in scales:
                scales.append(4.0)
            return scales
        filtered = [scale for scale in scales if scale <= 3.0]
        return filtered if filtered else scales[:1]

    def _build_roi_attempts(
        self,
        miss_streak: int = 0,
        scales: Optional[list[float]] = None,
        margins: Optional[list[float]] = None
    ) -> list[tuple[float, float]]:
        """Build ROI pyramid attempts from config."""
        scales = scales if scales is not None else (self.config.roi_zoom_scales or [self.config.roi_zoom_scale])
        margins = margins if margins is not None else (self.config.roi_margins or [self.config.roi_margin])
        if not scales or not margins:
            return []
        attempts: list[tuple[float, float]] = []
        scale_idx = 0
        margin_idx = 0
        attempts.append((float(scales[scale_idx]), float(margins[margin_idx])))
        while True:
            advanced = False
            if scale_idx + 1 < len(scales):
                scale_idx += 1
                attempts.append((float(scales[scale_idx]), float(margins[margin_idx])))
                advanced = True
            if margin_idx + 1 < len(margins):
                margin_idx += 1
                attempts.append((float(scales[scale_idx]), float(margins[margin_idx])))
                advanced = True
            if not advanced:
                break
        if self.config.roi_expand_on_miss and miss_streak >= 2 and len(margins) > 1:
            min_margin = float(margins[1])
            attempts = [attempt for attempt in attempts if attempt[1] >= min_margin]
        attempts_max = self.config.roi_attempts_max
        if attempts_max is not None:
            attempts = attempts[: max(1, int(attempts_max))]
        return attempts

    def _get_conf_thresholds(
        self,
        face_h_ref: Optional[float]
    ) -> tuple[float, float, Optional[float]]:
        """Select adaptive confidence thresholds based on face height."""
        face_h_value = None
        if face_h_ref is not None:
            try:
                face_h_value = float(face_h_ref)
            except (TypeError, ValueError):
                face_h_value = None
        if face_h_value is not None and face_h_value > 0 and face_h_value <= self.config.tiny_face_px:
            return self.config.tiny_keep_conf, self.config.tiny_acquire_conf, face_h_value
        return self.config.keep_conf_base, self.config.acquire_conf_base, face_h_value

    def _format_roi_attempts(self, roi_attempts: list[dict]) -> str:
        """Format ROI attempt log entries for compact logging."""
        if not roi_attempts:
            return "[]"
        parts = []
        for attempt in roi_attempts:
            margin = attempt.get("m")
            scale = attempt.get("s")
            ok = attempt.get("ok", 0)
            fields = []
            if margin is not None:
                fields.append(f"m={margin:.2f}")
            if scale is not None:
                fields.append(f"s={scale:g}")
            fields.append(f"ok={int(ok)}")
            if attempt.get("conf") is not None:
                fields.append(f"conf={attempt['conf']:.2f}")
            if attempt.get("face_h") is not None:
                fields.append(f"face_h={attempt['face_h']:.0f}")
            parts.append(f"({','.join(fields)})")
        return "[" + ",".join(parts) + "]"

    def _log_state_transition_to_lost(
        self,
        dt_since_good: Optional[float],
        detected_ok: bool,
        kf_update_ok: bool,
        kf_reinit_done: bool,
        kf_d2: Optional[float],
        gate_mode: str,
        used_keep_thr: Optional[float],
        used_acquire_thr: Optional[float],
        face_h_ref: Optional[float],
        roi_attempts_log: list[dict]
    ) -> None:
        """Log transition from tracking to lost with context."""
        last_conf = f"{self.last_good_conf:.2f}" if self.last_good_conf is not None else "n/a"
        last_face_h = "n/a"
        if self.last_good_bbox is not None:
            try:
                last_face_h = f"{self.last_good_bbox[3]:.0f}"
            except (TypeError, ValueError):
                last_face_h = "n/a"
        dt_str = f"{dt_since_good:.2f}s" if dt_since_good is not None else "n/a"
        keep_thr = float(used_keep_thr) if used_keep_thr is not None else self.config.keep_conf_base
        acquire_thr = float(used_acquire_thr) if used_acquire_thr is not None else self.config.acquire_conf_base
        face_h_ref_str = f"{face_h_ref:.0f}px" if face_h_ref is not None else "n/a"
        roi_attempts_str = self._format_roi_attempts(roi_attempts_log)
        d2_str = f"{kf_d2:.2f}" if kf_d2 is not None else "n/a"
        kalman_h_str = "n/a"
        kalman_sigma_x_str = "n/a"
        kalman_sigma_y_str = "n/a"
        if self._kalman_last_state is not None:
            kalman_h_str = f"{self._kalman_last_state['h']:.0f}px"
            kalman_sigma_x_str = f"{self._kalman_last_state['sigma_x']:.1f}"
            kalman_sigma_y_str = f"{self._kalman_last_state['sigma_y']:.1f}"
        logger.info(
            "TRACKING->LOST (miss_streak=%d, dt=%s, detected_ok=%s, kf_update_ok=%s, kf_reinit_done=%s, d2=%s, "
            "gate_mode=%s, last_good_conf=%s, last_good_face_h=%spx, source_last_good=%s, "
            "keep_thr=%.2f, acquire_thr=%.2f, face_h_ref=%s, roi_attempts=%s, "
            "h_pred=%s, sigma_x=%s, sigma_y=%s)",
            self.miss_streak,
            dt_str,
            int(detected_ok),
            int(kf_update_ok),
            int(kf_reinit_done),
            d2_str,
            gate_mode,
            last_conf,
            last_face_h,
            self.last_good_source or "n/a",
            keep_thr,
            acquire_thr,
            face_h_ref_str,
            roi_attempts_str,
            kalman_h_str,
            kalman_sigma_x_str,
            kalman_sigma_y_str
        )

    def _log_state_transition_to_tracking(
        self,
        detected_confidence: Optional[float],
        face_height_px: Optional[float],
        source: Optional[str],
        detected_ok: bool,
        kf_update_ok: bool,
        kf_reinit_done: bool,
        kf_d2: Optional[float],
        gate_mode: str,
        used_keep_thr: Optional[float],
        used_acquire_thr: Optional[float],
        face_h_ref: Optional[float],
        roi_attempts_log: list[dict]
    ) -> None:
        """Log transition from lost to tracking with context."""
        conf_str = f"{detected_confidence:.2f}" if detected_confidence is not None else "n/a"
        face_h_str = f"{face_height_px:.0f}px" if face_height_px is not None else "n/a"
        keep_thr = float(used_keep_thr) if used_keep_thr is not None else self.config.keep_conf_base
        acquire_thr = float(used_acquire_thr) if used_acquire_thr is not None else self.config.acquire_conf_base
        face_h_ref_str = f"{face_h_ref:.0f}px" if face_h_ref is not None else "n/a"
        roi_attempts_str = self._format_roi_attempts(roi_attempts_log)
        d2_str = f"{kf_d2:.2f}" if kf_d2 is not None else "n/a"
        kalman_h_str = "n/a"
        kalman_sigma_x_str = "n/a"
        kalman_sigma_y_str = "n/a"
        if self._kalman_last_state is not None:
            kalman_h_str = f"{self._kalman_last_state['h']:.0f}px"
            kalman_sigma_x_str = f"{self._kalman_last_state['sigma_x']:.1f}"
            kalman_sigma_y_str = f"{self._kalman_last_state['sigma_y']:.1f}"
        logger.info(
            "LOST->TRACKING (hit_streak=%d, miss_streak=%d, detected_ok=%s, kf_update_ok=%s, kf_reinit_done=%s, "
            "d2=%s, gate_mode=%s, conf=%s, face_h=%s, source=%s, keep_thr=%.2f, acquire_thr=%.2f, "
            "face_h_ref=%s, roi_attempts=%s, h_pred=%s, sigma_x=%s, sigma_y=%s)",
            self.hit_streak,
            self.miss_streak,
            int(detected_ok),
            int(kf_update_ok),
            int(kf_reinit_done),
            d2_str,
            gate_mode,
            conf_str,
            face_h_str,
            source or "n/a",
            keep_thr,
            acquire_thr,
            face_h_ref_str,
            roi_attempts_str,
            kalman_h_str,
            kalman_sigma_x_str,
            kalman_sigma_y_str
        )

    def _log_summary(self, now: float) -> None:
        """Log periodic summary metrics."""
        interval = max(0.1, float(self.config.summary_interval_sec))
        if now - self._last_summary_time < interval:
            return
        processed = self._stats["processed"]
        attempted = self._stats["attempted"]
        if processed == 0 and attempted == 0:
            self._last_summary_time = now
            return
        elapsed = now - self._last_summary_time
        fps = processed / elapsed if elapsed > 0 else 0.0
        detect_ms_avg = (
            self._stats["detect_ms_sum"] / self._stats["detect_count"]
            if self._stats["detect_count"] > 0 else 0.0
        )
        roi_ms_avg = (
            self._stats["roi_ms_sum"] / self._stats["roi_count"]
            if self._stats["roi_count"] > 0 else 0.0
        )
        full_ms_avg = (
            self._stats["full_ms_sum"] / self._stats["full_count"]
            if self._stats["full_count"] > 0 else 0.0
        )
        found_ratio = self._stats["found"] / max(1, attempted)
        detected_ok_ratio = self._stats["detected_ok"] / max(1, processed)
        last_good_face_h = None
        if self.last_good_bbox is not None:
            try:
                last_good_face_h = float(self.last_good_bbox[3])
            except (TypeError, ValueError):
                last_good_face_h = None
        last_good_face_h_str = f"{last_good_face_h:.0f}px" if last_good_face_h is not None else "n/a"
        last_good_conf_str = f"{self.last_good_conf:.2f}" if self.last_good_conf is not None else "n/a"
        full_redetect_count = int(self._stats["full_count"])
        roi_attempt_count = int(self._stats["roi_count"])
        kalman_initialized = self.kalman is not None
        kalman_sigma_x = "n/a"
        kalman_sigma_y = "n/a"
        kalman_sigma_s = "n/a"
        if self._kalman_last_state is not None:
            kalman_sigma_x = f"{self._kalman_last_state['sigma_x']:.1f}"
            kalman_sigma_y = f"{self._kalman_last_state['sigma_y']:.1f}"
            kalman_sigma_s = f"{self._kalman_last_state['sigma_s']:.3f}"
        kalman_gate_rejects = int(self._stats["kalman_rejects"])
        kalman_updates_ok = int(self._stats["kalman_accepts"])
        kalman_updates_total = kalman_gate_rejects + kalman_updates_ok
        kf_accept_ratio = (
            kalman_updates_ok / max(1, kalman_updates_total)
            if kalman_updates_total > 0 else 0.0
        )
        kalman_predicted = int(self._stats["kalman_predicted"])
        kalman_reinits = int(self._stats["kalman_reinits"])
        kalman_soft_updates = int(self._stats["kalman_soft_updates"])
        avg_d2 = (
            self._stats["kalman_d2_sum"] / self._stats["kalman_d2_count"]
            if self._stats["kalman_d2_count"] > 0 else 0.0
        )
        avg_residual = (
            self._stats["kalman_residual_sum"] / self._stats["kalman_residual_count"]
            if self._stats["kalman_residual_count"] > 0 else 0.0
        )
        dt_avg = (
            self._stats["kalman_dt_sum"] / self._stats["kalman_dt_count"]
            if self._stats["kalman_dt_count"] > 0 else 0.0
        )
        kalman_last_d2 = f"{self._kalman_last_d2:.2f}" if self._kalman_last_d2 is not None else "n/a"
        logger.info(
            "Summary: fps=%.1f detect_ms_avg=%.1f roi_detect_ms_avg=%.1f full_detect_ms_avg=%.1f "
            "found_ratio=%.2f detected_ok_ratio=%.2f kf_accept_ratio=%.2f kf_soft_updates_count=%d "
            "avg_d2=%.2f avg_residual_px=%.1f dt_avg=%.3f state=%s miss_streak=%d hit_streak=%d "
            "reject_streak=%d last_good_face_h=%s last_good_conf=%s "
            "full_redetect_count=%d roi_attempt_count=%d kalman_init=%s sigma_x=%s sigma_y=%s sigma_s=%s "
            "gate_rejects_count=%d accepted_updates_count=%d predicted_frames_count=%d reinits_count=%d last_d2=%s",
            fps,
            detect_ms_avg,
            roi_ms_avg,
            full_ms_avg,
            found_ratio,
            detected_ok_ratio,
            kf_accept_ratio,
            kalman_soft_updates,
            avg_d2,
            avg_residual,
            dt_avg,
            self.state,
            self.miss_streak,
            self.hit_streak,
            self._kalman_reject_streak,
            last_good_face_h_str,
            last_good_conf_str,
            full_redetect_count,
            roi_attempt_count,
            str(kalman_initialized),
            kalman_sigma_x,
            kalman_sigma_y,
            kalman_sigma_s,
            kalman_gate_rejects,
            kalman_updates_ok,
            kalman_predicted,
            kalman_reinits,
            kalman_last_d2
        )
        self._stats = {
            "processed": 0,
            "attempted": 0,
            "found": 0,
            "detect_ms_sum": 0.0,
            "detect_count": 0,
            "roi_ms_sum": 0.0,
            "roi_count": 0,
            "full_ms_sum": 0.0,
            "full_count": 0,
            "kalman_accepts": 0,
            "kalman_rejects": 0,
            "kalman_soft_updates": 0,
            "kalman_predicted": 0,
            "kalman_reinits": 0,
            "kalman_d2_sum": 0.0,
            "kalman_d2_count": 0,
            "kalman_dt_sum": 0.0,
            "kalman_dt_count": 0,
            "kalman_residual_sum": 0.0,
            "kalman_residual_count": 0,
            "detected_ok": 0,
        }
        self._last_summary_time = now

    def _log_tracking_event(
        self,
        event: str,
        bbox: Optional[tuple[float, float, float, float]],
        point: Optional[tuple[float, float]],
        frame_height: int,
        confidence: Optional[float],
        source: Optional[str],
        model: Optional[str]
    ) -> None:
        """Log tracking events with distance estimates."""
        if bbox is None:
            return
        _, _, w, h = bbox
        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        point_str = "n/a"
        if point is not None:
            try:
                px, py = point
                point_str = f"({int(px)}, {int(py)})"
            except (TypeError, ValueError):
                point_str = "n/a"
        if self.config.enable_distance_estimation:
            if self.config.effective_focal_px is not None and not self._focal_length_logged:
                logger.info(
                    "Using calibrated focal length: effective_focal_px=%.2f",
                    self.config.effective_focal_px
                )
                self._focal_length_logged = True
            if self.config.effective_focal_px is None and not self._distance_assumption_logged:
                logger.info(
                    "Distance estimate assumes face height %.2fm and camera FOV %.1f deg.",
                    DEFAULT_FACE_HEIGHT_M,
                    DEFAULT_CAMERA_FOV_DEG
                )
                self._distance_assumption_logged = True
            distance = estimate_distance_m(
                h,
                frame_height,
                effective_focal_px=self.config.effective_focal_px
            )
            distance_str = f"{distance:.2f}m" if distance is not None else "n/a"
            logger.info(
                "Head %s: point=%s source=%s model=%s face_h=%.0fpx face_w=%.0fpx conf=%s est_dist=%s (approx)",
                event,
                point_str,
                source or "n/a",
                model or "n/a",
                h,
                w,
                conf_str,
                distance_str
            )
        else:
            logger.info(
                "Head %s: point=%s source=%s model=%s face_h=%.0fpx face_w=%.0fpx conf=%s",
                event,
                point_str,
                source or "n/a",
                model or "n/a",
                h,
                w,
                conf_str
            )

    def _throttled_log_point(self, point: tuple[int, int]) -> None:
        """Log point coordinates with throttling."""
        if self.config.log_state_transitions_only:
            return
        now = time.monotonic()
        if now - self._last_point_log_time < 0.2:
            return
        self._last_point_log_time = now
        try:
            px, py = point
            if self._last_output_predicted:
                logger.info(f"Forehead point (pred): ({px}, {py})")
            else:
                logger.info(f"Forehead point: ({px}, {py})")
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to log forehead point: {e}")
    
    def _apply_smoothing(self, heads: list[HeadInfo]) -> list[HeadInfo]:
        """Apply exponential moving average smoothing to forehead points.
        
        Args:
            heads: List of detected heads.
            
        Returns:
            List of heads with smoothed forehead points.
        """
        if not heads:
            return heads
        
        smoothed_heads = []
        alpha = self.config.smoothing_alpha
        
        if alpha is None:
            return heads
        
        try:
            for head in heads:
                if head is None:
                    continue
                
                # Safely get forehead point
                if not hasattr(head, 'forehead_point') or head.forehead_point is None:
                    logger.warning("Head missing forehead_point, skipping smoothing")
                    smoothed_heads.append(head)
                    continue
                
                try:
                    fx, fy = head.forehead_point
                    
                    # Validate coordinates
                    if not isinstance(fx, (int, float)) or not isinstance(fy, (int, float)):
                        logger.warning(f"Invalid forehead point coordinates: ({fx}, {fy})")
                        smoothed_heads.append(head)
                        continue
                    
                    head_id = self._assign_head_id(head, int(fx), int(fy))
                    
                    # Safely get or create state (defaultdict should handle this, but be explicit)
                    if head_id not in self.smoothing_states:
                        self.smoothing_states[head_id] = {'x': None, 'y': None}
                    state = self.smoothing_states[head_id]
                    
                    if state['x'] is None or state['y'] is None:
                        state['x'] = float(fx)
                        state['y'] = float(fy)
                    else:
                        state['x'] = alpha * fx + (1 - alpha) * state['x']
                        state['y'] = alpha * fy + (1 - alpha) * state['y']
                    
                    smoothed_point = (int(state['x']), int(state['y']))
                    
                    smoothed_head = HeadInfo(
                        bbox=head.bbox,
                        forehead_point=smoothed_point,
                        landmarks=head.landmarks,
                        confidence=head.confidence
                    )
                    smoothed_heads.append(smoothed_head)
                    
                except (TypeError, ValueError, AttributeError) as e:
                    logger.warning(f"Error smoothing head: {e}, using original")
                    smoothed_heads.append(head)
            
            # Clean up old smoothing states (only remove states that are not in current active heads)
            # Don't remove states we just created in this iteration
            try:
                active_ids = set()
                for h in heads:
                    if h and hasattr(h, 'forehead_point') and h.forehead_point:
                        try:
                            fx, fy = h.forehead_point
                            if isinstance(fx, (int, float)) and isinstance(fy, (int, float)):
                                hid = self._assign_head_id(h, int(fx), int(fy))
                                active_ids.add(hid)
                        except Exception:
                            continue
                
                # Only remove states that are not in active_ids
                # Keep states that were just created in this iteration
                keys_to_remove = [k for k in self.smoothing_states.keys() if k not in active_ids]
                for key in keys_to_remove:
                    del self.smoothing_states[key]
                    
            except Exception as e:
                logger.warning(f"Error cleaning up smoothing states: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in _apply_smoothing: {e}", exc_info=True)
            return heads  # Return original heads on error
        
        return smoothed_heads
    
    def _assign_head_id(self, head: HeadInfo, fx: int, fy: int) -> int:
        """Assign or find ID for a head based on proximity to existing states.
        
        Args:
            head: Head information.
            fx, fy: Forehead point coordinates.
            
        Returns:
            Head ID.
        """
        threshold = 50  # pixels
        
        for head_id, state in self.smoothing_states.items():
            if state['x'] is not None and state['y'] is not None:
                dx = abs(fx - state['x'])
                dy = abs(fy - state['y'])
                distance = (dx**2 + dy**2)**0.5
                
                if distance < threshold:
                    return head_id
        
        new_id = self.head_id_counter
        self.head_id_counter += 1
        return new_id
    
    def release(self) -> None:
        """Release all resources."""
        if self.detector is not None:
            self.detector.release()
            self.detector = None
        
        if self.video_source is not None:
            self.video_source.release()
            self.video_source = None
        
        self.smoothing_states.clear()
        self.state = "lost"
        self.miss_streak = 0
        self.hit_streak = 0
        self.last_good_bbox = None
        self.last_good_point = None
        self.last_good_conf = None
        self.last_good_time = None
        self.last_good_source = None
        self.last_good_model = None
        self.last_output_point = None
        self.last_output_bbox = None
        self.last_output_conf = None
        self.last_output_source = None
        self.last_output_model = None
        self._last_output_predicted = False
        self.kalman = None
        self._kalman_last_time = None
        self._kalman_last_state = None
        self._kalman_last_d2 = None
        self._kalman_last_aspect = None
        self._kalman_reject_streak = 0
        self._kalman_reacquire_cooldown = 0
        self._kalman_soft_reinit_cooldown = 0
        self._kalman_reject_reinit_cooldown = 0
        self._frame_index = 0
        self._process_index = 0
        self._last_process_time = 0.0
        self._last_point_log_time = 0.0
        self._last_summary_time = time.monotonic()
        self._last_kalman_dt = None
        self._last_kf_reject_log_time = 0.0
        self._lost_process_counter = 0
        self._roi_center_override = None
        self._last_detected_count = 0
        self._stats = {
            "processed": 0,
            "attempted": 0,
            "found": 0,
            "detect_ms_sum": 0.0,
            "detect_count": 0,
            "roi_ms_sum": 0.0,
            "roi_count": 0,
            "full_ms_sum": 0.0,
            "full_count": 0,
            "kalman_accepts": 0,
            "kalman_rejects": 0,
            "kalman_soft_updates": 0,
            "kalman_predicted": 0,
            "kalman_reinits": 0,
            "kalman_d2_sum": 0.0,
            "kalman_d2_count": 0,
            "kalman_dt_sum": 0.0,
            "kalman_dt_count": 0,
            "kalman_residual_sum": 0.0,
            "kalman_residual_count": 0,
            "detected_ok": 0,
        }
        self._distance_assumption_logged = False
        self._focal_length_logged = False
        logger.info("HeadTracker released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
