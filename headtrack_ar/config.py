"""
Configuration classes for headtrack_ar package.
"""

from dataclasses import dataclass, field
from typing import Union, Optional


@dataclass
class OverlayConfig:
    """Configuration for overlay rendering.
    
    Attributes:
        color: Color in BGR format (blue, green, red) as tuple.
        size: Size of the crosshair marker in pixels.
        thickness: Thickness of the marker lines in pixels.
    """
    color: tuple[int, int, int] = (0, 255, 0)  # Green in BGR
    size: int = 20
    thickness: int = 2


@dataclass
class TrackerConfig:
    """Main configuration for HeadTracker.
    
    Attributes:
        source: Video source - int for camera index or str for video file path.
        target_resolution: Target resolution as (width, height) or None to keep original.
        draw_overlay: Whether to draw AR markers on frames.
        overlay_config: Configuration for overlay rendering.
        enable_roi_zoom: Enable ROI zoom pipeline to reduce full-frame detection cost.
        roi_zoom_scale: Zoom factor applied to ROI before detection.
        roi_margin: Margin around last bbox for ROI extraction (as fraction of bbox size).
        process_every_n_frames: Run detection every Nth frame (legacy alias for tracking_process_every_n_frames).
        tracking_process_every_n_frames: Run detection every Nth frame while tracking.
        lost_process_every_n_frames: Run detection every Nth frame while lost (if boosted).
        boost_processing_when_lost: Increase processing rate while lost.
        full_redetect_every_n: Force full-frame detection no more than every N frames.
        full_redetect_on_lost_every_n: Full-frame detect interval while in lost state.
        min_face_size_px: Minimum face size to consider as "near" in pixels.
        use_full_range_for_small_faces: Use full-range model when face is small or unknown.
        enable_smoothing: Enable EMA smoothing for forehead point.
        smoothing_alpha: Alpha parameter for EMA smoothing (0.0 to 1.0).
                        Higher values mean less smoothing. Set to None to disable smoothing.
        max_num_faces: Maximum number of faces to track.
        min_detection_confidence: Minimum confidence for face detection (0.0 to 1.0).
                                 Lower values = more sensitive but may have false positives.
        model_selection: MediaPipe face detection model selection (0=short-range up to 2m,
                        1=full-range up to 5m). Used as default when not auto-switching.
        power_mode: Power mode preset (eco, balanced, quality).
        roi_zoom_scales: List of ROI zoom scales for pyramid search.
        roi_margins: List of ROI margins for pyramid search.
        roi_attempts_max: Max ROI attempts per processed frame.
        roi_expand_on_miss: Expand ROI margin when misses accumulate.
        roi_k_size: ROI half-size multiplier for Kalman face height.
        roi_k_unc: ROI half-size multiplier for Kalman uncertainty.
        roi_k_unc_lost_mult: ROI uncertainty multiplier while lost.
        roi_miss_expand_factor: ROI expansion factor per miss.
        roi_miss_expand_cap: Max miss streak used for ROI expansion.
        lost_after_misses: Declare LOST after N consecutive misses.
        reacquire_after_hits: Declare FOUND after N consecutive hits.
        keep_conf_base: Confidence threshold to keep tracking (base for non-tiny faces).
        acquire_conf_base: Confidence threshold to reacquire when lost (base for non-tiny faces).
        tiny_face_px: Face height threshold for tiny faces in pixels.
        very_tiny_face_px: Face height threshold for very tiny faces in pixels.
        tiny_keep_conf: Keep threshold for tiny faces.
        tiny_acquire_conf: Acquire threshold for tiny faces.
        face_aspect: Default face bbox aspect ratio (width/height) for Kalman outputs.
        kalman_q_pos: Kalman process noise for position.
        kalman_q_vel: Kalman process noise for velocity.
        kalman_q_scale: Kalman process noise for log-scale.
        kalman_q_scale_vel: Kalman process noise for log-scale velocity.
        kalman_r_pos_base: Kalman measurement noise base for position.
        kalman_r_scale_base: Kalman measurement noise base for log-scale.
        kalman_r_conf_floor: Confidence floor used for adaptive measurement noise.
        kalman_gate_threshold: Kalman gating threshold (Mahalanobis distance^2).
        kalman_gate_mode: Kalman gating mode ("xy" or "xys").
        kalman_gate_threshold_xys: Kalman gating threshold when using position+scale.
        kalman_r_pos_base_lost: Kalman measurement noise base for position while lost.
        kalman_r_scale_base_lost: Kalman measurement noise base for log-scale while lost.
        kalman_gate_threshold_lost: Kalman gating threshold while lost.
        kalman_gate_threshold_xys_lost: Kalman gating threshold while lost (position+scale).
        gate_thr_base: Base gate threshold for adaptive gating.
        gate_thr_k: Adaptive gating factor for sigma_x + sigma_y.
        kalman_accept_high_conf_in_lost: Accept high-confidence measurements while lost.
        kalman_accept_conf_threshold: Confidence threshold for forced accept in lost.
        kalman_reinit_after_rejects: Reinitialize Kalman after N consecutive rejects in lost.
        kalman_reinit_conf_threshold: Confidence threshold to allow reinit in lost.
        kalman_reacquire_cooldown_frames: Extra smoothing frames after LOST->TRACKING.
        kalman_reacquire_r_scale: Measurement noise scale during reacquire cooldown.
        kalman_init_pos_var: Initial position variance for Kalman.
        kalman_init_vel_var: Initial velocity variance for Kalman.
        kalman_init_scale_var: Initial log-scale variance for Kalman.
        kalman_init_scale_vel_var: Initial log-scale velocity variance for Kalman.
        kf_use_adaptive_update: Use adaptive update instead of hard reject on gate.
        kf_d2_soft: Mahalanobis distance^2 to mark suspicious measurement.
        kf_d2_hard: Mahalanobis distance^2 to mark strong outliers.
        kf_R_inflate_soft: Measurement noise inflation for soft outliers.
        kf_R_inflate_hard: Measurement noise inflation for hard outliers.
        kf_residual_clip_px: Clip residual (dx, dy) before adaptive update.
        kf_soft_reinit_conf: Confidence threshold to allow soft reinit on gate reject.
        kf_soft_reinit_d2: Mahalanobis distance^2 threshold to trigger soft reinit.
        kf_soft_reinit_inflate_P: Inflate Kalman covariance by this factor on soft reinit.
        kf_soft_reinit_cooldown_frames: Cooldown frames between soft reinits.
        kf_max_speed_px_per_sec: Max velocity clamp for soft reinit (px/sec).
        kf_reject_streak_reinit: Reject streak count to trigger soft reinit.
        kf_reinit_conf_min: Minimum confidence to allow streak reinit.
        kf_reinit_inflate_P: Inflate Kalman covariance by this factor on streak reinit.
        kf_reinit_cooldown_frames: Cooldown frames between streak reinits.
        kf_cap_sigma_xy: Max sigma for x/y after predict/update.
        kf_cap_speed_px_per_sec: Max velocity clamp after predict/update (px/sec).
        roi_very_tiny_boost: Extra ROI size multiplier for very tiny faces.
        lost_timeout_sec: Declare LOST if no valid detection for T seconds.
        freeze_point_on_miss: Keep last output point during misses.
        log_state_transitions_only: Log only state transitions (reduce noise).
        max_process_fps: Max processing FPS for detector.
        capture_width: Capture width for camera.
        capture_height: Capture height for camera.
        force_capture_resolution: Force camera capture resolution to capture_width/height.
        enable_distance_estimation: Log distance estimates if enabled.
        effective_focal_px: Calibrated focal length in pixels (optional).
        summary_interval_sec: Summary log interval in seconds.
        disable_heavy_overlays: Skip extra overlays/text for lower CPU.
    """
    source: Union[int, str] = 0
    target_resolution: Optional[tuple[int, int]] = None
    draw_overlay: bool = True
    overlay_config: Optional[OverlayConfig] = None
    enable_roi_zoom: bool = True
    roi_zoom_scale: float = 3.0
    roi_margin: float = 0.25
    process_every_n_frames: Optional[int] = None
    tracking_process_every_n_frames: Optional[int] = None
    lost_process_every_n_frames: int = 1
    boost_processing_when_lost: bool = True
    min_face_size_px: int = 40
    use_full_range_for_small_faces: bool = True
    enable_smoothing: bool = True
    smoothing_alpha: Optional[float] = 0.3
    max_num_faces: int = 1
    min_detection_confidence: float = 0.6
    model_selection: int = 0
    power_mode: str = "balanced"
    roi_zoom_scales: Optional[list[float]] = field(default_factory=lambda: [2.0, 3.0, 4.0])
    roi_margins: Optional[list[float]] = field(default_factory=lambda: [0.25, 0.45, 0.70])
    roi_attempts_max: Optional[int] = None
    roi_expand_on_miss: bool = True
    roi_k_size: float = 1.2
    roi_k_unc: float = 2.5
    roi_k_unc_lost_mult: float = 1.5
    roi_miss_expand_factor: float = 0.35
    roi_miss_expand_cap: int = 5
    full_redetect_every_n: Optional[int] = None
    full_redetect_on_lost_every_n: int = 4
    lost_after_misses: int = 6
    reacquire_after_hits: int = 3
    keep_conf_base: float = 0.50
    acquire_conf_base: float = 0.65
    tiny_face_px: int = 80
    very_tiny_face_px: int = 55
    tiny_keep_conf: float = 0.45
    tiny_acquire_conf: float = 0.60
    face_aspect: float = 0.8
    kalman_q_pos: float = 40.0
    kalman_q_vel: float = 160.0
    kalman_q_scale: float = 0.15
    kalman_q_scale_vel: float = 0.50
    kalman_r_pos_base: float = 120.0
    kalman_r_scale_base: float = 0.30
    kalman_r_conf_floor: float = 0.2
    kalman_gate_threshold: float = 49.0
    kalman_gate_mode: str = "xy"
    kalman_gate_threshold_xys: float = 60.0
    kalman_r_pos_base_lost: float = 150.0
    kalman_r_scale_base_lost: float = 0.32
    kalman_gate_threshold_lost: float = 60.0
    kalman_gate_threshold_xys_lost: float = 75.0
    gate_thr_base: float = 25.0
    gate_thr_k: float = 0.25
    kalman_accept_high_conf_in_lost: bool = True
    kalman_accept_conf_threshold: float = 0.93
    kalman_reinit_after_rejects: int = 8
    kalman_reinit_conf_threshold: float = 0.85
    kalman_reacquire_cooldown_frames: int = 3
    kalman_reacquire_r_scale: float = 1.8
    kalman_init_pos_var: float = 200.0
    kalman_init_vel_var: float = 500.0
    kalman_init_scale_var: float = 1.0
    kalman_init_scale_vel_var: float = 1.0
    kf_use_adaptive_update: bool = True
    kf_d2_soft: float = 25.0
    kf_d2_hard: float = 200.0
    kf_R_inflate_soft: float = 6.0
    kf_R_inflate_hard: float = 20.0
    kf_residual_clip_px: float = 120.0
    kf_soft_reinit_conf: float = 0.90
    kf_soft_reinit_d2: float = 25.0
    kf_soft_reinit_inflate_P: float = 10.0
    kf_soft_reinit_cooldown_frames: int = 10
    kf_max_speed_px_per_sec: float = 2500.0
    kf_reject_streak_reinit: int = 4
    kf_reinit_conf_min: float = 0.80
    kf_reinit_inflate_P: float = 8.0
    kf_reinit_cooldown_frames: int = 15
    kf_cap_sigma_xy: float = 120.0
    kf_cap_speed_px_per_sec: float = 2500.0
    roi_very_tiny_boost: float = 1.25
    lost_timeout_sec: float = 1.2
    freeze_point_on_miss: bool = True
    log_state_transitions_only: bool = True
    max_process_fps: float = 30.0
    capture_width: int = 1280
    capture_height: int = 720
    force_capture_resolution: bool = True
    enable_distance_estimation: bool = True
    effective_focal_px: Optional[float] = None
    summary_interval_sec: float = 2.0
    disable_heavy_overlays: Optional[bool] = None
    
    def __post_init__(self):
        """Initialize overlay config if not provided."""
        if self.overlay_config is None:
            self.overlay_config = OverlayConfig()
        if self.smoothing_alpha is None:
            self.enable_smoothing = False
        if self.roi_zoom_scales is None:
            self.roi_zoom_scales = [2.0, 3.0, 4.0]
        if self.roi_margins is None:
            self.roi_margins = [0.25, 0.45, 0.70]
        self._apply_power_mode_defaults()
        self._apply_process_every_n_aliases()
        self._apply_kalman_gate_defaults()

    def _apply_power_mode_defaults(self) -> None:
        """Apply power mode defaults for processing-related parameters."""
        mode_defaults = {
            "eco": {
                "process_every_n_frames": 3,
                "roi_attempts_max": 2,
                "full_redetect_every_n": 45,
                "disable_heavy_overlays": True,
            },
            "balanced": {
                "process_every_n_frames": 1,
                "roi_attempts_max": 3,
                "full_redetect_every_n": 40,
                "disable_heavy_overlays": False,
            },
            "quality": {
                "process_every_n_frames": 1,
                "roi_attempts_max": 4,
                "full_redetect_every_n": 15,
                "disable_heavy_overlays": False,
            },
        }
        mode = (self.power_mode or "balanced").lower()
        if mode not in mode_defaults:
            mode = "balanced"
        defaults = mode_defaults[mode]
        if self.process_every_n_frames is None and self.tracking_process_every_n_frames is None:
            self.process_every_n_frames = defaults["process_every_n_frames"]
        if self.roi_attempts_max is None:
            self.roi_attempts_max = defaults["roi_attempts_max"]
        if self.full_redetect_every_n is None:
            self.full_redetect_every_n = defaults["full_redetect_every_n"]
        if self.disable_heavy_overlays is None:
            self.disable_heavy_overlays = defaults["disable_heavy_overlays"]
        self.power_mode = mode

    def _apply_process_every_n_aliases(self) -> None:
        """Keep process_every_n_frames and tracking_process_every_n_frames in sync."""
        if self.tracking_process_every_n_frames is None and self.process_every_n_frames is not None:
            self.tracking_process_every_n_frames = self.process_every_n_frames
        if self.process_every_n_frames is None and self.tracking_process_every_n_frames is not None:
            self.process_every_n_frames = self.tracking_process_every_n_frames

    def _apply_kalman_gate_defaults(self) -> None:
        """Normalize Kalman gating defaults and validate mode."""
        mode = (self.kalman_gate_mode or "xy").lower()
        if mode not in ("xy", "xys"):
            mode = "xy"
        self.kalman_gate_mode = mode
