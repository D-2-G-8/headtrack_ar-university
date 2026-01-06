# headtrack_ar

Python-пакет для трекинга головы человека в видеопотоке и наложения AR-маркеров в реальном времени.

## Описание

`headtrack_ar` — это библиотека на Python для детекции человеческих лиц, вычисления положения головы и визуализации AR-маркеров (крестиков) в области лба. Модуль полностью работает локально, не требует облачных сервисов или интернет-соединения. Использует ROI-zoom и Kalman-фильтр (2D + scale) для устойчивого трекинга и прогнозирования на кадрах без детекта.

## Области применения

Модуль предназначен для различных мирных приложений:

- **AR-фильтры и интерфейсы** — использование точки на лбу как якоря для наложения виртуальных объектов и эффектов
- **Авто-наведение камер** — автоматическое фокусирование и кадрирование на лице человека в видеопотоке
- **Автоматическое управление осветительными приборами** — следящее освещение для видео-конференций и стриминга
- **Образовательные приложения** — обучение правильной осанке
- **Игровые приложения** — интерактивные игры с управлением через положение головы
- **Стриминг и видеопроизводство** — автоматическое позиционирование виртуальных элементов интерфейса

## Требования

- Python 3.10 или выше
- Камера (для работы с живым видеопотоком) или видеофайл
- Операционная система: Windows, macOS, Linux

## Установка

### Установка зависимостей

```bash
pip install -r requirements.txt
```

Или установите пакет с зависимостями:

```bash
pip install -e .
```

### Локальная разработка

Для разработки с тестами:

```bash
pip install -e ".[dev]"
```

## Быстрый старт

### Использование в коде

```python
import cv2
from headtrack_ar import HeadTracker, TrackerConfig

# Создание конфигурации
config = TrackerConfig(
    source=0,                 # индекс камеры (или путь к видеофайлу)
    draw_overlay=True,       # включить отрисовку маркеров
    target_resolution=None,  # нативное разрешение камеры
)

# Инициализация трекера
tracker = HeadTracker(config)

# Обработка видеопотока
for frame_info in tracker.run():
    frame = frame_info.frame
    heads = frame_info.heads
  
    # Отображение количества обнаруженных голов
    print(f"Обнаружено голов: {len(heads)}")
  
    # Доступ к координатам точки на лбу
    for head in heads:
        forehead_x, forehead_y = head.forehead_point
        print(f"Точка на лбу: ({forehead_x}, {forehead_y})")
  
    # Отображение кадра
    cv2.imshow("HeadTrack AR", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов
tracker.release()
cv2.destroyAllWindows()
```

### Запуск демо через CLI

После установки пакета можно запустить демо-приложение:

```bash
headtrack-demo
```

Или напрямую:

```bash
python -m headtrack_ar.demo
```

#### Параметры командной строки

```bash
python -m headtrack_ar.demo --help
```

Доступные опции:

- `--source`: Источник видео (индекс камеры или путь к файлу, по умолчанию: 0)
- `--width`: Ширина кадра (если не указано — нативное)
- `--height`: Высота кадра (если не указано — нативное)
- `--no-overlay`: Отключить отрисовку маркеров
- `--color`: Цвет маркера (green, red, blue, yellow, cyan, magenta, white, по умолчанию: green)
- `--size`: Размер маркера в пикселях (по умолчанию: 20)
- `--thickness`: Толщина линий маркера в пикселях (по умолчанию: 2)
- `--model`: Модель детекции лиц (0=short-range, 1=full-range, по умолчанию: 0)
- `--power-mode`: Режим энергопотребления (eco, balanced, quality, по умолчанию: balanced)
- `--confidence`: Минимальная уверенность детекции (по умолчанию: 0.6)
- `--keep-conf-base`/`--keep-conf-threshold`: Базовый порог уверенности для удержания трека (по умолчанию: 0.50)
- `--acquire-conf-base`/`--acquire-conf-threshold`: Базовый порог уверенности для повторного захвата (по умолчанию: 0.65)
- `--tiny-face-px`: Порог tiny-face по высоте лица в пикселях (по умолчанию: 80)
- `--tiny-keep-conf`: Порог keep для tiny-face (по умолчанию: 0.45)
- `--tiny-acquire-conf`: Порог acquire для tiny-face (по умолчанию: 0.60)
- `--lost-after-misses`: LOST после N промахов (по умолчанию: 6)
- `--reacquire-after-hits`: FOUND после N попаданий (по умолчанию: 3)
- `--lost-timeout-sec`: LOST по таймауту (по умолчанию: 1.2)
- `--freeze-point-on-miss`/`--freeze-output-when-missed`: Заморозка последней точки при промахах
- `--log-state-transitions-only`/`--no-log-state-transitions-only`: Логировать только переходы состояний
- `--no-roi-zoom`: Отключить ROI-зум пайплайн
- `--roi-zoom-scale`: Масштаб ROI перед детекцией (по умолчанию: 3.0)
- `--roi-margin`: Отступы вокруг last bbox для ROI (по умолчанию: 0.25)
- `--roi-zoom-scales`: Список масштабов ROI-pyramid (через пробел)
- `--roi-margins`: Список отступов ROI-pyramid (через пробел)
- `--roi-attempts-max`: Максимум попыток ROI на кадр
- `--roi-expand-on-miss`/`--no-roi-expand-on-miss`: Расширять ROI при серии промахов
- `--tracking-process-every-n-frames`/`--process-every-n-frames`: Запуск детектора каждые N кадров в tracking (по умолчанию: от power-mode)
- `--lost-process-every-n-frames`: Запуск детектора каждые N кадров в lost (по умолчанию: 1)
- `--boost-processing-when-lost`/`--no-boost-processing-when-lost`: Ускорять обработку в lost
- `--full-frame-redetect-every-n`/`--full-redetect-every-n`: Полный детект не чаще чем раз в N кадров (по умолчанию: от power-mode)
- `--full-redetect-on-lost-every-n`: Полный детект при LOST раз в N обработок (по умолчанию: 6)
- `--min-face-size-px`: Минимальный размер лица для "near" режима (по умолчанию: 40)
- `--no-full-range-for-small-faces`: Не включать full-range для маленьких/неизвестных лиц
- `--no-smoothing`: Отключить сглаживание точки
- `--smoothing-alpha`: Коэффициент EMA для сглаживания (по умолчанию: 0.3)
- `--max-process-fps`: Лимит частоты обработки (по умолчанию: 15.0)
- `--max-num-faces`: Максимум лиц для трекинга (по умолчанию: 1)
- `--capture-width`: Ширина захвата камеры (по умолчанию: 1280)
- `--capture-height`: Высота захвата камеры (по умолчанию: 720)
- `--force-capture-resolution`/`--no-force-capture-resolution`: Принудительно задавать capture-разрешение
- `--enable-distance-estimation`/`--no-enable-distance-estimation`: Логи дистанции (approx)
- `--effective-focal-px`: Использовать калиброванную фокусную длину
- `--calibration-file`: Загрузить калибровку из файла
- `--calibrate-distance`: Запустить простую калибровку дистанции

По умолчанию capture-разрешение принудительно выставляется в 1280x720.
Используйте `--no-force-capture-resolution`, чтобы вернуть авто-выбор максимального разрешения камеры.
Параметры `--capture-width/--capture-height` задают нативное разрешение захвата камеры.

#### Примеры использования

```bash
# Запуск с камерой по умолчанию
python -m headtrack_ar.demo

# Запуск с видеофайлом
python -m headtrack_ar.demo --source path/to/video.mp4

# Запуск с кастомными параметрами
python -m headtrack_ar.demo --source 0 --width 1280 --height 720 --color red --size 30

# Без отрисовки маркеров (только детекция)
python -m headtrack_ar.demo --no-overlay
```

#### Калибровка дистанции

Для более честной оценки дистанции используйте простой режим калибровки:

- `--calibrate-distance` — попросит ввести реальную дистанцию и сохранит `effective_focal_px` рядом с логом.
- `--calibration-file` или `--effective-focal-px` — использовать сохраненную калибровку.

## API документация

### Основные классы

#### `HeadTracker`

Главный класс для трекинга голов в видеопотоке.

```python
tracker = HeadTracker(config: TrackerConfig)
```

**Методы:**

- `run()`: Генератор, возвращающий `FrameInfo` для каждого кадра
- `process_frame(frame: np.ndarray) -> FrameInfo`: Обработка одного кадра
- `release()`: Освобождение ресурсов

#### `TrackerConfig`

Конфигурация трекера.

```python
config = TrackerConfig(
    source: Union[int, str] = 0,              # источник видео
    target_resolution: Optional[tuple] = None,
    draw_overlay: bool = True,                # отрисовка маркеров
    overlay_config: Optional[OverlayConfig] = None,
    enable_roi_zoom: bool = True,             # включить ROI-зум
    roi_zoom_scale: float = 3.0,              # масштаб ROI перед детекцией
    roi_margin: float = 0.25,                 # отступ ROI вокруг bbox
    roi_zoom_scales: list[float] = [2.0, 3.0, 4.0],
    roi_margins: list[float] = [0.25, 0.45, 0.70],
    power_mode: str = "balanced",             # eco/balanced/quality
    tracking_process_every_n_frames: int = 1, # детект в tracking (от режима)
    lost_process_every_n_frames: int = 1,     # детект в lost
    boost_processing_when_lost: bool = True,
    roi_attempts_max: int = 3,                # попыток ROI на кадр (от режима)
    roi_expand_on_miss: bool = True,
    roi_k_unc_lost_mult: float = 1.5,
    roi_k_size: float = 1.2,                  # размер ROI по высоте лица (Kalman)
    roi_k_unc: float = 2.5,                   # вклад неопределенности в ROI (Kalman)
    roi_miss_expand_factor: float = 0.35,
    roi_miss_expand_cap: int = 5,
    roi_very_tiny_boost: float = 1.25,
    full_redetect_every_n: int = 40,          # полный детект раз в N обработок (от режима)
    full_redetect_on_lost_every_n: int = 4,   # полный детект при LOST
    min_face_size_px: int = 40,               # порог "маленького" лица
    use_full_range_for_small_faces: bool = True,
    lost_after_misses: int = 6,
    reacquire_after_hits: int = 3,
    keep_conf_base: float = 0.50,
    acquire_conf_base: float = 0.65,
    tiny_face_px: int = 80,
    very_tiny_face_px: int = 55,
    tiny_keep_conf: float = 0.45,
    tiny_acquire_conf: float = 0.60,
    face_aspect: float = 0.8,                 # аспект bbox для Kalman-выхода
    kalman_q_pos: float = 40.0,
    kalman_q_vel: float = 160.0,
    kalman_q_scale: float = 0.15,
    kalman_q_scale_vel: float = 0.50,
    kalman_r_pos_base: float = 120.0,
    kalman_r_scale_base: float = 0.30,
    kalman_r_conf_floor: float = 0.2,
    kalman_gate_threshold: float = 49.0,
    kalman_gate_mode: str = "xy",
    kalman_gate_threshold_xys: float = 60.0,
    kalman_r_pos_base_lost: float = 150.0,
    kalman_r_scale_base_lost: float = 0.32,
    kalman_gate_threshold_lost: float = 60.0,
    kalman_gate_threshold_xys_lost: float = 75.0,
    gate_thr_base: float = 25.0,
    gate_thr_k: float = 0.25,
    kalman_accept_high_conf_in_lost: bool = True,
    kalman_accept_conf_threshold: float = 0.93,
    kalman_reinit_after_rejects: int = 8,
    kalman_reinit_conf_threshold: float = 0.85,
    kalman_reacquire_cooldown_frames: int = 3,
    kalman_reacquire_r_scale: float = 1.8,
    kalman_init_pos_var: float = 200.0,
    kalman_init_vel_var: float = 500.0,
    kalman_init_scale_var: float = 1.0,
    kalman_init_scale_vel_var: float = 1.0,
    kf_use_adaptive_update: bool = True,
    kf_d2_soft: float = 25.0,
    kf_d2_hard: float = 200.0,
    kf_R_inflate_soft: float = 6.0,
    kf_R_inflate_hard: float = 20.0,
    kf_residual_clip_px: float = 120.0,
    kf_soft_reinit_conf: float = 0.90,
    kf_soft_reinit_d2: float = 25.0,
    kf_soft_reinit_inflate_P: float = 10.0,
    kf_soft_reinit_cooldown_frames: int = 10,
    kf_max_speed_px_per_sec: float = 2500.0,
    kf_reject_streak_reinit: int = 4,
    kf_reinit_conf_min: float = 0.80,
    kf_reinit_inflate_P: float = 8.0,
    kf_reinit_cooldown_frames: int = 15,
    kf_cap_sigma_xy: float = 120.0,
    kf_cap_speed_px_per_sec: float = 2500.0,
    lost_timeout_sec: float = 1.2,
    freeze_point_on_miss: bool = True,
    log_state_transitions_only: bool = True,
    enable_smoothing: bool = True,            # включить сглаживание
    smoothing_alpha: Optional[float] = 0.3,   # EMA сглаживание (None = отключено)
    max_num_faces: int = 1,                   # максимум лиц
    min_detection_confidence: float = 0.6,    # минимальная уверенность детекции
    model_selection: int = 0,                 # 0=short-range, 1=full-range
    max_process_fps: float = 30.0,
    capture_width: int = 1280,
    capture_height: int = 720,
    force_capture_resolution: bool = True,
    enable_distance_estimation: bool = True,
    effective_focal_px: Optional[float] = None,
    summary_interval_sec: float = 2.0,
    disable_heavy_overlays: bool = False,
)
```

#### Параметры TrackerConfig

- `source`: Источник видео (индекс камеры или путь к файлу).
- `target_resolution`: Целевое разрешение кадра или None для нативного/auto-режима захвата.
- `draw_overlay`: Рисовать AR-маркер (плюсик) на кадре.
- `overlay_config`: Цвет/размер/толщина маркера.
- `enable_roi_zoom`: Включить ROI-зум пайплайн для экономии CPU.
- `roi_zoom_scale`: Масштаб увеличения ROI перед детекцией.
- `roi_margin`: Отступы вокруг последнего bbox при формировании ROI.
- `roi_zoom_scales`: Список масштабов для ROI-pyramid.
- `roi_margins`: Список отступов для ROI-pyramid.
- `roi_attempts_max`: Максимум попыток ROI на обработку.
- `roi_expand_on_miss`: Расширять ROI при серии промахов.
- `roi_k_size`: Размер ROI как доля высоты лица (Kalman).
- `roi_k_unc`: Вклад неопределенности Kalman в размер ROI.
- `roi_k_unc_lost_mult`: Множитель неопределенности ROI в состоянии LOST.
- `roi_miss_expand_factor`: Насколько расширять ROI при промахах.
- `roi_miss_expand_cap`: Максимум промахов, учитываемых в расширении ROI.
- `roi_very_tiny_boost`: Дополнительный множитель размера ROI для very tiny лиц.
- `power_mode`: Режим энергопотребления (eco/balanced/quality).
- `tracking_process_every_n_frames`: Запуск детектора каждые N кадров в tracking (по умолчанию от режима).
- `lost_process_every_n_frames`: Запуск детектора каждые N кадров в lost.
- `boost_processing_when_lost`: Ускорять обработку в lost.
- `process_every_n_frames`: Алиас для `tracking_process_every_n_frames`.
- `full_redetect_every_n`: Как часто делать полный детект на всем кадре (по умолчанию от режима).
- `full_redetect_on_lost_every_n`: Полный детект в режиме LOST.
- `min_face_size_px`: Если лицо меньше этого порога, включается дальний режим.
- `use_full_range_for_small_faces`: Использовать full-range при маленьких/неизвестных лицах.
- `lost_after_misses`: LOST после N промахов.
- `reacquire_after_hits`: FOUND после N попаданий.
- `keep_conf_base`: Базовый порог удержания трека.
- `acquire_conf_base`: Базовый порог повторного захвата.
- `tiny_face_px`: Порог tiny-face по высоте лица.
- `very_tiny_face_px`: Порог very tiny-face по высоте лица.
- `tiny_keep_conf`: Порог keep для tiny-face.
- `tiny_acquire_conf`: Порог acquire для tiny-face.
- `face_aspect`: Аспект лица (w/h) для реконструкции bbox из Kalman-состояния.
- `kalman_q_pos`: Шум процесса для позиции Kalman.
- `kalman_q_vel`: Шум процесса для скорости Kalman.
- `kalman_q_scale`: Шум процесса для масштаба (log).
- `kalman_q_scale_vel`: Шум процесса для скорости масштаба (log).
- `kalman_r_pos_base`: Базовый шум измерения для позиции.
- `kalman_r_scale_base`: Базовый шум измерения для масштаба (log).
- `kalman_r_conf_floor`: Минимальный confidence для расчета шумов измерения.
- `kalman_gate_threshold`: Базовый порог гейтинга (Mahalanobis distance^2).
- `kalman_gate_mode`: Режим гейтинга ("xy" или "xys").
- `kalman_gate_threshold_xys`: Порог гейтинга для режима "xys".
- `kalman_r_pos_base_lost`: Базовый шум измерения для позиции в LOST.
- `kalman_r_scale_base_lost`: Базовый шум измерения для масштаба в LOST.
- `kalman_gate_threshold_lost`: Порог гейтинга в LOST.
- `kalman_gate_threshold_xys_lost`: Порог гейтинга в LOST для режима "xys".
- `gate_thr_base`: База для адаптивного порога гейтинга.
- `gate_thr_k`: Коэффициент для адаптивного порога гейтинга по (sigma_x + sigma_y).
- `kalman_accept_high_conf_in_lost`: Принимать высококонф. измерения в LOST.
- `kalman_accept_conf_threshold`: Порог confidence для forced accept в LOST.
- `kalman_reinit_after_rejects`: Реинициализация Kalman после N reject в LOST.
- `kalman_reinit_conf_threshold`: Порог confidence для реинициализации в LOST.
- `kalman_reacquire_cooldown_frames`: Количество кадров с усиленным сглаживанием после LOST->TRACKING.
- `kalman_reacquire_r_scale`: Множитель шумов измерения в периоде стабилизации.
- `kalman_init_pos_var`: Начальная дисперсия позиции.
- `kalman_init_vel_var`: Начальная дисперсия скорости.
- `kalman_init_scale_var`: Начальная дисперсия масштаба (log).
- `kalman_init_scale_vel_var`: Начальная дисперсия скорости масштаба (log).
- `kf_use_adaptive_update`: Использовать adaptive update вместо жесткого reject.
- `kf_d2_soft`: Порог Mahalanobis distance^2 для "подозрительного" измерения.
- `kf_d2_hard`: Порог Mahalanobis distance^2 для сильного выброса.
- `kf_R_inflate_soft`: Множитель R для soft-режима.
- `kf_R_inflate_hard`: Множитель R для hard-режима.
- `kf_residual_clip_px`: Клип residual (dx, dy) перед adaptive update.
- `kf_soft_reinit_conf`: Порог confidence для soft reinit по reject.
- `kf_soft_reinit_d2`: Порог d2 для soft reinit.
- `kf_soft_reinit_inflate_P`: Множитель P при soft reinit.
- `kf_soft_reinit_cooldown_frames`: Кулдаун между soft reinit.
- `kf_max_speed_px_per_sec`: Лимит скорости при soft reinit (px/sec).
- `kf_reject_streak_reinit`: Порог reject_streak для reinit.
- `kf_reinit_conf_min`: Минимальный confidence для reinit по streak.
- `kf_reinit_inflate_P`: Множитель P при reinit по streak.
- `kf_reinit_cooldown_frames`: Кулдаун между reinit по streak.
- `kf_cap_sigma_xy`: Максимум sigma для x/y после predict/update.
- `kf_cap_speed_px_per_sec`: Максимум скорости после predict/update (px/sec).
- `lost_timeout_sec`: Таймаут для перехода в LOST.
- `freeze_point_on_miss`: Замораживать последнюю точку при промахах.
- `log_state_transitions_only`: Логировать только переходы состояний.
- `enable_smoothing`: Включить EMA сглаживание координат.
- `smoothing_alpha`: Сила сглаживания (больше — менее плавно).
- `max_num_faces`: Максимальное число лиц для трекинга.
- `min_detection_confidence`: Минимальная уверенность детекции.
- `model_selection`: Базовый выбор модели (0=short-range, 1=full-range).
- `max_process_fps`: Лимит частоты обработки.
- `capture_width`: Ширина захвата камеры.
- `capture_height`: Высота захвата камеры.
- `force_capture_resolution`: Принудительно задавать capture-разрешение.
- `enable_distance_estimation`: Логи дистанции (approx).
- `effective_focal_px`: Калиброванная фокусная длина.
- `summary_interval_sec`: Интервал summary-логов.
- `disable_heavy_overlays`: Отключить дополнительный текстовый оверлей.

#### `HeadInfo`

Информация об обнаруженной голове.

- `bbox: tuple[int, int, int, int]` — ограничивающий прямоугольник (x, y, width, height)
- `forehead_point: tuple[int, int]` — координаты точки на лбу (x, y)
- `landmarks: Optional[list[tuple[int, int]]]` — ключевые точки лица
- `confidence: Optional[float]` — уверенность детекции (0.0 до 1.0)

#### `FrameInfo`

Информация об обработанном кадре.

- `frame: np.ndarray` — обработанный кадр (с маркерами, если включено)
- `raw_frame: Optional[np.ndarray]` — исходный кадр
- `heads: list[HeadInfo]` — список обнаруженных голов
- `detected_count: int` — количество голов, найденных на последней попытке детекта

## Производительность

Модуль оптимизирован для работы в реальном времени:

- **Целевая производительность**: 15+ FPS на среднем ноутбуке при разрешении 640x480
- **Технологии**: Использует MediaPipe для эффективной детекции лиц и ключевых точек
- **Сглаживание**: Экспоненциальное скользящее среднее для стабилизации координат
- **Kalman 2D+scale**: Сглаживание и прогноз cx/cy/scale, плюс adaptive-gating/soft update для защиты от выбросов

### Рекомендации по производительности

- Для лучшей производительности используйте разрешение 640x480 или ниже
- Сглаживание координат улучшает стабильность, но добавляет небольшую задержку
- Работает на CPU, поддержка GPU не требуется (но может улучшить производительность при наличии)

## Тестирование

Запуск тестов:

```bash
pytest
```

Или с покрытием:

```bash
pytest --cov=headtrack_ar
```

## Структура проекта

```
headtrack_ar/
├── __init__.py          # Экспорты пакета
├── config.py            # Конфигурация
├── types.py             # Структуры данных
├── video_source.py      # Работа с видеопотоком
├── detector.py          # Детекция лиц (MediaPipe)
├── tracker.py           # Основная логика трекинга
├── kalman.py            # Kalman фильтр 2D+scale
├── overlay.py           # Отрисовка AR-маркеров
└── demo.py              # CLI демо-приложение

tests/
└── test_basic.py        # Базовые тесты
```

## Технические детали

### Алгоритм вычисления точки на лбу

1. На каждом обработанном кадре Kalman предсказывает состояние (cx, cy, log(h))
2. ROI строится вокруг прогноза с учетом масштаба и неопределенности
3. Детекция выполняется в ROI, а full-frame включается только при необходимости
4. Из bbox строится измерение (cx, cy, log(h)), Kalman обновляется с гейтингом
5. При отсутствии измерения используется прогноз Kalman
6. Точка на лбу вычисляется по фильтрованному bbox (x + 0.5w, y + 0.2h) либо по прогнозу
7. Применяется сглаживание координат для стабилизации (экспоненциальное скользящее среднее)

### Поддержка нескольких голов

Модуль поддерживает одновременный трекинг нескольких голов в кадре. Каждая голова получает отдельный маркер и индивидуальное сглаживание координат.

## Примечания

- Модуль полностью работает локально, без необходимости интернет-соединения
- Не требует API ключей или облачных сервисов
- Использует открытые библиотеки: OpenCV и MediaPipe
