"""Interactive asset creation for the current base-photo workflow.

Run this once to capture the assets used by ScrumSurvivor.

Captures in order:
  1 x base photo               (neutral, mouth closed)
  6 x named idle animation clips

Controls (all in the preview window):
    SPACE  - capture photo / start or stop recording
    ENTER  - capture photo / start or stop recording / confirm and keep
  R      - retake
  ESC    - quit

Usage:
    .venv/Scripts/python.exe -m scrumsurvivor.create_assets
    .venv/Scripts/python.exe src/scrumsurvivor/create_assets.py
"""

from __future__ import annotations

import cv2
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path

try:
    from cv2_enumerate_cameras import enumerate_cameras as _enumerate_cameras_helper
except ImportError:
    _enumerate_cameras_helper = None


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ASSETS_DIR = _PROJECT_ROOT / "assets"
IDLE_DIR = _PROJECT_ROOT / "assets" / "idle_clips"
BASE_PHOTO = _PROJECT_ROOT / "assets" / "base_photo.png"
THEMES_DIR = _PROJECT_ROOT / "assets" / "themes"


@dataclass(frozen=True, slots=True)
class _AssetPaths:
    base_photo: Path
    idle_dir: Path
    theme_name: str | None

    def __str__(self) -> str:
        return self.theme_name or "(default)"

WINDOW = "ScrumSurvivor - Asset Creator"
_UNMIRROR_CAMERA_FEED = True
_KEY_ENTER = {10, 13}
_KEY_SPACE = {32}
_KEY_ESCAPE = 27
_PREVIOUS_SELECTION_KEYS = {ord("w"), ord("W"), 0x250000, 0x260000}
_NEXT_SELECTION_KEYS = {ord("s"), ord("S"), 0x270000, 0x280000}


@dataclass(frozen=True, slots=True)
class IdleClipSpec:
    slug: str
    title: str
    instruction: str
    duration_hint: str


@dataclass(frozen=True, slots=True)
class CameraOption:
    slot: int
    capture_index: int
    name: str
    backend: int = cv2.CAP_ANY
    path: str = ""


IDLE_CLIPS = [
    IdleClipSpec(
        slug="blink_standard",
        title="Blink (standard)",
        instruction="Neutral -> blink once -> neutral.",
        duration_hint="~1.0 s",
    ),
    IdleClipSpec(
        slug="blink_slow_drowsy",
        title="Blink (slow/drowsy)",
        instruction="A slightly heavier, slower blink, then back to neutral.",
        duration_hint="~1.5 s",
    ),
    IdleClipSpec(
        slug="breathing_shift",
        title="Breathing shift",
        instruction="One natural breath; shoulders rise and fall, head stays still.",
        duration_hint="~3.0 s",
    ),
    IdleClipSpec(
        slug="single_slow_nod",
        title="Single slow nod",
        instruction="One small acknowledgement nod, then return to neutral.",
        duration_hint="~1.5 s",
    ),
    IdleClipSpec(
        slug="head_tilt_return",
        title="Slight head tilt, return",
        instruction="Tilt your head slightly to one side, then return to center.",
        duration_hint="~2.0 s",
    ),
    IdleClipSpec(
        slug="glance_down_return",
        title="Brief glance down, return",
        instruction="Eyes drop as if checking notes, then return to camera.",
        duration_hint="~2.0 s",
    ),
]


def _load_face_detector():
    _ensure_src_root_on_path()
    try:
        from scrumsurvivor.detection.face_detector import detect_face_once
    except ModuleNotFoundError:
        from scrumsurvivor.detection.face_detector import detect_face_once
    return detect_face_once


def _ensure_src_root_on_path() -> None:
    src_root = _PROJECT_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _camera_identity_key(name: str, path: str) -> str:
    normalized_name = name.strip().lower()
    normalized_path = path.strip().lower()
    if normalized_path:
        return normalized_path.split("#{", 1)[0]
    return f"name:{normalized_name}"


def _build_camera_options(cameras: list[object]) -> list[CameraOption]:
    deduped: dict[str, CameraOption] = {}
    for camera in cameras:
        capture_index = int(getattr(camera, "index"))
        name = str(getattr(camera, "name", "")).strip() or f"Camera {capture_index}"
        backend = int(getattr(camera, "backend", cv2.CAP_ANY) or cv2.CAP_ANY)
        path = str(getattr(camera, "path", "") or "")
        key = _camera_identity_key(name, path)
        option = CameraOption(
            slot=0,
            capture_index=capture_index,
            name=name,
            backend=backend,
            path=path,
        )
        existing = deduped.get(key)
        if existing is None or option.capture_index < existing.capture_index:
            deduped[key] = option

    ordered = sorted(deduped.values(), key=lambda option: option.capture_index)
    return [
        CameraOption(
            slot=position,
            capture_index=option.capture_index,
            name=option.name,
            backend=option.backend,
            path=option.path,
        )
        for position, option in enumerate(ordered, start=1)
    ]


def _probe_camera_options() -> list[CameraOption]:
    options: list[CameraOption] = []
    for position, capture_index in enumerate(range(10), start=1):
        cap = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                options.append(
                    CameraOption(
                        slot=position,
                        capture_index=capture_index,
                        name=f"Camera {capture_index}",
                        backend=cv2.CAP_DSHOW,
                    )
                )
        cap.release()
    return options


def _camera_selection_canvas(
    options: list[CameraOption],
    selected_pos: int,
    width: int = 960,
    height: int = 540,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    panel = canvas.copy()
    cv2.rectangle(panel, (24, 24), (width - 24, height - 24), (35, 35, 35), -1)
    cv2.addWeighted(panel, 0.75, canvas, 0.25, 0, canvas)

    cv2.putText(
        canvas,
        "Available cameras",
        (48, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Use Up/Down or W/S to highlight, Enter/Space to select, 0-9 for direct pick",
        (48, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    y = 148
    for position, option in enumerate(options):
        if position == selected_pos:
            cv2.rectangle(canvas, (42, y - 24), (width - 42, y + 8), (0, 120, 215), -1)
            color = (255, 255, 255)
            prefix = ">"
        else:
            color = (220, 220, 220)
            prefix = " "
        label = f"{prefix} [{option.slot}] {option.name}"
        cv2.putText(
            canvas,
            label,
            (54, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2 if position == selected_pos else 1,
            cv2.LINE_AA,
        )
        y += 48

    cv2.putText(
        canvas,
        "Selection numbers are wizard slots. The actual capture index is resolved internally.",
        (48, height - 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "After selection you get a live preview confirmation before recording begins.",
        (48, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _wrap_canvas_text(
    text: str,
    max_width: int,
    font_scale: float = 0.62,
    thickness: int = 1,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0][0]
        if candidate_width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _instruction_canvas(
    title: str,
    lines: list[str],
    footer: str = "ENTER/SPACE to continue | ESC to quit",
    width: int = 960,
    height: int = 540,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    panel = canvas.copy()
    cv2.rectangle(panel, (24, 24), (width - 24, height - 24), (35, 35, 35), -1)
    cv2.addWeighted(panel, 0.78, canvas, 0.22, 0, canvas)

    cv2.putText(
        canvas,
        title,
        (48, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    y = 116
    for line in lines:
        wrapped = _wrap_canvas_text(line, width - 110)
        for wrapped_line in wrapped:
            cv2.putText(
                canvas,
                wrapped_line,
                (54, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            y += 30
        y += 6

    cv2.putText(
        canvas,
        footer,
        (48, height - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _wait_key(delay_ms: int) -> int:
    return cv2.waitKeyEx(delay_ms)


def _window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _quit_program(cap: cv2.VideoCapture | None = None) -> None:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\n  Quit.")
    sys.exit(0)


def _is_action_key(key: int) -> bool:
    return key in _KEY_ENTER or key in _KEY_SPACE


def _selection_delta_for_key(key: int) -> int:
    if key in _PREVIOUS_SELECTION_KEYS:
        return -1
    if key in _NEXT_SELECTION_KEYS:
        return 1
    return 0


def _show_instruction_screen(
    title: str,
    lines: list[str],
    footer: str = "ENTER/SPACE to continue | ESC to quit",
) -> None:
    while True:
        cv2.imshow(WINDOW, _instruction_canvas(title, lines, footer=footer))
        key = _wait_key(30)
        if _window_closed() or key == _KEY_ESCAPE:
            _quit_program()
        if _is_action_key(key):
            return


def _normalize_capture_frame(frame: np.ndarray) -> np.ndarray:
    if _UNMIRROR_CAMERA_FEED:
        return cv2.flip(frame, 1)
    return frame.copy()


def _read_capture_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok:
        return None
    return _normalize_capture_frame(frame)


def _idle_clip_output_path(spec: IdleClipSpec, idle_dir: Path = IDLE_DIR) -> Path:
    return idle_dir / f"{spec.slug}.mp4"


def _alignment_region(
    frame_shape: tuple[int, int, int] | tuple[int, int],
    face_rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = face_rect
    x1 = max(0, x - int(round(w * 1.4)))
    y1 = max(0, y - int(round(h * 0.9)))
    x2 = min(frame_w, x + w + int(round(w * 1.4)))
    y2 = min(frame_h, y + h + int(round(h * 3.2)))
    return x1, y1, x2, y2


def _build_alignment_guide(base_photo: np.ndarray) -> np.ndarray:
    guide = np.zeros_like(base_photo)
    detect_face_once = _load_face_detector()
    face_rect = detect_face_once(base_photo)

    gray = cv2.cvtColor(base_photo, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 150)

    if face_rect is not None:
        x1, y1, x2, y2 = _alignment_region(base_photo.shape, face_rect)
        roi_edges = edges[y1:y2, x1:x2].copy()
    else:
        x1, y1, x2, y2 = 0, 0, base_photo.shape[1], base_photo.shape[0]
        roi_edges = edges.copy()

    roi_edges = cv2.dilate(roi_edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(200, int(round((x2 - x1) * (y2 - y1) * 0.01)))
    significant = [c for c in contours if cv2.contourArea(c) >= min_area]

    if significant:
        shifted = [c + np.array([[[x1, y1]]], dtype=c.dtype) for c in significant[:6]]
        cv2.drawContours(guide, shifted, -1, (0, 215, 255), 2)
    else:
        cv2.rectangle(guide, (x1, y1), (x2, y2), (0, 215, 255), 2)

    if face_rect is not None:
        x, y, w, h = face_rect
        cv2.rectangle(guide, (x, y), (x + w, y + h), (255, 180, 0), 1)

    return guide


def _build_positioning_guide_from_face(
    h: int,
    w: int,
    face_rect: tuple[int, int, int, int],
) -> np.ndarray:
    """Build a realistic positioning guide anchored to a detected live face."""
    guide = np.zeros((h, w, 3), dtype=np.uint8)
    color_body = (0, 215, 80)
    color_cross = (0, 215, 180)

    x, y, fw, fh = face_rect
    face_cx = x + fw // 2

    # Tuned from user alignment feedback: fuller oval around head/headset,
    # with the center slightly lower than the raw face-box midpoint.
    head_ax = max(42, int(round(fw * 0.68)))
    head_ay = max(52, int(round(fh * 0.92)))
    head_cy = y + int(round(fh * 0.52))
    cv2.ellipse(
        guide,
        (face_cx, head_cy),
        (head_ax, head_ay),
        0,
        0,
        360,
        color_body,
        2,
        cv2.LINE_AA,
    )

    neck_top_y = min(h - 1, head_cy + head_ay - int(round(fh * 0.04)))
    neck_bot_y = min(h - 1, neck_top_y + int(round(fh * 0.20)))
    neck_half = max(10, int(round(fw * 0.15)))
    cv2.line(
        guide,
        (face_cx - neck_half, neck_top_y),
        (face_cx - neck_half, neck_bot_y),
        color_body,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        guide,
        (face_cx + neck_half, neck_top_y),
        (face_cx + neck_half, neck_bot_y),
        color_body,
        1,
        cv2.LINE_AA,
    )

    shoulder_y = min(h - 1, neck_bot_y + int(round(fh * 0.15)))
    shoulder_lx = max(0, face_cx - int(round(fw * 1.20)))
    shoulder_rx = min(w - 1, face_cx + int(round(fw * 1.20)))
    cv2.line(
        guide,
        (face_cx - neck_half, neck_bot_y),
        (shoulder_lx, shoulder_y),
        color_body,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        guide,
        (face_cx + neck_half, neck_bot_y),
        (shoulder_rx, shoulder_y),
        color_body,
        2,
        cv2.LINE_AA,
    )

    torso_bot_y = min(h - 1, shoulder_y + int(round(fh * 1.45)))
    cv2.line(guide, (shoulder_lx, shoulder_y), (shoulder_lx, torso_bot_y), color_body, 1, cv2.LINE_AA)
    cv2.line(guide, (shoulder_rx, shoulder_y), (shoulder_rx, torso_bot_y), color_body, 1, cv2.LINE_AA)
    cv2.line(guide, (shoulder_lx, torso_bot_y), (shoulder_rx, torso_bot_y), color_body, 1, cv2.LINE_AA)

    cv2.line(guide, (face_cx, 0), (face_cx, h), (0, 60, 20), 1)

    # Slightly below eye-line for easier "center your face" alignment.
    eye_y = y + int(round(fh * 0.46))
    cross = 16
    cv2.line(guide, (face_cx - cross, eye_y), (face_cx + cross, eye_y), color_cross, 2, cv2.LINE_AA)
    cv2.line(guide, (face_cx, eye_y - cross), (face_cx, eye_y + cross), color_cross, 2, cv2.LINE_AA)

    return guide


def _build_static_positioning_guide(h: int, w: int) -> np.ndarray:
    """Draw a static, always-centred head/shoulders/torso positioning guide.

    Proportions are based on realistic video-call framing for lipsync:
    - Head occupies the upper third of the frame with natural headroom.
    - Head ellipse is taller than wide (human proportion ~1:1.3).
    - Shoulders span roughly 55% of frame width (normal seated posture).
    - Eyes land on the upper rule-of-thirds line via the cyan crosshair.

    Using frame height for both axes keeps the ellipse shape consistent
    regardless of whether the camera is 4:3 or 16:9.
    """
    guide = np.zeros((h, w, 3), dtype=np.uint8)
    color_body = (0, 215, 80)    # green — head + body outline
    color_cross = (0, 215, 180)  # cyan — eye-level crosshair

    cx = w // 2

    # Head ellipse — centred horizontally.
    # Lowered further to match realistic desk-meeting webcam posture where
    # the face centre sits near mid-frame rather than upper-third framing.
    head_cy = int(h * 0.50)       # centre at 50 % from top
    head_ax = int(h * 0.16)       # half-width
    head_ay = int(h * 0.21)       # half-height  (~1 : 1.3 ratio)
    cv2.ellipse(guide, (cx, head_cy), (head_ax, head_ay), 0, 0, 360, color_body, 2, cv2.LINE_AA)

    # Neck — two short parallel lines below the head ellipse.
    neck_top_y = head_cy + head_ay
    neck_bot_y = int(h * 0.79)
    neck_half  = int(w * 0.05)     # ~10 % of frame width total
    cv2.line(guide, (cx - neck_half, neck_top_y), (cx - neck_half, neck_bot_y), color_body, 1, cv2.LINE_AA)
    cv2.line(guide, (cx + neck_half, neck_top_y), (cx + neck_half, neck_bot_y), color_body, 1, cv2.LINE_AA)

    # Shoulders — gentle slope from neck base to shoulder tips.
    # Span ~56 % of frame width, typical for an adult seated at a desk.
    shoulder_y  = int(h * 0.83)
    shoulder_lx = int(w * 0.22)
    shoulder_rx = int(w * 0.78)
    cv2.line(guide, (cx - neck_half, neck_bot_y), (shoulder_lx, shoulder_y), color_body, 2, cv2.LINE_AA)
    cv2.line(guide, (cx + neck_half, neck_bot_y), (shoulder_rx, shoulder_y), color_body, 2, cv2.LINE_AA)

    # Upper torso box — shoulder tips down close to frame bottom.
    torso_bot_y = int(h * 0.97)
    cv2.line(guide, (shoulder_lx, shoulder_y),  (shoulder_lx, torso_bot_y), color_body, 1, cv2.LINE_AA)
    cv2.line(guide, (shoulder_rx, shoulder_y),  (shoulder_rx, torso_bot_y), color_body, 1, cv2.LINE_AA)
    cv2.line(guide, (shoulder_lx, torso_bot_y), (shoulder_rx, torso_bot_y), color_body, 1, cv2.LINE_AA)

    # Faint vertical centre line to help with left/right centering.
    cv2.line(guide, (cx, 0), (cx, h), (0, 60, 20), 1)

    # Eye-level crosshair — place near head midpoint for real webcam eye line.
    eye_y = int(head_cy + head_ay * 0.02)
    cross = 18
    cv2.line(guide, (cx - cross, eye_y), (cx + cross, eye_y), color_cross, 2, cv2.LINE_AA)
    cv2.line(guide, (cx, eye_y - cross), (cx, eye_y + cross), color_cross, 2, cv2.LINE_AA)

    return guide


def _apply_alignment_guide(frame: np.ndarray, guide: np.ndarray | None) -> np.ndarray:
    if guide is None:
        return frame

    out = frame.copy()
    guide_mask = np.any(guide > 0, axis=2)
    blended = cv2.addWeighted(out, 1.0, guide, 0.9, 0)
    out[guide_mask] = blended[guide_mask]

    mask = out.copy()
    cv2.rectangle(mask, (0, 0), (out.shape[1], 42), (20, 20, 20), -1)
    cv2.addWeighted(mask, 0.25, out, 0.75, 0, out)
    hint = (
        "Position guide: align head to ellipse, eyes to crosshair, shoulders to angled lines"
        if np.any(guide[:, :, 1] == 80)   # green body guide = static
        else "Align your head, shoulders and torso to the gold outline"
    )
    cv2.putText(
        out,
        hint,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return out


def _overlay(
    frame: np.ndarray,
    lines: list[str],
    rec: bool = False,
    guide: np.ndarray | None = None,
) -> np.ndarray:
    """Draw alignment guidance plus the instruction bar for a frame."""
    out = frame.copy()
    out = _apply_alignment_guide(out, guide)
    h, w = out.shape[:2]
    bar_h = 22 + len(lines) * 30
    mask = out.copy()
    cv2.rectangle(mask, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(mask, 0.65, out, 0.35, 0, out)
    y = h - bar_h + 24
    for line in lines:
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (220, 220, 220), 1, cv2.LINE_AA)
        y += 30
    if rec:
        cv2.circle(out, (w - 28, 68), 12, (0, 0, 220), -1)
        cv2.putText(out, "REC", (w - 68, 76), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 220), 2, cv2.LINE_AA)
    return out


def _live_wait(
    cap: cv2.VideoCapture,
    lines: list[str],
    rec: bool = False,
    guide: np.ndarray | None = None,
) -> np.ndarray | None:
    """Show live feed with overlay until Enter or Space is pressed.

    Returns the last captured frame, or None if ESC is pressed.
    """
    last: np.ndarray | None = None
    while True:
        frame = _read_capture_frame(cap)
        if frame is not None:
            last = frame
            cv2.imshow(WINDOW, _overlay(frame, lines, rec, guide=guide))
        k = _wait_key(1)
        if _window_closed() or k == _KEY_ESCAPE:
            return None
        if _is_action_key(k):
            return last


def _confirm(cap: cv2.VideoCapture, saved: np.ndarray, label: str) -> bool:
    """Show saved frame with a confirmation prompt.

    Returns True to keep, False to retake. ESC quits the program.
    """
    h, w = saved.shape[:2]
    display = saved.copy()
    mask = display.copy()
    cv2.rectangle(mask, (0, h // 2 - 48), (w, h // 2 + 48), (0, 60, 0), -1)
    cv2.addWeighted(mask, 0.72, display, 0.28, 0, display)
    cv2.putText(display, label, (14, h // 2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(display, "ENTER = keep   R = retake   ESC = quit",
                (14, h // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (220, 220, 220), 1, cv2.LINE_AA)
    while True:
        cv2.imshow(WINDOW, display)
        k = _wait_key(30)
        if _window_closed() or k == _KEY_ESCAPE:
            _quit_program(cap)
        if k in _KEY_ENTER:
            return True
        if k in (ord('r'), ord('R')):
            return False


# ── Camera selection ───────────────────────────────────────────────────────────

def _detect_cameras() -> list[CameraOption]:
    """Return helper-enumerated cameras, falling back to index probing."""
    if _enumerate_cameras_helper is not None:
        try:
            options = _build_camera_options(list(_enumerate_cameras_helper()))
        except Exception:
            options = []
        if options:
            return options
    return _probe_camera_options()


def _open_camera_capture(option: CameraOption) -> cv2.VideoCapture:
    if option.backend == cv2.CAP_ANY:
        cap = cv2.VideoCapture(option.capture_index)
    else:
        cap = cv2.VideoCapture(option.capture_index, option.backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def _confirm_camera_preview(cap: cv2.VideoCapture, option: CameraOption) -> bool:
    while True:
        frame = _read_capture_frame(cap)
        if frame is not None:
            cv2.imshow(
                WINDOW,
                _overlay(
                    frame,
                    [
                        f"SELECTED CAMERA - [{option.slot}] {option.name}",
                        f"OpenCV index {option.capture_index} | Preview is unmirrored | ENTER/SPACE keep | R choose again",
                    ],
                ),
            )
        key = _wait_key(30)
        if _window_closed() or key == _KEY_ESCAPE:
            _quit_program(cap)
        if _is_action_key(key):
            return True
        if key in (ord("r"), ord("R")):
            return False


def select_camera() -> cv2.VideoCapture:
    print("\n  Detecting cameras...", end="", flush=True)
    options = _detect_cameras()
    print(f" found {len(options)}.\n")

    if not options:
        print("  ERROR: No cameras found. Connect a webcam and try again.")
        sys.exit(1)

    print("  Available cameras:")
    for option in options:
        print(f"    [{option.slot}]  {option.name} (OpenCV index {option.capture_index})")
    print()

    selected_pos = 0
    while True:
        while True:
            cv2.imshow(WINDOW, _camera_selection_canvas(options, selected_pos))
            key = _wait_key(30)
            if _window_closed() or key == _KEY_ESCAPE:
                _quit_program()
            if _is_action_key(key):
                chosen = options[selected_pos]
                break
            selection_delta = _selection_delta_for_key(key)
            if selection_delta:
                selected_pos = (selected_pos + selection_delta) % len(options)
                continue
            if ord("1") <= key <= ord("9"):
                requested_slot = int(chr(key))
                for position, option in enumerate(options):
                    if option.slot == requested_slot:
                        selected_pos = position
                        chosen = option
                        break
                else:
                    continue
                break

        cap = _open_camera_capture(chosen)
        if not cap.isOpened():
            print(
                f"  Could not open camera [{chosen.slot}] {chosen.name} "
                f"(OpenCV index {chosen.capture_index}). Choose another."
            )
            cap.release()
            continue

        if _confirm_camera_preview(cap, chosen):
            print(
                f"  Using camera [{chosen.slot}] {chosen.name} "
                f"(OpenCV index {chosen.capture_index}).\n"
            )
            return cap

        cap.release()


def step_base_photo(
    cap: cv2.VideoCapture,
    total_steps: int,
    base_photo_out: Path = BASE_PHOTO,
) -> np.ndarray:
    _show_instruction_screen(
        f"STEP 1/{total_steps} - Base Photo",
        [
            "Capture the reference frame for all later idle clips and lip-sync output.",
            "This is the image the app returns to whenever you are not in an idle clip.",
            "A GREEN GUIDE OUTLINE will appear on the live preview — position yourself to match it:",
            "  • Your eyes should meet the CYAN CROSSHAIR.",
            "  • Your head should fit inside the HEAD ELLIPSE.",
            "  • Your shoulders and upper body should align with the ANGLED/BOX LINES.",
            "  • Look directly into the camera. Head level — no tilt.",
            "Neutral expression - mouth firmly CLOSED. Lighting: no harsh shadows.",
            "Press ENTER or SPACE in the live preview to capture, then ENTER to keep or R to retake.",
        ],
    )

    # Always use the fully static, always-centred guide — no face detection.
    # The guide shows where the user *should* sit, not where they currently are.
    static_guide: np.ndarray | None = None
    prime = _read_capture_frame(cap)
    if prime is not None:
        static_guide = _build_static_positioning_guide(*prime.shape[:2])

    while True:
        frame = _live_wait(cap, [
            "BASE PHOTO  |  Eyes to crosshair  |  Head in ellipse  |  Mouth CLOSED",
            "Preview is unmirrored | ENTER/SPACE to capture | ESC to quit",
        ], guide=static_guide)
        if frame is None:
            _quit_program(cap)
        base_photo_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(base_photo_out), frame)
        if _confirm(cap, frame, f"Saved: {base_photo_out}"):
            print(f"  ✓  {base_photo_out}")
            return frame

def step_idle_clip(
    cap: cv2.VideoCapture,
    index: int,
    total_steps: int,
    spec: IdleClipSpec,
    alignment_guide: np.ndarray,
    idle_dir: Path = IDLE_DIR,
) -> None:
    _show_instruction_screen(
        f"STEP {index + 1}/{total_steps} - {spec.title}",
        [
            f"File name: {_idle_clip_output_path(spec, idle_dir).name}",
            f"Movement: {spec.instruction}",
            f"Target length: {spec.duration_hint}",
            "Return to your BASE PHOTO position before you start.",
            "Match your posture to the gold base-photo outline in the preview.",
            "Press ENTER or SPACE to START recording.",
            "Perform the movement once, then return to the same neutral pose.",
            "Keep your mouth CLOSED throughout the whole clip.",
            "Press ENTER or SPACE to STOP once you are back in neutral, then ENTER to keep or R to retake.",
        ],
    )

    idle_dir.mkdir(parents=True, exist_ok=True)
    out_path = _idle_clip_output_path(spec, idle_dir)

    while True:
        start_frame = _live_wait(cap, [
            f"{spec.title} - {spec.instruction[:56]}",
            f"Target {spec.duration_hint} | Preview is unmirrored | ENTER/SPACE to START recording",
        ], guide=alignment_guide)
        if start_frame is None:
            _quit_program(cap)

        fr_h, fr_w = start_frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fr_w, fr_h))

        last_frame = start_frame
        writer.write(start_frame)
        while True:
            fr = _read_capture_frame(cap)
            if fr is not None:
                last_frame = fr
                writer.write(fr)
                cv2.imshow(WINDOW, _overlay(fr, [
                    f"RECORDING — {spec.title}",
                    "Preview is unmirrored | Perform once, return to neutral, then ENTER/SPACE to STOP",
                ], rec=True, guide=alignment_guide))
            k = _wait_key(1)
            if _window_closed() or k == _KEY_ESCAPE:
                writer.release()
                _quit_program(cap)
            if _is_action_key(k):
                break

        writer.release()

        if _confirm(cap, last_frame, f"Saved: {out_path}"):
            break

        if out_path.exists():
            out_path.unlink()

    print(f"  ✓  {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def _prompt_theme_name() -> _AssetPaths:
    """Ask the user for a theme name and return resolved asset paths."""
    print()
    existing: list[str] = []
    if THEMES_DIR.exists():
        existing = sorted(
            d.name for d in THEMES_DIR.iterdir()
            if d.is_dir() and (d / "base_photo.png").exists()
        )

    if existing:
        print("  Existing themes:")
        for name in existing:
            print(f"    • {name}")
        print()

    raw = input(
        "  Theme name for this set of assets\n"
        "  (leave empty to overwrite the default, or type a name like 'casual'): "
    ).strip()

    if not raw:
        return _AssetPaths(
            base_photo=BASE_PHOTO,
            idle_dir=IDLE_DIR,
            theme_name=None,
        )

    # Sanitise: keep letters, digits, underscore, hyphen
    safe = "".join(c for c in raw if c.isalnum() or c in "-_").lower()
    if not safe:
        print("  Invalid name — using default.")
        return _AssetPaths(base_photo=BASE_PHOTO, idle_dir=IDLE_DIR, theme_name=None)

    theme_dir = THEMES_DIR / safe
    return _AssetPaths(
        base_photo=theme_dir / "base_photo.png",
        idle_dir=theme_dir / "idle_clips",
        theme_name=safe,
    )


def main():
    print("\n" + "=" * 60)
    print("  ScrumSurvivor - Asset Creator")
    print("=" * 60)

    asset_paths = _prompt_theme_name()
    if asset_paths.theme_name:
        print(f"\n  Theme: {asset_paths.theme_name}")
        print(f"  Saving to: {asset_paths.base_photo.parent}")
    else:
        print("\n  Saving to default asset paths.")
    print()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)

    _show_instruction_screen(
        "ScrumSurvivor Asset Creator",
        [
            "This wizard captures the base photo and the named idle clips used by ScrumSurvivor.",
            "All capture guidance is shown in the preview window.",
            "The live camera preview is shown unmirrored so positioning matches the final saved assets.",
            "Step 1 shows a GREEN POSITIONING GUIDE — align your head, eyes, and shoulders to it.",
            "After the base photo, the preview adds a GOLD outline derived from YOUR actual base photo.",
            "Controls: ENTER or SPACE start/stop/capture, ENTER keep/continue, R retake or reselect, ESC quit.",
            *((
                f"Active theme: '{asset_paths.theme_name}'  →  {asset_paths.base_photo.parent}",
            ) if asset_paths.theme_name else ()),
        ],
    )

    cap = select_camera()
    total_steps = 1 + len(IDLE_CLIPS)

    base_photo = step_base_photo(cap, total_steps, asset_paths.base_photo)
    alignment_guide = _build_alignment_guide(base_photo)

    for index, spec in enumerate(IDLE_CLIPS, start=1):
        step_idle_clip(cap, index, total_steps, spec, alignment_guide, asset_paths.idle_dir)

    cap.release()
    cv2.destroyAllWindows()

    clip_lines = "\n".join(
        f"    {_idle_clip_output_path(spec, asset_paths.idle_dir)}"
        for spec in IDLE_CLIPS
    )

    print("\n" + "=" * 60)
    print("  All assets captured!")
    print("=" * 60)

    theme_note = (
        f"  Theme name: '{asset_paths.theme_name}'\n"
        f"  Start with: .venv/Scripts/python.exe -m scrumsurvivor run --theme {asset_paths.theme_name}\n"
        if asset_paths.theme_name
        else ""
    )

    print(f"""
  Files saved:
        {asset_paths.base_photo}
{clip_lines}

{theme_note}  Next steps:
    1. Download wav2lip.pth → place at: models/wav2lip.pth
       (see README — Assets section D for the download link)

        2. Run:  .venv/Scripts/python.exe -m scrumsurvivor check-gpu
        3. Run:  .venv/Scripts/python.exe -m scrumsurvivor calibrate
        4. Run:  .venv/Scripts/python.exe -m scrumsurvivor run
""")


if __name__ == "__main__":
    main()
