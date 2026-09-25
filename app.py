"""
EYVORA V1 - Webcam Driver Monitoring Prototype

Detects:
- Face / no face
- Eyes open / closed
- Blinks
- Prolonged eye closure (drowsiness warning)
- Yawning
- Head direction
- Logs important events to SQLite

Uses:
- OpenCV for webcam/video
- Google MediaPipe Face Landmarker for face landmarks + blendshapes
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from settings import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    EYE_CLOSED_THRESHOLD,
    DROWSY_SECONDS,
    BLINK_MIN_SECONDS,
    BLINK_MAX_SECONDS,
    YAWN_JAW_THRESHOLD,
    YAWN_MIN_SECONDS,
    HEAD_YAW_THRESHOLD_DEG,
    HEAD_PITCH_THRESHOLD_DEG,
    YAW_SIGN,
    PITCH_SIGN,
    NO_FACE_ALERT_SECONDS,
    HEAD_AWAY_LOG_SECONDS,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
DB_PATH = APP_DIR / "eyvora_events.db"

# Official MediaPipe Face Landmarker model bundle.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# Key landmarks used only for visual overlay.
DISPLAY_LANDMARKS = [
    # left eye
    33, 160, 158, 133, 153, 144,
    # right eye
    362, 385, 387, 263, 373, 380,
    # mouth
    61, 13, 291, 14,
    # nose/chin
    1, 152,
]


def ensure_model() -> None:
    """Download Google's Face Landmarker model once if it is not present."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\nEYVORA: Face model is missing.")
    print("Downloading the official MediaPipe Face Landmarker model...")
    print("This happens only once.\n")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "\nCould not download the MediaPipe model.\n"
            "Check your internet connection and run the program again.\n"
            f"Technical error: {exc}"
        ) from exc


class EventLogger:
    """Tiny SQLite logger for future EYVORA baseline / incident work."""

    def __init__(self, db_path: Path):
        self.connection = sqlite3.connect(db_path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT
            )
            """
        )
        self.connection.commit()

    def log(self, event: str, details: str = "") -> None:
        self.connection.execute(
            "INSERT INTO events(timestamp, event, details) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), event, details),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def blendshape_dict(result) -> dict[str, float]:
    """Convert MediaPipe face blendshape output to {name: score}."""
    if not result.face_blendshapes:
        return {}

    categories = result.face_blendshapes[0]
    return {
        category.category_name: float(category.score)
        for category in categories
    }


def rotation_matrix_to_euler_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a 3x3 rotation matrix to pitch/yaw/roll in degrees.

    Returns:
        pitch_deg, yaw_deg, roll_deg
    """
    r = matrix[:3, :3].astype(np.float64)

    sy = math.sqrt((r[0, 0] * r[0, 0]) + (r[1, 0] * r[1, 0]))
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0

    return tuple(math.degrees(x) for x in (pitch, yaw, roll))


def get_head_angles(result) -> tuple[float, float, float] | None:
    """Read the facial transformation matrix returned by MediaPipe."""
    if not result.facial_transformation_matrixes:
        return None

    matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
    if matrix.shape != (4, 4):
        try:
            matrix = matrix.reshape(4, 4)
        except ValueError:
            return None

    return rotation_matrix_to_euler_degrees(matrix)


def draw_face_overlay(frame: np.ndarray, face_landmarks) -> None:
    """Draw a lightweight face box and important eye/mouth points."""
    h, w = frame.shape[:2]
    xs = [p.x for p in face_landmarks]
    ys = [p.y for p in face_landmarks]

    x1 = max(0, int(min(xs) * w))
    y1 = max(0, int(min(ys) * h))
    x2 = min(w - 1, int(max(xs) * w))
    y2 = min(h - 1, int(max(ys) * h))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 220, 120), 2)

    for index in DISPLAY_LANDMARKS:
        if index < len(face_landmarks):
            p = face_landmarks[index]
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (x, y), 2, (255, 220, 80), -1)


def draw_panel(
    frame: np.ndarray,
    status: dict,
    fps: float,
) -> None:
    """Draw the EYVORA information panel."""
    h, w = frame.shape[:2]

    # Header
    cv2.rectangle(frame, (0, 0), (w, 58), (16, 20, 28), -1)
    cv2.putText(
        frame, "EYVORA - DRIVER MONITOR V1",
        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame, f"{fps:4.1f} FPS",
        (w - 125, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 200, 210), 1,
        cv2.LINE_AA
    )

    panel_x = 18
    panel_y = 78
    panel_w = 330
    panel_h = 275

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 24, 32),
        -1,
    )
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    if not status["face"]:
        lines = [
            ("DRIVER", "NO FACE"),
            ("EYES", "--"),
            ("BLINKS", str(status["blinks"])),
            ("YAWNING", "--"),
            ("HEAD", "--"),
        ]
    else:
        eyes_value = "CLOSED" if status["eyes_closed"] else "OPEN"
        if status["eyes_closed"]:
            eyes_value += f"  {status['closed_seconds']:.1f}s"

        lines = [
            ("DRIVER", status["driver_state"]),
            ("EYES", eyes_value),
            ("BLINKS", str(status["blinks"])),
            ("YAWNING", "YES" if status["yawning"] else "NO"),
            ("YAWNS", str(status["yawns"])),
            ("HEAD", status["head_direction"]),
        ]

    y = panel_y + 35
    for label, value in lines:
        cv2.putText(
            frame, label,
            (panel_x + 16, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 180, 190), 1,
            cv2.LINE_AA
        )

        value_color = (235, 235, 235)
        if value in ("CLOSED", "YES", "NO FACE", "DROWSY"):
            value_color = (80, 100, 255)

        cv2.putText(
            frame, value,
            (panel_x + 132, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, value_color, 2,
            cv2.LINE_AA
        )
        y += 37

    if status["face"]:
        pose_text = (
            f"Pitch {status['pitch']:+.1f}  "
            f"Yaw {status['yaw']:+.1f}  "
            f"Roll {status['roll']:+.1f}"
        )
        cv2.putText(
            frame, pose_text,
            (18, h - 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 215, 220), 1,
            cv2.LINE_AA
        )

    cv2.putText(
        frame,
        "Q quit   C centre head   R reset counters",
        (18, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 215, 220), 1,
        cv2.LINE_AA,
    )

    # Full-width alert message.
    alert = status.get("alert", "")
    if alert:
        cv2.rectangle(frame, (0, h - 105), (w, h - 62), (20, 20, 210), -1)
        text_size = cv2.getTextSize(
            alert, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        )[0]
        x = max(10, (w - text_size[0]) // 2)
        cv2.putText(
            frame, alert, (x, h - 76),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
            cv2.LINE_AA
        )


class DriverMonitor:
    def __init__(self, logger: EventLogger):
        self.logger = logger

        self.blink_count = 0
        self.yawn_count = 0

        self.eye_closed_since: float | None = None
        self.yawn_since: float | None = None
        self.yawn_counted = False

        self.drowsy_active = False
        self.no_face_since: float | None = None
        self.no_face_alerted = False

        self.neutral_pitch: float | None = None
        self.neutral_yaw: float | None = None
        self.neutral_roll: float | None = None

        self.head_away_since: float | None = None
        self.head_away_logged = False

        self.last_face_present = False

    def reset_counters(self) -> None:
        self.blink_count = 0
        self.yawn_count = 0
        self.logger.log("counters_reset")

    def centre_head(self, pitch: float, yaw: float, roll: float) -> None:
        self.neutral_pitch = pitch
        self.neutral_yaw = yaw
        self.neutral_roll = roll
        self.logger.log("head_recentered")

    def process_no_face(self, now: float) -> dict:
        if self.last_face_present:
            self.logger.log("face_lost")

        self.last_face_present = False
        self.eye_closed_since = None
        self.yawn_since = None
        self.yawn_counted = False

        if self.no_face_since is None:
            self.no_face_since = now

        no_face_seconds = now - self.no_face_since
        alert = ""

        if no_face_seconds >= NO_FACE_ALERT_SECONDS:
            alert = "WARNING: DRIVER NOT DETECTED"
            if not self.no_face_alerted:
                self.logger.log(
                    "no_face_alert",
                    f"face missing for {no_face_seconds:.1f}s"
                )
                self.no_face_alerted = True

        return {
            "face": False,
            "driver_state": "NO FACE",
            "eyes_closed": False,
            "closed_seconds": 0.0,
            "blinks": self.blink_count,
            "yawning": False,
            "yawns": self.yawn_count,
            "head_direction": "--",
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "alert": alert,
        }

    def process_face(self, result, now: float) -> dict:
        if not self.last_face_present:
            self.logger.log("face_detected")

        self.last_face_present = True
        self.no_face_since = None
        self.no_face_alerted = False

        scores = blendshape_dict(result)

        left_blink = scores.get("eyeBlinkLeft", 0.0)
        right_blink = scores.get("eyeBlinkRight", 0.0)
        eye_blink_score = (left_blink + right_blink) / 2.0
        eyes_closed = eye_blink_score >= EYE_CLOSED_THRESHOLD

        # Blink / prolonged closure state machine.
        closed_seconds = 0.0

        if eyes_closed:
            if self.eye_closed_since is None:
                self.eye_closed_since = now

            closed_seconds = now - self.eye_closed_since
        else:
            if self.eye_closed_since is not None:
                duration = now - self.eye_closed_since

                if BLINK_MIN_SECONDS <= duration <= BLINK_MAX_SECONDS:
                    self.blink_count += 1
                    self.logger.log("blink", f"{duration:.3f}s")

            self.eye_closed_since = None

        drowsy_now = eyes_closed and closed_seconds >= DROWSY_SECONDS

        if drowsy_now and not self.drowsy_active:
            self.logger.log(
                "drowsiness_alert",
                f"eyes closed {closed_seconds:.2f}s"
            )
            self.drowsy_active = True
        elif not drowsy_now and self.drowsy_active:
            self.logger.log("drowsiness_cleared")
            self.drowsy_active = False

        # Yawn detection.
        jaw_open = scores.get("jawOpen", 0.0)
        mouth_open = jaw_open >= YAWN_JAW_THRESHOLD

        if mouth_open:
            if self.yawn_since is None:
                self.yawn_since = now

            yawn_duration = now - self.yawn_since
            yawning = yawn_duration >= YAWN_MIN_SECONDS

            if yawning and not self.yawn_counted:
                self.yawn_count += 1
                self.yawn_counted = True
                self.logger.log(
                    "yawn",
                    f"jawOpen={jaw_open:.2f}, duration={yawn_duration:.2f}s"
                )
        else:
            self.yawn_since = None
            self.yawn_counted = False
            yawning = False

        # Head pose.
        raw_angles = get_head_angles(result)
        pitch = yaw = roll = 0.0

        if raw_angles is not None:
            raw_pitch, raw_yaw, raw_roll = raw_angles

            if self.neutral_pitch is None:
                self.neutral_pitch = raw_pitch
                self.neutral_yaw = raw_yaw
                self.neutral_roll = raw_roll

            pitch = PITCH_SIGN * (raw_pitch - self.neutral_pitch)
            yaw = YAW_SIGN * (raw_yaw - self.neutral_yaw)
            roll = raw_roll - self.neutral_roll

        if yaw > HEAD_YAW_THRESHOLD_DEG:
            head_direction = "RIGHT"
        elif yaw < -HEAD_YAW_THRESHOLD_DEG:
            head_direction = "LEFT"
        elif pitch > HEAD_PITCH_THRESHOLD_DEG:
            head_direction = "DOWN"
        elif pitch < -HEAD_PITCH_THRESHOLD_DEG:
            head_direction = "UP"
        else:
            head_direction = "FORWARD"

        head_away = head_direction != "FORWARD"

        if head_away:
            if self.head_away_since is None:
                self.head_away_since = now

            away_seconds = now - self.head_away_since
            if (
                away_seconds >= HEAD_AWAY_LOG_SECONDS
                and not self.head_away_logged
            ):
                self.logger.log(
                    "head_away",
                    f"{head_direction}, {away_seconds:.1f}s"
                )
                self.head_away_logged = True
        else:
            if self.head_away_logged:
                self.logger.log("head_forward")
            self.head_away_since = None
            self.head_away_logged = False

        # High-level driver state.
        if drowsy_now:
            driver_state = "DROWSY"
            alert = "DROWSINESS DETECTED - EYES CLOSED TOO LONG"
        elif yawning:
            driver_state = "YAWNING"
            alert = "YAWNING DETECTED"
        elif head_away and self.head_away_since is not None and (
            now - self.head_away_since >= HEAD_AWAY_LOG_SECONDS
        ):
            driver_state = "DISTRACTED"
            alert = f"LOOKING {head_direction}"
        else:
            driver_state = "ALERT"
            alert = ""

        return {
            "face": True,
            "driver_state": driver_state,
            "eyes_closed": eyes_closed,
            "closed_seconds": closed_seconds,
            "blinks": self.blink_count,
            "yawning": yawning,
            "yawns": self.yawn_count,
            "head_direction": head_direction,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "alert": alert,
            "eye_score": eye_blink_score,
            "jaw_open": jaw_open,
        }


def open_camera(index: int):
    """Open webcam with a few cross-platform fallbacks."""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
    elif sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    return cap


def main() -> None:
    ensure_model()

    logger = EventLogger(DB_PATH)
    monitor = DriverMonitor(logger)

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    cap = open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        logger.close()
        raise RuntimeError(
            "EYVORA could not open your webcam.\n"
            "Close Zoom/FaceTime/Teams, allow camera permission, and try again.\n"
            "If you have multiple cameras, change CAMERA_INDEX in settings.py."
        )

    logger.log("eyvora_started")

    start_perf = time.perf_counter()
    last_timestamp_ms = -1
    last_frame_time = time.perf_counter()
    fps_smoothed = 0.0
    last_status = None

    print("\nEYVORA V1 running.")
    print("Q = quit | C = centre head | R = reset blink/yawn counters\n")

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Could not read a webcam frame.")
                    break

                # Mirror the camera so it feels natural to the user.
                frame = cv2.flip(frame, 1)

                now = time.perf_counter()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = int((now - start_perf) * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks:
                    status = monitor.process_face(result, now)
                    draw_face_overlay(frame, result.face_landmarks[0])
                else:
                    status = monitor.process_no_face(now)

                last_status = status

                frame_delta = max(1e-6, now - last_frame_time)
                instant_fps = 1.0 / frame_delta
                if fps_smoothed == 0.0:
                    fps_smoothed = instant_fps
                else:
                    fps_smoothed = (0.90 * fps_smoothed) + (0.10 * instant_fps)
                last_frame_time = now

                draw_panel(frame, status, fps_smoothed)

                cv2.imshow("EYVORA Driver Monitor V1", frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                if key in (ord("r"), ord("R")):
                    monitor.reset_counters()

                if key in (ord("c"), ord("C")) and status["face"]:
                    # Re-centre using the raw pose from this frame.
                    raw_angles = get_head_angles(result)
                    if raw_angles is not None:
                        monitor.centre_head(*raw_angles)

    finally:
        logger.log("eyvora_stopped")
        logger.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("\nEYVORA ERROR")
        print("------------")
        print(exc)
        input("\nPress Enter to close...")
