from collections import deque
from datetime import datetime
from pathlib import Path
import cv2

class RollingVideoBuffer:
    def __init__(self, seconds, max_fps):
        self.max_fps = int(max_fps)
        self.frames = deque(maxlen=max(1, int(seconds) * self.max_fps))
        self.last_added = 0.0

    def add(self, frame, now):
        interval = 1.0 / max(1, self.max_fps)
        if now - self.last_added >= interval:
            self.frames.append(frame.copy())
            self.last_added = now

    def save(self, output_dir: Path, fps=None):
        if not self.frames:
            return ""
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = list(self.frames)
        h, w = frames[0].shape[:2]
        fps = max(5.0, min(float(self.max_fps), float(fps or self.max_fps)))
        path = output_dir / f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            return ""
        for frame in frames:
            writer.write(frame)
        writer.release()
        return str(path)
