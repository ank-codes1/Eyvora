# EYVORA V1 settings
#
# Start with these values. We can calibrate them later for your face/camera.

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# MediaPipe blendshape threshold:
# 0 = open, 1 = strongly closed/blinking.
EYE_CLOSED_THRESHOLD = 0.55

# If the eyes remain closed this long, EYVORA flags drowsiness.
DROWSY_SECONDS = 1.50

# A closure within this range counts as a normal blink.
BLINK_MIN_SECONDS = 0.07
BLINK_MAX_SECONDS = 0.80

# MediaPipe "jawOpen" threshold and minimum duration for a yawn.
YAWN_JAW_THRESHOLD = 0.55
YAWN_MIN_SECONDS = 0.60

# Head direction thresholds after pressing C while looking straight.
HEAD_YAW_THRESHOLD_DEG = 15.0
HEAD_PITCH_THRESHOLD_DEG = 12.0

# If LEFT/RIGHT or UP/DOWN is reversed on your camera, change 1.0 to -1.0.
YAW_SIGN = 1.0
PITCH_SIGN = 1.0

# No-face and distraction timers.
NO_FACE_ALERT_SECONDS = 1.5
HEAD_AWAY_LOG_SECONDS = 2.0
