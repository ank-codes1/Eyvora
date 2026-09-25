# EYVORA V2

Software-only AI Driver Monitoring and Emergency Response prototype.

## Features
- Face / no-face detection
- Eye open / closed detection
- Blink counting
- Drowsiness detection
- Yawn detection
- Head direction / distraction detection
- Live 0–100 driver risk score
- SAFE / CAUTION / HIGH / CRITICAL states
- Driver profile in SQLite
- Session/event/incident database
- 8-second "ARE YOU OK?" countdown
- `I` or `SPACE` to cancel the emergency
- 15-second rolling video buffer
- Automatic incident clip if no response

## Python
Use Python 3.12.

## Dependencies
```text
mediapipe==0.10.21
opencv-contrib-python==4.10.0.84
numpy==2.5.2
```

## Run
```bash
python app.py
```

Controls:
- `C` centre head
- `I` or `SPACE` I'm OK
- `R` reset risk
- `Q` quit

This is a school/research prototype, not a medical diagnosis system.
