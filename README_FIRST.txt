EYVORA V1 — Webcam Driver Monitoring
====================================

WHAT THIS VERSION DOES
----------------------
✓ Detects a face
✓ Detects NO FACE
✓ Eyes OPEN / CLOSED
✓ Counts blinks
✓ Detects prolonged eye closure / drowsiness
✓ Detects yawning
✓ Detects head LEFT / RIGHT / UP / DOWN
✓ Logs important events into eyvora_events.db

This is V1 only. We will later add ESP32, MAX30102, thermal camera,
EDA/GSR, pressure sensor, buzzer, physical cancel button, GPS, etc.


BEST PYTHON VERSION
-------------------
Use Python 3.11 or Python 3.12.

Check:
    python3 --version


MAC — EASY SETUP
----------------
1. Open Terminal.
2. Drag the EYVORA_V1 folder into Terminal after typing:

    cd 

3. Press Enter.

4. Create a virtual environment:

    python3 -m venv .venv

5. Activate it:

    source .venv/bin/activate

6. Install packages:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

7. Run:

    python app.py

The first run downloads Google's official Face Landmarker model.
macOS may ask for CAMERA PERMISSION. Allow it.


WINDOWS — EASY SETUP
--------------------
Open Command Prompt inside the EYVORA_V1 folder, then:

    py -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python app.py


CONTROLS
--------
Q  = quit
C  = tell EYVORA "I am looking straight right now"
R  = reset blink/yawn counters

IMPORTANT:
When the camera opens, look STRAIGHT at it and press C once.
This gives head-direction detection a neutral position.


IF HEAD DIRECTIONS ARE REVERSED
-------------------------------
Open settings.py.

If LEFT and RIGHT are reversed:
    YAW_SIGN = -1.0

If UP and DOWN are reversed:
    PITCH_SIGN = -1.0


IF EYE DETECTION NEEDS TUNING
-----------------------------
Open settings.py.

Default:
    EYE_CLOSED_THRESHOLD = 0.55

If it says CLOSED when your eyes are open:
    raise it slightly, e.g. 0.65

If it does not detect closed eyes:
    lower it slightly, e.g. 0.45


DATABASE
--------
EYVORA automatically creates:

    eyvora_events.db

It logs important events such as:
- face detected/lost
- blink
- yawn
- drowsiness alert
- head-away event

This database is the beginning of the later EYVORA personal-baseline system.


NOT MEDICAL DIAGNOSIS
---------------------
This is a school/research prototype. Camera detections can be wrong.
Do not use this version to control a real vehicle or make medical decisions.
