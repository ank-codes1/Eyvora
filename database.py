import sqlite3
from datetime import datetime
from pathlib import Path

class Database:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS drivers(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            total_blinks INTEGER DEFAULT 0,
            total_yawns INTEGER DEFAULT 0,
            max_risk REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            details TEXT,
            risk_score REAL
        );
        CREATE TABLE IF NOT EXISTS incidents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT NOT NULL,
            incident_type TEXT NOT NULL,
            risk_score REAL,
            driver_response TEXT,
            video_path TEXT,
            details TEXT
        );
        """)
        self.conn.commit()

    def get_or_create_driver(self, name):
        name = name.strip() or "Default Driver"
        row = self.conn.execute("SELECT id FROM drivers WHERE lower(name)=lower(?)", (name,)).fetchone()
        if row:
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO drivers(name, created_at) VALUES (?,?)",
            (name, datetime.now().isoformat(timespec="seconds"))
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def start_session(self, driver_id):
        cur = self.conn.execute(
            "INSERT INTO sessions(driver_id, started_at) VALUES (?,?)",
            (driver_id, datetime.now().isoformat(timespec="seconds"))
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def end_session(self, session_id, blinks, yawns, max_risk):
        self.conn.execute(
            "UPDATE sessions SET ended_at=?, total_blinks=?, total_yawns=?, max_risk=? WHERE id=?",
            (datetime.now().isoformat(timespec="seconds"), blinks, yawns, float(max_risk), session_id)
        )
        self.conn.commit()

    def log_event(self, session_id, event_type, details="", severity="info", risk_score=None):
        self.conn.execute(
            "INSERT INTO events(session_id,timestamp,event_type,severity,details,risk_score) VALUES (?,?,?,?,?,?)",
            (session_id, datetime.now().isoformat(timespec="seconds"), event_type, severity, details, risk_score)
        )
        self.conn.commit()

    def log_incident(self, session_id, incident_type, risk_score, driver_response, video_path="", details=""):
        self.conn.execute(
            "INSERT INTO incidents(session_id,timestamp,incident_type,risk_score,driver_response,video_path,details) VALUES (?,?,?,?,?,?,?)",
            (session_id, datetime.now().isoformat(timespec="seconds"), incident_type, float(risk_score), driver_response, video_path, details)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
