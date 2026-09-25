from dataclasses import dataclass
from settings import *

@dataclass
class RiskResult:
    score: float
    level: str
    reasons: list

class RiskEngine:
    def __init__(self):
        self.score = 0.0
        self.max_score = 0.0

    def level_for(self, score):
        if score >= CRITICAL_RISK: return "CRITICAL"
        if score >= HIGH_RISK: return "HIGH"
        if score >= CAUTION_RISK: return "CAUTION"
        return "SAFE"

    def update(self, status, dt):
        self.score = max(0.0, self.score - RISK_DECAY_PER_SECOND * max(0.0, dt))
        reasons = []

        if not status.get("face", False):
            self.score = max(self.score, float(RISK_NO_FACE))
            reasons.append("driver not detected")
        else:
            if status.get("drowsy", False):
                extra = min(35.0, max(0.0, status.get("closed_seconds", 0)-1.5) * 18)
                self.score = max(self.score, RISK_EYES_CLOSED + extra)
                reasons.append("prolonged eye closure")
            if status.get("yawning", False):
                self.score = max(self.score, float(RISK_YAWNING))
                reasons.append("yawning")
            if status.get("head_away_alert", False):
                self.score = max(self.score, float(RISK_HEAD_AWAY))
                reasons.append("looking away")
            if status.get("yawns_recent", 0) >= 3:
                self.score = min(100.0, self.score + RISK_REPEATED_YAWN_BONUS * dt / 3.0)
                reasons.append("repeated yawns")

        self.score = max(0.0, min(100.0, self.score))
        self.max_score = max(self.max_score, self.score)
        return RiskResult(self.score, self.level_for(self.score), reasons)

    def reset(self):
        self.score = 0.0
