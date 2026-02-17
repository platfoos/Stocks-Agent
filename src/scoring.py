from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ScoreBreakdown:
    total: float
    components: Dict[str, float]
    notes: list[str]


class StockScorer:
    """Score candidate setups with a strict swing-trading rubric."""

    def score(self, metrics: Dict[str, float | bool | int]) -> ScoreBreakdown:
        components: Dict[str, float] = {}
        notes: list[str] = []

        trend_score = 0.0
        trend_distance = float(metrics.get("price_vs_sma150_pct", 0.0))
        if trend_distance >= 10:
            trend_score = 25
        elif trend_distance >= 5:
            trend_score = 21
        elif trend_distance >= 2:
            trend_score = 17
        elif trend_distance > 0:
            trend_score = 12
        components["trend"] = trend_score

        breakout_score = 0.0
        if bool(metrics.get("breakout_above_resistance", False)) and bool(metrics.get("volume_confirmed", False)):
            breakout_score = 25
            notes.append("Resistance breakout with volume confirmation")
        elif bool(metrics.get("breakout_above_sma150", False)):
            breakout_score = 18
            notes.append("SMA150 breakout")
        components["breakout"] = breakout_score

        relative_strength_score = 0.0
        rel_3m = float(metrics.get("relative_3m_pct", 0.0))
        rel_6m = float(metrics.get("relative_6m_pct", 0.0))
        if rel_3m > 8 and rel_6m > 12:
            relative_strength_score = 20
        elif rel_3m > 5 and rel_6m > 8:
            relative_strength_score = 16
        elif rel_3m > 2 and rel_6m > 4:
            relative_strength_score = 12
        elif rel_3m > 0 and rel_6m > 0:
            relative_strength_score = 8
        components["relative_strength"] = relative_strength_score

        risk_reward_score = 0.0
        rr = float(metrics.get("risk_reward", 0.0))
        if rr >= 3:
            risk_reward_score = 16
        elif rr >= 2.5:
            risk_reward_score = 13
        elif rr >= 2:
            risk_reward_score = 10
        components["risk_reward"] = risk_reward_score

        earnings_penalty = 0.0
        days_to_earnings = int(metrics.get("days_to_earnings", 999))
        if days_to_earnings <= 14:
            earnings_penalty = -6
            notes.append("Earnings in <=14 days")
        components["earnings_penalty"] = earnings_penalty

        base_score = sum(components.values())
        total = max(0.0, min(100.0, base_score + 14.0))

        if bool(metrics.get("multi_week_pattern", False)):
            total = min(100.0, total + 3)
            notes.append("Strong multi-week bullish base")

        return ScoreBreakdown(total=round(total, 1), components=components, notes=notes)
