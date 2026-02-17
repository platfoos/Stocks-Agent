from __future__ import annotations

import csv
import datetime as dt
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from scoring import StockScorer


SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}


@dataclass
class Candidate:
    ticker: str
    score: float
    days_to_earnings: int
    enter_now: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_reward: float
    atr14: float
    sl_in_atr: float
    rel_3m: float
    rel_6m: float
    benchmark: str
    rationale: str
    pattern: str


def safe_pct_change(start: float, end: float) -> float:
    if start <= 0:
        return 0.0
    return (end - start) / start * 100


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def most_recent_resistance(df: pd.DataFrame, lookback: int = 60) -> float:
    if len(df) < lookback:
        lookback = len(df)
    return float(df["High"].iloc[-lookback:-1].max())


def finnhub_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    time.sleep(0.25)
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def get_days_to_earnings(ticker: str, api_key: str) -> int:
    now = dt.datetime.now(dt.timezone.utc).date()
    future_limit = now + dt.timedelta(days=365)
    payload = finnhub_get(
        "https://finnhub.io/api/v1/calendar/earnings",
        {
            "symbol": ticker,
            "from": now.isoformat(),
            "to": future_limit.isoformat(),
            "token": api_key,
        },
    )
    earnings_rows = payload.get("earningsCalendar", []) or []
    if not earnings_rows:
        return 999

    next_days: list[int] = []
    for row in earnings_rows:
        date_text = row.get("date")
        if not date_text:
            continue
        earnings_date = dt.date.fromisoformat(date_text)
        delta = (earnings_date - now).days
        if delta >= 0:
            next_days.append(delta)

    return min(next_days) if next_days else 999


def get_sector_and_market_cap(ticker: str, api_key: str) -> tuple[Optional[str], Optional[float]]:
    profile = finnhub_get(
        "https://finnhub.io/api/v1/stock/profile2",
        {"symbol": ticker, "token": api_key},
    )
    market_cap = profile.get("marketCapitalization")
    finnhub_industry = profile.get("finnhubIndustry")
    return finnhub_industry, float(market_cap) * 1_000_000 if market_cap else None


def fetch_us_symbols(api_key: str) -> Iterable[str]:
    symbols = finnhub_get("https://finnhub.io/api/v1/stock/symbol", {"exchange": "US", "token": api_key})
    for row in symbols:
        symbol = row.get("symbol", "")
        symbol_type = row.get("type", "")
        if symbol and "." not in symbol and symbol_type == "Common Stock":
            yield symbol


def download_ohlcv(symbol: str, api_key: str, period_days: int = 365) -> Optional[pd.DataFrame]:
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=period_days)
    payload = finnhub_get(
        "https://finnhub.io/api/v1/stock/candle",
        {
            "symbol": symbol,
            "resolution": "D",
            "from": int(start.timestamp()),
            "to": int(now.timestamp()),
            "token": api_key,
        },
    )
    if payload.get("s") != "ok":
        return None

    df = pd.DataFrame(
        {
            "Open": payload.get("o", []),
            "High": payload.get("h", []),
            "Low": payload.get("l", []),
            "Close": payload.get("c", []),
            "Volume": payload.get("v", []),
        },
        index=pd.to_datetime(payload.get("t", []), unit="s", utc=True),
    )
    if df.empty:
        return None

    return df.dropna().copy()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_trade_levels(df: pd.DataFrame, atr14: float, resistance: float) -> tuple[float, float, float, float, float]:
    last_close = float(df["Close"].iloc[-1])
    entry = max(last_close, resistance * 1.002)

    risk = max(atr14 * 1.5, entry * 0.025)
    stop_loss = entry - risk

    sl_pct = (entry - stop_loss) / entry * 100
    if sl_pct > 5:
        # Keep stop below support but avoid over-wide risk.
        stop_loss = entry * 0.949
        risk = entry - stop_loss

    support = float(df["Low"].iloc[-30:-1].min())
    if math.isclose(stop_loss, support, rel_tol=0.002):
        stop_loss = support * 0.995
        risk = entry - stop_loss

    tp1 = entry + (risk * 2.0)
    tp2 = entry + (risk * 3.0)
    rr = (tp1 - entry) / risk if risk > 0 else 0

    return entry, stop_loss, tp1, tp2, rr


def scan() -> list[Candidate]:
    load_dotenv()
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is required")

    scorer = StockScorer()

    max_tickers = int(os.getenv("MAX_TICKERS", "0"))
    symbols = list(fetch_us_symbols(api_key))
    if max_tickers > 0:
        symbols = symbols[:max_tickers]

    results: list[Candidate] = []

    for idx, symbol in enumerate(symbols, start=1):
        try:
            sector, market_cap = get_sector_and_market_cap(symbol, api_key)
            if not market_cap or market_cap <= 1_000_000_000:
                continue

            benchmark = SECTOR_ETF_MAP.get(sector or "", "SPY")

            stock_df = download_ohlcv(symbol, api_key)
            benchmark_df = download_ohlcv(benchmark, api_key)
            if stock_df is None or benchmark_df is None:
                continue

            stock_df = flatten_columns(stock_df)
            benchmark_df = flatten_columns(benchmark_df)

            stock_df["SMA150"] = sma(stock_df["Close"], 150)
            stock_df["ATR14"] = atr(stock_df, 14)

            if pd.isna(stock_df["SMA150"].iloc[-1]) or pd.isna(stock_df["ATR14"].iloc[-1]):
                continue

            close = float(stock_df["Close"].iloc[-1])
            sma150_val = float(stock_df["SMA150"].iloc[-1])
            if close <= sma150_val:
                continue

            avg_volume_20 = float(stock_df["Volume"].tail(20).mean())
            latest_volume = float(stock_df["Volume"].iloc[-1])
            resistance = most_recent_resistance(stock_df)

            breakout_above_sma150 = (
                float(stock_df["Close"].iloc[-2]) <= float(stock_df["SMA150"].iloc[-2]) and close > sma150_val
            )
            breakout_above_resistance = close > resistance * 1.002
            volume_confirmed = latest_volume > avg_volume_20 * 1.2

            atr14_val = float(stock_df["ATR14"].iloc[-1])
            entry, stop_loss, tp1, tp2, rr = build_trade_levels(stock_df, atr14_val, resistance)
            sl_in_atr = (entry - stop_loss) / atr14_val if atr14_val > 0 else 0
            if sl_in_atr < 1.5:
                continue
            if rr < 2:
                continue

            stock_3m = safe_pct_change(float(stock_df["Close"].iloc[-63]), close) if len(stock_df) > 63 else 0
            stock_6m = safe_pct_change(float(stock_df["Close"].iloc[-126]), close) if len(stock_df) > 126 else 0

            bench_close = float(benchmark_df["Close"].iloc[-1])
            bench_3m = safe_pct_change(float(benchmark_df["Close"].iloc[-63]), bench_close) if len(benchmark_df) > 63 else 0
            bench_6m = safe_pct_change(float(benchmark_df["Close"].iloc[-126]), bench_close) if len(benchmark_df) > 126 else 0

            rel_3m = stock_3m - bench_3m
            rel_6m = stock_6m - bench_6m

            highs_60 = stock_df["High"].tail(60)
            lows_60 = stock_df["Low"].tail(60)
            base_tightness = (highs_60.max() - lows_60.min()) / close * 100
            multi_week_pattern = base_tightness < 18 and close >= highs_60.quantile(0.9)

            days_to_earnings = get_days_to_earnings(symbol, api_key)

            metrics: Dict[str, Any] = {
                "price_vs_sma150_pct": safe_pct_change(sma150_val, close),
                "breakout_above_resistance": breakout_above_resistance,
                "breakout_above_sma150": breakout_above_sma150,
                "volume_confirmed": volume_confirmed,
                "relative_3m_pct": rel_3m,
                "relative_6m_pct": rel_6m,
                "risk_reward": rr,
                "days_to_earnings": days_to_earnings,
                "multi_week_pattern": multi_week_pattern,
            }
            breakdown = scorer.score(metrics)
            if breakdown.total < 85:
                continue

            enter_now = "WAIT" if days_to_earnings <= 14 else "YES"
            pattern = "multi-week bullish base" if multi_week_pattern else "short swing breakout"
            rationale = "; ".join(
                [
                    f"Close above SMA150 ({close:.2f}>{sma150_val:.2f})",
                    "Resistance breakout + volume" if breakout_above_resistance and volume_confirmed else "SMA150 breakout",
                    f"Rel 3M/6M vs {benchmark}: {rel_3m:.1f}%/{rel_6m:.1f}%",
                ]
            )

            results.append(
                Candidate(
                    ticker=symbol,
                    score=breakdown.total,
                    days_to_earnings=days_to_earnings,
                    enter_now=enter_now,
                    entry=entry,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    risk_reward=rr,
                    atr14=atr14_val,
                    sl_in_atr=sl_in_atr,
                    rel_3m=rel_3m,
                    rel_6m=rel_6m,
                    benchmark=benchmark,
                    rationale=rationale,
                    pattern=pattern,
                )
            )

            if idx % 40 == 0:
                time.sleep(1.2)
        except (requests.RequestException, KeyError, ValueError):
            continue

    return sorted(results, key=lambda c: c.score, reverse=True)


def fmt_money_pct(ref: float, value: float) -> str:
    pct = safe_pct_change(ref, value)
    sign = "+" if pct >= 0 else ""
    return f"${value:.2f} ({sign}{pct:.2f}%)"


def write_outputs(candidates: list[Candidate]) -> None:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    md_path = output_dir / "results.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Ticker",
                "Score/100",
                "Days to Earnings",
                "Enter now?",
                "Entry",
                "SL",
                "TP1",
                "TP2",
                "RR",
                "ATR14",
                "SL in ATR",
                "Benchmark",
                "Rel 3M %",
                "Rel 6M %",
                "Rationale",
            ]
        )
        for c in candidates:
            writer.writerow(
                [
                    c.ticker,
                    c.score,
                    c.days_to_earnings,
                    c.enter_now,
                    c.entry,
                    c.stop_loss,
                    c.tp1,
                    c.tp2,
                    round(c.risk_reward, 2),
                    round(c.atr14, 2),
                    round(c.sl_in_atr, 2),
                    c.benchmark,
                    round(c.rel_3m, 2),
                    round(c.rel_6m, 2),
                    c.rationale,
                ]
            )

    lines: list[str] = ["# Daily US Stock Scanner Results", ""]

    if not candidates:
        lines.extend(
            [
                "No setups above 85 today.",
                "",
                "Likely causes: insufficient volume-confirmed breakouts, relative strength below threshold, or earnings within 14 days.",
                "Potential relaxations: allow breakout-without-volume if trend is very strong, reduce minimum score to 80, or allow wider SL up to 6-7% while keeping >=1.5 ATR.",
            ]
        )
    else:
        for c in candidates:
            row = (
                f"- {c.ticker} | {c.score:.1f}/100 | {c.days_to_earnings}d to earnings"
                f" | Enter now? {c.enter_now} | Entry {fmt_money_pct(c.entry, c.entry)}"
                f" | SL {fmt_money_pct(c.entry, c.stop_loss)} | TP1 {fmt_money_pct(c.entry, c.tp1)}"
                f" | TP2 {fmt_money_pct(c.entry, c.tp2)} | RR {c.risk_reward:.2f}"
                f" | ATR14 {c.atr14:.2f} | SL in ATR {c.sl_in_atr:.2f}"
                f" | Rel3M/6M vs {c.benchmark}: {c.rel_3m:.1f}%/{c.rel_6m:.1f}%"
            )
            lines.append(row)
            lines.append(f"  - Level rationale: {c.rationale}. Pattern: {c.pattern}.")
            if c.days_to_earnings <= 14:
                lines.append("  - ⚠️ Earnings in <=14 days, score already penalized and entry flagged as WAIT.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    candidates = scan()
    write_outputs(candidates)
    print(f"Generated {len(candidates)} high-score candidate(s).")


if __name__ == "__main__":
    main()
