# Daily US Stock Scanning Agent (Finnhub + Python)

This project scans US-listed common stocks once per day and outputs **only high-quality swing setups** (score >= 85).

## Strategy constraints enforced

Hard filters:
- Market Cap > $1B
- Price > SMA150

Candidate computations:
- ATR(14)
- SMA150
- Breakout signal (SMA150 breakout OR resistance breakout with volume confirmation)
- Relative return vs sector ETF benchmark (3M and 6M)

Risk/trade constraints:
- Stop-loss distance >= 1.5x ATR14
- Prefer stop <= 5% below entry (if not possible, fallback logic documented in output rationale)
- Stop not placed exactly on obvious support (nudged beyond support)
- Risk:Reward >= 1:2 (setups below are excluded)
- Earnings <= 14 days: score penalty and `Enter now? WAIT`

## Files

- `src/scan.py`: Main scanner and output generator
- `src/scoring.py`: Scoring model
- `output/results.csv`: Machine-readable results
- `output/results.md`: Human-readable rows and rationale
- `.github/workflows/daily_scan.yml`: Daily scheduled run + auto-commit outputs

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment:
   ```bash
   cp config.example.env .env
   ```
3. Add your `FINNHUB_API_KEY` in `.env`.

## Run

```bash
python src/scan.py
```

## Output format

Each qualifying row in markdown includes:

`TICKER | Score/100 | Days to next earnings | Enter now? (YES/WAIT) | Entry | SL | TP1 | TP2 | RR | ATR14 | SL in ATR`

If no setups pass (score >= 85), the file starts with:

`No setups above 85 today`

followed by a concise explanation and what constraints to relax.

## Notes on data quality

- Primary universe and company profile source: Finnhub.
- All profile, candles, earnings calendar, and financial metric fallbacks are sourced from Finnhub endpoints only (no Yahoo scraping).
- Company profile (`/stock/profile2`), OHLCV candles (`/stock/candle`), and earnings calendar (`/calendar/earnings`) are fetched directly from Finnhub with rate-limit protection (0.25s sleep between requests).
- No fabricated values are used.
