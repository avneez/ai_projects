# Price Movement Detector

## Purpose
Monitor real-time price data and trigger prediction pipeline when significant price movements are detected.

## Libraries Used
- **Redis** - Subscribe to market data pub/sub
- **NumPy** - Statistical calculations
- **multiprocessing** - Parallel monitoring of multiple symbols

## Detection Logic

### Trigger Conditions
```
Significant movement if ANY condition met:

1. Price Change: |current_price - prev_price| / prev_price > 2%
2. Volume Spike: current_volume > 2x average_volume_20bars
3. Volatility: ATR (Average True Range) > 1.5x historical_ATR
4. Gap: |open - prev_close| / prev_close > 1%
```

**Why these thresholds?**
- 2% price move: Significant for most stocks (not noise)
- 2x volume: Indicates unusual activity
- 1.5x ATR: Abnormal volatility
- 1% gap: Overnight news impact

### Algorithm Flow
```
Subscribe to Redis: market:bars:{SYMBOL}
  ↓
For each new bar (1-minute OHLCV):
  ↓
Calculate metrics:
  - price_change_pct
  - volume_ratio (vs 20-bar MA)
  - current_ATR
  ↓
Check trigger conditions
  ↓
If triggered:
  ├─ Fetch historical features (TimescaleDB)
  ├─ Query RAG for news context
  └─ Send to prediction pipeline
Else:
  └─ Continue monitoring
```

### Rolling Window Calculations
```python
# Pseudo-logic for volume spike detection
class PriceMovementDetector:
    def __init__(self):
        self.volume_history = deque(maxlen=20)  # Last 20 bars

    def check_volume_spike(self, current_volume):
        if len(self.volume_history) < 20:
            return False  # Need warmup period

        avg_volume = np.mean(self.volume_history)
        spike = current_volume > (2 * avg_volume)

        self.volume_history.append(current_volume)

        return spike
```

## Multiprocessing Architecture

### Process Pool for Scalability
```
Main Process (Orchestrator)
  ↓
Fork Worker Processes (1 per symbol or batch)
  ├─ Worker 1: Monitor AAPL, MSFT, GOOGL (batch)
  ├─ Worker 2: Monitor TSLA, NVDA, META
  ├─ Worker 3: Monitor AMZN, JPM, BAC
  └─ Worker N: ...

Each worker:
  - Subscribes to Redis channels
  - Maintains own state (volume history, ATR)
  - Independent trigger detection
```

**Why multiprocessing?**
- Python GIL: Blocks multi-threading for CPU work
- Each worker process has own Python interpreter
- True parallelism on multi-core CPUs
- Can monitor 100+ symbols simultaneously

### Worker Process Logic
```python
# Conceptual
def worker_process(symbols):
    redis_client = Redis()
    detectors = {sym: PriceMovementDetector(sym) for sym in symbols}

    pubsub = redis_client.pubsub()
    channels = [f"market:bars:{sym}" for sym in symbols]
    pubsub.subscribe(*channels)

    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel']
            symbol = channel.split(':')[-1]
            bar_data = json.loads(message['data'])

            if detectors[symbol].check_trigger(bar_data):
                # Send to prediction queue
                trigger_prediction(symbol, bar_data)
```

## State Management

### Per-Symbol State
```python
{
  "symbol": "AAPL",
  "last_price": 150.50,
  "volume_history": [1M, 1.2M, 0.9M, ...],  # deque(maxlen=20)
  "atr_history": [2.1, 2.3, 2.0, ...],      # deque(maxlen=14)
  "last_trigger_time": "2024-11-19T10:30:00Z",
  "cooldown_until": "2024-11-19T10:35:00Z"   # Prevent spam triggers
}
```

**Cooldown Period:**
- After trigger: Wait 5 minutes before next trigger
- Prevents multiple triggers for same event
- Reduces redundant predictions

## Performance Considerations

### Throughput
- **50 symbols**: 50 bars/minute = 0.83 bars/sec → Single process
- **500 symbols**: 500 bars/minute → 10 worker processes
- **CPU usage**: <5% per worker (mostly I/O waiting)

### Memory Usage
- **Per symbol**: ~10 KB (rolling windows)
- **500 symbols**: 5 MB total
- **Redis connection**: ~1 MB per worker

## Integration Points

### Input
- **Source**: Redis pub/sub `market:bars:{SYMBOL}`
- **Format**: JSON with OHLCV data
- **Frequency**: 1-minute bars

### Output
- **Destination**: Internal queue (multiprocessing.Queue)
- **Consumer**: Feature engineering + prediction pipeline
- **Format**: `{"symbol": "AAPL", "timestamp": "...", "trigger_reason": "price_change_3.2%"}`

## False Positive Handling

### Market Open Volatility
```
First 15 minutes after market open (9:30-9:45 AM):
  - Increase threshold to 3% (vs 2%)
  - Higher volume threshold (3x vs 2x)
  - Reason: Normal volatility at open
```

### Earnings Days
```
If today = earnings date:
  - Increase thresholds by 50%
  - Or disable detector (rely on scheduled analysis)
```

### Pre-Market/After-Hours
```
Outside 9:30 AM - 4:00 PM ET:
  - Use separate thresholds (5% price move)
  - Lower volume data quality
```

## Interview Q&A

**Q: Why monitor price movements instead of continuous prediction?**
A: Cost/efficiency. Continuous prediction = 500 symbols × 60 predictions/hour = 30K predictions/day. Event-driven = ~1K predictions/day (only on significant moves). 30x cost reduction.

**Q: What if multiple symbols trigger simultaneously?**
A: Queue-based processing. Prediction pipeline consumes from queue at its own pace (e.g., 10 predictions/minute). Overflow → queue depth increases → autoscaling triggers more workers.

**Q: How do you backtest trigger logic?**
A: Historical replay:
1. Load historical OHLCV data
2. Simulate Redis stream
3. Count triggers vs actual opportunities
4. Optimize thresholds (minimize false positives/negatives)

**Q: What about flash crashes?**
A: Circuit breaker logic:
- If price drops >10% in <5 minutes → Flag as anomaly
- Pause triggers for that symbol
- Alert for manual review
- Prevents bad trades on data errors

**Q: How do you tune thresholds?**
A: Backtesting + optimization:
1. Grid search over threshold combinations
2. Maximize: True signals / Total triggers
3. A/B test in production (shadow mode)

**Q: Multiprocessing vs asyncio?**
A:
- **Multiprocessing**: True parallelism, CPU-bound calculations
- **Asyncio**: Better for I/O-bound (network calls)
- Choice: Multiprocessing (calculations + isolation per symbol)

**Q: What if Redis goes down?**
A: Fall back to polling TimescaleDB every 1 minute. Higher latency (1-min delay) but system continues. Alert monitoring team.

**Q: How do you test this without real market data?**
A: Mock Redis publisher:
```python
# Test script
publish_fake_bars(symbol="AAPL", pattern="spike")  # Inject test data
assert detector.triggered == True
```
