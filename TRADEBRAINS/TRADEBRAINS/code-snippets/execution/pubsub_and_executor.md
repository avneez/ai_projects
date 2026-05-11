# Pub/Sub Architecture & Trade Executor

## Purpose
Decouple prediction model from trade execution using pub/sub messaging. Enables horizontal scaling of executors and fault isolation.

## Architecture Overview

```
Prediction Model
  ↓
Publish Signal → Redis Pub/Sub Topic: "trading_signals"
  ↓ (Broadcast to all subscribers)
├─ Trade Executor Instance 1
├─ Trade Executor Instance 2
├─ Trade Executor Instance 3
└─ Trade Executor Instance N

Each executor:
- Receives ALL signals
- Distributed lock ensures only ONE executes
- Fault tolerance: If one fails, others continue
```

## Libraries Used

###Publisher Side
- **Redis** - Pub/sub messaging
- **json** - Signal serialization

### Executor Side
- **Redis** - Pub/sub + distributed locking
- **Alpaca Trade API** - Broker integration
- **psycopg2** - PostgreSQL for trade logging

## Signal Format

### Published Message
```json
{
  "signal_id": "uuid-1234-5678",
  "timestamp": "2024-11-19T10:30:15Z",
  "ticker": "AAPL",
  "action": "BUY",  // BUY, SELL, HOLD
  "confidence": 0.72,
  "position_size_usd": 5000,
  "position_size_shares": 33,
  "model_version": "ensemble_v2.3",
  "contributing_factors": {
    "xgboost_prob": 0.68,
    "lstm_prob": 0.75,
    "llm_sentiment": 0.65
  },
  "trigger_reason": "price_spike_2.5%",
  "expires_at": "2024-11-19T10:35:15Z"  // 5-min TTL
}
```

### Why Include Metadata?
- **signal_id**: Deduplication + audit trail
- **confidence**: Risk management decisions
- **expires_at**: Ignore stale signals
- **contributing_factors**: Debugging + analysis

## Publisher Implementation Logic

```python
# Conceptual pseudo-code
class SignalPublisher:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.channel = "trading_signals"

    def publish_signal(self, prediction_result):
        signal = {
            "signal_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "ticker": prediction_result.symbol,
            "action": prediction_result.action,
            "confidence": prediction_result.confidence,
            "position_size_usd": prediction_result.position_size,
            "expires_at": (datetime.now() + timedelta(minutes=5)).isoformat(),
            # ... other fields
        }

        # Publish to Redis
        self.redis.publish(
            self.channel,
            json.dumps(signal)
        )

        # Also log to database for audit
        log_signal_to_db(signal)
```

## Trade Executor Architecture

### Distributed Locking (Critical!)

**Problem:** Multiple executor instances receive same signal
**Solution:** Distributed lock ensures only ONE executes

```python
def trade_executor_worker():
    pubsub = redis_client.pubsub()
    pubsub.subscribe("trading_signals")

    for message in pubsub.listen():
        if message['type'] == 'message':
            signal = json.loads(message['data'])

            # Try to acquire lock
            lock_key = f"lock:signal:{signal['signal_id']}"
            locked = redis_client.set(
                lock_key,
                "1",
                nx=True,  # Only set if not exists
                ex=60     # Expire after 60 seconds
            )

            if locked:
                # This instance won the lock
                try:
                    process_signal(signal)
                except Exception as e:
                    logger.error(f"Execution failed: {e}")
                finally:
                    # Release lock
                    redis_client.delete(lock_key)
            else:
                # Another instance already processing
                logger.debug(f"Signal {signal['signal_id']} already locked")
```

**Lock Parameters:**
- **nx=True**: Set only if key doesn't exist (atomic operation)
- **ex=60**: Auto-expire after 60s (prevents deadlock if executor crashes)

### Signal Processing Pipeline

```
Receive Signal
  ↓
Validate Signal
  ├─ Check expiry (ignore if expired)
  ├─ Check action (BUY/SELL/HOLD)
  └─ Check confidence (skip if < threshold)
  ↓
Risk Management Checks
  ├─ Position limits (max 5% per stock)
  ├─ Drawdown check (pause if -10% daily)
  ├─ Blacklist check (avoid problematic stocks)
  └─ Market hours check (9:30-16:00 ET)
  ↓
Execute Trade (Broker API)
  ├─ Market order (fast execution)
  ├─ Or limit order (better price)
  └─ Get order confirmation
  ↓
Log Trade
  ├─ PostgreSQL (persistent audit trail)
  ├─ Redis (real-time monitoring)
  └─ Metrics (Prometheus)
  ↓
Acknowledge Signal
```

### Risk Management Checks

```python
def passes_risk_checks(signal):
    # 1. Position size limit
    current_positions = get_portfolio_positions()
    if signal['position_size_usd'] > 0.05 * portfolio_value:
        return False, "Position size exceeds 5% limit"

    # 2. Daily drawdown limit
    daily_pnl = get_daily_pnl()
    if daily_pnl < -0.10 * portfolio_value:
        return False, "Daily drawdown limit reached (-10%)"

    # 3. Maximum open positions
    if len(current_positions) >= 20:
        return False, "Max 20 open positions"

    # 4. Duplicate position check
    if signal['ticker'] in current_positions:
        return False, f"Already have position in {signal['ticker']}"

    # 5. Market hours
    if not is_market_open():
        return False, "Market closed"

    # 6. Confidence threshold
    if signal['confidence'] < 0.60:
        return False, "Confidence below 60% threshold"

    return True, None
```

### Broker Integration (Alpaca Example)

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class BrokerExecutor:
    def __init__(self, api_key, api_secret):
        self.client = TradingClient(api_key, api_secret)

    def execute_trade(self, signal):
        # Prepare order
        order_data = MarketOrderRequest(
            symbol=signal['ticker'],
            qty=signal['position_size_shares'],
            side=OrderSide.BUY if signal['action'] == 'BUY' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        # Submit order
        order = self.client.submit_order(order_data)

        return {
            "order_id": order.id,
            "status": order.status,
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
        }
```

## Horizontal Scaling

### Load Balancing
```
Scenario: High signal volume (100 signals/minute)

Single Executor: Can handle ~10-20 orders/minute
Solution: Deploy 5-10 executor instances

Redis pub/sub broadcasts to all → Distributed lock ensures no duplicates
```

### Scaling Triggers
```
Metrics to monitor:
- Signal processing latency
- Queue depth (Redis list if implementing buffering)
- Executor CPU usage

Auto-scale rules:
- If latency > 5s: Add 2 instances
- If CPU > 80%: Add 1 instance
- If signals < 10/min for 10 min: Remove instances
```

## Fault Tolerance

### Executor Instance Failure
```
Executor crashes mid-execution:
  ↓
Distributed lock expires after 60s
  ↓
Another instance can acquire lock
  ↓
Retry signal processing

Idempotency: Check if order already placed (via order_id in DB)
```

### Redis Pub/Sub Failure
```
If Redis connection lost:
  ↓
Executor enters reconnect loop
  ↓
Buffered signals in memory (up to 100)
  ↓
Reconnect → Flush buffer
  ↓
Resume normal operation

Alert: Page ops team if down > 1 minute
```

### Broker API Failure
```
If Alpaca API returns error:
  ↓
Retry with exponential backoff (3 attempts)
  ↓
If still failing:
    ├─ Log error
    ├─ Increment failure counter
    └─ Alert if failure rate > 5%

Signal marked as "failed" in database
```

## Trade Logging & Audit

### PostgreSQL Schema
```sql
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    signal_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- BUY, SELL
    confidence DECIMAL(5,4),
    position_size_shares INT,
    order_price DECIMAL(12,4),
    filled_price DECIMAL(12,4),
    order_id VARCHAR(100),  -- Broker order ID
    status VARCHAR(20),  -- FILLED, PARTIAL, REJECTED
    executor_instance VARCHAR(50),  -- Which instance executed
    execution_time_ms INT,  -- Latency metric
    model_version VARCHAR(50)
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_ticker ON trades(ticker);
CREATE INDEX idx_trades_signal_id ON trades(signal_id);  -- For deduplication
```

### Real-Time Metrics (Prometheus)
```
# Counters
trades_executed_total{action="BUY|SELL", status="FILLED|REJECTED"}
signals_received_total
signals_rejected_total{reason="expired|low_confidence|risk_check"}

# Histograms
trade_execution_latency_seconds
signal_processing_duration_seconds

# Gauges
active_positions_count
portfolio_value_usd
daily_pnl_usd
```

## Monitoring & Alerting

### Critical Alerts
```
1. Executor down (no heartbeat for 1 minute)
2. Trade rejection rate > 20%
3. Execution latency > 10 seconds
4. Daily drawdown > 10%
5. Position limit exceeded
6. Broker API errors > 5/minute
```

### Dashboard Metrics
```
- Trades per minute (real-time chart)
- Win rate (rolling 24 hours)
- P&L curve
- Signal distribution (BUY/SELL/HOLD ratio)
- Execution latency (p50, p95, p99)
- Active positions heatmap
```

## Interview Q&A

**Q: Why Redis Pub/Sub over Kafka?**
A:
- **Redis**: Lower latency (<1ms), simpler setup, good for small messages
- **Kafka**: Better for high throughput, replay capability, multi-consumer patterns
- Choice: Redis (low latency critical for trading, moderate volume)

**Q: What if two executors acquire lock simultaneously?**
A: Impossible. Redis SET NX is atomic (implemented at server level). Only one client can set a non-existent key.

**Q: How do you handle partial fills?**
A: Log partial fill, create new signal for remaining quantity. Or use limit orders with fill-or-kill (FOK) to ensure atomic execution.

**Q: What about slippage (price moves before execution)?**
A: Accept slippage <0.5% (market orders). For larger orders: split into chunks, use limit orders, or implement TWAP/VWAP execution algorithms.

**Q: How do you test executors without real money?**
A: Paper trading mode (Alpaca provides paper trading accounts). Same code, fake money. Test for 1-2 weeks before live deployment.

**Q: What if signal expires while locked (processing takes >5min)?**
A: Lock TTL (60s) < signal expiry (5min). If processing takes that long, there's a bigger problem. Alert and investigate.

**Q: How to roll out new executor version without downtime?**
A:
1. Deploy new version alongside old (both subscribe)
2. New instances process 10% of signals (distributed lock naturally balances)
3. Monitor for 1 hour
4. If stable: Scale up new, scale down old
5. Zero downtime

**Q: Can you prioritize high-confidence signals?**
A: Redis pub/sub doesn't have priority. Solutions:
1. Separate channels (high_confidence_signals, normal_signals)
2. Or use Redis Streams (supports consumer groups + priority)
3. Or executor filters by confidence first
