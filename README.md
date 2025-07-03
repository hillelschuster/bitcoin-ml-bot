# 🧠 Bitcoin ML Trading Bot
skibidi toilet

A real-time, 24/7 Bitcoin scalping bot using machine learning and price action patterns. Built to be modular, efficient, and emotionless.

---

## ⚠️ IMPORTANT FOR AUGMENT AGENT

> **Before modifying or generating code, YOU MUST read this entire file.**
>
> ✅ This file defines: architecture, file roles, execution flow, and coding standards.
> ✅ Work only within the structure below unless a clear, justified enhancement is needed.
> 🚫 DO NOT duplicate or freestyle logic — explain your reasoning first.
> 🚫 DO NOT HIDE CORE LOGIC BUGS WITH SUPERFICIAL SOLUTIONS. Fix the actual root cause.
> 🚫 NEVER override user guidelines or make changes without explicit permission.
> ✅ Always ask for permission before making significant changes to the codebase.

---

## 🧩 PROJECT STRUCTURE

Bitcoin_ML_Trading_Bot/
│
├── run.py                      # Main bot runner - launches loop + model monitor + reloader
│
├── /scripts                   # Utility scripts
│   └── update_price_feed.py   # Updates live price data for regime detection
├── config.json                # Core config values (symbol, ATR, cooldown, risk, etc.)
├── /config                    # Configuration files
│   ├── .env                   # API keys + Telegram secrets (not committed to git)
│   └── .env.example           # Example environment variables template
├── requirements.txt           # Python deps
│
├── /data                      # Price data storage
│   └── price_feed.csv         # Recent price data for regime detection
│
├── /logs                      # Log files and database
│   ├── trades.db              # SQLite database of trade history
│   ├── active_position.json   # Current position state
│   └── bot.log                # Main bot log file
│
├── /model_artifacts           # Trained ML models
│   ├── lstm_model.h5          # LSTM model file
│   ├── lstm_scaler.pkl        # Scaler for model features
│   └── /history               # Model version history
│
├── /src                       # All core logic modules live here
│   ├── exchange.py            # Executes trades (mock or live)
│   ├── logger.py              # Logs trades into SQLite
│   ├── notifier.py            # Sends Telegram alerts
│   ├── reporter.py            # Generates daily PnL summary
│   ├── risk_manager.py        # Sets position size + stop-loss (ATR-based)
│   ├── __init__.py            # Package initialization
│   │
│   ├── /core                  # Core trading components
│   │   ├── /bot               # Modular trading bot implementation
│   │   │   ├── orchestrator.py # Main bot orchestrator
│   │   │   ├── state_manager.py # Position state management
│   │   │   ├── safety_monitor.py # Drawdown and circuit breaker monitoring
│   │   │   ├── exit_handler.py # Position exit logic
│   │   │   ├── position_manager.py # Position entry logic
│   │   │   ├── strategy.py     # Trading strategy logic
│   │   │   ├── risk_manager.py # Advanced risk management
│   │   │   └── regime_filter.py # Market regime detection (trending/choppy)
│   │   ├── exchange.py        # Exchange API interaction
│   │   └── /utils             # Core utility functions
│   │       ├── sentiment_utils.py # Sentiment analysis
│   │       ├── retry_utils.py # Retry logic
│   │       ├── strategy_utils.py # Strategy utilities
│   │       └── reason_explainer.py # Trade reasoning generation
│   │
│   ├── /models                # ML model components
│   │   ├── auto_model_hot_reload.py # Reloads model file every X min
│   │   ├── model_lstm_core.py  # LSTM model implementation
│   │   └── model_retrain_loop.py # Model retraining + hot reload engine
│   │
│   ├── /monitoring            # Monitoring and reporting
│   │   ├── integrity_checker.py # Validates trade data and model health
│   │   ├── model_monitor.py   # Monitors model score periodically
│   │   └── reporter.py        # Generates reports and alerts
│   │
│   └── /utils                 # Utility modules
│       ├── backtest.py        # Backtesting engine with modular design
│       ├── config.py          # Configuration management
│       ├── data.py            # Data fetching and processing
│       ├── error_handler.py   # Error handling utilities
│       ├── logger.py          # Logging utilities
│       └── notifier.py        # Notification utilities

---

## 🔄 BOT FLOW

1. `run.py` starts:
   - `start_model_reload()` - Reloads model when file changes
   - `start_monitoring()` - Monitors model performance
   - `start_periodic_reporting()` - Sends status reports via Telegram
   - `run_trading_bot()` - Main trading loop
2. `core/bot/orchestrator.py`:
   - Fetches live 1m candles
   - Checks market sentiment via `sentiment_utils.py`
   - Feeds to model → gets signal
   - Uses modular components for position management and exit handling
   - Uses `risk_manager.py` for sizing/stop with sentiment adjustment
   - Logs trade, sends Telegram notification
3. Monitors run in background, reloading model and logging metrics

---

## 🔧 DEVELOPMENT GUIDELINES

### Structure Guidelines
- ✅ You **MAY add new modules inside `/src/`** for improvements (e.g. advanced ML, trade filters, indicators)
- 🚫 DO NOT create alternate runners like `bot_cli.py`, etc
- 🚫 DO NOT duplicate model logic, logging, or strategy handlers
- 🚫 DO NOT create any new files without explicit permission from the user
- ✅ When in doubt, **explain the proposed change first** (e.g., "adding a new module for liquidation feed parser")

### Code Organization
- ✅ All core functionality is in the following directories:
  - `src/core/` - Core trading logic and exchange functionality
  - `src/utils/` - Utility functions and logging
  - `src/monitoring/` - Monitoring and reporting tools
  - `src/models/` - Machine learning models

- ✅ The following files are thin wrappers for backward compatibility:
  - `src/exchange.py` → imports from `src/core/exchange.py`
  - `src/risk_manager.py` → imports from `src/core/risk_manager.py`
  - `src/logger.py` → imports from `src/utils/logger.py`
  - `src/notifier.py` → imports from `src/utils/notifier.py`
  - `src/reporter.py` → imports from `src/monitoring/reporter.py`

- 🚫 DO NOT add new functionality to wrapper files - add it to the corresponding implementation files instead
- ✅ `run.py` is the unified runner that launches all components (model reloader, monitor, reporter, trading loop, sentiment analysis)

### Position Sizing Guidelines
- ✅ All position sizing caps should be based on position value as a percentage of account balance (e.g., 5%), not on a fixed BTC amount
- ✅ This ensures the bot adapts to changing account sizes and market prices
- ✅ Only use fixed BTC caps if explicitly requested in the strategy logic
- ✅ Always log `[POSITION CAPPED]` with the calculated values for transparency
- ✅ Ensure minimum notional value of $100 (Binance requirement) is always met

### Balance Fetching Guidelines
- ✅ Always fetch the current account balance using the bot's active exchange interface
- ✅ Use `balance = self.exchange.get_account_balance()` to ensure all modules operate on live portfolio value
- ✅ Never use manual balance fetching or hardcoded fallback unless connection fails
- ✅ Apply this across risk_manager, strategy, execution, and reporting layers
- ✅ The exchange interface will log `[BALANCE FETCHED]` and `[FALLBACK BALANCE]` in all cases

---

## 🧪 TESTING & MONITORING

- Trades logged to `logs/trades.db`
- Model checked every 5 min
- Daily reports printed to console
- Telegram notifications for trades and performance reports

## 📈 MARKET REGIME FILTER

The bot includes a market regime detection system that classifies market conditions as either:

- **TRENDING**: Strong directional movement with sufficient volatility
- **CHOPPY**: Sideways or erratic price action with weak directional movement

The bot will skip trades during choppy markets to reduce false signals and losses.

📖 **Additional Docs**

- Market Regime Filter → `docs/README_regime_filter.md`

To keep the price feed updated for regime detection, run:
```bash
python scripts/update_price_feed.py
```

---

## 🔑 API Keys Setup

### Binance Testnet
1. Register at https://testnet.binancefuture.com/
2. Go to API Management and create a new API key
3. Copy the API key and secret to your `config/.env` file:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=True
```

### Important Notes
- The bot requires valid Binance Testnet API keys to function properly
- API keys must have trading permissions enabled
- Minimum order size on Binance is $100 notional value

## 💬 Telegram

Set `config/.env` with:
```env
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
```

## 🌐 Environment Variables

All environment variables are centrally managed through `config/Config.py`. This module:

- Loads variables from `config/.env` using dotenv
- Provides default values for optional variables
- Exports variables for use throughout the application
- Centralizes all environment-based configuration

To use environment variables in your code:

```python
from config.Config import BINANCE_API_KEY, TEST_MODE, RISK_PER_TRADE

# Variables are already properly typed (bool, float, etc.)
if TEST_MODE:
    # Test mode logic
    pass
```

## 📦 Install
```bash
pip install -r requirements.txt
```

## Run bot:
```bash
python run.py
```

## 🔧 Debugging & Troubleshooting

### API Connection Issues
- If you see `Invalid API-key` errors, verify your API keys in the `config/.env` file
- Make sure your API keys have the correct permissions (trading enabled)
- Check that your IP is allowed in the Binance Testnet API settings

### Minimum Order Size
- Binance requires a minimum order size of $100 notional value
- The bot will automatically adjust order sizes to meet this requirement
- Position sizing is based on percentage of account balance (up to 5%), not fixed BTC amounts
- Both minimum ($100) and maximum (5% of balance) limits are enforced automatically
- If you see `Order's notional must be no smaller than 100` errors, check your position sizing logic

### ⚠️ Leverage Behavior (Binance Futures)

The bot includes leverage in its position size formula:

```
position_size = (balance × risk_percent) × leverage ÷ price
```

This does **not** increase your risk. Leverage is applied by Binance.

The bot needs this value only to calculate how much BTC/ETH/SOL to buy.

If you change leverage on Binance manually, you must also update it in the bot (for now).

Later, we may implement auto-sync with Binance's leverage setting via API.

### Testing
- Always test with Binance Testnet before using real funds
- Monitor the logs for any errors or warnings

Augment agent, read above before proceeding. Ask before any structural change.

