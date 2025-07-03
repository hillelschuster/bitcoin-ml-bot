# Market Regime Filter

This document explains how to use the Market Regime Filter feature.

## How It Works

The Market Regime Filter helps the bot determine if the market is in a trending or choppy state. This can help avoid trading during choppy markets, which often lead to false signals and losses.

1. The filter analyzes price data using three indicators:
   - ADX (Average Directional Index) - measures trend strength
   - Bollinger Band Width - measures volatility
   - Moving Average Slope - measures direction and momentum

2. Based on these indicators, it classifies the market as either:
   - TRENDING: Strong directional movement with sufficient volatility
   - CHOPPY: Sideways or erratic price action with weak directional movement

3. The bot will skip trades during choppy markets to reduce false signals.

## Live Price Feed

The bot needs recent price data to determine the market regime. This is stored in `data/price_feed.csv`.

To keep this file updated:

```bash
# Run this script regularly (e.g., every minute)
python scripts/update_price_feed.py
```

You can set up a scheduled task or cron job to run this script automatically.

## Monitoring

The bot's performance and market regime status can be monitored through:

### Features:

- Console logs showing the current market regime (Trending/Choppy)
- Telegram notifications for trades and performance reports
- Trade history stored in the SQLite database (`logs/trades.db`)
- Daily performance reports

## Integration with Bot

The Market Regime Filter is already integrated into the bot's trading logic. It will automatically skip trades during choppy markets and log the current regime status.

You can see these logs in the bot's output and in the Telegram notifications.
