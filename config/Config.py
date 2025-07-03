"""
Centralized configuration and environment variable management for the Bitcoin ML Trading Bot.

This module provides a centralized way to load and access environment variables
and configuration settings for the trading bot.
"""

from dotenv import load_dotenv
import os

# Load environment variables from config/.env
load_dotenv(dotenv_path="config/.env")

# Binance API credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Trading configuration
TEST_MODE = os.getenv("TEST_MODE", "True").lower() == "true"
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "5"))

# No API tokens needed for now

# Export all environment variables for easy access
__all__ = [
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TEST_MODE",
    "RISK_PER_TRADE",
    "MAX_DAILY_TRADES"
]
