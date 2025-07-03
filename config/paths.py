"""
Centralized path definitions for the Bitcoin ML Trading Bot.

This module provides a single source of truth for all file paths
used throughout the application, making it easier to maintain
and update paths when needed.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# State files
POSITION_FILE = os.path.join(LOGS_DIR, "active_position.json")
CIRCUIT_BREAKER_FILE = os.path.join(LOGS_DIR, "circuit_breaker.json")
EMERGENCY_EXIT_METRICS_FILE = os.path.join(LOGS_DIR, "emergency_exit_metrics.json")

# Database files
TRADES_DB_PATH = os.path.join(LOGS_DIR, "trades.db")

# Model files
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_FILE = os.path.join(MODEL_DIR, "lstm_scaler.pkl")

# Data files
PRICE_FEED_FILE = os.path.join(DATA_DIR, "price_feed.csv")

# Config files
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
ENV_FILE = os.path.join(CONFIG_DIR, ".env")

# Export all path variables
__all__ = [
    "BASE_DIR",
    "LOGS_DIR",
    "DATA_DIR",
    "MODEL_DIR",
    "CONFIG_DIR",
    "POSITION_FILE",
    "CIRCUIT_BREAKER_FILE",
    "EMERGENCY_EXIT_METRICS_FILE",
    "TRADES_DB_PATH",
    "MODEL_FILE",
    "SCALER_FILE",
    "PRICE_FEED_FILE",
    "CONFIG_FILE",
    "ENV_FILE"
]
