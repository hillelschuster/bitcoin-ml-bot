"""
Trading bot module for the Bitcoin ML Trading Bot.

This package provides a modular implementation of the trading bot,
with separate components for state management, safety monitoring,
position management, and exit handling.
"""

from .orchestrator import TradingOrchestrator, run_trading_bot
from .state_manager import StateManager
from .safety_monitor import SafetyMonitor
from .exit_handler import ExitHandler
from .position_manager import PositionManager

__all__ = [
    "TradingOrchestrator",
    "run_trading_bot",
    "StateManager",
    "SafetyMonitor",
    "ExitHandler",
    "PositionManager"
]
