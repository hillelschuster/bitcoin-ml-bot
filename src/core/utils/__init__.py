"""
Utility modules for the trading bot.

This package provides utility functions for sentiment analysis,
retry logic, strategy feature preparation, and trade reasoning.
"""

from .sentiment_utils import fetch_sentiment_score, classify_sentiment
from .retry_utils import retry_exchange_call
from .strategy_utils import prepare_features, check_veto_conditions, build_feature_bundle, get_adaptive_cooldown
from .reason_explainer import generate_trade_reason, log_reasoning, get_veto_info

__all__ = [
    "fetch_sentiment_score",
    "classify_sentiment",
    "retry_exchange_call",
    "prepare_features",
    "check_veto_conditions",
    "build_feature_bundle",
    "get_adaptive_cooldown",
    "generate_trade_reason",
    "log_reasoning",
    "get_veto_info"
]
