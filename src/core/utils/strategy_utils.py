"""
Strategy utilities for the trading bot.

This module provides functions for feature preparation, veto conditions,
and feature bundle construction.
"""

import logging
from typing import Dict, List, Optional, Any
from src.models.model_lstm_core import load_model_metadata

logger = logging.getLogger("core.utils.strategy_utils")


def prepare_features(market_data, sentiment_score=None, regime=None, model_confidence=None, signal=None):
    """
    Prepare features for confidence calculation and strategy execution.

    Args:
        market_data: Market data DataFrame with price information
        sentiment_score: Market sentiment score (optional)
        regime: Market regime classification (optional)
        model_confidence: Model confidence score (optional)
        signal: Trading signal (UP, DOWN, HOLD) (optional)

    Returns:
        dict: Dictionary of features for strategy execution
    """
    try:
        # Extract price data from market data
        if market_data is not None and not market_data.empty and 'close' in market_data:
            price_data = market_data['close'].values
        else:
            logger.warning("Invalid market data for feature preparation")
            price_data = []

        # Set default values for missing features
        if sentiment_score is None:
            try:
                from src.core.utils.sentiment_utils import fetch_sentiment_score
                sentiment_score = fetch_sentiment_score()
            except Exception as e:
                logger.debug(f"Could not fetch sentiment for feature preparation: {e}")
                sentiment_score = 50  # Default neutral sentiment

        # Prepare feature dictionary
        features = {
            'price_data': price_data,
            'sentiment_score': sentiment_score,
            'regime': regime or 'UNKNOWN',
            'expected_trend': signal or 'HOLD',
            'model_confidence': model_confidence or 0.5
        }

        logger.debug(f"Prepared features: sentiment={sentiment_score}, regime={regime}, signal={signal}")
        return features
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        # Return minimal feature set on error
        return {
            'price_data': [],
            'sentiment_score': 50,
            'regime': 'UNKNOWN',
            'expected_trend': 'HOLD',
            'model_confidence': 0.5
        }


def check_veto_conditions(confidence, min_confidence=0.65, regime=None, allowed_regimes=None, signal=None,
                   sentiment_score=None, model=None, use_fallback=True):
    """
    Check if a trade should be vetoed based on confidence and regime.

    Args:
        confidence: Model confidence score
        min_confidence: Minimum required confidence
        regime: Current market regime
        allowed_regimes: List of allowed market regimes
        signal: Current trading signal
        sentiment_score: Current market sentiment score (0-100)
        model: The model being used for predictions
        use_fallback: Whether to use fallback strategy for untrained models

    Returns:
        dict: Veto information including whether trade was vetoed and why
    """
    # Load model metadata to check if model is trained
    metadata = load_model_metadata()

    # Initialize veto info
    veto_info = {
        "veto": False,
        "min_confidence": min_confidence,
        "allowed_regimes": allowed_regimes or ["TRENDING_UP", "TRENDING_DOWN", "TRENDING"]
    }

    # Check if model is untrained based on metadata
    if use_fallback and not metadata.get("trained", False):
        logger.info("⚠️ Model is untrained — using fallback strategy")

        # Use fallback strategy based on sentiment and regime
        if sentiment_score is not None and regime is not None:
            # Allow trade only if sentiment >= 55 and regime is TRENDING, TRENDING_UP, or TRENDING_DOWN
            if sentiment_score >= 55 and regime in ["TRENDING", "TRENDING_UP", "TRENDING_DOWN"]:
                logger.info("✅ Fallback conditions met — bypassing confidence veto.")
                veto_info["veto"] = False
                veto_info["using_fallback"] = True
                veto_info["sentiment_score"] = sentiment_score
                return veto_info
            else:
                veto_info["veto"] = True
                veto_info["veto_reason"] = "Fallback conditions not satisfied (need sentiment >=55 and TRENDING regime)"
                veto_info["using_fallback"] = True
                logger.info("🚫 Fallback conditions not met — veto trade.")
                return veto_info

    # Standard veto checks if model is trained or fallback not applicable

    # Check confidence threshold
    if confidence < min_confidence:
        veto_info["veto"] = True
        veto_info["veto_reason"] = f"Confidence too low: {confidence:.2f} < {min_confidence:.2f}"
        logger.info(f"Trade vetoed: {veto_info['veto_reason']}")
        return veto_info

    # Check regime if provided
    if regime and allowed_regimes:
        if regime not in allowed_regimes:
            veto_info["veto"] = True
            veto_info["veto_reason"] = f"Regime not allowed: {regime} not in {allowed_regimes}"
            logger.info(f"Trade vetoed: {veto_info['veto_reason']}")
            return veto_info

    # Check signal if provided
    if signal and signal == "HOLD":
        veto_info["veto"] = True
        veto_info["veto_reason"] = f"Signal is HOLD, no trade needed"
        logger.info(f"Trade vetoed: {veto_info['veto_reason']}")
        return veto_info

    return veto_info


def build_feature_bundle(market_data, sentiment_score=None, regime=None, signal=None, confidence=None, price=None):
    """
    Build a comprehensive feature bundle for strategy execution and reasoning.

    Args:
        market_data: Market data DataFrame with price information
        sentiment_score: Market sentiment score (optional)
        regime: Market regime classification (optional)
        signal: Trading signal (UP, DOWN, HOLD) (optional)
        confidence: Model confidence score (optional)
        price: Current price (optional)

    Returns:
        dict: Comprehensive feature bundle for strategy execution
    """
    # Prepare basic features
    features = prepare_features(market_data, sentiment_score, regime, confidence, signal)

    # Extract current price if not provided
    if price is None and market_data is not None and not market_data.empty and 'close' in market_data:
        price = market_data['close'].iloc[-1]

    # Add additional features
    features.update({
        'current_price': price,
        'signal': signal,
        'confidence': confidence
    })

    # Calculate adaptive cooldown based on confidence
    if confidence is not None:
        cooldown_time = get_adaptive_cooldown(confidence)
        features['cooldown'] = cooldown_time

    return features


def get_adaptive_cooldown(confidence):
    """
    Calculate adaptive cooldown period based on prediction confidence.

    Higher confidence trades have shorter cooldown periods:
    - Confidence ≥ 0.95: 60 seconds
    - Confidence ≥ 0.85: 180 seconds
    - Confidence < 0.85: 600 seconds

    Args:
        confidence (float): Prediction confidence (0.0-1.0)

    Returns:
        int: Cooldown period in seconds
    """
    if confidence >= 0.95:
        cooldown = 60  # 1 minute for high confidence
    elif confidence >= 0.85:
        cooldown = 180  # 3 minutes for medium confidence
    else:
        cooldown = 600  # 10 minutes for low confidence

    logger.info(f"[COOLDOWN] Confidence: {confidence:.4f} → Waiting {cooldown} seconds")
    return cooldown
