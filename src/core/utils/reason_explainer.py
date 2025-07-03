"""
Trade reasoning explainer for the Bitcoin ML Trading Bot.

This module generates structured explanations for trade decisions,
capturing the full context and rationale behind each trade or skipped trade.
It provides both machine-readable data for analysis and human-readable
summaries for monitoring.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Union, List

# Configure logging
logger = logging.getLogger('core.utils.reason_explainer')

def generate_trade_reason(
    symbol: str,
    timeframe: str,
    price: float,
    confidence: float,
    regime: str,
    sentiment: float,
    signal: str,
    threshold_info: Optional[Dict[str, Any]] = None,
    additional_factors: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a structured explanation for a trade decision.

    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT')
        timeframe (str): Timeframe of the analysis (e.g., '1m', '5m')
        price (float): Current price at decision time
        confidence (float): Model confidence score (0 to 1)
        regime (str): Market regime classification (e.g., 'TRENDING_UP', 'CHOPPY')
        sentiment (float): Market sentiment score (-1 to 1 or 0 to 100)
        signal (str): Model signal (e.g., 'BUY', 'SELL', 'HOLD')
        threshold_info (dict, optional): Information about decision thresholds
        additional_factors (dict, optional): Any additional factors affecting the decision

    Returns:
        dict: JSON-serializable dictionary with comprehensive reasoning details
    """
    # Create base reasoning structure
    reasoning = {
        "timestamp": datetime.utcnow().isoformat(),
        "market_context": {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": price
        },
        "decision_factors": {
            "confidence": confidence,
            "regime": regime,
            "sentiment": sentiment,
            "signal": signal
        },
        "thresholds": threshold_info or {},
        "additional_factors": additional_factors or {}
    }
    
    # Add human-readable summary
    reasoning["summary"] = generate_summary(
        confidence, regime, sentiment, signal, 
        threshold_info, additional_factors
    )
    
    # Add decision explanation
    reasoning["explanation"] = generate_explanation(
        confidence, regime, sentiment, signal,
        threshold_info, additional_factors
    )
    
    return reasoning

def generate_summary(
    confidence: float,
    regime: str,
    sentiment: Union[float, str],
    signal: str,
    threshold_info: Optional[Dict[str, Any]] = None,
    additional_factors: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a concise human-readable summary of the trade decision.
    
    Args:
        confidence (float): Model confidence score
        regime (str): Market regime classification
        sentiment (float or str): Market sentiment score or classification
        signal (str): Model signal
        threshold_info (dict, optional): Information about decision thresholds
        additional_factors (dict, optional): Any additional factors affecting the decision
        
    Returns:
        str: A concise summary string
    """
    # Classify confidence level
    if confidence > 0.85:
        confidence_desc = "High confidence"
    elif confidence > 0.65:
        confidence_desc = "Moderate confidence"
    else:
        confidence_desc = "Low confidence"
    
    # Format sentiment appropriately based on type
    if isinstance(sentiment, (int, float)):
        if sentiment > 70:
            sentiment_desc = f"Extreme Greed ({sentiment})"
        elif sentiment > 55:
            sentiment_desc = f"Greed ({sentiment})"
        elif sentiment > 45:
            sentiment_desc = f"Neutral ({sentiment})"
        elif sentiment > 30:
            sentiment_desc = f"Fear ({sentiment})"
        else:
            sentiment_desc = f"Extreme Fear ({sentiment})"
    else:
        sentiment_desc = str(sentiment)
    
    # Create summary parts
    summary_parts = [
        f"{signal} signal",
        confidence_desc,
        f"Regime: {regime}",
        f"Sentiment: {sentiment_desc}"
    ]
    
    # Add veto information if present
    if threshold_info and threshold_info.get("veto", False):
        veto_reason = threshold_info.get("veto_reason", "Unknown reason")
        summary_parts.insert(0, f"VETOED: {veto_reason}")
    
    return " | ".join(summary_parts)

def generate_explanation(
    confidence: float,
    regime: str,
    sentiment: Union[float, str],
    signal: str,
    threshold_info: Optional[Dict[str, Any]] = None,
    additional_factors: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate a detailed explanation of the trade decision as a list of reasoning statements.
    
    Args:
        confidence (float): Model confidence score
        regime (str): Market regime classification
        sentiment (float or str): Market sentiment score or classification
        signal (str): Model signal
        threshold_info (dict, optional): Information about decision thresholds
        additional_factors (dict, optional): Any additional factors affecting the decision
        
    Returns:
        list: A list of explanation statements
    """
    explanation = []
    
    # Signal explanation
    if signal == "BUY":
        explanation.append(f"The LSTM model predicted an upward price movement with {confidence:.2%} confidence.")
    elif signal == "SELL":
        explanation.append(f"The LSTM model predicted a downward price movement with {confidence:.2%} confidence.")
    elif signal == "HOLD":
        explanation.append(f"The model suggested holding with {confidence:.2%} confidence.")
    
    # Regime explanation
    if "TRENDING" in regime:
        if "UP" in regime:
            explanation.append("The market is in a strong upward trend.")
        elif "DOWN" in regime:
            explanation.append("The market is in a strong downward trend.")
        else:
            explanation.append("The market is in a trending regime.")
    elif "CHOPPY" in regime:
        explanation.append("The market is in a choppy/sideways regime with low directional movement.")
    
    # Sentiment explanation
    if isinstance(sentiment, (int, float)):
        if sentiment > 70:
            explanation.append(f"Market sentiment is extremely greedy ({sentiment}/100), suggesting potential overvaluation.")
        elif sentiment > 55:
            explanation.append(f"Market sentiment is greedy ({sentiment}/100), showing bullish bias.")
        elif sentiment > 45:
            explanation.append(f"Market sentiment is neutral ({sentiment}/100).")
        elif sentiment > 30:
            explanation.append(f"Market sentiment shows fear ({sentiment}/100), suggesting potential undervaluation.")
        else:
            explanation.append(f"Market sentiment shows extreme fear ({sentiment}/100), indicating possible overselling.")
    
    # Threshold explanations
    if threshold_info:
        if threshold_info.get("veto", False):
            explanation.append(f"Trade was VETOED: {threshold_info.get('veto_reason', 'Unknown reason')}")
        
        # Confidence threshold
        if "min_confidence" in threshold_info:
            min_conf = threshold_info["min_confidence"]
            if confidence < min_conf:
                explanation.append(f"Confidence ({confidence:.2f}) is below minimum threshold ({min_conf:.2f}).")
            else:
                explanation.append(f"Confidence ({confidence:.2f}) exceeds minimum threshold ({min_conf:.2f}).")
    
    # Additional factors
    if additional_factors:
        for factor, value in additional_factors.items():
            if factor == "cooldown":
                explanation.append(f"Trade cooldown period: {value} seconds based on confidence.")
            elif factor == "position_size":
                explanation.append(f"Position size adjusted to {value} based on confidence and risk parameters.")
            elif factor == "higher_timeframe":
                explanation.append(f"Higher timeframe ({value.get('timeframe', 'unknown')}) shows {value.get('signal', 'unknown')} signal.")
    
    return explanation

def log_reasoning(
    reasoning: Dict[str, Any],
    logger_func: Optional[Callable] = None,
    db_logger: Optional[Any] = None,
    file_path: Optional[str] = None
) -> None:
    """
    Log the trade reasoning to multiple destinations.

    Args:
        reasoning (dict): Reasoning dictionary
        logger_func (callable, optional): If provided, will be called with the reasoning
        db_logger (object, optional): Database logger object with log_data method
        file_path (str, optional): Path to a JSON file for appending reasoning data
    """
    # Convert to JSON string
    reasoning_json = json.dumps(reasoning)
    
    # Log to logger if provided
    if logger_func:
        logger_func(f"Trade reasoning: {reasoning['summary']}")
        logger_func(f"Full reasoning: {reasoning_json}")
    else:
        logger.info(f"Trade reasoning: {reasoning['summary']}")
        logger.debug(f"Full reasoning: {reasoning_json}")
    
    # Log to database if provided
    if db_logger and hasattr(db_logger, 'log_data'):
        try:
            db_logger.log_data('trade_reasoning', reasoning)
        except Exception as e:
            logger.error(f"Failed to log reasoning to database: {e}")
    
    # Append to file if provided
    if file_path:
        try:
            with open(file_path, 'a') as f:
                f.write(reasoning_json + '\n')
        except Exception as e:
            logger.error(f"Failed to write reasoning to file {file_path}: {e}")

def get_veto_info(
    confidence: float,
    min_confidence: float,
    regime: str,
    allowed_regimes: List[str],
    higher_tf_signal: Optional[str] = None,
    current_signal: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate veto information based on decision thresholds.
    
    Args:
        confidence (float): Model confidence score
        min_confidence (float): Minimum required confidence
        regime (str): Current market regime
        allowed_regimes (list): List of allowed market regimes
        higher_tf_signal (str, optional): Signal from higher timeframe
        current_signal (str, optional): Current signal
        
    Returns:
        dict: Veto information including whether trade was vetoed and why
    """
    veto_info = {
        "veto": False,
        "min_confidence": min_confidence,
        "allowed_regimes": allowed_regimes
    }
    
    # Check confidence threshold
    if confidence < min_confidence:
        veto_info["veto"] = True
        veto_info["veto_reason"] = f"Confidence ({confidence:.2f}) below threshold ({min_confidence:.2f})"
        return veto_info
    
    # Check regime
    if regime not in allowed_regimes:
        veto_info["veto"] = True
        veto_info["veto_reason"] = f"Regime '{regime}' not in allowed regimes {allowed_regimes}"
        return veto_info
    
    # Check higher timeframe alignment if provided
    if higher_tf_signal and current_signal and higher_tf_signal != current_signal:
        veto_info["veto"] = True
        veto_info["veto_reason"] = f"Higher timeframe signal '{higher_tf_signal}' conflicts with current '{current_signal}'"
        veto_info["higher_timeframe_signal"] = higher_tf_signal
        return veto_info
    
    return veto_info
