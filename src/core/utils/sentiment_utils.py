"""
Sentiment analysis utilities for the trading bot.

This module provides functions for fetching and analyzing market sentiment data
from the Fear & Greed Index.
"""

import logging
import requests
import time
from datetime import datetime, timedelta

logger = logging.getLogger("core.utils.sentiment_utils")

# Cache for sentiment data to avoid excessive API calls
_sentiment_cache = {
    'score': None,  # No default sentiment - will force fetch on first call
    'classification': None,  # No default classification
    'last_updated': datetime.now() - timedelta(hours=2),  # Force initial update
    'update_interval': timedelta(seconds=5),  # Ultra-fast refresh for scalping (5 seconds)
    'last_successful_score': None,  # Last successfully fetched score (for fallback)
    'last_successful_classification': None  # Last successfully fetched classification (for fallback)
}


def fetch_sentiment_score(force_refresh=False):
    """
    Fetch the current market sentiment score from the Fear & Greed Index.

    Uses caching to avoid excessive API calls and implements retry logic for reliability.
    Optimized for ultra-fast refresh rates (1-5 seconds) for scalping strategies.

    Args:
        force_refresh (bool): If True, bypass cache and force a fresh API call

    Returns:
        int: Sentiment score (0-100), higher values indicate more positive sentiment

    Note:
        If API calls fail, returns the last known good value with a warning log
    """
    # Check cache first to avoid unnecessary API calls (unless force_refresh is True)
    current_time = datetime.now()
    cache_age = current_time - _sentiment_cache['last_updated']

    # Use cache if it's fresh enough and not forcing refresh
    if not force_refresh and cache_age < _sentiment_cache['update_interval']:
        if _sentiment_cache['score'] is not None:
            logger.debug(f"Using cached sentiment score: {_sentiment_cache['score']} ({_sentiment_cache['classification']}), age: {cache_age.total_seconds():.1f}s, refresh in: {(_sentiment_cache['update_interval'] - cache_age).total_seconds():.1f}s")
            return _sentiment_cache['score']

    # Implement retry logic with backoff
    max_retries = 2  # Reduced retries for faster response
    backoff_times = [0.5, 1]  # Shorter backoff times (seconds)

    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching Fear & Greed Index (attempt {attempt+1}/{max_retries})")
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=3)  # Reduced timeout for faster response

            if response.status_code == 200:
                data = response.json().get('data', [{}])[0]
                score = int(data.get('value', 0))  # Convert to int
                classification = data.get('value_classification', 'Unknown')

                # Update cache and last successful values
                _sentiment_cache.update({
                    'score': score,
                    'classification': classification,
                    'last_updated': datetime.now(),
                    'last_successful_score': score,
                    'last_successful_classification': classification
                })

                # Log the sentiment with classification
                logger.info(f"Current sentiment: {score} ({classification})")
                return score
            else:
                error_msg = f"API returned status code {response.status_code}"
                logger.warning(f"Fear & Greed API error: {error_msg} (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            logger.warning(f"Error fetching Fear & Greed Index (attempt {attempt+1}/{max_retries}): {str(e)}")

        # Don't sleep on the last attempt
        if attempt < max_retries - 1:
            sleep_time = backoff_times[attempt]
            logger.debug(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    # If we get here, all attempts failed - use last known good value if available
    if _sentiment_cache['last_successful_score'] is not None:
        logger.warning(f"[SENTIMENT] API fetch failed, using cached score ({_sentiment_cache['last_successful_score']})")
        return _sentiment_cache['last_successful_score']

    # If no last known good value, use a neutral default
    logger.error(f"Failed to fetch Fear & Greed Index after {max_retries} attempts and no previous value available")
    return 50  # Neutral sentiment as fallback


def classify_sentiment(score=None):
    """
    Classify a sentiment score into a descriptive category.
    
    Args:
        score (int, optional): Sentiment score (0-100). If None, fetches current score.
        
    Returns:
        str: Sentiment classification (e.g., 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed')
    """
    # If no score provided, use the cached classification or fetch a new one
    if score is None:
        # Use cached classification if available
        if _sentiment_cache['classification'] is not None:
            return _sentiment_cache['classification']
        
        # Otherwise, fetch the current score
        score = fetch_sentiment_score()
    
    # Classify the score
    if score <= 20:
        return "Extreme Fear"
    elif score <= 40:
        return "Fear"
    elif score <= 60:
        return "Neutral"
    elif score <= 80:
        return "Greed"
    else:
        return "Extreme Greed"


def get_sentiment_adjustment_factor(base_factor=1.0, force_refresh=False):
    """
    Calculate a position sizing adjustment factor based on sentiment.
    
    Args:
        base_factor (float): Base adjustment factor (default: 1.0)
        force_refresh (bool): If True, bypass cache and force a fresh API call
        
    Returns:
        float: Adjustment factor (0.5-1.5) where higher values allow larger positions
    """
    score = fetch_sentiment_score(force_refresh=force_refresh)
    classification = classify_sentiment(score)
    
    # Calculate adjustment factor: maps 0-100 score to 0.5-1.5 range
    adjustment = base_factor * (0.5 + score / 100)
    
    logger.info(f"Sentiment adjustment: {score} ({classification}) -> factor {adjustment:.2f}")
    return adjustment


def should_trade_based_on_sentiment(threshold=50, force_refresh=False):
    """
    Determine if trading should proceed based on current market sentiment.
    
    Args:
        threshold (int): Minimum sentiment score required to trade (default: 50)
        force_refresh (bool): If True, bypass cache and force a fresh API call
        
    Returns:
        bool: True if sentiment is favorable for trading, False otherwise
    """
    score = fetch_sentiment_score(force_refresh=force_refresh)
    classification = classify_sentiment(score)
    
    result = score >= threshold
    logger.info(f"Sentiment check: {score} ({classification}) vs threshold {threshold} -> {'PASS' if result else 'FAIL'}")
    
    return result
