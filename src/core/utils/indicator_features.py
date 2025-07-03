"""
Technical indicator features for the Bitcoin ML Trading Bot.

This module provides functions for calculating technical indicators
that can be used as features for the machine learning models.
"""

import pandas as pd
import numpy as np


def compute_atr(df, period=14):
    """
    Calculate Average True Range (ATR) for a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        period (int): ATR period
        
    Returns:
        pandas.DataFrame: DataFrame with ATR added
    """
    df = df.copy()
    
    # Calculate True Range
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate ATR
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # Drop temporary columns
    df = df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1)
    
    return df


def compute_rsi(df, period=14):
    """
    Calculate Relative Strength Index (RSI) for a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        period (int): RSI period
        
    Returns:
        pandas.DataFrame: DataFrame with RSI added
    """
    df = df.copy()
    
    # Calculate price changes
    delta = df['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df


def compute_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) for a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period
        
    Returns:
        pandas.DataFrame: DataFrame with MACD indicators added
    """
    df = df.copy()
    
    # Calculate MACD line
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    
    # Calculate signal line
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df


def compute_bollinger_width(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands width for a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        period (int): Bollinger Bands period
        std_dev (int): Number of standard deviations
        
    Returns:
        pandas.DataFrame: DataFrame with Bollinger Bands width added
    """
    df = df.copy()
    
    # Calculate middle band (SMA)
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = df['close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
    
    # Calculate bandwidth
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    return df
