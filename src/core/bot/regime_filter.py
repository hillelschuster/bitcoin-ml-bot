import pandas as pd
import logging
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger('core.bot.regime_filter')


class MarketRegimeFilter:
    # Default threshold values
    DEFAULT_THRESHOLDS = {
        'adx_threshold': 20.0,        # Minimum ADX value for trending market
        'bb_width_threshold': 0.005,  # Minimum Bollinger Band width for trending market
        'ma_slope_threshold': 0.05,   # Minimum MA slope magnitude for trending market
        'adx_window': 14,             # ADX calculation window
        'bb_window': 20,              # Bollinger Bands calculation window
        'bb_window_dev': 2,           # Bollinger Bands standard deviation
        'ma_window': 10               # Moving average window for slope calculation
    }

    def __init__(self, price_data: Optional[pd.DataFrame] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 verbose: bool = False):
        """
        Classifies market regime based on ADX, Bollinger Band width, and MA slope.

        Args:
            price_data: DataFrame with ['high', 'low', 'close'] columns. Can be None if provided later.
            thresholds: Dictionary of threshold values to override defaults
            verbose: Whether to enable detailed debug logging
        """
        # Set up thresholds with defaults and any overrides
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)

        # Set verbose flag
        self.verbose = verbose
        if verbose:
            logger.info(f"Initialized MarketRegimeFilter with thresholds: {self.thresholds}")

        # Initialize dataframe if provided
        self.df = None
        if price_data is not None:
            self.update_data(price_data)

    def update_data(self, price_data: pd.DataFrame) -> None:
        """
        Update the price data and recalculate indicators.

        Args:
            price_data: DataFrame with ['high', 'low', 'close'] columns
        """
        if not all(col in price_data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Price data must include 'high', 'low', and 'close' columns")

        self.df = price_data.copy()
        self._calculate_indicators()

        if self.verbose:
            logger.debug(f"Updated price data with {len(self.df)} rows")

    def _calculate_indicators(self) -> None:
        """
        Calculate technical indicators used for regime classification.
        Handles NaN values to ensure safe access to the latest values.
        """
        # Extract threshold parameters
        adx_window = self.thresholds['adx_window']
        bb_window = self.thresholds['bb_window']
        bb_window_dev = self.thresholds['bb_window_dev']
        ma_window = self.thresholds['ma_window']

        # Calculate ADX
        self.df['adx'] = ADXIndicator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=adx_window
        ).adx()

        # Calculate Bollinger Bands width
        bb = BollingerBands(
            close=self.df['close'],
            window=bb_window,
            window_dev=bb_window_dev
        )
        self.df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()

        # Calculate MA slope
        self.df['ma_slope'] = self.df['close'].rolling(window=ma_window).mean().diff()

        # Fill NaN values with appropriate defaults to prevent issues
        # For ADX and BB width, use 0 (indicating no trend/volatility)
        # For MA slope, use 0 (indicating flat price action)
        self.df['adx'] = self.df['adx'].fillna(0)
        self.df['bb_width'] = self.df['bb_width'].fillna(0)
        self.df['ma_slope'] = self.df['ma_slope'].fillna(0)

        if self.verbose:
            # Log the last few rows of indicators for debugging
            last_rows = min(5, len(self.df))
            logger.debug(f"Last {last_rows} rows of indicators:\n{self.df[['adx', 'bb_width', 'ma_slope']].tail(last_rows)}")

    def is_trending(self, price_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Determine if the market is currently in a trending regime.

        Args:
            price_data: Optional DataFrame with price data. If provided, updates internal data first.
                        This parameter is for backward compatibility.

        Returns:
            bool: True if market is trending, False if choppy
        """
        # Update data if provided (for backward compatibility)
        if price_data is not None:
            self.update_data(price_data)

        if self.df is None or len(self.df) == 0:
            logger.warning("No price data available for regime detection")
            return False

        # Safely get the latest values, ensuring we have valid data
        try:
            # Get the latest row with valid indicator values
            latest = self.df.iloc[-1]

            # Extract current indicator values
            current_adx = latest['adx']
            current_bb_width = latest['bb_width']
            current_ma_slope = latest['ma_slope']

            # Extract threshold values
            adx_threshold = self.thresholds['adx_threshold']
            bb_width_threshold = self.thresholds['bb_width_threshold']
            ma_slope_threshold = self.thresholds['ma_slope_threshold']

            # Determine if trending based on thresholds
            is_trending = (
                current_adx > adx_threshold
                and current_bb_width > bb_width_threshold
                and abs(current_ma_slope) > ma_slope_threshold
            )

            # Log detailed information if verbose mode is enabled
            if self.verbose:
                logger.debug(f"Regime detection: ADX={current_adx:.2f} (threshold={adx_threshold})")
                logger.debug(f"Regime detection: BB Width={current_bb_width:.6f} (threshold={bb_width_threshold})")
                logger.debug(f"Regime detection: MA Slope={current_ma_slope:.6f} (threshold={ma_slope_threshold})")
                logger.debug(f"Market regime: {'TRENDING' if is_trending else 'CHOPPY'}")

            return is_trending

        except (IndexError, KeyError) as e:
            logger.error(f"Error accessing indicator data: {e}")
            return False

    def get_regime(self) -> str:
        """
        Get the current market regime as a string.

        Returns:
            str: 'TRENDING' or 'CHOPPY'
        """
        return "TRENDING" if self.is_trending() else "CHOPPY"

    def get_indicator_values(self) -> Dict[str, Any]:
        """
        Get the current values of all indicators used for regime detection.

        Returns:
            dict: Dictionary containing current indicator values
        """
        if self.df is None or len(self.df) == 0:
            return {
                'adx': 0.0,
                'bb_width': 0.0,
                'ma_slope': 0.0,
                'regime': 'UNKNOWN'
            }

        try:
            latest = self.df.iloc[-1]
            return {
                'adx': latest['adx'],
                'bb_width': latest['bb_width'],
                'ma_slope': latest['ma_slope'],
                'regime': self.get_regime()
            }
        except (IndexError, KeyError):
            return {
                'adx': 0.0,
                'bb_width': 0.0,
                'ma_slope': 0.0,
                'regime': 'UNKNOWN'
            }

    def classify_regime(self, price_data: Optional[pd.DataFrame] = None) -> str:
        """
        Classify the market regime based on the provided price data.

        Args:
            price_data: Optional DataFrame with price data. If provided, updates internal data first.

        Returns:
            str: 'TRENDING' or 'CHOPPY'
        """
        # Update data if provided
        if price_data is not None:
            self.update_data(price_data)

        # Return the regime classification
        return self.get_regime()
