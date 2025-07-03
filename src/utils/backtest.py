#!/usr/bin/env python
"""
Backtest module for the Bitcoin ML Trading Bot.

This module provides functionality for backtesting trading strategies
on historical data, evaluating performance, and generating reports.

Usage:
    python -m src.utils.backtest --symbol BTCUSDT --timeframe 1m
"""

import os
import argparse
import logging
import pandas as pd
from datetime import datetime  # Used for timestamp handling in the backtest engine
from typing import Dict, List, Optional, Union, Any

from src.models.model_lstm_core import load_model, predict_signal
from src.core.bot.strategy import should_enter_trade, should_exit_trade
from src.core.bot.risk_manager import calculate_position_size_standalone, get_stop_loss
from src.utils.config import Config
from src.utils.data import get_latest_data, DataHandler
from src.utils.trade_stats import log_trade_stats, generate_trade_stats_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest')


class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.

    This class provides functionality for simulating trading on historical data,
    evaluating performance, and generating reports.
    """

    def __init__(self, config: Optional[Config] = None, strategy=None, data_loader=None, verbose: bool = True):
        """
        Initialize the backtest engine.

        Args:
            config: Configuration object (optional)
            strategy: Trading strategy to use (optional)
            data_loader: Data loader to use (optional)
            verbose: Whether to log detailed information (default: True)
        """
        self.config = config if config is not None else Config()
        self.strategy = strategy
        self.data_loader = data_loader
        self.verbose = verbose

        # Get trading configuration
        trading_config = self.config.get('trading', {})
        self.symbol = trading_config.get('symbol', 'BTCUSDT')
        self.timeframe = trading_config.get('timeframe', '1h')
        self.initial_balance = trading_config.get('initial_balance', 10000.0)
        self.risk_per_trade = trading_config.get('risk_per_trade', 0.02)
        self.debug_mode = trading_config.get('debug_mode', False)

        # Initialize model
        self.model, self.scaler = load_model(model_path="model_artifacts/lstm_model.h5", scaler_path="model_artifacts/lstm_scaler.pkl")

        # Initialize results storage
        self.trades = []
        self.positions = []
        self.balance_history = []
        self.current_balance = self.initial_balance
        self.position = None
        self.entry_time = None

        # Set up logging based on verbosity
        self._setup_logging()

        if self.verbose:
            logger.info(f"Initialized backtest engine for {self.symbol} ({self.timeframe})")

    def _setup_logging(self):
        """
        Configure logging based on verbosity settings.
        """
        if not self.verbose:
            # Reduce logging output when not in verbose mode
            logging.getLogger('backtest').setLevel(logging.WARNING)
        elif self.debug_mode:
            # Increase logging detail in debug mode
            logging.getLogger('backtest').setLevel(logging.DEBUG)
        else:
            # Default logging level
            logging.getLogger('backtest').setLevel(logging.INFO)

    def load_data(self, symbol: Optional[str] = None, timeframe: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Load historical data for backtesting.

        Args:
            symbol: Trading pair symbol (optional, uses config value if None)
            timeframe: Candle timeframe (optional, uses config value if None)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol or self.symbol
        timeframe = timeframe or self.timeframe

        logger.info(f"Loading {limit} candles of {timeframe} data for {symbol}")

        if self.data_loader is not None:
            # Use provided data loader
            df = self.data_loader.collect_data(limit=limit)
            df = self.data_loader.process_data(df)
        else:
            # Use default data loading method
            df, _ = get_latest_data(symbol=symbol, timeframe=timeframe, limit=limit, use_mock_data=True)

            # Process data with DataHandler
            data_handler = DataHandler(config=self.config)
            df = data_handler.process_data(df)

        logger.info(f"Loaded {len(df)} candles for backtesting")
        return df

    def _process_candle(self, row_df: pd.DataFrame) -> None:
        """
        Process a single candle in the backtest.

        Args:
            row_df: DataFrame with data up to the current candle
        """
        current_time = row_df.index[-1]
        current_price = row_df["close"].iloc[-1]

        # Generate signal
        signal = predict_signal(row_df, self.model, self.scaler)

        # Check for entry signal
        if not self.position and should_enter_trade(signal, row_df):
            self._enter_trade(row_df, current_time, current_price)

        # Check for exit signal or stop loss
        elif self.position:
            self._check_exit_conditions(row_df, signal, current_time, current_price)

    def _enter_trade(self, row_df: pd.DataFrame, current_time: datetime, current_price: float) -> None:
        """
        Enter a new trade position.

        Args:
            row_df: DataFrame with data up to the current candle
            current_time: Current candle timestamp
            current_price: Current closing price
        """
        # Calculate position size based on current balance
        position_size = calculate_position_size_standalone(
            row_df,
            balance=self.current_balance,
            confidence=0.75  # Default confidence for backtest
        )

        # Calculate stop loss
        stop_loss = get_stop_loss(row_df)

        # Record entry
        entry_price = current_price
        self.entry_time = current_time
        self.position = {
            "entry_time": self.entry_time,
            "entry_price": entry_price,
            "size": position_size,
            "stop_loss": stop_loss,
            "position_side": "long"  # Default to long for now
        }

        # Log entry
        if self.verbose:
            logger.info(f"[{current_time}] ENTRY at ${entry_price:.2f}, size: {position_size:.6f}, stop: ${stop_loss:.2f}")

        # Log trade to CSV
        trade_data = {
            "type": "entry",
            "symbol": self.symbol,
            "price": entry_price,
            "amount": position_size,
            "position_side": "long",
            "confidence": 0.75,
            "model_used": "backtest",
            "entry_time": self.entry_time
        }
        log_trade_stats(trade_data)

        # Store position
        self.positions.append(self.position)

    def _check_exit_conditions(self, row_df: pd.DataFrame, signal: int, current_time: datetime, current_price: float) -> None:
        """
        Check if exit conditions are met and exit the trade if necessary.

        Args:
            row_df: DataFrame with data up to the current candle
            signal: Trading signal from the model
            current_time: Current candle timestamp
            current_price: Current closing price
        """
        exit_reason = None

        # Check for stop loss hit
        if current_price <= self.position["stop_loss"]:
            exit_reason = "stop_loss"
        # Check for exit signal
        elif should_exit_trade(signal, row_df):
            exit_reason = "signal"

        if exit_reason:
            self._exit_trade(current_time, current_price, exit_reason)

    def _exit_trade(self, current_time: datetime, current_price: float, exit_reason: str) -> None:
        """
        Exit the current trade position.

        Args:
            current_time: Current candle timestamp
            current_price: Current closing price
            exit_reason: Reason for exiting the trade
        """
        # Calculate PnL
        exit_price = current_price
        pnl = (exit_price - self.position["entry_price"]) * self.position["size"]

        # Update balance
        self.current_balance += pnl
        self.balance_history.append(self.current_balance)

        # Calculate time held
        exit_time = current_time
        time_held = (exit_time - self.entry_time).total_seconds() if self.entry_time else 0

        # Record trade
        trade = {
            "entry_time": self.position["entry_time"],
            "exit_time": exit_time,
            "entry_price": self.position["entry_price"],
            "exit_price": exit_price,
            "size": self.position["size"],
            "pnl": pnl,
            "reason": exit_reason,
            "time_held": time_held
        }
        self.trades.append(trade)

        # Log exit
        if self.verbose:
            logger.info(f"[{current_time}] EXIT at ${exit_price:.2f}, PnL: ${pnl:.2f}, reason: {exit_reason}")

        # Log trade to CSV
        trade_data = {
            "type": "close",
            "symbol": self.symbol,
            "price": exit_price,
            "amount": self.position["size"],
            "entry_price": self.position["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "confidence": 0.75,
            "model_used": "backtest",
            "reason": exit_reason,
            "position_side": "long",
            "entry_time": self.position["entry_time"],
            "exit_time": exit_time,
            "time_held": time_held
        }
        log_trade_stats(trade_data)

        # Reset position
        self.position = None
        self.entry_time = None

    def run(self, data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Run the backtest simulation.

        Args:
            data: Historical market data (optional, will be loaded if None)

        Returns:
            List of trade results
        """
        # Load data if not provided
        df = data if data is not None else self.load_data()

        if df is None or df.empty:
            logger.error("No data available for backtesting")
            return []

        # Reset results
        self.trades = []
        self.positions = []
        self.balance_history = [self.initial_balance]
        self.current_balance = self.initial_balance
        self.position = None
        self.entry_time = None

        # Ensure we have enough data for indicators
        min_candles = 20
        if len(df) <= min_candles:
            logger.error(f"Insufficient data for backtesting: {len(df)} candles (need at least {min_candles})")
            return []

        if self.verbose:
            logger.info(f"Starting backtest with initial balance: ${self.initial_balance:.2f}")

        # Iterate through each candle
        for i in range(min_candles, len(df)):
            # Get data up to current candle
            row_df = df.iloc[:i+1]

            # Process this candle
            self._process_candle(row_df)

            # Update balance history even if no trade
            if len(self.balance_history) <= i - min_candles:
                self.balance_history.append(self.current_balance)

        # Close any open position at the end of the backtest
        if self.position:
            self._close_final_position(df)

        # Log final results
        self._log_backtest_summary()

        return self.trades

    def _close_final_position(self, df: pd.DataFrame) -> None:
        """
        Close any open position at the end of the backtest.

        Args:
            df: Full DataFrame with historical data
        """
        exit_price = df["close"].iloc[-1]
        exit_time = df.index[-1]
        pnl = (exit_price - self.position["entry_price"]) * self.position["size"]
        time_held = (exit_time - self.position["entry_time"]).total_seconds() if self.position["entry_time"] else 0

        # Update balance
        self.current_balance += pnl

        # Record trade
        trade = {
            "entry_time": self.position["entry_time"],
            "exit_time": exit_time,
            "entry_price": self.position["entry_price"],
            "exit_price": exit_price,
            "size": self.position["size"],
            "pnl": pnl,
            "reason": "end_of_backtest",
            "time_held": time_held
        }
        self.trades.append(trade)

        # Log exit
        if self.verbose:
            logger.info(f"[{exit_time}] FINAL EXIT at ${exit_price:.2f}, PnL: ${pnl:.2f}, reason: end_of_backtest")

        # Log trade to CSV
        trade_data = {
            "type": "close",
            "symbol": self.symbol,
            "price": exit_price,
            "amount": self.position["size"],
            "entry_price": self.position["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "confidence": 0.75,
            "model_used": "backtest",
            "reason": "end_of_backtest",
            "position_side": "long",
            "entry_time": self.position["entry_time"],
            "exit_time": exit_time,
            "time_held": time_held
        }
        log_trade_stats(trade_data)

    def _log_backtest_summary(self) -> None:
        """
        Log a summary of backtest results.
        """
        if not self.verbose:
            return

        total_pnl = sum(trade["pnl"] for trade in self.trades)
        final_balance = self.initial_balance + total_pnl
        logger.info(f"Backtest completed with {len(self.trades)} trades")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}, Final balance: ${final_balance:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f} ({(total_pnl / self.initial_balance) * 100:.2f}%)")

    def evaluate(self) -> Dict[str, Union[int, float, str]]:
        """
        Evaluate backtest results and generate performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            logger.warning("No trades to evaluate")
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0
            }

        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade["pnl"] > 0)
        losing_trades = sum(1 for trade in self.trades if trade["pnl"] < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = sum(trade["pnl"] for trade in self.trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Calculate average time held
        time_held_values = [trade["time_held"] for trade in self.trades if trade["time_held"] is not None]
        avg_time_held = sum(time_held_values) / len(time_held_values) if time_held_values else 0

        # Calculate max drawdown
        max_drawdown = 0.0
        peak_balance = self.initial_balance
        for balance in self.balance_history:
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate profit factor
        gross_profit = sum(trade["pnl"] for trade in self.trades if trade["pnl"] > 0)
        gross_loss = abs(sum(trade["pnl"] for trade in self.trades if trade["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Return comprehensive metrics
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_time_held": avg_time_held,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "initial_balance": self.initial_balance,
            "final_balance": self.initial_balance + total_pnl
        }

    def generate_report(self) -> str:
        """
        Generate a comprehensive backtest report.

        Returns:
            Formatted report as a string
        """
        # Get trade statistics report
        stats_report = generate_trade_stats_report()

        # Get additional metrics from evaluate()
        metrics = self.evaluate()

        # Create custom report with additional metrics
        report = "📈 BACKTEST RESULTS 📈\n\n"
        report += f"Symbol: {self.symbol}\n"
        report += f"Timeframe: {self.timeframe}\n"
        report += f"Initial Balance: ${metrics['initial_balance']:.2f}\n"
        report += f"Final Balance: ${metrics['final_balance']:.2f}\n"
        report += f"Total Return: {((metrics['final_balance'] / metrics['initial_balance']) - 1) * 100:.2f}%\n"
        report += f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%\n"
        report += f"Profit Factor: {metrics['profit_factor']:.2f}\n\n"

        # Append trade statistics
        report += stats_report

        return report


# Note: The legacy backtest function has been removed as it's no longer used.
# If you need the original functionality, use the BacktestEngine class directly:
#
# engine = BacktestEngine()
# engine.run(df)
# pnl_series = pd.Series([trade["pnl"] for trade in engine.trades])


def main():
    """
    Main entry point for CLI usage.
    """
    parser = argparse.ArgumentParser(description="Backtest trading strategy on historical data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candles to fetch")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance for backtest")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode with minimal output")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with extra output")

    args = parser.parse_args()

    # Create config with CLI arguments
    config = Config(args.config)

    # Override config with CLI arguments
    trading_config = config.get('trading', {})
    trading_config['symbol'] = args.symbol
    trading_config['timeframe'] = args.timeframe
    trading_config['initial_balance'] = args.balance
    trading_config['debug_mode'] = args.debug
    config.config['trading'] = trading_config

    # Create and run backtest engine
    engine = BacktestEngine(config=config, verbose=not args.quiet)
    engine.run()

    # Print report
    report = engine.generate_report()
    print("\n" + report)


if __name__ == "__main__":
    main()
