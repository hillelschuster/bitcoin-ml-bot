"""
Trading bot orchestrator.

This module serves as the main entry point for the trading bot,
initializing all components and orchestrating the trading cycle.
"""

import os
import time
import logging
import platform
import threading
import traceback
from datetime import datetime

# File locking support
HAS_FILE_LOCKING = platform.system() != "Windows"
if not HAS_FILE_LOCKING:
    import msvcrt
else:
    import fcntl

# Import core components
from src.utils.config import Config
from src.utils.data import DataHandler
from src.models.model_lstm_core import PredictionModel
from src.core.bot.strategy import TradingStrategy
from src.utils.logger import TradeLogger
from src.utils.notifier import Notifier
from src.core.bot.risk_manager import RiskManager
from src.core.bot.regime_filter import MarketRegimeFilter
from src.core.utils.reason_explainer import generate_trade_reason, log_reasoning, get_veto_info

# Import modular components
from src.core.bot.state_manager import StateManager
from src.core.bot.safety_monitor import SafetyMonitor
from src.core.bot.exit_handler import ExitHandler
from src.core.bot.position_manager import PositionManager

# Import utilities
from src.core.utils.sentiment_utils import fetch_sentiment_score, classify_sentiment
from src.core.utils.strategy_utils import prepare_features, check_veto_conditions, build_feature_bundle

logger = logging.getLogger("core.bot.orchestrator")


class TradingOrchestrator:
    """
    Main orchestrator for the trading bot.

    This class initializes all components and orchestrates the trading cycle.
    """

    def __init__(self, config_path='config/config.json'):
        """
        Initialize the trading orchestrator.

        Args:
            config_path: Path to the configuration file
        """
        # Log file locking support status
        if not HAS_FILE_LOCKING:
            logger.warning("⚠️ File locking not available on this platform. Multiple instances may cause race conditions.")

        # Initialize file lock
        self._trade_lock = threading.RLock()

        # Initialize configuration
        self.config = Config(config_path)

        # Initialize exchange client
        from src.core.exchange.exchange_base import get_client
        self.client = get_client()

        # Initialize data handler
        self.data_handler = DataHandler(self.config, self.client)

        # Initialize model
        self.model = PredictionModel(self.config)

        # Initialize trade logger
        self.trade_logger = TradeLogger()

        # Initialize notifier
        self.notifier = Notifier(self.config)

        # Initialize risk manager
        self.risk_manager = RiskManager(self.client, self.trade_logger, self.config)

        # Initialize state manager
        self.state_manager = StateManager(self.notifier)
        self.state_manager._trade_lock = self._trade_lock  # Share the lock

        # Initialize safety monitor
        self.safety_monitor = SafetyMonitor(self.client, self.config, self.state_manager, self.notifier)

        # Initialize exit handler
        self.exit_handler = ExitHandler(self.client, self.state_manager, self.safety_monitor, self.notifier, self.config)

        # Initialize position manager
        self.position_manager = PositionManager(self.client, self.state_manager, self.exit_handler, self.notifier, self.config, self.risk_manager)

        # Initialize strategy
        self.strategy = TradingStrategy(self.config, self.client, self.trade_logger, self.risk_manager)

        # Initialize regime filter
        self.regime_filter = MarketRegimeFilter()

        # Initialize trading state
        self.last_trade_time = 0
        self.trades_today = 0
        self.position_closing = False

        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        # Load state
        self.state_manager.load_circuit_breaker_state()
        self.state_manager.load_emergency_exit_metrics()
        symbol = self.config.get('symbol', 'BTC/USDT')
        self.state_manager.load_position_state(self.client, symbol)

        # Initialize emergency exit tracking
        self.emergency_exit_attempts = 0

    def run(self):
        """
        Run the trading bot in a continuous loop.
        """
        logger.info("🚀 Starting trading bot...")
        self.notifier.send("🚀 Trading bot started")

        try:
            # Main trading loop
            while True:
                try:
                    # Run a single trading cycle
                    self.run_cycle()

                    # Sleep for the configured cycle interval
                    cycle_interval = self.config.get('cycle_interval', 60)
                    logger.debug(f"Sleeping for {cycle_interval} seconds...")
                    time.sleep(cycle_interval)
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt detected. Exiting...")
                    break
                except (ValueError, KeyError, AttributeError) as e:
                    logger.error(f"Error in trading cycle: {e}")
                    logger.error(traceback.format_exc())
                    self.notifier.send(f"⚠️ Error in trading cycle: {e}")
                    time.sleep(30)  # Sleep for 30 seconds before retrying
        finally:
            # Clean up resources
            self.cleanup()

    def run_cycle(self):
        """
        Execute a single trading cycle.
        """
        # Get the trading symbol
        symbol = self.config.get('symbol', 'BTC/USDT')

        # Validate position state against exchange at the start of each cycle
        if self.state_manager.position:
            try:
                # Validate that our position state matches the exchange
                from src.core.exchange.position_utils import get_position_amount
                position_amount = get_position_amount(self.client, symbol)
                logger.info(f"Position validation: {symbol} position amount = {position_amount}")

                # Check if position state matches exchange
                if (self.state_manager.position == "long" and position_amount <= 0) or \
                   (self.state_manager.position == "short" and position_amount >= 0) or \
                   (abs(position_amount) < 0.0001):
                    logger.warning(f"Position state mismatch: file={self.state_manager.position}, exchange={position_amount}")
                    # Reset position state if exchange shows no position
                    if abs(position_amount) < 0.0001:
                        logger.warning("Exchange shows no position, resetting position state")
                        self.state_manager.reset_position_state()
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Position validation error: {e}")

        # Check if trading is suspended due to circuit breaker
        if self.state_manager.trading_suspended:
            logger.info("🚨 Trading suspended by circuit breaker")
            return

        # Get current market data
        try:
            market_data = self.data_handler.get_market_data()
            if market_data is None or market_data.empty:
                logger.error("Failed to get market data")
                return
        except (IOError, OSError) as e:
            logger.error(f"Error getting market data: {e}")
            return

        # Get current price
        try:
            from src.core.exchange.price_utils import get_market_price
            current_price = get_market_price(self.client, symbol)
            if current_price <= 0:
                logger.error(f"Invalid current price: {current_price}")
                return
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Error getting current price: {e}")
            return

        # Check for drawdown if we have an open position
        if self.state_manager.position:
            # Monitor for excessive drawdown
            if self.safety_monitor.monitor_drawdown(symbol, self.state_manager.position_entry_time):
                logger.warning("Drawdown limit exceeded - initiating emergency exit")
                self.exit_handler.emergency_exit(symbol, "Drawdown limit exceeded")
                return

            # Check if it's time to take profit or exit based on prediction
            prediction = self.model.predict(market_data)
            confidence = prediction.get('confidence', 0)
            signal = prediction.get('signal', 'HOLD')

            # Get current position side
            position_side = self.state_manager.position

            # Check if we should exit based on prediction
            if (position_side == "long" and signal == "DOWN") or (position_side == "short" and signal == "UP"):
                logger.info(f"Signal ({signal}) suggests exiting {position_side} position")
                self.exit_handler.exit_position(symbol, current_price)
                return
        else:
            # No position - check if we should enter one

            # Get sentiment data
            sentiment_score = fetch_sentiment_score()
            sentiment_class = classify_sentiment(sentiment_score)
            logger.info(f"Current sentiment: {sentiment_score} ({sentiment_class})")

            # Get market regime
            regime = self.regime_filter.classify_regime(market_data)
            logger.info(f"Current market regime: {regime}")

            # Get prediction
            prediction = self.model.predict(market_data)

            # Handle different prediction formats (tuple or dict)
            if isinstance(prediction, tuple) and len(prediction) >= 2:
                signal, confidence = prediction
            elif isinstance(prediction, dict):
                confidence = prediction.get('confidence', 0)
                signal = prediction.get('signal', 'HOLD')
            else:
                logger.warning(f"Unexpected prediction format: {type(prediction)}")
                signal = 'HOLD'
                confidence = 0

            logger.info(f"Prediction: {signal} with confidence {confidence:.4f}")

            # Check veto conditions
            veto_info = check_veto_conditions(
                confidence,
                min_confidence=self.config.get('min_confidence', 0.65),
                regime=regime,
                allowed_regimes=self.config.get('allowed_regimes', ["TRENDING_UP", "TRENDING_DOWN"]),
                signal=signal,
                sentiment_score=sentiment_score
            )

            # If trade is vetoed, log reason and return
            if veto_info.get('veto', False):
                logger.info(f"Trade vetoed: {veto_info.get('veto_reason', 'Unknown reason')}")
                return

            # Check cooldown period
            cooldown_time = self.config.get('trade_cooldown_seconds', 300)
            time_since_last_trade = time.time() - self.last_trade_time
            if time_since_last_trade < cooldown_time:
                logger.info(f"Cooldown active: {time_since_last_trade:.1f}s / {cooldown_time}s")
                return

            # Generate trade reason
            features = build_feature_bundle(market_data, sentiment_score, regime, signal, confidence, current_price)
            reason = generate_trade_reason(
                symbol=symbol,
                timeframe=self.config.get('timeframe', '1m'),
                price=current_price,
                confidence=confidence,
                regime=regime,
                sentiment=sentiment_score,
                signal=signal,
                threshold_info=veto_info
            )

            # Log reasoning
            log_reasoning(reason)

            # Execute trade based on signal
            if signal == "UP":
                logger.info("Signal suggests entering LONG position")
                if self.position_manager.enter_long(confidence, current_price, symbol):
                    self.last_trade_time = time.time()
                    self.trades_today += 1
            elif signal == "DOWN":
                logger.info("Signal suggests entering SHORT position")
                if self.position_manager.enter_short(confidence, current_price, symbol):
                    self.last_trade_time = time.time()
                    self.trades_today += 1

    def cleanup(self):
        """
        Clean up resources before exiting.
        """
        logger.info("Cleaning up resources...")

        # Save state
        self.state_manager.save_position_state()
        self.state_manager.save_emergency_exit_metrics()

        # Send notification
        self.notifier.send("🛑 Trading bot stopped")

        logger.info("Cleanup complete. Exiting...")


def run_trading_bot(config_path='config/config.json'):
    """
    Run the trading bot with the specified configuration.

    Args:
        config_path: Path to the configuration file
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/trading_bot.log")
        ]
    )

    # Create and run the trading orchestrator
    orchestrator = TradingOrchestrator(config_path)
    orchestrator.run()


if __name__ == "__main__":
    run_trading_bot()
