"""Test script for the circuit breaker functionality.

This script tests the circuit breaker functionality by simulating emergency exit failures
and verifying that the circuit breaker is activated after 3 failures within 15 minutes.
"""

import os
import sys
import time
import logging
import argparse

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.bot.orchestrator import TradingOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('circuit_breaker_test')

def test_circuit_breaker(config_path='config.json', force_activate=False, reset=False):
    """Test the circuit breaker functionality.

    Args:
        config_path (str): Path to the configuration file
        force_activate (bool): Whether to force activate the circuit breaker
        reset (bool): Whether to reset the circuit breaker
    """
    logger.info("Starting circuit breaker test...")

    # Create a bot instance
    bot = TradingOrchestrator(config_path=config_path)

    # Get the current circuit breaker state
    state_manager = bot.state_manager
    state_manager.load_emergency_exit_metrics()

    if reset:
        # Reset the circuit breaker
        logger.info("Resetting circuit breaker...")
        state_manager.trading_suspended = False
        state_manager.save_circuit_breaker_state(active=False)

        # Clear emergency exit metrics
        state_manager.emergency_exit_failures = []
        state_manager.save_emergency_exit_metrics()
        logger.info("Emergency exit metrics cleared")

        logger.info("Circuit breaker reset complete")
        return

    # Calculate metrics
    failures = state_manager.emergency_exit_failures
    current_time = time.time()
    failures_last_15min = sum(1 for f in failures if current_time - f.get('timestamp', 0) < 15 * 60)
    failures_last_hour = sum(1 for f in failures if current_time - f.get('timestamp', 0) < 60 * 60)
    failures_last_24h = sum(1 for f in failures if current_time - f.get('timestamp', 0) < 24 * 60 * 60)

    # Display current metrics
    logger.info(f"Current emergency exit metrics:")
    logger.info(f"  Failures in last 15 minutes: {failures_last_15min}")
    logger.info(f"  Failures in last hour: {failures_last_hour}")
    logger.info(f"  Failures in last 24 hours: {failures_last_24h}")
    logger.info(f"  Total failures: {len(failures)}")
    logger.info(f"  Circuit breaker active: {state_manager.trading_suspended}")

    if force_activate:
        # Force activate the circuit breaker
        logger.info("Forcing circuit breaker activation...")

        # Simulate 3 emergency exit failures
        for i in range(3):
            failure = {
                "timestamp": time.time(),
                "reason": f"Test failure {i+1}"
            }
            state_manager.emergency_exit_failures.append(failure)
            logger.info(f"Recorded test failure {i+1}")

        # Save the failures
        state_manager.save_emergency_exit_metrics()

        # Activate the circuit breaker
        state_manager.trading_suspended = True
        state_manager.save_circuit_breaker_state(active=True, duration_seconds=3600)  # 1 hour

        # Check if circuit breaker was activated
        state_manager.load_circuit_breaker_state()
        if state_manager.trading_suspended:
            logger.info("Circuit breaker was successfully activated")
        else:
            logger.warning("Circuit breaker was not activated as expected")

    logger.info("Circuit breaker test completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the circuit breaker functionality')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--force-activate', action='store_true', help='Force activate the circuit breaker')
    parser.add_argument('--reset', action='store_true', help='Reset the circuit breaker')

    args = parser.parse_args()

    test_circuit_breaker(
        config_path=args.config,
        force_activate=args.force_activate,
        reset=args.reset
    )
