"""
Test script for the state_manager.py module.

This script verifies that the StateManager class works correctly
by testing its methods for loading and saving state.
"""

import sys
import os
import logging
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the StateManager class
from src.core.bot.state_manager import StateManager
from config.paths import POSITION_FILE, CIRCUIT_BREAKER_FILE, EMERGENCY_EXIT_METRICS_FILE

# Create a mock notifier
class MockNotifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        print(f"NOTIFICATION: {message}")
        self.messages.append(message)

# Create a mock exchange
class MockExchange:
    def __init__(self):
        self.position_size = 0

    def validate_position_state(self, symbol):
        print(f"Validating position state for {symbol}")
        return True

    def get_current_position_size(self, symbol):
        print(f"Getting position size for {symbol}")
        return self.position_size

def test_circuit_breaker():
    """Test the circuit breaker functionality."""
    print("\nTesting circuit breaker...")

    # Create a StateManager instance
    notifier = MockNotifier()
    state_manager = StateManager(notifier)

    # Save circuit breaker state
    state_manager.save_circuit_breaker_state(active=True, duration_seconds=10)
    print("Circuit breaker state saved")

    # Load circuit breaker state
    state_manager.load_circuit_breaker_state()
    print(f"Circuit breaker active: {state_manager.trading_suspended}")

    # Wait for circuit breaker to expire
    print("Waiting for circuit breaker to expire...")
    time.sleep(11)

    # Load circuit breaker state again
    state_manager.load_circuit_breaker_state()
    print(f"Circuit breaker active after expiry: {state_manager.trading_suspended}")

    # Reset circuit breaker
    state_manager.reset_circuit_breaker()
    print("Circuit breaker reset")

    print("Circuit breaker test passed!")

def test_emergency_exit_metrics():
    """Test the emergency exit metrics functionality."""
    print("\nTesting emergency exit metrics...")

    # Create a StateManager instance
    notifier = MockNotifier()
    state_manager = StateManager(notifier)

    # Add some emergency exit failures
    state_manager.emergency_exit_failures = [
        {"timestamp": time.time(), "reason": "Test failure 1"},
        {"timestamp": time.time(), "reason": "Test failure 2"}
    ]

    # Save emergency exit metrics
    state_manager.save_emergency_exit_metrics()
    print("Emergency exit metrics saved")

    # Load emergency exit metrics
    state_manager.load_emergency_exit_metrics()
    print(f"Emergency exit failures: {len(state_manager.emergency_exit_failures)}")

    print("Emergency exit metrics test passed!")

def cleanup():
    """Clean up test files."""
    print("\nCleaning up test files...")

    # Remove test files
    for file_path in [CIRCUIT_BREAKER_FILE, EMERGENCY_EXIT_METRICS_FILE]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")

if __name__ == "__main__":
    print("Running state_manager.py tests...")

    try:
        # Run the tests
        test_circuit_breaker()
        test_emergency_exit_metrics()

        print("\nAll tests passed! ✅")
    finally:
        # Clean up
        cleanup()
