"""
Test script for the reason_explainer module.

This script tests the functionality of the reason_explainer module
by generating example trade reasons and logging them.
"""

import unittest
from unittest.mock import patch
import sys
import os
import logging
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the reason_explainer module
from src.core.utils.reason_explainer import (
    generate_trade_reason,
    log_reasoning,
    get_veto_info
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_reason_explainer')

class TestReasonExplainer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for testing
        self.test_file = "test_reasoning.jsonl"

        # Remove the test file if it exists
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        # Create a test log file path
        self.test_log_file = "logs/test_reasoning.jsonl"

        # Remove the test log file if it exists
        if os.path.exists(self.test_log_file):
            try:
                os.remove(self.test_log_file)
                logger.info(f"Removed existing test log file: {self.test_log_file}")
            except Exception as e:
                logger.warning(f"Could not remove existing test log file: {e}")

    def tearDown(self):
        """Clean up after tests."""
        # Remove the test file
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

        # Remove the test log file
        if os.path.exists(self.test_log_file):
            try:
                os.remove(self.test_log_file)
                logger.info(f"Cleaned up test log file: {self.test_log_file}")
            except Exception as e:
                logger.warning(f"Could not clean up test log file: {e}")

    def test_generate_reason_from_confidence_high(self):
        """Test generating trade reason with high confidence."""
        reason = generate_trade_reason(
            symbol="BTCUSDT",
            timeframe="1m",
            price=50000.0,
            confidence=0.92,
            regime="TRENDING_UP",
            sentiment=65,
            signal="BUY"
        )

        # Verify the structure
        self.assertIn("timestamp", reason)
        self.assertIn("market_context", reason)
        self.assertIn("decision_factors", reason)
        self.assertIn("summary", reason)
        self.assertIn("explanation", reason)

        # Verify high confidence is reflected
        self.assertIn("High confidence", reason["summary"])

    def test_generate_reason_from_confidence_medium(self):
        """Test generating trade reason with medium confidence."""
        reason = generate_trade_reason(
            symbol="BTCUSDT",
            timeframe="1m",
            price=50000.0,
            confidence=0.75,
            regime="TRENDING_UP",
            sentiment=65,
            signal="BUY"
        )

        # Verify medium confidence is reflected
        self.assertIn("Moderate confidence", reason["summary"])

    def test_generate_reason_from_confidence_low(self):
        """Test generating trade reason with low confidence."""
        reason = generate_trade_reason(
            symbol="BTCUSDT",
            timeframe="1m",
            price=50000.0,
            confidence=0.55,
            regime="TRENDING_UP",
            sentiment=65,
            signal="BUY"
        )

        # Verify low confidence is reflected
        self.assertIn("Low confidence", reason["summary"])

    def test_veto_reasoning(self):
        """Test reasoning with veto conditions."""
        # Get veto info for a trade that should be vetoed due to low confidence
        veto_info = get_veto_info(
            confidence=0.55,
            min_confidence=0.65,
            regime="TRENDING_UP",
            allowed_regimes=["TRENDING_UP", "TRENDING_DOWN"]
        )

        # Generate reasoning with veto info
        reason = generate_trade_reason(
            symbol="BTCUSDT",
            timeframe="1m",
            price=50000.0,
            confidence=0.55,
            regime="TRENDING_UP",
            sentiment=65,
            signal="BUY",
            threshold_info=veto_info
        )

        # Verify the veto
        self.assertTrue(veto_info["veto"])
        self.assertIn("veto_reason", veto_info)
        self.assertIn("VETOED", reason["summary"])

    def test_higher_timeframe_veto(self):
        """Test reasoning with higher timeframe veto."""
        # Get veto info for a trade that should be vetoed due to higher timeframe conflict
        veto_info = get_veto_info(
            confidence=0.75,
            min_confidence=0.65,
            regime="TRENDING_UP",
            allowed_regimes=["TRENDING_UP", "TRENDING_DOWN"],
            higher_tf_signal="SELL",
            current_signal="BUY"
        )

        # Additional factors with higher timeframe information
        additional_factors = {
            "higher_timeframe": {
                "timeframe": "1h",
                "signal": "SELL",
                "regime": "TRENDING_DOWN"
            }
        }

        # Generate reasoning with veto info and additional factors
        reason = generate_trade_reason(
            symbol="BTCUSDT",
            timeframe="1m",
            price=50000.0,
            confidence=0.75,
            regime="TRENDING_UP",
            sentiment=65,
            signal="BUY",
            threshold_info=veto_info,
            additional_factors=additional_factors
        )

        # Verify the higher timeframe veto
        self.assertTrue(veto_info["veto"])
        self.assertIn("higher_timeframe_signal", veto_info)
        self.assertIn("conflicts", veto_info["veto_reason"])

    def test_file_logging(self):
        """Test logging reasoning to a file."""
        # Generate a basic trade reason
        reason = generate_trade_reason(
            symbol="BTCUSDT",
            timeframe="1m",
            price=50000.0,
            confidence=0.78,
            regime="TRENDING_UP",
            sentiment=65,
            signal="BUY"
        )

        # Log the reasoning to the file
        log_reasoning(reason, file_path=self.test_file)

        # Verify the file was created and contains valid JSON
        with open(self.test_file, 'r') as f:
            line = f.readline().strip()
            loaded_reason = json.loads(line)
            self.assertEqual(loaded_reason["market_context"]["symbol"], "BTCUSDT")

    def test_log_to_file(self):
        """Test logging reasoning to the specified file."""
        # Generate a basic trade reason
        reason = generate_trade_reason(
            symbol="BTCUSDT",
            timeframe="1m",
            price=50000.0,
            confidence=0.78,
            regime="TRENDING_UP",
            sentiment=65,
            signal="BUY"
        )

        # Log the reasoning to the file
        log_reasoning(reason, file_path=self.test_log_file)

        # Verify the file was created and contains valid JSON
        with open(self.test_log_file, 'r') as f:
            line = f.readline().strip()
            loaded_reason = json.loads(line)
            self.assertEqual(loaded_reason["market_context"]["symbol"], "BTCUSDT")

if __name__ == "__main__":
    unittest.main()
