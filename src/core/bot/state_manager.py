"""
State management for position tracking, circuit breaker, and exit metrics.
"""

import os
import json
import time
import tempfile
import shutil
import logging
import platform

HAS_FILE_LOCKING = platform.system() != "Windows"
if not HAS_FILE_LOCKING:
    import msvcrt
else:
    import fcntl

from config.paths import POSITION_FILE, CIRCUIT_BREAKER_FILE, EMERGENCY_EXIT_METRICS_FILE

logger = logging.getLogger("core.bot.state_manager")


class StateManager:
    def __init__(self, notifier):
        self.notifier = notifier
        self.position = None
        self.position_entry_time = 0
        self.trading_suspended = False
        self.emergency_exit_failures = []
        self.emergency_exit_failures_last_24h = 0
        self._trade_lock = None  # Assigned externally for file-locking reuse

    def load_position_state(self, exchange, symbol):
        if self._trade_lock:
            with self._trade_lock:
                try:
                    try:
                        exchange.validate_position_state(symbol)
                        logger.info("Position state validated against exchange")
                    except (ValueError, KeyError, AttributeError) as validate_error:
                        logger.warning(f"Could not validate position state: {validate_error}")

                    if os.path.exists(POSITION_FILE):
                        with open(POSITION_FILE, "r") as f:
                            position_data = json.load(f)
                            if isinstance(position_data, dict):
                                self.position = position_data.get("position")
                                self.position_entry_time = position_data.get("entry_time", 0)
                                if self.position:
                                    size = exchange.get_current_position_size(symbol)
                                    if size == 0:
                                        logger.warning("Resetting stale position state.")
                                        self.position = None
                                        self.position_entry_time = 0
                                        self.save_position_state()
                except (IOError, OSError, json.JSONDecodeError) as e:
                    logger.error(f"Error loading position state: {e}")

    def save_position_state(self):
        if self._trade_lock:
            with self._trade_lock:
                try:
                    os.makedirs(os.path.dirname(POSITION_FILE), exist_ok=True)
                    data = {
                        "position": self.position,
                        "entry_time": self.position_entry_time,
                        "updated_at": time.time(),
                    }
                    if self.position is None:
                        if os.path.exists(POSITION_FILE):
                            os.remove(POSITION_FILE)
                            logger.debug("Removed position file (position closed)")
                        return

                    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(POSITION_FILE))
                    with os.fdopen(fd, "w") as f:
                        json.dump(data, f)
                    shutil.move(temp_path, POSITION_FILE)
                    logger.debug("Position state saved (atomic)")
                except (IOError, OSError) as e:
                    logger.error(f"Error saving position state: {e}")
                    self.notifier.send(f"⚠️ Warning: Failed to save position state: {e}")

    def load_circuit_breaker_state(self):
        try:
            if os.path.exists(CIRCUIT_BREAKER_FILE):
                with open(CIRCUIT_BREAKER_FILE, "r") as f:
                    data = json.load(f)
                    if data.get("active", False):
                        expires_at = data.get("expires_at", 0)
                        if time.time() < expires_at:
                            self.trading_suspended = True
                            logger.critical("🚨 Circuit breaker active!")
                            self.notifier.send("🚨 Circuit breaker is active. Trading suspended.")
                        else:
                            logger.info("Circuit breaker expired. Resuming.")
                            self.reset_circuit_breaker()
        except (IOError, OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading circuit breaker: {e}")

    def save_circuit_breaker_state(self, active=True, duration_seconds=900):
        try:
            os.makedirs(os.path.dirname(CIRCUIT_BREAKER_FILE), exist_ok=True)
            now = time.time()
            data = {
                "active": active,
                "activated_at": now,
                "expires_at": now + duration_seconds,
                "updated_at": now,
            }
            with open(CIRCUIT_BREAKER_FILE, "w") as f:
                json.dump(data, f)
            logger.info("Circuit breaker state saved.")
        except (IOError, OSError) as e:
            logger.error(f"Error saving circuit breaker: {e}")

    def reset_circuit_breaker(self):
        try:
            if os.path.exists(CIRCUIT_BREAKER_FILE):
                with open(CIRCUIT_BREAKER_FILE, "w") as f:
                    json.dump({"active": False, "updated_at": time.time()}, f)
                logger.info("Circuit breaker reset.")
            self.trading_suspended = False
        except (IOError, OSError) as e:
            logger.error(f"Error resetting circuit breaker: {e}")

    def load_emergency_exit_metrics(self):
        try:
            if os.path.exists(EMERGENCY_EXIT_METRICS_FILE):
                with open(EMERGENCY_EXIT_METRICS_FILE, "r") as f:
                    data = json.load(f)
                    failures = data.get("failures", [])
                    cutoff = time.time() - 86400
                    self.emergency_exit_failures = [f for f in failures if f.get("timestamp", 0) > cutoff]
                    self.emergency_exit_failures_last_24h = len(self.emergency_exit_failures)
        except (IOError, OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load emergency metrics: {e}")

    def save_emergency_exit_metrics(self):
        try:
            os.makedirs(os.path.dirname(EMERGENCY_EXIT_METRICS_FILE), exist_ok=True)
            failures = self.emergency_exit_failures[-100:]  # cap to 100
            data = {
                "failures": failures,
                "total_failures": len(failures),
                "updated_at": time.time()
            }
            with open(EMERGENCY_EXIT_METRICS_FILE, "w") as f:
                json.dump(data, f)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save emergency metrics: {e}")

    def reset_position_state(self):
        """Reset position state to None and save the state."""
        self.position = None
        self.position_entry_time = 0
        self.save_position_state()
        logger.info("Position state reset to None")
