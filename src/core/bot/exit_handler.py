"""
Exit handling for the trading bot.

This module handles normal and emergency position exits,
with retry logic and verification of position closure.
"""

import time
import logging

logger = logging.getLogger("core.bot.exit_handler")

# Constants
MAX_EMERGENCY_EXIT_ATTEMPTS = 5  # Maximum number of attempts for emergency exit


class ExitHandler:
    def __init__(self, exchange, state_manager, safety_monitor, notifier, config):
        """
        Initialize the exit handler.
        
        Args:
            exchange: Exchange client
            state_manager: State manager for position state
            safety_monitor: Safety monitor for circuit breaker logic
            notifier: Notifier for sending alerts
            config: Configuration object
        """
        self.exchange = exchange
        self.state_manager = state_manager
        self.safety_monitor = safety_monitor
        self.notifier = notifier
        self.config = config
        self.position_closing = False
        
    def exit_position(self, symbol, exit_price=None):
        """
        Exit a position with proper error handling and verification.
        
        Args:
            symbol: Trading pair symbol
            exit_price: Exit price for logging (optional)
            
        Returns:
            bool: True if position was successfully closed, False otherwise
        """
        # Use the state manager's trade lock to prevent concurrent entry/exit operations
        if self.state_manager._trade_lock:
            with self.state_manager._trade_lock:
                return self._exit_position_internal(symbol, exit_price)
        else:
            return self._exit_position_internal(symbol, exit_price)
            
    def _exit_position_internal(self, symbol, exit_price=None):
        """
        Internal implementation of exit_position with proper error handling and verification.
        
        Args:
            symbol: Trading pair symbol
            exit_price: Exit price for logging (optional)
            
        Returns:
            bool: True if position was successfully closed, False otherwise
        """
        # Verify we have a position to exit
        if not self.state_manager.position:
            logger.warning("Cannot exit position - no position is open")
            return False

        # Store position for logging
        position_side = self.state_manager.position

        # Set flag to prevent drawdown monitoring during exit
        self.position_closing = True
        self.safety_monitor.position_closing = True

        try:
            # Get current price if not provided
            if exit_price is None:
                try:
                    exit_price = self.exchange.get_market_price(symbol)
                except (ValueError, KeyError, AttributeError) as e:
                    logger.warning(f"Could not get market price for exit: {e}")
                    exit_price = 0

            # Close the position
            close_result = self.exchange.close_position(symbol)
            
            # Check if close was successful
            if not close_result:
                # Verify position size to see if it's actually closed despite error
                position_size = self.exchange.get_current_position_size(symbol)
                
                if position_size <= 0:
                    logger.warning(f"Position appears to be closed despite error response. Resetting state.")
                    # Reset position state since it's actually closed
                    self.state_manager.position = None
                    self.state_manager.position_entry_time = 0
                    self.state_manager.save_position_state()
                    self.notifier.send(f"📤 {position_side.upper()} position appears closed despite error")
                    return True
                else:
                    logger.critical(f"❌ Failed to close {position_side} position. Size still open: {position_size}")
                    self.notifier.send(f"⚠️ Failed to close {position_side.upper()} position. Manual intervention required!")
                    return False

            # Log the trade
            try:
                from src.utils.logger import TradeLogger
                trade_logger = TradeLogger()
                trade_logger.log_trade(
                    type="close",
                    symbol=symbol,
                    price=exit_price,
                    side="SELL" if position_side == "long" else "BUY"
                )
            except (IOError, OSError, json.JSONDecodeError) as log_error:
                logger.error(f"Error logging trade: {log_error}")

            # Send notification
            self.notifier.send(f"📤 {position_side.upper()} position closed at {exit_price}")

            # Verify position is actually closed
            position_size = self.exchange.get_current_position_size(symbol)
            if position_size > 0:
                # Try one more time with explicit size
                try:
                    logger.warning(f"Position still shows size {position_size} after closing. Attempting force close.")
                    # Execute force close
                    self.exchange.close_position(symbol, size=position_size)
                    
                    # Verify again after retry
                    position_size_after_retry = self.exchange.get_current_position_size(symbol)
                    if position_size_after_retry <= 0:
                        logger.info(f"Force close successful on retry - position is now closed")
                    else:
                        logger.critical(f"❌ Force close attempt failed - position size still {position_size_after_retry}")
                        self.notifier.send(f"⚠️ WARNING: Failed to fully close {position_side.upper()} position. Manual intervention may be required!")
                except (ValueError, KeyError, AttributeError) as e:
                    logger.critical(f"Error during force close attempt: {e}")
                    self.notifier.send(f"⚠️ ERROR: Exception during force close: {e}. Manual intervention required!")

            # Verify one final time that position is actually closed before resetting state
            final_position_size = self.exchange.get_current_position_size(symbol)
            if final_position_size <= 0:
                # Only reset position state if we've confirmed it's closed
                logger.info(f"Confirmed {position_side.upper()} position is closed. Resetting state.")
                self.state_manager.position = None
                self.state_manager.position_entry_time = 0
                self.state_manager.save_position_state()
                return True
            else:
                logger.critical(f"❌ Cannot reset position state - position still shows size {final_position_size}")
                self.notifier.send(f"⚠️ CRITICAL: Position appears to still be open with size {final_position_size}. Manual intervention required!")
                return False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error during position exit: {e}")
            # Don't reset position state here - better to think we have a position when we don't
            # than to think we don't have a position when we do
            return False
        finally:
            # Always reset the closing flag
            self.position_closing = False
            self.safety_monitor.position_closing = False
            
    def emergency_exit(self, symbol, reason):
        """
        Emergency exit from a position with enhanced safety features.
        
        This method includes:
        1. Max retry cap (5 attempts)
        2. Circuit breaker activation if too many failures occur
        3. Panic mode force-close as last resort
        4. Failure metrics logging
        
        Args:
            symbol: Trading pair symbol
            reason: Reason for the emergency exit
            
        Returns:
            bool: True if position was successfully closed, False otherwise
        """
        # Use the state manager's trade lock to prevent concurrent entry/exit operations
        if self.state_manager._trade_lock:
            with self.state_manager._trade_lock:
                return self._emergency_exit_internal(symbol, reason)
        else:
            return self._emergency_exit_internal(symbol, reason)
            
    def _emergency_exit_internal(self, symbol, reason):
        """
        Internal implementation of emergency_exit with proper error handling and verification.
        
        Args:
            symbol: Trading pair symbol
            reason: Reason for the emergency exit
            
        Returns:
            bool: True if position was successfully closed, False otherwise
        """
        # Verify we have a position to exit
        if not self.state_manager.position:
            logger.warning("Cannot perform emergency exit - no position is open")
            return False

        # Store position for logging
        position_side = self.state_manager.position

        # Set flag to prevent drawdown monitoring during exit
        self.position_closing = True
        self.safety_monitor.position_closing = True

        # Reset attempt counter for this emergency exit session
        emergency_exit_attempts = 0
        exit_successful = False

        try:
            # Send notification immediately for emergency
            self.notifier.send(f"❌ EMERGENCY EXIT TRIGGERED: {reason}")

            # Main exit loop with retry cap
            while emergency_exit_attempts < MAX_EMERGENCY_EXIT_ATTEMPTS and not exit_successful:
                emergency_exit_attempts += 1
                logger.warning(f"[EMERGENCY EXIT] Attempt {emergency_exit_attempts}/{MAX_EMERGENCY_EXIT_ATTEMPTS}")

                # If this is the last attempt, use panic mode
                is_final_attempt = emergency_exit_attempts >= MAX_EMERGENCY_EXIT_ATTEMPTS

                # Close the position
                try:
                    if is_final_attempt:
                        # PANIC MODE: Last resort attempt with force close
                        logger.critical(f"[FORCE CLOSE ATTEMPT] Final emergency exit attempt - using panic mode")
                        # Get current position size for explicit close
                        panic_size = self.exchange.get_current_position_size(symbol)
                        if panic_size > 0:
                            # Use exchange directly with reduceOnly=True
                            side = "SELL" if position_side == "long" else "BUY"
                            logger.critical(f"[FORCE CLOSE ATTEMPT] Executing {side} order for {panic_size} with reduceOnly=True")
                            try:
                                # Direct exchange call with minimal parameters
                                close_result = self.exchange.place_order(side, quantity=panic_size, reduce_only=True)
                                close_success = close_result and close_result.get('success', False)
                            except (RuntimeError, ValueError) as panic_error:
                                logger.critical(f"[FORCE CLOSE ATTEMPT] Failed: {panic_error}")
                                close_success = False
                                close_result = None
                        else:
                            logger.info("[FORCE CLOSE ATTEMPT] No position to close")
                            close_success = True
                            close_result = {'success': True}
                    else:
                        # Normal attempt
                        close_result = self.exchange.close_position(symbol)
                        close_success = close_result is not None
                except (RuntimeError, ValueError) as e:
                    logger.critical(f"Failed to close position in emergency exit attempt {emergency_exit_attempts}: {e}")
                    close_success = False
                    close_result = None

                # Verify position after close attempt
                position_size = self.exchange.get_current_position_size(symbol)
                
                # If position is closed, update state and exit
                if position_size <= 0:
                    # Verify one final time before resetting state
                    time.sleep(1)  # Brief pause to ensure exchange state is updated
                    final_check_size = self.exchange.get_current_position_size(symbol)
                    if final_check_size <= 0:
                        logger.info(f"Final verification confirms position is closed. Resetting state.")
                        self.state_manager.position = None
                        self.state_manager.position_entry_time = 0
                        self.state_manager.save_position_state()
                        self.notifier.send(f"✅ Emergency exit CONFIRMED for {position_side.upper()} position")
                        exit_successful = True
                        break  # Exit the retry loop
                    else:
                        logger.critical(f"❌ Final verification shows position still open with size {final_check_size}!")
                        # Continue to next attempt
                else:
                    # Position is still open
                    if not close_success:
                        logger.critical(f"Close operation failed and position still open with size {position_size}")
                        
                        # Try with explicit size if not the final attempt
                        if not is_final_attempt:
                            logger.warning(f"Attempting force close with explicit size {position_size}")
                            try:
                                # Execute force close
                                self.exchange.close_position(symbol, size=position_size)
                                
                                # Verify position after force close
                                verify_size = self.exchange.get_current_position_size(symbol)
                                if verify_size <= 0:
                                    logger.info(f"Force close successful - position is now closed")
                                    self.state_manager.position = None
                                    self.state_manager.position_entry_time = 0
                                    self.state_manager.save_position_state()
                                    self.notifier.send(f"✅ Emergency exit completed with force close")
                                    exit_successful = True
                                    break  # Exit the retry loop
                            except (RuntimeError, ValueError) as e:
                                logger.critical(f"Failed to force close position: {e}")
                    else:
                        # Close reported success but position still exists - inconsistent state
                        logger.warning(f"Exchange reported successful close but position size is still {position_size}. Continuing retry.")

                # If we've reached the max attempts and still haven't succeeded, break the loop
                if emergency_exit_attempts >= MAX_EMERGENCY_EXIT_ATTEMPTS and not exit_successful:
                    logger.critical(f"[MAX RETRY REACHED] Emergency Exit Aborted after {MAX_EMERGENCY_EXIT_ATTEMPTS} attempts")
                    self.notifier.send(f"⚠️ CRITICAL: Max retry reached ({MAX_EMERGENCY_EXIT_ATTEMPTS} attempts). Emergency exit aborted.")
                    # Record the failure for circuit breaker evaluation
                    self.safety_monitor.record_exit_failure(f"Max retry reached: {reason}", position_side, emergency_exit_attempts)
                    break

                # If not successful and not at max attempts, wait before retrying
                if not exit_successful and emergency_exit_attempts < MAX_EMERGENCY_EXIT_ATTEMPTS:
                    wait_time = 2 ** (emergency_exit_attempts - 1)  # Exponential backoff
                    logger.warning(f"Retrying emergency exit in {wait_time} seconds...")
                    time.sleep(wait_time)

            # Final verification after all attempts
            if not exit_successful:
                # Position is still open after all attempts
                final_size = self.exchange.get_current_position_size(symbol)
                if final_size > 0:
                    logger.critical(f"❌ CRITICAL: Emergency exit failed after {emergency_exit_attempts} attempts. Position still open: {final_size}")
                    self.notifier.send(f"⚠️ URGENT: Emergency exit FAILED after {emergency_exit_attempts} attempts. Manual intervention required!")
                else:
                    # Position appears to be closed despite reporting failure
                    logger.warning(f"Position appears to be closed despite reporting failure. Resetting state.")
                    self.state_manager.position = None
                    self.state_manager.position_entry_time = 0
                    self.state_manager.save_position_state()
                    self.notifier.send(f"✅ Position appears closed despite reported failures. State reset.")
                    exit_successful = True
        except (RuntimeError, ValueError) as e:
            logger.critical(f"Error during emergency exit: {e}")
            self.notifier.send(f"⚠️ Error during emergency exit: {e}. Manual intervention may be required!")
            # Record the failure
            self.safety_monitor.record_exit_failure(f"Exception: {str(e)[:100]}", position_side, emergency_exit_attempts)
            return False
        finally:
            # Always reset the closing flag
            self.position_closing = False
            self.safety_monitor.position_closing = False

            # If exit was not successful, check circuit breaker condition
            if not exit_successful:
                self.safety_monitor.check_circuit_breaker_condition()
                
        return exit_successful
        
    def verify_position_closure(self, symbol):
        """
        Verify that a position is actually closed.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if position is closed, False otherwise
        """
        try:
            position_size = self.exchange.get_current_position_size(symbol)
            if position_size <= 0:
                return True
            else:
                logger.warning(f"Position verification failed - size: {position_size}")
                return False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error verifying position closure: {e}")
            return False
