"""
Retry utilities for the trading bot.

This module provides functions and decorators for retrying operations with backoff.
"""

import time
import logging
from functools import wraps

logger = logging.getLogger("core.utils.retry_utils")

# Default retry configuration
MAX_RETRIES = 3
BACKOFF_TIMES = [1, 2, 4]  # seconds


def retry_with_backoff(func, max_retries=3, initial_delay=1, backoff_factor=2, *args, **kwargs):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: The function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The return value of the function if successful
        
    Raises:
        Exception: The last exception encountered if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"Retry attempt {attempt+1}/{max_retries} failed: {str(e)}")

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_retries} retry attempts failed")

    # If we get here, all retries failed
    raise last_exception


def retry_exchange_call(method_name=None, max_retries=MAX_RETRIES, backoff_times=None):
    """
    Decorator for retrying exchange API calls with backoff.
    
    This can be used in two ways:
    1. As a function that returns a decorator: retry_exchange_call('method_name')
    2. As a decorator directly: @retry_exchange_call
    
    Args:
        method_name: Name of the method to retry (optional)
        max_retries: Maximum number of retry attempts
        backoff_times: List of backoff times in seconds
        
    Returns:
        Decorated function that will retry on failure
    """
    if backoff_times is None:
        backoff_times = BACKOFF_TIMES
        
    def decorator(func_or_method_name):
        # Handle case where this is used as a function that returns a decorator
        if isinstance(func_or_method_name, str):
            # This is being used as retry_exchange_call('method_name')
            method_name_to_use = func_or_method_name
            
            # Return a function that will wrap the method when called
            def method_wrapper(instance):
                method = getattr(instance, method_name_to_use)
                
                @wraps(method)
                def wrapper(*args, **kwargs):
                    last_error = None
                    for attempt in range(max_retries):
                        try:
                            logger.debug(f"Calling {method_name_to_use} (attempt {attempt+1}/{max_retries})")
                            result = method(*args, **kwargs)
                            if result is not None:  # None is often an error indicator
                                return result
                            logger.warning(f"{method_name_to_use} returned None (attempt {attempt+1}/{max_retries})")
                        except Exception as e:
                            last_error = e
                            logger.warning(f"Error in {method_name_to_use} (attempt {attempt+1}/{max_retries}): {e}")

                        # Don't sleep on the last attempt
                        if attempt < max_retries - 1:
                            sleep_time = backoff_times[attempt]
                            logger.info(f"Retrying {method_name_to_use} in {sleep_time}s...")
                            time.sleep(sleep_time)

                    # If we get here, all attempts failed
                    error_msg = f"Failed to execute {method_name_to_use} after {max_retries} attempts"
                    if last_error:
                        error_msg += f": {last_error}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
                return wrapper
                
            return method_wrapper
        else:
            # This is being used as a decorator directly: @retry_exchange_call
            @wraps(func_or_method_name)
            def wrapper(*args, **kwargs):
                last_error = None
                func_name = func_or_method_name.__name__
                
                for attempt in range(max_retries):
                    try:
                        logger.debug(f"Calling {func_name} (attempt {attempt+1}/{max_retries})")
                        result = func_or_method_name(*args, **kwargs)
                        if result is not None:  # None is often an error indicator
                            return result
                        logger.warning(f"{func_name} returned None (attempt {attempt+1}/{max_retries})")
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Error in {func_name} (attempt {attempt+1}/{max_retries}): {e}")

                    # Don't sleep on the last attempt
                    if attempt < max_retries - 1:
                        sleep_time = backoff_times[attempt]
                        logger.info(f"Retrying {func_name} in {sleep_time}s...")
                        time.sleep(sleep_time)

                # If we get here, all attempts failed
                error_msg = f"Failed to execute {func_name} after {max_retries} attempts"
                if last_error:
                    error_msg += f": {last_error}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            return wrapper
            
    # Handle case where this is used as a decorator directly
    if callable(method_name):
        func = method_name
        method_name = None
        return decorator(func)
        
    # Handle case where this is used as a function that returns a decorator
    return decorator
