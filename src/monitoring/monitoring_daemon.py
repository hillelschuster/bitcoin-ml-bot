"""Unified monitoring interface for the Bitcoin ML Trading Bot.

This module provides a simple interface to start all monitoring services:
- Model monitoring: Checks model scores and evaluates model performance
- Integrity monitoring: Validates trade data and model health
- Reporting: Generates and sends status reports via Telegram

It handles thread management, error handling, and provides a single entry point
for starting all monitoring services with configurable intervals.
"""
import threading
import logging
from time import sleep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monitoring')

# Import monitoring components
from src.monitoring.model_monitor import start_monitoring, start_integrity_monitoring
from src.monitoring.reporter import start_periodic_reporting

# Thread factory functions to enable restarting
def create_model_thread(interval):
    return threading.Thread(
        target=start_monitoring,
        args=(interval,),
        daemon=True,
        name="ModelMonitorThread"
    )

def create_integrity_thread(interval):
    return threading.Thread(
        target=start_integrity_monitoring,
        args=(interval,),
        daemon=True,
        name="IntegrityMonitorThread"
    )

def create_reporting_thread(interval):
    return threading.Thread(
        target=start_periodic_reporting,
        args=(interval,),
        daemon=True,
        name="ReportingThread"
    )

def start_all_monitoring(
    model_check_interval=300,     # 5 minutes
    integrity_check_interval=3600,  # 1 hour
    reporting_interval=1800       # 30 minutes
):
    """Start all monitoring services in separate threads.

    This function initializes and starts all monitoring services as daemon threads,
    allowing them to run in the background while the main program continues execution.
    Each service runs with its own configurable interval.

    Args:
        model_check_interval (int): Time in seconds between model checks
        integrity_check_interval (int): Time in seconds between integrity checks
        reporting_interval (int): Time in seconds between status reports

    Returns:
        dict: Dictionary containing the thread objects for each monitoring service
    """
    logger.info("Starting all monitoring services...")

    # Create and start model monitoring thread
    model_thread = create_model_thread(model_check_interval)
    model_thread.start()
    logger.info(f"Model monitoring started (interval: {model_check_interval}s)")

    # Create and start integrity monitoring thread
    integrity_thread = create_integrity_thread(integrity_check_interval)
    integrity_thread.start()
    logger.info(f"Integrity monitoring started (interval: {integrity_check_interval}s)")

    # Create and start reporting thread
    reporting_thread = create_reporting_thread(reporting_interval)
    reporting_thread.start()
    logger.info(f"Periodic reporting started (interval: {reporting_interval}s)")

    # Return threads and their factory functions with intervals
    return {
        "model_thread": {
            "thread": model_thread,
            "factory": create_model_thread,
            "interval": model_check_interval
        },
        "integrity_thread": {
            "thread": integrity_thread,
            "factory": create_integrity_thread,
            "interval": integrity_check_interval
        },
        "reporting_thread": {
            "thread": reporting_thread,
            "factory": create_reporting_thread,
            "interval": reporting_interval
        }
    }

def watch_monitoring_threads(threads, check_interval=60):
    """Monitor the status of monitoring threads and restart them if needed.

    This function continuously checks if monitoring threads are still alive
    and restarts any threads that have stopped using their factory functions.
    It's designed to run in the main thread to provide oversight of the monitoring services.

    Args:
        threads (dict): Dictionary containing thread info (thread object, factory function, interval)
        check_interval (int): Time in seconds between thread status checks

    Note:
        This function runs in an infinite loop and should be executed in the main thread.
    """
    logger.info("Starting thread monitoring...")

    while True:
        for name, thread_info in threads.items():
            thread = thread_info["thread"]
            factory = thread_info["factory"]
            interval = thread_info["interval"]

            if not thread.is_alive():
                logger.warning(f"Thread {name} is not alive. Restarting...")
                # Create a new thread using the factory function
                new_thread = factory(interval)
                new_thread.start()

                # Update the thread reference in the dictionary
                threads[name]["thread"] = new_thread
                logger.info(f"Thread {name} restarted successfully")

        sleep(check_interval)

if __name__ == "__main__":
    # Start all monitoring services
    threads = start_all_monitoring()
    logger.info("All monitoring services started successfully")

    try:
        # Monitor threads and keep the main thread alive
        watch_monitoring_threads(threads)
    except KeyboardInterrupt:
        logger.info("Monitoring services shutting down...")
        logger.info("Press Ctrl+C again to force exit")
