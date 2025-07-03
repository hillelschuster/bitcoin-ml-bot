"""Reporter module for the Bitcoin ML Trading Bot.

This module provides functionality for generating and sending reports
about the trading bot's performance via Telegram. It includes functions
for creating daily reports, status updates, and trade summaries.
"""

import os
import time
import json
import logging
import sqlite3
import threading
import pandas as pd
import requests
from src.utils.trade_stats import generate_trade_stats_report
from config.Config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reporter')

# Database path
DB_PATH = os.path.join("logs", "trades.db")
TRADES_DB_PATH = DB_PATH

# Create a lock for thread-safe operations
config_lock = threading.Lock()

# Dictionary to store command handlers
command_handlers = {}

# Flag to indicate if command checking is active
command_checking_active = False

def register_command(command_name):
    """Decorator to register a command handler function.

    Args:
        command_name (str): The command name without the leading slash

    Returns:
        function: Decorator function
    """
    def decorator(func):
        command_handlers[command_name.lower()] = func
        return func
    return decorator

def load_config():
    """Load the configuration from the config file.

    Returns:
        dict: The configuration dictionary, or None if loading fails
    """
    try:
        with open('config/config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None

def save_config(config):
    """Save the configuration to the config file.

    Args:
        config (dict): The configuration dictionary

    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        with config_lock:
            with open('config/config.json', 'w') as f:
                json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

def generate_daily_report():
    """
    Generate a daily report of trading activity.

    Returns:
        str: A formatted report string with trade count and total PnL
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()

    today = pd.Timestamp.utcnow().normalize()
    today_trades = df[pd.to_datetime(df["timestamp"]) >= today]

    if today_trades.empty:
        return "No trades today."

    # Check if pnl column exists
    if "pnl" in today_trades.columns:
        total_pnl = today_trades["pnl"].sum()
    else:
        # Fallback to estimated PnL if pnl column doesn't exist
        logger.warning("PnL column not found, using estimated calculation")
        total_pnl = today_trades.eval("amount * price").sum()

    return f"Daily Report:\nTrades: {len(today_trades)}\nTotal PnL: ${total_pnl:.2f}"

def send_telegram_message(message, max_retries=3, retry_delay=5):
    """
    Send a message to a Telegram chat with retry logic.

    Args:
        message (str): The message to send, can include Markdown formatting
        max_retries (int): Maximum number of retry attempts if sending fails
        retry_delay (int): Delay in seconds between retry attempts

    Returns:
        bool: True if the message was sent successfully, False otherwise
    """
    # Get Telegram credentials from Config
    telegram_token = TELEGRAM_BOT_TOKEN
    telegram_chat_id = TELEGRAM_CHAT_ID

    if not telegram_token or not telegram_chat_id:
        logger.warning("Telegram credentials not found in Config")
        return False

    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    data = {
        "chat_id": telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    # Implement retry logic
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, data=data, timeout=10)  # Add timeout
            success = response.status_code == 200

            if success:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.warning(f"Attempt {attempt}/{max_retries}: Failed to send Telegram message: {response.text}")

                # If we have more retries, wait before trying again
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay *= 2
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries}: Error sending Telegram message: {e}")

            # If we have more retries, wait before trying again
            if attempt < max_retries:
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay *= 2

    logger.error("All attempts to send Telegram message failed")
    return False

def fetch_trade_summary():
    """
    Fetch a summary of trading activity from the database.

    Returns:
        tuple: (total_trades, open_trades, closed_trades, total_pnl)
    """
    try:
        conn = sqlite3.connect(TRADES_DB_PATH)
        cursor = conn.cursor()

        # Total trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]

        # Check table schema to determine the best way to identify open/closed trades
        cursor.execute("PRAGMA table_info(trades)")
        columns = [col[1] for col in cursor.fetchall()]

        # Determine the best way to identify open/closed trades based on available columns
        if "is_closed" in columns:
            # If we have a dedicated is_closed column, use it
            cursor.execute("SELECT COUNT(*) FROM trades WHERE is_closed = 1")
            closed_trades = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trades WHERE is_closed = 0 OR is_closed IS NULL")
            open_trades = cursor.fetchone()[0]

            # Total PNL
            cursor.execute("SELECT SUM(pnl) FROM trades WHERE is_closed = 1")
            total_pnl = cursor.fetchone()[0] or 0
        elif "exit_price" in columns:
            # If we have exit_price, use it to determine closed trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL AND exit_price > 0")
            closed_trades = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NULL OR exit_price = 0")
            open_trades = cursor.fetchone()[0]

            # Total PNL
            if "pnl" in columns:
                cursor.execute("SELECT SUM(pnl) FROM trades WHERE exit_price IS NOT NULL AND exit_price > 0")
                total_pnl = cursor.fetchone()[0] or 0
            else:
                # Estimate PNL if no dedicated column exists
                cursor.execute("SELECT SUM((exit_price - entry_price) * amount) FROM trades WHERE exit_price IS NOT NULL AND exit_price > 0")
                total_pnl = cursor.fetchone()[0] or 0
        elif "pnl" in columns:
            # If we have pnl column, assume trades with non-null, non-zero PNL are closed
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL AND pnl != 0")
            closed_trades = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl IS NULL OR pnl = 0")
            open_trades = cursor.fetchone()[0]

            # Total PNL
            cursor.execute("SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL")
            total_pnl = cursor.fetchone()[0] or 0
        else:
            # Fallback to type-based identification if no better options exist
            logger.warning("Using trade type to determine open/closed status (less reliable)")
            cursor.execute("SELECT COUNT(*) FROM trades WHERE type IN ('buy', 'sell')")
            open_trades = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trades WHERE type = 'close'")
            closed_trades = cursor.fetchone()[0]

            # Estimate PNL from price and amount
            cursor.execute("SELECT SUM(price * amount) FROM trades WHERE type = 'close'")
            total_pnl = cursor.fetchone()[0] or 0

        conn.close()
        return total_trades, open_trades, closed_trades, round(total_pnl, 2)
    except Exception as e:
        logger.error(f"Error fetching trade summary: {e}")
        return 0, 0, 0, 0.0


def generate_status_report():
    """
    Generate a comprehensive status report for the bot.

    Returns:
        str: A formatted status report with trade statistics
    """
    total_trades, open_trades, closed_trades, total_pnl = fetch_trade_summary()

    # Basic status report
    msg = (
        f"📊 *Bot Status Update*\n\n"
        f"🔁 Total Trades: {total_trades}\n"
        f"📂 Open Trades: {open_trades}\n"
        f"✅ Closed Trades: {closed_trades}\n"
        f"💰 Total PnL: `{total_pnl} USDT`\n"
    )

    # Add detailed trade statistics report
    try:
        trade_stats = generate_trade_stats_report()
        msg += "\n\n" + trade_stats
    except Exception as e:
        logger.error(f"Error generating trade statistics report: {e}")
        msg += "\n\n⚠️ Trade statistics report unavailable"

    return msg

def send_status_report():
    """Generate and send a status report via Telegram.

    Creates a status report and sends it to the configured Telegram channel.

    Returns:
        bool: True if the report was sent successfully, False otherwise
    """
    report = generate_status_report()
    success = send_telegram_message(report)

    if success:
        logger.info("[Reporter] Status report sent successfully")
    else:
        logger.warning("[Reporter] Failed to send status report")

    return success

@register_command('help')
def handle_help_command(chat_id):
    """Handle the /help command.

    Args:
        chat_id (str): The chat ID to send the response to

    Returns:
        bool: True if the message was sent successfully, False otherwise
    """
    help_text = """
🤖 *Bitcoin ML Trading Bot Commands*

*/status* - Get current bot status
*/confidence* - Get current confidence mode
*/set_confidence_mode [auto|model|fallback]* - Set confidence calculation mode
*/help* - Show this help message

*Confidence Modes:*
- *auto*: Automatically choose between model and fallback
- *model*: Force use of model output (fails if model is broken)
- *fallback*: Force use of fallback calculation based on sentiment, momentum, and regime
    """
    return send_telegram_message(help_text)

@register_command('status')
def handle_status_command(chat_id):
    """Handle the /status command.

    Args:
        chat_id (str): The chat ID to send the response to

    Returns:
        bool: True if the message was sent successfully, False otherwise
    """
    # Generate and send the status report
    return send_status_report()

@register_command('confidence')
def handle_confidence_command(chat_id):
    """Handle the /confidence command.

    Args:
        chat_id (str): The chat ID to send the response to

    Returns:
        bool: True if the message was sent successfully, False otherwise
    """
    config = load_config()
    if not config:
        return send_telegram_message("❌ Error: Could not load configuration")

    confidence_mode = config.get('force_confidence_mode', 'auto')
    confidence_text = f"""
🧠 *Confidence Settings*

*Current Mode:* {confidence_mode}
*Min Confidence:* {config.get('min_confidence', 0.58)}

*Available Modes:*
- *auto*: Automatically choose between model and fallback
- *model*: Force use of model output (fails if model is broken)
- *fallback*: Force use of fallback calculation based on sentiment, momentum, and regime

To change mode, use:
/set_confidence_mode [auto|model|fallback]
    """
    return send_telegram_message(confidence_text)

@register_command('set_confidence_mode')
def handle_confidence_mode_command(chat_id, args):
    """Handle the /set_confidence_mode command.

    Args:
        chat_id (str): The chat ID to send the response to
        args (list): The command arguments

    Returns:
        bool: True if the message was sent successfully, False otherwise
    """
    try:
        # Check if we have arguments
        if not args:
            return send_telegram_message("❌ Usage: /set_confidence_mode [auto|model|fallback]")

        # Get the mode from arguments
        mode = args[0].strip().lower()
        if mode not in ["auto", "model", "fallback"]:
            return send_telegram_message("❌ Invalid mode. Use 'auto', 'model', or 'fallback'")

        # Load the config
        config = load_config()
        if not config:
            return send_telegram_message("❌ Error: Could not load configuration")

        # Update the config
        config['force_confidence_mode'] = mode

        # Save the config
        if save_config(config):
            return send_telegram_message(f"✅ Confidence mode set to: {mode}")
        else:
            return send_telegram_message("❌ Error: Could not save configuration")
    except Exception as e:
        logger.error(f"Error handling set_confidence_mode command: {e}")
        return send_telegram_message(f"❌ Error: {str(e)}")

def check_for_commands():
    """Check for new commands from Telegram.

    This function polls the Telegram API for new messages and processes
    any commands it finds.
    """
    global command_checking_active

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not found, command checking disabled")
        return

    # Mark as active
    command_checking_active = True

    # Get the last update ID we've processed
    last_update_id = 0

    logger.info("Starting Telegram command checking")

    while command_checking_active:
        try:
            # Get updates from Telegram
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {
                "offset": last_update_id + 1,
                "timeout": 30
            }

            response = requests.get(url, params=params, timeout=35)

            if response.status_code == 200:
                updates = response.json()

                if updates.get("ok") and updates.get("result"):
                    for update in updates["result"]:
                        # Process the update
                        process_update(update)

                        # Update the last update ID
                        last_update_id = max(last_update_id, update["update_id"])
            else:
                logger.warning(f"Failed to get updates from Telegram: {response.text}")

        except Exception as e:
            logger.error(f"Error checking for commands: {e}")

        # Sleep for a bit before checking again
        time.sleep(5)

def process_update(update):
    """Process a Telegram update.

    Args:
        update (dict): The update from Telegram
    """
    try:
        # Check if this is a message
        if "message" not in update:
            return

        message = update["message"]

        # Check if this is a command
        if "text" not in message or not message["text"].startswith("/"):
            return

        # Get the command and arguments
        text = message["text"]
        parts = text.split()
        command = parts[0][1:].lower()  # Remove the leading slash
        args = parts[1:] if len(parts) > 1 else []

        # Get the chat ID
        chat_id = message["chat"]["id"]

        # Check if we have a handler for this command
        if command in command_handlers:
            logger.info(f"Processing command: {command} with args: {args}")

            # Call the handler
            handler = command_handlers[command]
            if len(args) > 0:
                handler(chat_id, args)
            else:
                handler(chat_id)
        else:
            logger.warning(f"Unknown command: {command}")
            send_telegram_message(f"Unknown command: {command}. Use /help for available commands.")

    except Exception as e:
        logger.error(f"Error processing update: {e}")

def start_command_checking():
    """Start the command checking thread.

    Returns:
        threading.Thread: The command checking thread
    """
    thread = threading.Thread(target=check_for_commands, daemon=True)
    thread.start()
    return thread

def start_periodic_reporting(interval=1800):
    """Start a background process that periodically sends status reports.

    This function runs continuously in a separate thread, generating and
    sending status reports at regular intervals via Telegram.

    Args:
        interval (int): Time in seconds between reports (default: 1800 - 30 minutes)

    Note:
        This function runs in an infinite loop and should be executed in a separate thread.
    """
    logger.info(f"[Reporter] Starting periodic reporting with interval of {interval} seconds")

    # Start the command checking thread
    start_command_checking()

    while True:
        try:
            send_status_report()
            # Logging is handled inside send_status_report()
        except Exception as e:
            logger.error(f"[Reporter] Error in periodic reporting: {e}")

        # Sleep until next report
        time.sleep(interval)
