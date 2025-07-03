#!/usr/bin/env python
"""
Test script to view the contents of the trades.db file.

This script connects to the trades.db file in the logs directory,
lists all available tables, and displays the first 20 rows from
any table that looks like a trade log.
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_view_trades')

# Default database path
DEFAULT_DB_PATH = os.path.join("logs", "trades.db")

def view_trades_db(db_path=DEFAULT_DB_PATH, limit=20):
    """
    View the contents of the trades.db file.
    
    Args:
        db_path (str): Path to the database file
        limit (int): Maximum number of rows to display per table
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if database file exists
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            return False
            
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            logger.warning(f"No tables found in database: {db_path}")
            conn.close()
            return False
            
        logger.info(f"Found {len(tables)} tables in database: {', '.join(tables)}")
        
        # Look for trade-related tables
        trade_tables = [table for table in tables if any(keyword in table.lower() for keyword in ['trade', 'execution', 'order', 'position'])]
        
        if not trade_tables:
            # If no trade-specific tables found, use all tables
            trade_tables = tables
            logger.info("No trade-specific tables found, showing all tables")
        else:
            logger.info(f"Found {len(trade_tables)} trade-related tables: {', '.join(trade_tables)}")
        
        # Display rows from each trade table
        for table in trade_tables:
            logger.info(f"\n{'='*80}\nTable: {table}\n{'='*80}")
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            logger.info(f"Columns: {', '.join(columns)}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            logger.info(f"Total rows: {row_count}")
            
            # Get rows
            cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT {limit}")
            rows = cursor.fetchall()
            
            if not rows:
                logger.info(f"No rows found in table: {table}")
                continue
                
            # Display rows
            logger.info(f"\nShowing {len(rows)} most recent rows:")
            
            # Format rows as dictionaries for better readability
            for i, row in enumerate(rows):
                row_dict = {}
                for j, col in enumerate(columns):
                    # Format timestamp if it looks like one
                    if col.lower() in ['timestamp', 'time', 'date'] and isinstance(row[j], str):
                        try:
                            # Try to parse and format the timestamp
                            dt = datetime.fromisoformat(row[j].replace('Z', '+00:00'))
                            row_dict[col] = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except (ValueError, TypeError):
                            row_dict[col] = row[j]
                    else:
                        row_dict[col] = row[j]
                
                # Print row with index
                logger.info(f"Row {i+1}:")
                for key, value in row_dict.items():
                    logger.info(f"  {key}: {value}")
                logger.info("")  # Empty line between rows
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error viewing trades database: {e}")
        return False

if __name__ == "__main__":
    logger.info("Viewing trades database...")
    
    # Check if a custom database path was provided
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        logger.info(f"Using custom database path: {db_path}")
    else:
        db_path = DEFAULT_DB_PATH
        logger.info(f"Using default database path: {db_path}")
    
    # Check if a custom limit was provided
    if len(sys.argv) > 2:
        try:
            limit = int(sys.argv[2])
            logger.info(f"Using custom row limit: {limit}")
        except ValueError:
            limit = 20
            logger.warning(f"Invalid limit provided, using default: {limit}")
    else:
        limit = 20
        logger.info(f"Using default row limit: {limit}")
    
    if view_trades_db(db_path, limit):
        logger.info("✅ Successfully viewed trades database!")
        sys.exit(0)
    else:
        logger.error("❌ Failed to view trades database!")
        sys.exit(1)
