# Initializing DuckDB database schema

import duckdb
import logging
from pathlib import Path
from datetime import datetime
from config import DATA_DIR

# Setting up the logger
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Create log filename with timestamp
log_filename = LOGS_DIR / f"init_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Logging to: {log_filename}")

# Defining the database path

DB_PATH = DATA_DIR / "github_evolution.duckdb"

def init_database():
    """Creating database and tables."""
    try:
        logger.info(f"Initializing database at {DB_PATH}")
        
        con = duckdb.connect(str(DB_PATH))
        
        # Repos table
        con.execute("""
            CREATE TABLE IF NOT EXISTS repos (
                repo_id INTEGER PRIMARY KEY,
                owner VARCHAR,
                name VARCHAR,
                positioning VARCHAR,
                UNIQUE(owner, name)
            )
        """)
        logger.info("Created repos table")
        
        # Commits table
        con.execute("""
            CREATE TABLE IF NOT EXISTS commits (
                sha VARCHAR PRIMARY KEY,
                repo_id INTEGER,
                author_login VARCHAR,
                author_date TIMESTAMP,
                commit_date TIMESTAMP,
                message TEXT,
                additions INTEGER,
                deletions INTEGER,
                total_changes INTEGER
            )
        """)
        logger.info("Created commits table")
        
        # Pull requests table
        con.execute("""
            CREATE TABLE IF NOT EXISTS pull_requests (
                pr_id INTEGER PRIMARY KEY,
                repo_id INTEGER,
                number INTEGER,
                author_login VARCHAR,
                created_at TIMESTAMP,
                closed_at TIMESTAMP,
                merged_at TIMESTAMP,
                state VARCHAR,
                title TEXT
            )
        """)
        logger.info("Created pull_requests table")
        
        con.close()
        logger.info("Database initialization complete")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    init_database()