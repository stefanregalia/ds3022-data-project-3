# Prefect flow to ingest pull requests from GitHub to DuckDB

# Importing necessary libraries
import duckdb
import logging
from prefect import flow, task
from typing import List, Dict, Optional
from config import load_repos, DATA_DIR
from github_client import client

# Setting up the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Defining the database path
DB_PATH = DATA_DIR / "github_evolution.duckdb"

# Auto-initialize database 
def ensure_database_exists():
    """Creating database and tables if they don't exist."""
    if not DB_PATH.exists():
        logger.info("Database not found. Initializing...")
        from init_db import init_database
        init_database()

ensure_database_exists()

# Task definitions
@task(retries=3, retry_delay_seconds=60)
def fetch_repo_prs(owner: str, name: str, max_pages: int = 50) -> List[Dict]:
    """Fetching pull requests for a single repository."""
    try:
        logger.info(f"Fetching PRs for {owner}/{name}")
        prs = client.fetch_pull_requests(owner, name, state="all", per_page=100, max_pages=max_pages)
        logger.info(f"Successfully fetched {len(prs)} PRs for {owner}/{name}")
        return prs
    except Exception as e:
        logger.error(f"Failed to fetch PRs for {owner}/{name}: {str(e)}")
        raise

@task
def get_repo_id(owner: str, name: str) -> Optional[int]:
    """Getting repo_id from database."""
    con = None
    try:
        logger.info(f"Looking up repo_id for {owner}/{name}")
        con = duckdb.connect(str(DB_PATH))
        result = con.execute(
            "SELECT repo_id FROM repos WHERE owner = ? AND name = ?",
            [owner, name]
        ).fetchone()
        
        if result:
            logger.info(f"Found repo_id: {result[0]}")
            return result[0]
        else:
            logger.error(f"Repo {owner}/{name} not found in database")
            raise ValueError(f"Repo not found: {owner}/{name}")
            
    except Exception as e:
        logger.error(f"Failed to get repo_id: {str(e)}")
        raise
    finally:
        if con:
            con.close()

@task
def save_prs_to_db(prs: List[Dict], repo_id: int, owner: str, name: str):
    """Saving pull requests to DuckDB."""
    con = None
    try:
        if not prs:
            logger.warning(f"No PRs to save for {owner}/{name}")
            return
        
        logger.info(f"Saving {len(prs)} PRs to database for {owner}/{name}")
        con = duckdb.connect(str(DB_PATH))
        
        # Preparing data
        rows = []
        skipped = 0
        for pr in prs:
            try:
                row = {
                    "pr_id": pr["id"],
                    "repo_id": repo_id,
                    "number": pr["number"],
                    "author_login": pr.get("user", {}).get("login") if pr.get("user") else None,
                    "created_at": pr.get("created_at"),
                    "closed_at": pr.get("closed_at"),
                    "merged_at": pr.get("merged_at"),
                    "state": pr.get("state", ""),
                    "title": pr.get("title", "")
                }
                rows.append(row)
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping malformed PR: {str(e)}")
                continue
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} malformed PRs")
        
        # Inserting into database
        inserted = 0
        duplicates = 0
        for row in rows:
            try:
                con.execute("""
                    INSERT INTO pull_requests (pr_id, repo_id, number, author_login, 
                                              created_at, closed_at, merged_at, state, title)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row["pr_id"], row["repo_id"], row["number"], row["author_login"],
                    row["created_at"], row["closed_at"], row["merged_at"], 
                    row["state"], row["title"]
                ])
                inserted += 1
            except Exception:
                # Skipping duplicates
                duplicates += 1
                continue
        
        logger.info(f"Saved {inserted} PRs for {owner}/{name} (skipped {duplicates} duplicates)")
        
    except Exception as e:
        logger.error(f"Failed to save PRs: {str(e)}")
        raise
    finally:
        if con:
            con.close()

# Flow definition
@flow(name="Ingest GitHub Pull Requests")
def ingest_prs_flow(max_pages: int = 200):
    """Main flow to ingest PRs from all repos."""
    total_prs = 0
    
    try:
        repos = load_repos()
        logger.info(f"Starting PR ingestion for {len(repos)} repositories")
        logger.info(f"Max pages per repo: {max_pages}")
        
        for repo in repos:
            owner = repo["owner"]
            name = repo["name"]
            positioning = repo["positioning"]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing PRs for {owner}/{name} ({positioning})")
            logger.info(f"{'='*60}")
            
            # Getting repo_id
            repo_id = get_repo_id(owner, name)
            
            # Fetching PRs
            prs = fetch_repo_prs(owner, name, max_pages)
            total_prs += len(prs)
            
            # Saving to database
            save_prs_to_db(prs, repo_id, owner, name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PR Ingestion Complete")
        logger.info(f"Total repos processed: {len(repos)}")
        logger.info(f"Total PRs fetched: {total_prs}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"PR Ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Fetching 200 pages of PRs per repo
    ingest_prs_flow(max_pages=200)