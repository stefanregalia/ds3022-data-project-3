# Prefect flow to ingest commits from GitHub to DuckDB

# Importing libraries
import duckdb
import logging
from prefect import flow, task
from datetime import datetime
from typing import List, Dict
from config import load_repos, DATA_DIR
from github_client import client
import sys


# Defining the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Adding stream handler to output logs to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

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
def fetch_repo_commits(owner: str, name: str, max_pages: int = 10) -> List[Dict]:
    """Fetching commits for a single repository."""
    try:
        commits = client.fetch_commits(owner, name, per_page=100, max_pages=max_pages)
        return commits
    except Exception as e:
        logger.error(f"Failed to fetch commits for {owner}/{name}: {str(e)}")
        raise

@task
def insert_repo(owner: str, name: str, positioning: str) -> int:
    """Inserting or getting repo_id."""
    try:
        con = duckdb.connect(str(DB_PATH))
        
        # Checking if repo exists
        result = con.execute(
            "SELECT repo_id FROM repos WHERE owner = ? AND name = ?",
            [owner, name]
        ).fetchone()
        
        # If exists, return repo_id
        if result:
            repo_id = result[0]
            logger.info(f"Repo {owner}/{name} already exists with ID {repo_id}")
        else:
            # Get max repo_id and increment
            max_id = con.execute("SELECT COALESCE(MAX(repo_id), 0) FROM repos").fetchone()[0]
            repo_id = max_id + 1
            
            con.execute(
                "INSERT INTO repos (repo_id, owner, name, positioning) VALUES (?, ?, ?, ?)",
                [repo_id, owner, name, positioning]
            )
            logger.info(f"Inserted repo {owner}/{name} with ID {repo_id}")
        
        con.close()
        return repo_id
        
        # Handling exceptions
    except Exception as e:
        logger.error(f"Failed to insert repo: {str(e)}")
        raise

@task
def save_commits_to_db(commits: List[Dict], repo_id: int, owner: str, name: str):
    """Saving commits to DuckDB."""
    try:
        if not commits:
            logger.warning(f"No commits to save for {owner}/{name}")
            return
        
        con = duckdb.connect(str(DB_PATH))
        
        # Preparing data
        rows = []
        for commit in commits:
            try:
                commit_data = commit.get("commit", {})
                author = commit_data.get("author", {})
                stats = commit.get("stats", {})
                
                row = {
                    "sha": commit["sha"],
                    "repo_id": repo_id,
                    "author_login": commit.get("author", {}).get("login") if commit.get("author") else None,
                    "author_date": author.get("date"),
                    "commit_date": commit_data.get("committer", {}).get("date"),
                    "message": commit_data.get("message", ""),
                    "additions": stats.get("additions", 0),
                    "deletions": stats.get("deletions", 0),
                    "total_changes": stats.get("total", 0)
                }
                rows.append(row)
            except Exception as e:
                logger.warning(f"Skipping malformed commit: {str(e)}")
                continue
        
        # Inserting into database 
        for row in rows:
            try:
                con.execute("""
                    INSERT INTO commits (sha, repo_id, author_login, author_date, commit_date, 
                                       message, additions, deletions, total_changes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row["sha"], row["repo_id"], row["author_login"], 
                    row["author_date"], row["commit_date"], row["message"],
                    row["additions"], row["deletions"], row["total_changes"]
                ])
            except:
                # Skipping duplicates
                pass
        
        con.close()
        logger.info(f"Saved {len(rows)} commits for {owner}/{name}")
        
    except Exception as e:
        logger.error(f"Failed to save commits: {str(e)}")
        raise

@flow(name="Ingest GitHub Commits")
def ingest_commits_flow(max_pages: int = 200):
    """Flow to ingest commits from all repos."""
    try:
        repos = load_repos()
        print(f"Starting ingestion for {len(repos)} repositories")
        
        for repo in repos:
            owner = repo["owner"]
            name = repo["name"]
            positioning = repo["positioning"]
            
            print(f"\n{'='*60}")
            print(f"Processing {owner}/{name} ({positioning})")
            print(f"{'='*60}")
            
            # Inserting repo
            print(f"Task 1/3: Inserting repository metadata")
            repo_id = insert_repo(owner, name, positioning)
            
            # Fetching commits
            print(f"Task 2/3: Fetching commits from GitHub API")
            commits = fetch_repo_commits(owner, name, max_pages)
            
            # Saving to database
            print(f"Task 3/3: Saving commits to database")
            save_commits_to_db(commits, repo_id, owner, name)
        
        print(f"\n{'='*60}")
        print("Ingestion complete")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Ingesting 200 pages per repo
    ingest_commits_flow(max_pages=200)