# GitHub API client with rate limiting and error handling
import httpx
import logging
import time
from typing import Optional, Dict, List
from config import GITHUB_TOKEN, API_BASE_URL

# Defining the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GitHub API Client
class GitHubClient:
    """Client for interacting with GitHub API."""
    
    # Initializing with authentication token
    def __init__(self, token: str):
        self.token = token
        self.base_url = API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
    
    def _handle_rate_limit(self, response: httpx.Response):
        """Handle rate limiting"""
        remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
        
        if remaining == 0:
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            wait_time = max(reset_time - time.time(), 0) + 10  # Adding 10 second buffer
            logger.warning(f"Rate limit reached. Waiting {wait_time:.0f} seconds until reset.")
            time.sleep(wait_time)
    
    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Making GET request"""
        try:
            response = httpx.get(url, headers=self.headers, params=params, timeout=30.0)
            
            # Handle rate limiting
            self._handle_rate_limit(response)
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            return response.json()
        
        # Handling HTTP errors
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {url}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def fetch_commits(self, owner: str, repo: str, per_page: int = 100, max_pages: int = 10) -> List[Dict]:
        """Fetching commits for a repository."""
        logger.info(f"Fetching commits for {owner}/{repo}")
        
        commits = []
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        page = 1
        
        # Implementing pagination
        try:
            while page <= max_pages:
                params = {"per_page": per_page, "page": page}
                data = self._get(url, params)
                
                if not data:
                    break
                
                commits.extend(data)
                logger.info(f"  Fetched page {page}: {len(data)} commits")
                
                if len(data) < per_page:
                    break
                
                page += 1
            
            logger.info(f"Total commits fetched: {len(commits)}")
            return commits
        
        except Exception as e:
            logger.error(f"Failed to fetch commits: {str(e)}")
            raise
    
    def fetch_pull_requests(self, owner: str, repo: str, state: str = "all", per_page: int = 100, max_pages: int = 10) -> List[Dict]:
        """Fetch pull requests for a repository."""
        logger.info(f"Fetching pull requests for {owner}/{repo}")
        
        prs = []
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        page = 1
        
        # Implementing pagination
        try:
            while page <= max_pages:
                params = {"state": state, "per_page": per_page, "page": page}
                data = self._get(url, params)
                
                if not data:
                    break
                
                prs.extend(data)
                logger.info(f"  Fetched page {page}: {len(data)} PRs")
                
                if len(data) < per_page:
                    break
                
                page += 1
            
            logger.info(f"Total PRs fetched: {len(prs)}")
            return prs
        
        except Exception as e:
            logger.error(f"Failed to fetch PRs: {str(e)}")
            raise

# Create a singleton instance
client = GitHubClient(GITHUB_TOKEN)