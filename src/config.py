"""Configuration loader for the project."""
import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

# Logging set up
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Loading environment variables
load_dotenv()
logger.info("Environment variables loaded")

# GitHub API configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    logger.error("GITHUB_TOKEN not found in environment variables")
    raise ValueError("GITHUB_TOKEN must be set in .env file")

API_BASE_URL = "https://api.github.com"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

def load_repos():
    """Loading repository configuration from YAML file."""
    try:
        config_path = CONFIG_DIR / "repos.yaml"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"repos.yaml not found at {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if not config or "repos" not in config:
            logger.error("Invalid config format: 'repos' key not found")
            raise ValueError("Config file must contain 'repos' key")
        
        repos = config["repos"]
        logger.info(f"Successfully loaded {len(repos)} repositories from config")
        return repos
    
    except Exception as e:
        logger.error(f"Failed to load repos config: {str(e)}")
        raise