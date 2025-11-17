#Data transformations for GitHub repository analysis.

# Importing necessary libraries

import duckdb
import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import linregress
import logging
from pathlib import Path
import random

random.seed(42)  
np.random.seed(42)

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "github_evolution.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

def drop_duplicates_and_nulls():
    """
    Removing duplicate records and handle null values before analysis
    in commits and pull_requests tables.
    """
    logger.info("Starting data cleaning: removing duplicates and nulls")
    
    con = None
    try:
        con = duckdb.connect(str(DB_PATH))
        
        # Checking for duplicate commits
        dup_commits = con.execute("""
            SELECT COUNT(*) - COUNT(DISTINCT sha) as duplicates 
            FROM commits
        """).fetchone()[0]
        logger.info(f"Found {dup_commits} duplicate commits")
        
        # Checking for duplicate PRs
        dup_prs = con.execute("""
            SELECT COUNT(*) - COUNT(DISTINCT pr_id) as duplicates 
            FROM pull_requests
        """).fetchone()[0]
        logger.info(f"Found {dup_prs} duplicate PRs")
        
        # Counting all missing values in critical fields 
        null_dates = con.execute("""
            SELECT COUNT(*) FROM commits WHERE author_date IS NULL
        """).fetchone()[0]
        logger.info(f"Found {null_dates} commits with null dates")
        
        null_messages = con.execute("""
            SELECT COUNT(*) FROM commits WHERE message IS NULL OR message = ''
        """).fetchone()[0]
        logger.info(f"Found {null_messages} commits with null messages")
        
        logger.info("Data cleaning check complete")
        return dup_commits, dup_prs, null_dates, null_messages
        
    except Exception as e:
        logger.error(f"Error during data cleaning check: {str(e)}")
        raise
    finally:
        if con:
            con.close()


def calculate_sentiment_scores():
    """
    Calculating sentiment scores for commit messages, returns average sentiment per repository
    between -1 (negative) to +1 (positive).
    """
    logger.info("Calculating sentiment scores")
    
    con = None
    try:
        con = duckdb.connect(str(DB_PATH))
        
        # Fetching commits with valid messages
        # Excluding nulls and messages shorter than 10 characters
        query = """
            SELECT c.sha, c.repo_id, c.message, r.name, r.positioning
            FROM commits c
            JOIN repos r ON c.repo_id = r.repo_id
            WHERE c.message IS NOT NULL 
            AND LENGTH(c.message) > 10
        """
        
        df = con.execute(query).df()
        logger.info(f"Loaded {len(df)} commits for sentiment analysis")
        
        if len(df) == 0:
            logger.error("No valid commits found for sentiment analysis")
            raise ValueError("No commits with valid messages found")
        
        # Sampling commits with n = 5000 for efficiency and to ensure each repo is represented proportionally

        sample_size = min(5000, len(df))
        df_sample = df.groupby('repo_id', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // df['repo_id'].nunique()))
        )
        logger.info(f"Analyzing sentiment for {len(df_sample)} sampled commits")
        
        # Calculate sentiment using TextBlob: -1 (negative) to +1 (positive)

        def get_sentiment(text):
            try:
                return TextBlob(text).sentiment.polarity
            except:
                # Returns neutral for any parsing errors
                return 0.0
        
        df_sample['sentiment_score'] = df_sample['message'].apply(get_sentiment)
        
        # Calculate average sentiment per repository
        sentiment_by_repo = df_sample.groupby(['repo_id', 'name', 'positioning'])['sentiment_score'].mean().reset_index()
        sentiment_by_repo.columns = ['repo_id', 'repo_name', 'positioning', 'avg_sentiment']
        
        logger.info("Sentiment calculation complete")
        logger.info(f"\n{sentiment_by_repo}")
        
        return sentiment_by_repo
        
    except Exception as e:
        logger.error(f"Error calculating sentiment scores: {str(e)}")
        raise
    finally:
        if con:
            con.close()


def calculate_gini_coefficient(commit_counts):
    """
    Calculating Gini coefficient to measure contributor concentration from 0 (perfect equality) to 1 (perfect inequality).
    Higher values mean more concentrated contributions with fewer contributors.
    """
    try:
        if len(commit_counts) == 0:
            return None
        
        commit_counts = sorted(commit_counts)
        n = len(commit_counts)
        
        # Calculating Gini coefficient
        cumsum = 0
        for i, count in enumerate(commit_counts):
            cumsum += (2 * (i + 1) - n - 1) * count
        
        return cumsum / (n * sum(commit_counts))
    except Exception as e:
        logger.error(f"Error calculating Gini coefficient: {str(e)}")
        return None


def calculate_contributor_concentration():
    """
    Calculating Gini coefficient for each repository to measure contributor concentration.
    """
    logger.info("Calculating contributor concentration (Gini coefficient)")
    
    con = None
    try:
        con = duckdb.connect(str(DB_PATH))
        
        # Getting commit counts per contributor per repository
        # Excluding null authors since we cannot measure their contribution
        query = """
            SELECT c.repo_id, r.name, r.positioning, c.author_login, COUNT(*) as commit_count
            FROM commits c
            JOIN repos r ON c.repo_id = r.repo_id
            WHERE c.author_login IS NOT NULL
            GROUP BY c.repo_id, r.name, r.positioning, c.author_login
        """
        
        df = con.execute(query).df()
        logger.info(f"Loaded contributor data for {df['repo_id'].nunique()} repositories")
        
        if len(df) == 0:
            logger.error("No contributor data found")
            raise ValueError("No contributors with valid data found")
        
        # Calculating Gini coefficient for each repository
        gini_results = []
        for repo_id in df['repo_id'].unique():
            try:
                repo_data = df[df['repo_id'] == repo_id]
                repo_name = repo_data['name'].iloc[0]
                positioning = repo_data['positioning'].iloc[0]
                commit_counts = repo_data['commit_count'].tolist()
                
                gini = calculate_gini_coefficient(commit_counts)
                
                gini_results.append({
                    'repo_id': repo_id,
                    'repo_name': repo_name,
                    'positioning': positioning,
                    'gini_coefficient': gini,
                    'num_contributors': len(commit_counts)
                })
            except Exception as e:
                logger.warning(f"Error calculating Gini for repo {repo_id}: {str(e)}")
                continue
        
        gini_df = pd.DataFrame(gini_results)
        logger.info("Gini coefficient calculation complete")
        logger.info(f"\n{gini_df}")
        
        return gini_df
        
    except Exception as e:
        logger.error(f"Error calculating contributor concentration: {str(e)}")
        raise
    finally:
        if con:
            con.close()


def calculate_weekend_activity():
    """
    Calculating proportion of weekend activity (commits and PRs on Saturday = 7, Sunday = 1)
    """
    logger.info("Calculating weekend activity proportions")
    
    con = None
    try:
        con = duckdb.connect(str(DB_PATH))
        
        # Getting day of week for commits
        commits_query = """
            SELECT c.repo_id, r.name, r.positioning,
                   DAYOFWEEK(c.author_date) as day_of_week,
                   COUNT(*) as count
            FROM commits c
            JOIN repos r ON c.repo_id = r.repo_id
            WHERE c.author_date IS NOT NULL
            GROUP BY c.repo_id, r.name, r.positioning, day_of_week
        """
        
        commits_df = con.execute(commits_query).df()
        
        # Getting day of week for PRs
        prs_query = """
            SELECT p.repo_id, r.name, r.positioning,
                   DAYOFWEEK(p.created_at) as day_of_week,
                   COUNT(*) as count
            FROM pull_requests p
            JOIN repos r ON p.repo_id = r.repo_id
            WHERE p.created_at IS NOT NULL
            GROUP BY p.repo_id, r.name, r.positioning, day_of_week
        """
        
        prs_df = con.execute(prs_query).df()
        
        # Combining commits and PRs for total activity
        combined_df = pd.concat([commits_df, prs_df])
        combined_df = combined_df.groupby(['repo_id', 'name', 'positioning', 'day_of_week'])['count'].sum().reset_index()
        
        if len(combined_df) == 0:
            logger.error("No activity data found for weekend calculation")
            raise ValueError("No activity data with valid dates found")
        
        # Calculating weekend proportion for each repository
        weekend_results = []
        for repo_id in combined_df['repo_id'].unique():
            try:
                repo_data = combined_df[combined_df['repo_id'] == repo_id]
                repo_name = repo_data['name'].iloc[0]
                positioning = repo_data['positioning'].iloc[0]
                
                weekend_count = repo_data[repo_data['day_of_week'].isin([1, 7])]['count'].sum()
                total_count = repo_data['count'].sum()
                weekend_proportion = weekend_count / total_count if total_count > 0 else 0
                
                weekend_results.append({
                    'repo_id': repo_id,
                    'repo_name': repo_name,
                    'positioning': positioning,
                    'weekend_proportion': weekend_proportion,
                    'weekend_count': weekend_count,
                    'total_count': total_count
                })
            except Exception as e:
                logger.warning(f"Error calculating weekend activity for repo {repo_id}: {str(e)}")
                continue
        
        weekend_df = pd.DataFrame(weekend_results)
        logger.info("Weekend activity calculation complete")
        logger.info(f"\n{weekend_df}")
        
        return weekend_df
        
    except Exception as e:
        logger.error(f"Error calculating weekend activity: {str(e)}")
        raise
    finally:
        if con:
            con.close()


def calculate_temporal_trends():
    """
    Calculating PR-to-commit ratio over time and trend slopes by aggregating by year and 
    calculating the ratio of PRs to commits.
    """
    logger.info("Calculating temporal trends")
    
    con = None
    try:
        con = duckdb.connect(str(DB_PATH))
        
        # Aggregating commits by year
        commits_query = """
            SELECT c.repo_id, r.name, r.positioning,
                   EXTRACT(YEAR FROM c.author_date) as year,
                   COUNT(*) as commit_count
            FROM commits c
            JOIN repos r ON c.repo_id = r.repo_id
            WHERE c.author_date IS NOT NULL
            GROUP BY c.repo_id, r.name, r.positioning, year
        """
        
        commits_df = con.execute(commits_query).df()
        
        # Aggregating PRs by year
        prs_query = """
            SELECT p.repo_id, r.name, r.positioning,
                   EXTRACT(YEAR FROM p.created_at) as year,
                   COUNT(*) as pr_count
            FROM pull_requests p
            JOIN repos r ON p.repo_id = r.repo_id
            WHERE p.created_at IS NOT NULL
            GROUP BY p.repo_id, r.name, r.positioning, year
        """
        
        prs_df = con.execute(prs_query).df()
        
        # Merging commits and PRs by year
        temporal_df = pd.merge(
            commits_df, 
            prs_df, 
            on=['repo_id', 'name', 'positioning', 'year'], 
            how='outer'
        ).fillna(0)
        
        if len(temporal_df) == 0:
            logger.error("No temporal data found")
            raise ValueError("No temporal data with valid dates found")
        
        # Calculating formalization ratio (how much work goes through formal review as opposed to direct commits) 
        # PRs / (PRs + Commits)

        temporal_df['formalization_ratio'] = temporal_df['pr_count'] / (temporal_df['pr_count'] + temporal_df['commit_count'])
        
        # Calculating trend slope for each repository using linear regression
        trend_results = []
        for repo_id in temporal_df['repo_id'].unique():
            try:
                repo_data = temporal_df[temporal_df['repo_id'] == repo_id].sort_values('year')
                repo_name = repo_data['name'].iloc[0]
                positioning = repo_data['positioning'].iloc[0]
                
                if len(repo_data) >= 2:
                    years = repo_data['year'].values
                    ratios = repo_data['formalization_ratio'].values
                    
                    # Calculating linear regression slope
                    slope, intercept, r_value, p_value, std_err = linregress(years, ratios)
                    
                    trend_results.append({
                        'repo_id': repo_id,
                        'repo_name': repo_name,
                        'positioning': positioning,
                        'trend_slope': slope,
                        'r_squared': r_value ** 2,
                        'num_years': len(repo_data)
                    })
                else:
                    logger.warning(f"Not enough data points for trend calculation for repo {repo_id}")
            except Exception as e:
                logger.warning(f"Error calculating trend for repo {repo_id}: {str(e)}")
                continue
        
        trend_df = pd.DataFrame(trend_results)
        logger.info("Temporal trend calculation complete")
        logger.info(f"\n{trend_df}")
        
        return trend_df, temporal_df
        
    except Exception as e:
        logger.error(f"Error calculating temporal trends: {str(e)}")
        raise
    finally:
        if con:
            con.close()


def create_summary_metrics():
    """
    Combining all metrics into a single summary table.
    """
    logger.info("Creating summary metrics table")
    
    try:
        # Calculating all metrics
        sentiment_df = calculate_sentiment_scores()
        gini_df = calculate_contributor_concentration()
        weekend_df = calculate_weekend_activity()
        trend_df, temporal_df = calculate_temporal_trends()
        
        # Merging all metrics
        summary = sentiment_df.copy()
        
        # Adding Gini coefficient
        summary = pd.merge(
            summary,
            gini_df[['repo_id', 'gini_coefficient', 'num_contributors']],
            on='repo_id',
            how='left'
        )
        
        # Adding weekend proportion
        summary = pd.merge(
            summary,
            weekend_df[['repo_id', 'weekend_proportion']],
            on='repo_id',
            how='left'
        )
        
        # Adding trend slope
        summary = pd.merge(
            summary,
            trend_df[['repo_id', 'trend_slope']],
            on='repo_id',
            how='left'
        )
        
        # Saving to CSV
        output_path = OUTPUT_DIR / "repo_metrics.csv"
        summary.to_csv(output_path, index=False)
        logger.info(f"Summary metrics saved to {output_path}")
        
        # Saving temporal data for visualization
        temporal_path = OUTPUT_DIR / "temporal_data.csv"
        temporal_df.to_csv(temporal_path, index=False)
        logger.info(f"Temporal data saved to {temporal_path}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating summary metrics: {str(e)}")
        raise


def main():
    """
    Function to run all transformations.
    """
    logger.info("Starting transformation pipeline")
    
    try:
        # Checking for duplicates and nulls
        drop_duplicates_and_nulls()
        
        # Creating summary metrics
        summary = create_summary_metrics()
        
        logger.info("\n" + "-"*80)
        logger.info("FINAL SUMMARY METRICS")
        logger.info("="*80)
        logger.info(f"\n{summary}")
        
        # Grouping by positioning for preview before hypothesis testing
        logger.info("\n" + "-"*80)
        logger.info("GROUPED BY POSITIONING")
        logger.info("="*80)
        
        performance = summary[summary['positioning'] == 'performance']
        simplicity = summary[summary['positioning'] == 'simplicity']
        
        logger.info("\nPerformance Repos:")
        logger.info(f"  Avg Sentiment: {performance['avg_sentiment'].mean():.4f}")
        logger.info(f"  Avg Gini: {performance['gini_coefficient'].mean():.4f}")
        logger.info(f"  Avg Weekend %: {performance['weekend_proportion'].mean():.4f}")
        logger.info(f"  Avg Trend Slope: {performance['trend_slope'].mean():.4f}")
        
        logger.info("\nSimplicity Repos:")
        logger.info(f"  Avg Sentiment: {simplicity['avg_sentiment'].mean():.4f}")
        logger.info(f"  Avg Gini: {simplicity['gini_coefficient'].mean():.4f}")
        logger.info(f"  Avg Weekend %: {simplicity['weekend_proportion'].mean():.4f}")
        logger.info(f"  Avg Trend Slope: {simplicity['trend_slope'].mean():.4f}")
        
        logger.info("\n" + "-"*80)
        logger.info("Transformation pipeline complete")
        
    except Exception as e:
        logger.error(f"Transformation pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()