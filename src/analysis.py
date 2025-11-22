# Statistical analysis and hypothesis testing

# Imports
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from pathlib import Path
import logging
from datetime import datetime

# Creating logger object
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Creating log filename with timestamp
log_filename = LOGS_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configuring logging to both file and console
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

# Defining paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR

# Statistical functions

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size to measure magnitude of difference between groups regardless of p-value.
    """
    try:
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            logger.warning("Insufficient data for Cohen's d calculation")
            return None
        
        # Calculating pooled standard deviation using degrees of freedom correction
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            logger.warning("Pooled standard deviation is zero, cannot calculate Cohen's d")
            return None
        
        # Calculating standardized mean difference
        return (np.mean(group1) - np.mean(group2)) / pooled_std
        
    except Exception as e:
        logger.error(f"Error calculating Cohen's d: {str(e)}")
        return None


def confidence_interval(group1, group2, confidence=0.95):
    """
    Calculating confidence interval for difference in means of the performance and simplicity groups.
    """
    try:
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            logger.warning("Insufficient data for confidence interval calculation")
            return None, None
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Calculating standard error of the difference
        se_diff = np.sqrt(var1/n1 + var2/n2)
        
        # Calculating degrees of freedom using Welch-Satterthwaite equation, accounting for potentially unequal variances
        df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Calculating t-critical value for confidence level
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # Calculating confidence interval
        diff = mean1 - mean2
        margin = t_crit * se_diff
        
        return diff - margin, diff + margin
        
    except Exception as e:
        logger.error(f"Error calculating confidence interval: {str(e)}")
        return None, None


def interpret_cohens_d(d):
    """
    Providing interpretation of Cohen's d effect size.
    """
    if d is None:
        return "unable to calculate"
    
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def test_hypothesis(group1, group2, group1_name, group2_name, metric_name, hypothesis_direction):
    """
    Running complete hypothesis test with t-test, effect size, and confidence interval.
    """
    try:
        logger.info(f"Testing {metric_name}: {group1_name} vs {group2_name}")
        
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        # Calculating descriptive statistics
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)
        
        # Running two-sample t-test to detect differences in either direction
        t_stat, p_value = ttest_ind(group1, group2)
        
        # Calculating effect size
        d = cohens_d(group1, group2)
        d_interpretation = interpret_cohens_d(d)
        
        # Calculating confidence interval
        ci_lower, ci_upper = confidence_interval(group1, group2)
        
        # Determining if the result is statistically significant when alpha = 0.05
        is_significant = p_value < 0.05
        
        # Determining if the direction matches the null hypothesis
        if hypothesis_direction == 'greater':
            matches_hypothesis = mean1 > mean2
        elif hypothesis_direction == 'less':
            matches_hypothesis = mean1 < mean2
        else:
            matches_hypothesis = True 
        
        results = {
            'metric': metric_name,
            'group1_name': group1_name,
            'group2_name': group2_name,
            'n1': len(group1),
            'n2': len(group2),
            'mean1': mean1,
            'mean2': mean2,
            'std1': std1,
            'std2': std2,
            'difference': mean1 - mean2,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'effect_size_interpretation': d_interpretation,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_significant': is_significant,
            'matches_hypothesis': matches_hypothesis
        }
        
        logger.info(f"  Mean difference: {mean1 - mean2:.6f}")
        logger.info(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        logger.info(f"  Cohen's d: {d:.4f} ({d_interpretation})")
        logger.info(f"  Significant: {is_significant}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in hypothesis test for {metric_name}: {str(e)}")
        raise


def load_repo_metrics():
    """
    Loading processed repository metrics from CSV file
    """
    try:
        metrics_path = DATA_DIR / "repo_metrics.csv"
        
        if not metrics_path.exists():
            logger.error(f"Metrics file not found: {metrics_path}")
            raise FileNotFoundError(f"Cannot find {metrics_path}. Run transformations.py first.")
        
        df = pd.read_csv(metrics_path)
        logger.info(f"Loaded metrics for {len(df)} repositories")
        
        # Verify required columns exist
        required_cols = ['repo_name', 'positioning', 'avg_sentiment', 'gini_coefficient', 
                        'weekend_proportion', 'trend_slope']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading repo metrics: {str(e)}")
        raise


def run_all_hypothesis_tests():
    """
    Executing all four hypothesis tests and compiling results.
    """
    try:
        logger.info("Starting hypothesis testing")
        
        # Loading data
        df = load_repo_metrics()
        
        # Splitting into performance and simplicity groups
        performance = df[df['positioning'] == 'performance']
        simplicity = df[df['positioning'] == 'simplicity']
        
        logger.info(f"Performance repos: {list(performance['repo_name'])}")
        logger.info(f"Simplicity repos: {list(simplicity['repo_name'])}")
        
        all_results = {}
        
        # Sentiment Analysis
        # Hypothesis: Performance repos have LOWER sentiment than simplicity repos
        logger.info("\n" + "-"*80)
        logger.info("H1: SENTIMENT ANALYSIS")
        logger.info("-"*80)
        
        h1_results = test_hypothesis(
            group1=performance['avg_sentiment'].values,
            group2=simplicity['avg_sentiment'].values,
            group1_name='Performance',
            group2_name='Simplicity',
            metric_name='Average Sentiment',
            hypothesis_direction='less'
        )
        all_results['H1_Sentiment'] = h1_results
        
        # Contributor Concentration 
        # Hypothesis: Performance repos have a higher Gini than simplicity repos for contributor concentration
        logger.info("\n" + "-"*80)
        logger.info("H2: CONTRIBUTOR CONCENTRATION (GINI COEFFICIENT)")
        logger.info("-"*80)
        
        h2_results = test_hypothesis(
            group1=performance['gini_coefficient'].values,
            group2=simplicity['gini_coefficient'].values,
            group1_name='Performance',
            group2_name='Simplicity',
            metric_name='Gini Coefficient',
            hypothesis_direction='greater'
        )
        all_results['H2_Gini'] = h2_results
        
        # Weekend Activity
        # Hypothesis: Simplicity repos have a higher weekend activity proportion (weekend activity/total activity) than performance repos
   
        logger.info("\n" + "-"*80)
        logger.info("H3: WEEKEND ACTIVITY")
        logger.info("-"*80)
        
        h3_results = test_hypothesis(
            group1=performance['weekend_proportion'].values,
            group2=simplicity['weekend_proportion'].values,
            group1_name='Performance',
            group2_name='Simplicity',
            metric_name='Weekend Proportion',
            hypothesis_direction='less'
        )
        all_results['H3_Weekend'] = h3_results
        
        # Temporal Trend (Formalization)
        # Hypothesis: Performance repos have an increasing trend in PRs over time (positive slope), unlike simplicity repos (negative slope)

        logger.info("\n" + "-"*80)
        logger.info("H4: FORMALIZATION TREND")
        logger.info("-"*80)
        
        h4_results = test_hypothesis(
            group1=performance['trend_slope'].values,
            group2=simplicity['trend_slope'].values,
            group1_name='Performance',
            group2_name='Simplicity',
            metric_name='Formalization Trend Slope',
            hypothesis_direction='greater'
        )
        all_results['H4_Temporal'] = h4_results
        
        logger.info("\n" + "-"*80)
        logger.info("All hypothesis tests complete")
        logger.info("-"*80)
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error running hypothesis tests: {str(e)}")
        raise

def save_results(results):
    """
    Saving results to CSV and text formats
    """
    try:
        
        # Saving summary table as CSV
        summary_data = []
        for hyp_name, result in results.items():
            summary_data.append({
                'hypothesis': hyp_name,
                'metric': result['metric'],
                'performance_mean': result['mean1'],
                'simplicity_mean': result['mean2'],
                'difference': result['difference'],
                't_statistic': result['t_statistic'],
                'p_value': result['p_value'],
                'cohens_d': result['cohens_d'],
                'effect_size': result['effect_size_interpretation'],
                'significant': result['is_significant'],
                'matches_hypothesis': result['matches_hypothesis']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = OUTPUT_DIR / "hypothesis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary CSV to {summary_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def main():
    """
    Main function to run complete statistical analysis pipeline.
    """
    try:
        logger.info("Starting statistical analysis pipeline")
        
        # Run all hypothesis tests
        results = run_all_hypothesis_tests()
        
        # Save results
        save_results(results)
        
        logger.info("Statistical analysis completed")
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()