
# Creating visualizations for hypothesis testing results

# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

# Creating logger
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Creating log filename with timestamp
log_filename = LOGS_DIR / f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Setting style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Defining consistent color scheme for all visualizations
COLORS = {
    'performance': '#2E86AB',  
    'simplicity': '#A23B72'   
}


def load_data():
    """
    Loading processed metrics and hypothesis results for visualization.
    """
    try:
        metrics_path = DATA_DIR / "repo_metrics.csv"
        temporal_path = DATA_DIR / "temporal_data.csv"
        results_path = DATA_DIR / "hypothesis_summary.csv"
        
        if not metrics_path.exists():
            logger.error(f"Metrics file not found: {metrics_path}")
            raise FileNotFoundError(f"Run transformations.py first")
        
        metrics_df = pd.read_csv(metrics_path)
        temporal_df = pd.read_csv(temporal_path)
        results_df = pd.read_csv(results_path)
        
        logger.info(f"Loaded data: {len(metrics_df)} repos, {len(temporal_df)} temporal records")
        
        return metrics_df, temporal_df, results_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def plot_sentiment(metrics_df, results_df):
    """
    Creating bar chart for sentiment analysis showing mean sentiment by repository type with error bars.
    """
    try:
        logger.info("Creating sentiment visualization")
        
        # Calculating group statistics for performance and simplicity repos
        performance = metrics_df[metrics_df['positioning'] == 'performance']
        simplicity = metrics_df[metrics_df['positioning'] == 'simplicity']
        
        perf_mean = performance['avg_sentiment'].mean()
        simp_mean = simplicity['avg_sentiment'].mean()
        perf_std = performance['avg_sentiment'].std()
        simp_std = simplicity['avg_sentiment'].std()
        
        # Getting statistical results from hypothesis test
        h1_results = results_df[results_df['hypothesis'] == 'H1_Sentiment'].iloc[0]
        p_value = h1_results['p_value']
        cohens_d = h1_results['cohens_d']
        
        # Creating figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Creating bars with error bars showing standard deviation
        x = [0, 1]
        means = [perf_mean, simp_mean]
        stds = [perf_std, simp_std]
        labels = ['Performance', 'Simplicity']
        colors = [COLORS['performance'], COLORS['simplicity']]
        
        bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Adding horizontal line at y = 0 to show neutral sentiment
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Adding value labels on top of bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.001,
                   f'{mean:.4f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Formatting plot
        ax.set_ylabel('Average Sentiment Score', fontsize=14, fontweight='bold')
        ax.set_title('Commit Message Sentiment by Repository Type\n' + 
                    f'p={p_value:.3f} (not significant), Cohen\'s d={cohens_d:.2f}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylim(bottom=min(means) - max(stds) - 0.005, 
                   top=max(means) + max(stds) + 0.005)
        
        # Adding grid for readability
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Saving figure
        output_path = OUTPUT_DIR / "h1_sentiment.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sentiment plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating sentiment plot: {str(e)}")
        raise


def plot_gini(metrics_df, results_df):
    """
    Creating bar chart with individual repo points overlaid to show contributor concentration.
    """
    try:
        logger.info("Creating Gini coefficient visualization")
        
        # Calculating group statistics
        performance = metrics_df[metrics_df['positioning'] == 'performance']
        simplicity = metrics_df[metrics_df['positioning'] == 'simplicity']
        
        perf_mean = performance['gini_coefficient'].mean()
        simp_mean = simplicity['gini_coefficient'].mean()
        
        # Getting statistical results
        h2_results = results_df[results_df['hypothesis'] == 'H2_Gini'].iloc[0]
        p_value = h2_results['p_value']
        cohens_d = h2_results['cohens_d']
        
        # Creating figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Creating bars for group averages
        x = [0, 1]
        means = [perf_mean, simp_mean]
        labels = ['Performance', 'Simplicity']
        colors = [COLORS['performance'], COLORS['simplicity']]
        
        bars = ax.bar(x, means, color=colors, alpha=0.3, 
                     edgecolor='black', linewidth=2)
        
        # Overlaying individual repo points with slight jitter for visibility
        np.random.seed(42) 
        jitter_strength = 0.05
        
        for i, (repos, color, x_pos) in enumerate([(performance, COLORS['performance'], 0), 
                                                     (simplicity, COLORS['simplicity'], 1)]):
            x_jitter = np.random.normal(x_pos, jitter_strength, len(repos))
            ax.scatter(x_jitter, repos['gini_coefficient'], 
                      color=color, s=200, alpha=0.8, 
                      edgecolor='black', linewidth=1.5, zorder=3)
            
            # Adding repo name labels 
            for j, (idx, row) in enumerate(repos.iterrows()):
                ax.annotate(row['repo_name'], 
                           xy=(x_jitter[j], row['gini_coefficient']),
                           xytext=(8, 0), textcoords='offset points',
                           fontsize=9, va='center')
        
        # Adding reference line at 0.5 showing moderate concentration
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, 
                  alpha=0.5, label='Moderate concentration')
        
        # Formatting plot
        ax.set_ylabel('Gini Coefficient', fontsize=14, fontweight='bold')
        ax.set_title('Contributor Concentration by Repository Type\n' + 
                    f'p={p_value:.3f} (not significant), Cohen\'s d={cohens_d:.2f}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylim(0.7, 1.0)
        ax.legend(loc='lower right')
        
        # Adding grid for readability
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Saving figure
        output_path = OUTPUT_DIR / "h2_gini.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Gini plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating Gini plot: {str(e)}")
        raise


def plot_weekend_activity(metrics_df, results_df):
    """
    Creating stacked bar chart showing weekday vs weekend activity proportions.
    """
    try:
        logger.info("Creating weekend activity visualization")
        
        # Calculating group statistics
        performance = metrics_df[metrics_df['positioning'] == 'performance']
        simplicity = metrics_df[metrics_df['positioning'] == 'simplicity']
        
        perf_weekend = performance['weekend_proportion'].mean()
        simp_weekend = simplicity['weekend_proportion'].mean()
        
        perf_weekday = 1 - perf_weekend
        simp_weekday = 1 - simp_weekend
        
        # Getting statistical results
        h3_results = results_df[results_df['hypothesis'] == 'H3_Weekend'].iloc[0]
        p_value = h3_results['p_value']
        cohens_d = h3_results['cohens_d']
        
        # Creating figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data for stacked bars
        x = [0, 1]
        weekday_vals = [perf_weekday, simp_weekday]
        weekend_vals = [perf_weekend, simp_weekend]
        labels = ['Performance', 'Simplicity']
        
        # Creating stacked bars
        bar_width = 0.6
        bars1 = ax.bar(x, weekday_vals, bar_width, label='Weekday', 
                      color='#95A3A4', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, weekend_vals, bar_width, bottom=weekday_vals, 
                      label='Weekend', color='#F39C12', alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        
        # Adding percentage labels on each section
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Weekday label
            ax.text(bar1.get_x() + bar1.get_width()/2., 
                   weekday_vals[i]/2,
                   f'{weekday_vals[i]*100:.1f}%',
                   ha='center', va='center', fontweight='bold', fontsize=11)
            
            # Weekend label
            ax.text(bar2.get_x() + bar2.get_width()/2., 
                   weekday_vals[i] + weekend_vals[i]/2,
                   f'{weekend_vals[i]*100:.1f}%',
                   ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Formatting plot
        ax.set_ylabel('Proportion of Activity', fontsize=14, fontweight='bold')
        ax.set_title('Weekend vs Weekday Activity by Repository Type\n' + 
                    f'p={p_value:.3f} (not significant), Cohen\'s d={cohens_d:.2f}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(False)
        
        plt.tight_layout()
        
        # Saving figure
        output_path = OUTPUT_DIR / "h3_weekend.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved weekend activity plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating weekend activity plot: {str(e)}")
        raise


def plot_temporal_trends(metrics_df, temporal_df, results_df):
    """
    Creating line plot showing formalization trends over time for both repository types.
    """
    try:
        logger.info("Creating temporal trends visualization")
        
        # Calculating formalization ratio for each year and repo
        temporal_df['formalization_ratio'] = temporal_df['pr_count'] / (
            temporal_df['pr_count'] + temporal_df['commit_count']
        )
        
        # Getting statistical results
        h4_results = results_df[results_df['hypothesis'] == 'H4_Temporal'].iloc[0]
        p_value = h4_results['p_value']
        cohens_d = h4_results['cohens_d']
        perf_slope = h4_results['performance_mean']
        simp_slope = h4_results['simplicity_mean']
        
        # Creating figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plotting individual repo lines
        for repo_id in temporal_df['repo_id'].unique():
            repo_data = temporal_df[temporal_df['repo_id'] == repo_id].sort_values('year')
            positioning = repo_data['positioning'].iloc[0]
            color = COLORS[positioning]
            
            ax.plot(repo_data['year'], repo_data['formalization_ratio'],
                   color=color, alpha=0.2, linewidth=1)
        
        # Calculating and plotting group averages
        for positioning, color in COLORS.items():
            group_data = temporal_df[temporal_df['positioning'] == positioning]
            yearly_avg = group_data.groupby('year')['formalization_ratio'].mean().reset_index()
            
            ax.plot(yearly_avg['year'], yearly_avg['formalization_ratio'],
                   color=color, linewidth=3, label=positioning.capitalize(),
                   marker='o', markersize=6)
        
        # Adding trend line slopes as annotations
        ax.text(0.02, 0.98, 
               f'Performance slope: {perf_slope:.4f}\nSimplicity slope: {simp_slope:.4f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting plot
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Formalization Ratio\n(PRs / Total Activity)', 
                     fontsize=14, fontweight='bold')
        ax.set_title('Formalization Trends Over Time by Repository Type\n' + 
                    f'p={p_value:.3f} (not significant), Cohen\'s d={cohens_d:.2f}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=12)
        ax.grid(axis='both', alpha=0.3)
        
        # Setting y-axis to show proportions clearly
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Saving figure
        output_path = OUTPUT_DIR / "h4_temporal.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal trends plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating temporal trends plot: {str(e)}")
        raise


def create_all_visualizations():
    """
    Creating all four hypothesis visualizations.
    """
    try:
        logger.info("Starting visualization creation")
        
        # Loading data
        metrics_df, temporal_df, results_df = load_data()
        
        # Creating each visualization
        plot_sentiment(metrics_df, results_df)
        plot_gini(metrics_df, results_df)
        plot_weekend_activity(metrics_df, results_df)
        plot_temporal_trends(metrics_df, temporal_df, results_df)
        
        logger.info(f"All visualizations saved to {OUTPUT_DIR}")
        logger.info("Visualization creation complete")
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {str(e)}")
        raise


def main():
    """
    Main function to generate all visualizations.
    """
    try:
        logger.info("Starting visualization pipeline")
        create_all_visualizations()
        logger.info("Visualization pipeline complete")
        
    except Exception as e:
        logger.error(f"Visualization pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()