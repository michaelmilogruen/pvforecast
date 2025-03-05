import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from pathlib import Path

def load_data(parquet_path):
    """Load data from parquet file and extract relevant columns."""
    print("Loading data from parquet file...")
    df = pd.read_parquet(parquet_path)
    
    # Extract radiation and temperature data
    radiation_df = df[['GL [W m-2]', 'cglo']]
    temperature_df = df[['T2M [degree_Celsius]', 'tl']]
    
    print("\nData shape:", df.shape)
    print("\nSample of radiation data:")
    print(radiation_df.head())
    print("\nSample of temperature data:")
    print(temperature_df.head())
    
    return radiation_df, temperature_df

def calculate_correlations(data1, data2, label1, label2):
    """Calculate Pearson and Spearman correlations."""
    pearson_corr, pearson_p = stats.pearsonr(data1, data2)
    spearman_corr, spearman_p = stats.spearmanr(data1, data2)
    
    print(f"\nCorrelation Analysis for {label1} vs {label2}:")
    print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3e})")
    print(f"Spearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3e})")
    
    return {
        'pearson': (pearson_corr, pearson_p),
        'spearman': (spearman_corr, spearman_p)
    }

def create_correlation_plot(data, x_col, y_col, title, save_path):
    """Create and save correlation plot with regression line."""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with regression line
    sns.regplot(data=data, x=x_col, y=y_col, scatter_kws={'alpha':0.5})
    plt.title(f'{title} Correlation Plot')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    # Add correlation coefficients
    pearson_corr = data[x_col].corr(data[y_col])
    spearman_corr = stats.spearmanr(data[x_col], data[y_col])[0]
    
    plt.text(0.05, 0.95, f'Pearson correlation: {pearson_corr:.3f}\n'
             f'Spearman correlation: {spearman_corr:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_time_series_plot(radiation_df, temperature_df, save_path):
    """Create and save time series comparison plot."""
    plt.figure(figsize=(15, 10))
    
    # Plot radiation data
    plt.subplot(2, 1, 1)
    radiation_df.plot(ax=plt.gca())
    plt.title('Global Radiation Comparison')
    plt.legend(['GL [W m-2]', 'cglo'])
    
    # Plot temperature data
    plt.subplot(2, 1, 2)
    temperature_df.plot(ax=plt.gca())
    plt.title('Temperature Comparison')
    plt.legend(['T2M [degree_Celsius]', 'tl'])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_correlation_heatmap(df, save_path):
    """Create and save correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_distributions(df, save_dir):
    """Analyze and plot distributions of variables."""
    for col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(save_dir / f'distribution_{col.replace(" ", "_").replace("[", "").replace("]", "")}.png')
        plt.close()
        
        # Calculate basic statistics
        stats_dict = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        print(f"\nStatistics for {col}:")
        for stat, value in stats_dict.items():
            print(f"{stat}: {value:.3f}")

def main():
    # Create directories for saving plots
    vis_dir = Path('visualizations/correlation_analysis')
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    radiation_df, temperature_df = load_data('data/processed_training_data.parquet')
    
    # Calculate correlations
    rad_corr = calculate_correlations(
        radiation_df['GL [W m-2]'], radiation_df['cglo'],
        'GL [W m-2]', 'cglo'
    )
    temp_corr = calculate_correlations(
        temperature_df['T2M [degree_Celsius]'], temperature_df['tl'],
        'T2M [degree_Celsius]', 'tl'
    )
    
    # Create correlation plots
    create_correlation_plot(
        radiation_df, 'GL [W m-2]', 'cglo',
        'Global Radiation',
        vis_dir / 'radiation_correlation.png'
    )
    create_correlation_plot(
        temperature_df, 'T2M [degree_Celsius]', 'tl',
        'Temperature',
        vis_dir / 'temperature_correlation.png'
    )
    
    # Create time series plot
    create_time_series_plot(
        radiation_df, temperature_df,
        vis_dir / 'time_series_comparison.png'
    )
    
    # Create correlation heatmap
    combined_df = pd.concat([radiation_df, temperature_df], axis=1)
    create_correlation_heatmap(combined_df, vis_dir / 'correlation_heatmap.png')
    
    # Analyze distributions
    print("\nAnalyzing distributions...")
    analyze_distributions(combined_df, vis_dir)
    
    print(f"\nAnalysis complete. Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    main()