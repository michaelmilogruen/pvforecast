import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import datetime # Import the datetime module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataAnalyzer:
    def __init__(self, high_res_path: str, low_res_path: str):
        """
        Initialize the Data Analyzer with paths to the processed data files.

        Args:
            high_res_path: Path to the high-resolution (10min) parquet file.
            low_res_path: Path to the low-resolution (1h) parquet file.
        """
        self.high_res_path = high_res_path
        self.low_res_path = low_res_path
        self.high_res_df = None
        self.low_res_df = None

    def load_data(self):
        """Load the processed high-resolution and low-resolution data from parquet files."""
        logging.info("Loading data from parquet files...")
        try:
            if os.path.exists(self.high_res_path):
                self.high_res_df = pd.read_parquet(self.high_res_path)
                logging.info(f"Successfully loaded high-resolution data: {self.high_res_df.shape}")
                logging.info(f"High-res data columns: {self.high_res_df.columns.tolist()}")
                logging.info(f"High-res data index timezone: {self.high_res_df.index.tz}")
            else:
                logging.error(f"High-resolution file not found: {self.high_res_path}")
                self.high_res_df = pd.DataFrame() # Assign empty DataFrame on failure

            if os.path.exists(self.low_res_path):
                self.low_res_df = pd.read_parquet(self.low_res_path)
                logging.info(f"Successfully loaded low-resolution data: {self.low_res_df.shape}")
                logging.info(f"Low-res data columns: {self.low_res_df.columns.tolist()}")
                logging.info(f"Low-res data index timezone: {self.low_res_df.index.tz}")
            else:
                logging.error(f"Low-resolution file not found: {self.low_res_path}")
                self.low_res_df = pd.DataFrame() # Assign empty DataFrame on failure

        except Exception as e:
            logging.error(f"Error loading data: {e}", exc_info=True)
            self.high_res_df = pd.DataFrame()
            self.low_res_df = pd.DataFrame()

    def perform_correlation_analysis(self, df: pd.DataFrame, method: str = 'spearman', target_variable: str = 'power_w', exclude_derived: list = None, title_suffix: str = "") -> pd.DataFrame:
        """
        Calculate the full correlation matrix and visualize the correlation of the target variable
        with other relevant numerical variables in a heatmap.

        Args:
            df: DataFrame to analyze.
            method: Correlation method ('pearson', 'kendall', 'spearman').
            target_variable: The name of the target variable for the specific heatmap plot.
            exclude_derived: List of variable names to exclude from the target correlation heatmap.
            title_suffix: Suffix for plot titles.

        Returns:
            pd.DataFrame: The calculated full correlation matrix.
        """
        if df.empty:
            logging.warning(f"DataFrame is empty. Skipping correlation analysis{title_suffix}.")
            return pd.DataFrame()

        logging.info(f"Performing {method} correlation analysis for all numerical columns{title_suffix}...")

        # Select only numerical columns for correlation
        numerical_df = df.select_dtypes(include=np.number).copy()

        # Drop rows with any NaNs in numerical columns for correlation calculation
        initial_rows = len(numerical_df)
        numerical_df.dropna(inplace=True)
        if len(numerical_df) < initial_rows:
             logging.warning(f"Dropped {initial_rows - len(numerical_df)} rows with NaNs for correlation calculation{title_suffix}.")

        if numerical_df.empty:
             logging.warning(f"Numerical DataFrame is empty after dropping NaNs. Skipping correlation analysis{title_suffix}.")
             return pd.DataFrame()

        # Calculate the full correlation matrix (needed for get_top_correlations)
        try:
            correlation_matrix = numerical_df.corr(method=method)
            logging.info(f"Full {method.capitalize()} correlation matrix calculated{title_suffix}.")

            # --- Generate Heatmap for Correlation with Target Variable ---
            if target_variable not in correlation_matrix.columns:
                 logging.warning(f"Target variable '{target_variable}' not found in correlation matrix. Cannot generate target correlation heatmap{title_suffix}.")
            else:
                # Extract correlations of the target variable
                target_corr_series = correlation_matrix[target_variable].copy()

                # Remove the correlation of the target with itself
                if target_variable in target_corr_series.index:
                    target_corr_series = target_corr_series.drop(target_variable)

                # Remove specified derived variables
                if exclude_derived:
                    excluded_cols_present = [col for col in exclude_derived if col in target_corr_series.index]
                    if excluded_cols_present:
                        target_corr_series = target_corr_series.drop(excluded_cols_present)
                        logging.info(f"Excluded derived variables from target correlation heatmap: {excluded_cols_present}")


                if not target_corr_series.empty:
                    # Reshape the series into a DataFrame with one column for the heatmap
                    target_corr_df = target_corr_series.to_frame(name=target_variable)

                    plt.figure(figsize=(6, max(8, len(target_corr_df.index)*0.4))) # Adjust size
                    # Plot the heatmap with annotations
                    sns.heatmap(target_corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
                    plt.title(f'{method.capitalize()} Correlation with {target_variable}{title_suffix}')
                    plt.tight_layout()
                    plt.show()
                    logging.info(f"Heatmap of correlation with {target_variable} generated{title_suffix}.")
                else:
                    logging.warning(f"No other variables remaining after exclusion to plot correlation with {target_variable}{title_suffix}.")

            return correlation_matrix # Return the full matrix for the top correlations table

        except Exception as e:
            logging.error(f"Error calculating or plotting correlation matrix{title_suffix}: {e}", exc_info=True)
            return pd.DataFrame()


    def get_top_correlations(self, correlation_matrix: pd.DataFrame, target_variable: str = 'power_w', exclude_derived: list = None, n: int = None) -> pd.DataFrame:
        """
        Extract correlations between the target variable and other variables,
        excluding specified derived variables, sorted by absolute correlation.

        Args:
            correlation_matrix: The input correlation matrix.
            target_variable: The name of the target variable.
            exclude_derived: List of variable names to exclude from correlations with the target.
            n: The number of top correlations to retrieve. If None, return all.

        Returns:
            pd.DataFrame: DataFrame with correlations (Rank, Variable, Correlation, Absolute Correlation).
        """
        if correlation_matrix.empty or target_variable not in correlation_matrix.columns:
            logging.warning(f"Correlation matrix is empty or target variable '{target_variable}' not found. Cannot get correlations.")
            return pd.DataFrame(columns=['Rank', 'Variable', 'Correlation', 'Absolute Correlation']) # Return empty with expected columns

        try:
            # Get correlations of the target variable with all other variables
            target_corr = correlation_matrix[target_variable].copy()

            # Remove the correlation of the target with itself
            if target_variable in target_corr.index:
                target_corr = target_corr.drop(target_variable)

            # Remove specified derived variables from the correlations
            if exclude_derived:
                excluded_cols_present = [col for col in exclude_derived if col in target_corr.index]
                if excluded_cols_present:
                    target_corr = target_corr.drop(excluded_cols_present)
                    logging.info(f"Excluded derived variables from correlations with '{target_variable}': {excluded_cols_present}")

            if target_corr.empty:
                 logging.warning(f"No other variables remaining after exclusion to list correlations with '{target_variable}'.")
                 return pd.DataFrame(columns=['Rank', 'Variable', 'Correlation', 'Absolute Correlation'])


            # Calculate absolute correlation for sorting
            target_corr_abs = target_corr.abs()

            # Sort by absolute correlation in descending order
            sorted_corr_abs = target_corr_abs.sort_values(ascending=False)

            # Create a DataFrame from the sorted correlations
            corr_df = pd.DataFrame({
                'Variable': sorted_corr_abs.index,
                'Correlation': target_corr.loc[sorted_corr_abs.index], # Get original correlation value
                'Absolute Correlation': sorted_corr_abs.values
            })

            # Add Rank column
            corr_df['Rank'] = range(1, len(corr_df) + 1)

            # Reorder columns
            corr_df = corr_df[['Rank', 'Variable', 'Correlation', 'Absolute Correlation']]

            # Apply head(n) if n is specified
            if n is not None:
                 corr_df = corr_df.head(n)
                 logging.info(f"Extracted top {len(corr_df)} correlations with '{target_variable}' (requested {n}).")
            else:
                 logging.info(f"Extracted all {len(corr_df)} correlations with '{target_variable}'.")


            return corr_df.reset_index(drop=True) # Reset index for clean table output

        except Exception as e:
            logging.error(f"Error extracting correlations with '{target_variable}': {e}", exc_info=True)
            return pd.DataFrame(columns=['Rank', 'Variable', 'Correlation', 'Absolute Correlation']) # Return empty with expected columns


    def plot_scatter_relationships(self, df: pd.DataFrame, title_suffix: str = ""):
        """
        Generate scatter plots for key relationships.

        Args:
            df: DataFrame to plot from.
            title_suffix: Suffix for plot titles.
        """
        if df.empty:
            logging.warning(f"DataFrame is empty. Skipping scatter plots{title_suffix}.")
            return

        logging.info(f"Generating scatter plots{title_suffix}...")

        # Define key relationships to plot
        # Ensure these columns exist in the DataFrame before plotting
        relationships = [
            ('GlobalRadiation [W m-2]', 'power_w', 'PV Power vs. Measured Global Radiation'),
            ('ClearSkyGHI', 'power_w', 'PV Power vs. Clear Sky GHI'),
            ('AOI [degrees]', 'power_w', 'PV Power vs. AOI'),
            ('ClearSkyIndex', 'power_w', 'PV Power vs. Clear Sky Index'),
            ('GlobalRadiation [W m-2]', 'ClearSkyGHI', 'Clear Sky GHI vs. Measured Global Radiation'),
             ('Temperature [degree_Celsius]', 'power_w', 'PV Power vs. Temperature'),
             ('WindSpeed [m s-1]', 'power_w', 'PV Power vs. Wind Speed'),
             ('Pressure [hPa]', 'power_w', 'PV Power vs. Pressure'),
        ]

        # Filter relationships based on available columns
        available_relationships = [(x, y, title) for x, y, title in relationships if x in df.columns and y in df.columns]

        if not available_relationships:
             logging.warning(f"No valid relationships found for scatter plots in the DataFrame{title_suffix}.")
             return

        # Determine number of plots and layout
        n_plots = len(available_relationships)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() # Flatten the axes array for easy iteration

        # Generate plots
        for i, (x_col, y_col, plot_title) in enumerate(available_relationships):
            # Ensure columns are numeric and drop NaNs for plotting
            plot_df = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna().copy()

            if not plot_df.empty:
                sns.scatterplot(x=x_col, y=y_col, data=plot_df, alpha=0.6, s=10, ax=axes[i])
                axes[i].set_title(f'{plot_title}{title_suffix}')
                axes[i].set_xlabel(x_col)
                axes[i].set_ylabel(y_col)
                axes[i].grid(True)
            else:
                axes[i].set_title(f'Not enough valid data for {plot_title}{title_suffix}')
                axes[i].set_xlabel(x_col)
                axes[i].set_ylabel(y_col)


        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        logging.info(f"Scatter plots generated{title_suffix}.")

    def plot_time_series_comparison(self, df: pd.DataFrame, title_suffix: str = ""):
        """
        Plots key variables over time for visual comparison.

        Args:
            df: DataFrame to plot from.
            title_suffix: Suffix for plot titles.
        """
        if df.empty:
            logging.warning(f"DataFrame is empty. Skipping time series plot{title_suffix}.")
            return

        logging.info(f"Generating time series comparison plot{title_suffix}...")

        # Define columns to plot
        # Ensure these columns exist in the DataFrame
        plot_cols = [
            'power_w',
            'GlobalRadiation [W m-2]',
            'ClearSkyGHI',
            'AOI [degrees]',
            'ClearSkyIndex',
            'Temperature [degree_Celsius]', # Include temperature for context
            'WindSpeed [m s-1]',
            'Pressure [hPa]',
            'hour',
            'day_of_year',
            'isNight' # Include night mask
        ]
        available_plot_cols = [col for col in plot_cols if col in df.columns]

        if not available_plot_cols:
             logging.warning(f"No key columns found for time series plot in the DataFrame{title_suffix}.")
             # Print available columns for debugging
             logging.info(f"Available columns: {df.columns.tolist()}")
             return

        # Select a shorter time window for better visualization if the dataset is very long
        # Example: Plotting the first 7 days
        plot_df = df[available_plot_cols].copy()
        # Check if the index has a frequency before calculating time window
        if plot_df.index.freq is not None:
             time_window_limit = 7 * 24 * (60 // plot_df.index.freq.n) # Approx 7 days
        else:
             # If frequency is None, estimate based on the first few differences, or use a fixed number of points
             time_diffs = plot_df.index.to_series().diff().dropna()
             if not time_diffs.empty:
                  # Use the mode of time differences to estimate frequency interval in minutes
                  mode_diff_minutes = time_diffs.mode()[0].total_seconds() / 60
                  if mode_diff_minutes > 0:
                       time_window_limit = 7 * 24 * (60 // int(round(mode_diff_minutes)))
                  else:
                       time_window_limit = 7 * 24 * 6 # Default to 10min if mode is zero or negative
             else:
                  time_window_limit = 1000 # Fallback to a fixed number of points if no time differences

        if len(plot_df) > time_window_limit and not plot_df.empty:
             end_time = plot_df.index.min() + pd.Timedelta(days=7)
             plot_df = plot_df.loc[plot_df.index.min():end_time]
             logging.info(f"Plotting time series for a subset from {plot_df.index.min()} to {plot_df.index.max()}")


        if plot_df.empty:
             logging.warning(f"DataFrame subset for time series plot is empty{title_suffix}.")
             return

        # Create subplots for each variable
        n_plots = len(plot_df.columns)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3 * n_plots), sharex=True)

        # Ensure axes is an array even for 1 subplot
        if n_plots == 1:
            axes = [axes]

        for i, col in enumerate(plot_df.columns):
            plot_df[col].plot(ax=axes[i])
            axes[i].set_ylabel(col.split(' [')[0]) # Use part before unit as label
            axes[i].set_title(f'{col}{title_suffix}')
            axes[i].grid(True)

            # Set specific y-limits for relevant columns
            if col == 'AOI [degrees]':
                axes[i].set_ylim(0, 180)
            elif col == 'ClearSkyIndex':
                 # Allow y-axis to adapt to actual data range, but maybe set a reasonable upper bound
                 axes[i].set_ylim(0, max(plot_df[col].max() * 1.1, 2.0)) # Allow up to 2 or 1.1 * max
            elif col == 'isNight':
                 axes[i].set_ylim(-0.1, 1.1) # Binary values

        axes[-1].set_xlabel('Time (UTC)')
        plt.tight_layout()
        plt.show()
        logging.info(f"Time series comparison plot generated{title_suffix}.")


    def generate_markdown_report(self, dataset_name: str, correlation_matrix: pd.DataFrame, target_correlations: pd.DataFrame) -> str:
        """
        Generates a Markdown report string for the correlation analysis.

        Args:
            dataset_name: Name of the dataset (e.g., "High-Resolution (10min)").
            correlation_matrix: The full correlation matrix.
            target_correlations: DataFrame containing all correlations with the target variable, including rank.

        Returns:
            str: The Markdown formatted report.
        """
        report = f"# PV Data Correlation Analysis - {dataset_name}\n\n"
        report += "## Introduction\n"
        report += f"This report presents the results of a Spearman correlation analysis performed on the {dataset_name} PV and meteorological data.\n\n"

        report += "## Methodology\n"
        report += f"Spearman rank correlation was calculated for all numerical parameters in the dataset. Spearman correlation assesses monotonic relationships (whether variables tend to increase or decrease together, but not necessarily linearly). Missing values were handled by dropping rows containing NaNs before calculating the correlation matrix.\n\n"

        report += "## Results\n"
        if correlation_matrix.empty:
             report += f"Correlation matrix could not be calculated for the {dataset_name} dataset due to insufficient data.\n\n"
        else:
             report += f"A full Spearman correlation matrix was calculated for {len(correlation_matrix.columns)} numerical variables.\n\n"

             report += "### Correlations with PV Power (Excluding Derived Variables)\n"
             # Explicitly check if target_correlations is empty before trying to convert to markdown
             if target_correlations.empty:
                 report += "No significant correlations with PV Power found (excluding derived variables).\n\n"
             else:
                 report += "The table below shows the variables and their Spearman correlation values with 'power_w', ranked by the absolute correlation (descending order), excluding variables derived from power:\n\n"
                 report += target_correlations.to_markdown(index=False)
                 report += "\n\n"

             report += "### Visualizations\n"
             report += "The following plots provide visual insights into the data and its relationships:\n\n"
             report += "1.  **Correlation Heatmap with PV Power:** Shows the Spearman correlation coefficients between 'power_w' and other relevant numerical variables. Correlation values are displayed on the heatmap cells. (Generated by the script)\n\n"
             report += "2.  **Scatter Plots:** Illustrate the relationships between key variables, particularly focusing on PV Power against different solar and meteorological parameters. (Generated by the script)\n\n"
             report += "3.  **Time Series Comparison:** Displays the trends of key variables over a sample time period, allowing for visual inspection of their co-variation. (Generated by the script)\n\n"

        report += "## Conclusion\n"
        if not correlation_matrix.empty and not target_correlations.empty:
            report += f"The correlation analysis reveals key relationships within the {dataset_name} dataset, specifically highlighting the variables most strongly correlated with PV Power ('power_w'), excluding variables derived from power. These correlations, ranked by absolute value, are crucial for understanding the primary drivers of PV power generation and can guide feature selection for predictive modeling.\n\n"
            report += "Visualizations of the full correlation matrix and key scatter plots provide further context for these relationships.\n\n"
        elif not correlation_matrix.empty and target_correlations.empty:
             report += f"The correlation matrix was calculated for the {dataset_name} dataset, but no significant correlations with PV Power were found (excluding derived variables).\n\n"
        else:
             report += "Due to insufficient data, a comprehensive correlation analysis could not be performed.\n\n"


        report += "---\n"
        # Corrected: Use datetime.datetime.now()
        report += f"Analysis performed on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC."

        return report


    def analyze_and_plot(self):
        """Load data and perform all analyses and plotting, and generate report."""
        self.load_data()

        # Define variables likely derived from power that should be excluded from top correlations with power
        derived_power_vars = ['energy_wh', 'energy_production_wh'] # Add any other known derived variables here

        if not self.high_res_df.empty:
            logging.info("\n--- Analyzing High-Resolution Data ---")
            # Pass exclude_derived list to perform_correlation_analysis for heatmap filtering
            high_res_corr_matrix = self.perform_correlation_analysis(self.high_res_df, method='spearman', target_variable='power_w', exclude_derived=derived_power_vars, title_suffix=" (High-Res)")
            high_res_target_corr = pd.DataFrame() # Initialize as empty
            if not high_res_corr_matrix.empty:
                # get_top_correlations now returns all correlations with target, ranked
                high_res_target_corr = self.get_top_correlations(high_res_corr_matrix, target_variable='power_w', exclude_derived=derived_power_vars, n=None) # Set n=None to get all
                self.plot_scatter_relationships(self.high_res_df, title_suffix=" (High-Res)")
                self.plot_time_series_comparison(self.high_res_df, title_suffix=" (High-Res)")

            # Generate and display Markdown report for High-Res
            # Pass the DataFrame with all target correlations and rank
            high_res_report_markdown = self.generate_markdown_report("High-Resolution (10min)", high_res_corr_matrix, high_res_target_corr)
            print(high_res_report_markdown) # Print the markdown string to be captured by the immersive tag

            logging.info("--- Finished High-Resolution Analysis ---")

        else:
            logging.warning("High-resolution data is empty. Skipping analysis for high-res.")
            # Generate a report indicating no data
            empty_high_res_report = self.generate_markdown_report("High-Resolution (10min)", pd.DataFrame(), pd.DataFrame())
            print(empty_high_res_report)


        if not self.low_res_df.empty:
            logging.info("\n--- Analyzing Low-Resolution Data ---")
            # Pass exclude_derived list to perform_correlation_analysis for heatmap filtering
            low_res_corr_matrix = self.perform_correlation_analysis(self.low_res_df, method='spearman', target_variable='power_w', exclude_derived=derived_power_vars, title_suffix=" (Low-Res)")
            low_res_target_corr = pd.DataFrame() # Initialize as empty
            if not low_res_corr_matrix.empty:
                 # get_top_correlations now returns all correlations with target, ranked
                 # Note: Low-res data might not have 'energy_wh' or 'energy_production_wh' if they weren't aggregated
                 low_res_target_corr = self.get_top_correlations(low_res_corr_matrix, target_variable='power_w', exclude_derived=derived_power_vars, n=None) # Set n=None to get all
                 # Scatter plots might be less informative for low-res data due to aggregation, but can still be plotted
                 self.plot_scatter_relationships(self.low_res_df, title_suffix=" (Low-Res)")
                 self.plot_time_series_comparison(self.low_res_df, title_suffix=" (Low-Res)")

                 # Generate and display Markdown report for Low-Res
                 # Pass the DataFrame with all target correlations and rank
                 low_res_report_markdown = self.generate_markdown_report("Low-Resolution (1h)", low_res_corr_matrix, low_res_target_corr)
                 print(low_res_report_markdown) # Print the markdown string to be captured by the immersive tag

            logging.info("--- Finished Low-Resolution Analysis ---")
        else:
            logging.warning("Low-resolution data is empty. Skipping analysis for low-res.")
            # Generate a report indicating no data
            empty_low_res_report = self.generate_markdown_report("Low-Resolution (1h)", pd.DataFrame(), pd.DataFrame())
            print(empty_low_res_report)


if __name__ == "__main__":
    # File paths for the processed data (adjust if you saved them elsewhere)
    HIGH_RES_FILE = "data/processed/station_data_10min.parquet"
    LOW_RES_FILE = "data/processed/station_data_1h.parquet"

    # Initialize and run the analyzer
    analyzer = DataAnalyzer(HIGH_RES_FILE, LOW_RES_FILE)
    analyzer.analyze_and_plot()
