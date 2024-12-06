import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# User specifies which plots to generate
plots_to_generate = [4]#[1, 3, 4]  # Replace with the plot numbers you want

# Mapping of plot numbers to functions
#plot_functions = {
#    1: 'plot_time_series',
#    2: 'plot_boxplots',
#    3: 'plot_mean_barplots',
#    4: 'plot_mean_divided_by_humid_barplots',
#    # Add more plots as needed
#}

# Define your data directory
data_dir = 'ActiveExperiment'

# Output directory setup
output_dir = f'{data_dir}/outputs'     # Directory to save the outputs
# Create the outputs directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 1: Read and Combine Data

# Get a list of all CSV files in the data directory
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# Initialize a list to hold DataFrames
dfs = []

for file in csv_files:
    # Extract the filename without extension
    filename = os.path.basename(file)
    soil_type = os.path.splitext(filename)[0]
    
    # Read the CSV file
    df = pd.read_csv(file, sep=',')
    
    # Add the soil_type column
    df['soil_type'] = soil_type
    
    # Append the DataFrame to the list
    dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Convert Timestamp to numeric if necessary
combined_df['Timestamp'] = pd.to_numeric(combined_df['Timestamp'], errors='coerce')

# Replace -1 with NaN
combined_df.replace(-1, np.nan, inplace=True)
combined_df.replace("Invalid", np.nan, inplace=True)

# Step 3: Calculate Summary Statistics and Reshape the DataFrame

# Define the metrics
metrics = ['Temp', 'Humid', 'PH', 'EC', 'N', 'P', 'K', 'Latitude', 'Longitude', 'Satellites', 'HDOP']

# Group by soil_type and SensorID
grouped = combined_df.groupby(['soil_type', 'SensorID'])

# Calculate statistics, automatically skipping NaNs
stats_df = grouped[metrics].agg(['mean', 'median', 'std', 'max', 'min']).reset_index()

# Flatten the MultiIndex columns
stats_df.columns.names = [None, None]
stats_df.columns = [' '.join(col).strip() if col[1] else col[0] for col in stats_df.columns.values]

# Reshape the DataFrame
stats_df = pd.melt(stats_df, id_vars=['soil_type', 'SensorID'], var_name='metric_statistic', value_name='value')

# Split 'metric_statistic' into 'metric' and 'statistic'
stats_df[['metric', 'statistic']] = stats_df['metric_statistic'].str.rsplit(' ', n=1, expand=True)

# Drop 'metric_statistic' column
stats_df.drop(columns=['metric_statistic'], inplace=True)

# Pivot the DataFrame to get one column per statistic
stats_df = stats_df.pivot_table(index=['soil_type', 'SensorID', 'metric'], columns='statistic', values='value').reset_index()

# Reorder the columns if needed
stats_df = stats_df[['soil_type', 'SensorID', 'metric', 'mean', 'median', 'std', 'max', 'min']]

# Save the summary statistics to CSV
stats_filename = 'summary_statistics.csv'
stats_df.to_csv(os.path.join(output_dir, stats_filename), index=False)

# Step 6: Calculate Means for Each Experiment and Save in Easier Format

# Group by soil_type and SensorID and calculate the mean for each metric
mean_df = combined_df.groupby(['soil_type', 'SensorID'])[metrics].mean().reset_index()

# Sort the columns by metric names to have consistent ordering
mean_df = mean_df.sort_index(axis=1)

# Save the means to a CSV file where each row is an experiment/sensor pair and columns are the metrics
mean_filename = 'means_summary.csv'
mean_df.to_csv(os.path.join(output_dir, mean_filename), index=False)


# same thing but for mediasn 
median_df = combined_df.groupby(['soil_type', 'SensorID'])[metrics].median().reset_index()
median_df = median_df.sort_index(axis=1)
median_filename = 'means_summary.csv'
median_df.to_csv(os.path.join(output_dir, median_filename), index=False)


# Define plotting functions
def plot_time_series(combined_df, output_dir):
    metrics = ['Temp', 'Humid', 'PH', 'EC', 'N', 'P', 'K', 'Latitude', 'Longitude', 'Satellites', 'HDOP']

    for metric in metrics:
        # Drop NaNs only in the current metric
        plot_df = combined_df.dropna(subset=[metric, 'Timestamp', 'SensorID', 'soil_type'])
        
        # Sort the DataFrame by Timestamp
        plot_df = plot_df.sort_values('Timestamp')
        
        plt.figure(figsize=(12, 6))
        
        sns.lineplot(
            data=plot_df,
            x='Timestamp',
            y=metric,
            hue='soil_type',
            style='SensorID',
            markers=True,
            dashes=False,
            markersize=8
        )
        
        plt.title(f'{metric} over Time')
        plt.xlabel('Timestamp')
        plt.ylabel(metric)
        plt.legend(title='Soil Type / SensorID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'{metric}_timeseries.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

def plot_boxplots(combined_df, output_dir):
    metrics = ['Temp', 'Humid', 'PH', 'EC', 'N', 'P', 'K', 'Latitude', 'Longitude', 'Satellites', 'HDOP']

    for metric in metrics:
        # Drop NaNs only in the current metric
        boxplot_df = combined_df.dropna(subset=[metric, 'soil_type', 'SensorID'])
        
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(
            data=boxplot_df,
            x='soil_type',
            y=metric,
            hue='SensorID',
            palette='viridis'
        )
        
        plt.title(f'Boxplot of {metric} by Soil Type and SensorID')
        plt.xlabel('Soil Type')
        plt.ylabel(metric)
        plt.legend(title='SensorID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the boxplot
        boxplot_filename = f'{metric}_boxplot.png'
        plt.savefig(os.path.join(output_dir, boxplot_filename))
        plt.close()

def plot_mean_barplots(stats_df, output_dir):
    metrics = ['Temp', 'Humid', 'PH', 'EC', 'N', 'P', 'K', 'Latitude', 'Longitude', 'Satellites', 'HDOP']

    for metric in metrics:
        # Filter stats_df for the current metric
        barplot_df = stats_df[stats_df['metric'] == metric].sort_values(by='soil_type')  # Sort by soil_type
        
        plt.figure(figsize=(12, 6))
    
        sns.barplot(
            data=barplot_df,
            x='soil_type',
            y='mean',
            hue='SensorID',
            palette='viridis'
        )
    
        plt.title(f'Mean {metric} by Soil Type and SensorID')
        plt.xlabel('Soil Type')
        plt.ylabel(f'Mean {metric}')
        plt.legend(title='SensorID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)

        plt.tight_layout()
    
        # Save the bar plot
        barplot_filename = f'{metric}_mean_barplot.png'
        plt.savefig(os.path.join(output_dir, barplot_filename))
        plt.close()

def plot_mean_divided_by_humid_barplots(stats_df, output_dir):
    # First, get the mean Humid values for each soil_type and SensorID
    humid_means = stats_df[stats_df['metric'] == 'Humid'][['soil_type', 'SensorID', 'mean']]
    humid_means = humid_means.rename(columns={'mean': 'mean_humid'})
    
    # Merge humid_means back into stats_df
    stats_with_humid = stats_df.merge(humid_means, on=['soil_type', 'SensorID'], how='left')
    
    # Exclude the Humid metric itself
    metrics = ['Temp', 'PH', 'EC', 'N', 'P', 'K', 'Latitude', 'Longitude', 'Satellites', 'HDOP']
    
    for metric in metrics:
        # Filter stats_with_humid for the current metric
        plot_df = stats_with_humid[stats_with_humid['metric'] == metric].copy()
        
        # Avoid division by zero
        plot_df = plot_df[plot_df['mean_humid'] != 0]
        
        # Calculate the ratio
        plot_df['mean_divided_by_humid'] = plot_df['mean'] / plot_df['mean_humid']
        
        plt.figure(figsize=(12,6))
        
        sns.barplot(
            data=plot_df,
            x='soil_type',
            y='mean_divided_by_humid',
            hue='SensorID',
            palette='viridis'
        )
        
        plt.title(f'Mean {metric} divided by Humid by Soil Type and SensorID')
        plt.xlabel('Soil Type')
        plt.ylabel(f'Mean {metric} divided by Humid')
        plt.legend(title='SensorID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)

        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'{metric}_mean_divided_by_humid_barplot.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

# Mapping of plot numbers to functions
plot_functions = {
    1: plot_time_series,
    2: plot_boxplots,
    3: plot_mean_barplots,
    4: plot_mean_divided_by_humid_barplots,
    # Add more plots as needed
}

# Generate plots based on user's selection
for plot_num in plots_to_generate:
    plot_func = plot_functions.get(plot_num)
    if plot_func:
        if plot_num in [1, 2]:  # Functions that require combined_df
            plot_func(combined_df, output_dir)
        elif plot_num in [3, 4]:  # Functions that require stats_df
            plot_func(stats_df, output_dir)
        else:
            # Handle other functions if any
            pass
    else:
        print(f"Plot number {plot_num} is not recognized.")
