import pandas as pd

# Load the CSV files
csv1_path = '/Users/jflyn/Documents/projects/DRIVEsoilDisplay/MeasurementCSVs/outputs/means_summary.csv'
csv2_path = '/Users/jflyn/Documents/projects/DRIVEsoilDisplay/input_data.csv'

# Read the CSVs into DataFrames
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)
df2['soil_type'] = df2['soil_type'].astype(str)

# Find the common columns
common_columns = df1.columns.intersection(df2.columns)

# Subset the DataFrames to include only common columns
df1_common = df1[common_columns]
df2_common = df2[common_columns]


# Combine the two DataFrames (e.g., concatenation)
combined_df = pd.concat([df1_common, df2_common], ignore_index=True)
combined_df = combined_df.loc[combined_df['soil_type'].isin(df2['soil_type'].unique())]


# Define the mapping from letters to numbers
rating_mapping = {'vL': 1, 'L': 2, 'M': 3, 'H': 4, 'vH': 5}

# List of columns where ratings are present
rating_cols = ['PH', 'N', 'P', 'K', 'EC']

# Identify rows where SensorID is 'Lab(Rating)'
lab_rating_rows = combined_df['SensorID'] == 'Lab(Rating)'

# Apply the mapping to the specified columns in those rows
for col in rating_cols:
    combined_df.loc[lab_rating_rows, col] = combined_df.loc[lab_rating_rows, col].map(rating_mapping)

# Optional: If there are NaN values after mapping (e.g., unmapped values), handle them
combined_df[rating_cols] = combined_df[rating_cols].fillna('')

# Save the combined DataFrame to a new CSV
output_path = "combined.csv"  # Specify your output file path
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV saved to {output_path}")
