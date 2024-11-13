import pandas as pd

# Load the CSV files
csv1_path = '/Users/jflyn/Documents/projects/DRIVEsoilDisplay/MeasurementCSVs/outputs/means_summary.csv'
csv2_path = '/Users/jflyn/Documents/projects/DRIVEsoilDisplay/input_data.csv'

# Read the CSVs into DataFrames
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Find the common columns
common_columns = df1.columns.intersection(df2.columns)

# Subset the DataFrames to include only common columns
df1_common = df1[common_columns]
df2_common = df2[common_columns]

# Combine the two DataFrames (e.g., concatenation)
combined_df = pd.concat([df1_common, df2_common], ignore_index=True)

# Save the combined DataFrame to a new CSV
output_path = "combined.csv"  # Specify your output file path
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV saved to {output_path}")