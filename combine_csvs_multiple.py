
import pandas as pd

# Load the CSV files
csv1a_path = '/Users/jflyn/Documents/projects/DRIVEsoilDisplay/MeasurementCSVs/outputs/means_summary.csv' # our auto-csv collected
csv1b_path = '/Users/jflyn/Documents/projects/DRIVEsoilDisplay/input_variable_data.csv' #our manual writing collected
csv2_path = '/Users/jflyn/Documents/projects/DRIVEsoilDisplay/input_data.csv' # lab info

# Read the CSVs into DataFrames
df1a = pd.read_csv(csv1a_path)
df1b = pd.read_csv(csv1b_path)
df2 = pd.read_csv(csv2_path)
df2['soil_type'] = df2['soil_type'].astype(str)

# combine the autocollected and manual entry variable data 
df1 = pd.concat([df1a, df1b], axis=0) # will have nans but no worry

print(df1)

# Split df1 into non-flood and flood data
df1_non_flood = df1[~df1['soil_type'].str.endswith('_flood')]
df1_flood = df1[df1['soil_type'].str.endswith('_flood')]

# Remove '_flood' from soil_type in df1_flood to match df2
df1_flood['soil_type'] = df1_flood['soil_type'].str.replace('_flood', '', regex=False)

# Function to process and save DataFrames
def process_and_save(df1_part, output_path):
    # Find the common columns
    common_columns = df1_part.columns.intersection(df2.columns)
    
    # Subset the DataFrames to include only common columns
    df1_common = df1_part[common_columns]
    df2_common = df2[common_columns]
    
    # Combine the two DataFrames
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
    
    # Handle NaN values after mapping
    combined_df[rating_cols] = combined_df[rating_cols].fillna('')
    
    # Save the combined DataFrame to a new CSV
    combined_df.to_csv(output_path, index=False)
    
    print(f"Combined CSV saved to {output_path}")

# Process and save non-flood data
process_and_save(df1_non_flood, 'combined.csv')

# Process and save flood data
process_and_save(df1_flood, 'combined_flood.csv')
