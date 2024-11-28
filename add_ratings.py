import pandas as pd
import numpy as np


# Define the rating mapping dictionary
rating_dict = {
    'PH': [
        (-float('inf'), 4.5, 'vL'), 
        (4.5, 5.5, 'L'), 
        (5.5, 6.5, 'M'), 
        (6.5,7.5,'H'),
        (7.5, float('inf'), 'vH')
    ],
    'N': [
        (-float('inf'), .1, 'vL'),
        (.1,.19,'L'),
        (.19,.49,'M'),
        (.49,.99,'H'),
        (.99,float('inf'), 'vH')
    ],
    'P': [
        (-float('inf'), 5, 'vL'),
        (5, 14.9, 'L'),
        (14.9, 29.9, 'M'),
        (29.9, float('inf'), 'H'),
    ],
    'K': [
        (-float('inf'), 40, 'vL'),
        (40, 99, 'L'),
        (99, 199, 'M'),
        (199, 299, 'H'),
        (299, float('inf'), 'vH'),
    ],
    'EC': [
        (-float('inf'), 0.4, 'vL'),
        (0.4, .8, 'L'),
        (.8, 1.6, 'M'),
        (1.6, float('inf'), 'H'),
    ],
}
def get_rating(param, value):
    """Returns the rating for a given parameter value."""
    if param not in rating_dict or pd.isnull(value):
        return ''
    for (lower, upper, rating) in rating_dict[param]:
        if lower <= value < upper:
            return rating
    return 'Unknown'
def get_rating(param, value):
    """Returns the rating for a given parameter value."""
    if param not in rating_dict or pd.isnull(value):
        return ''
    for (lower, upper, rating) in rating_dict[param]:
        if lower <= value < upper:
            return rating
    return ''

# Paths to the input and output CSV files
input_path = 'input_data_20samples.csv'
output_path = 'updated_input_data_20samples.csv'

# Read the CSV file
df = pd.read_csv(input_path, header=None)

# Assign column names
column_names = ['soil_type', 'SensorID', 'PH', 'N', 'P', 'K', 'EC']
df.columns = column_names[:df.shape[1]]  # Adjust columns based on actual data

# Iterate over unique soil types
soil_types = df['soil_type'].dropna().unique()

for soil_type in soil_types:
    # Filter rows for the current soil type
    soil_rows = df[df['soil_type'] == soil_type]
    
    # Get the 'Lab(Value)' row
    lab_value_row = soil_rows[soil_rows['SensorID'] == 'Lab(Value)']
    if lab_value_row.empty:
        continue  # Skip if 'Lab(Value)' row is not found
    
    lab_value_row = lab_value_row.iloc[0]
    
    # Extract values for parameters
    params = ['PH', 'N', 'P', 'K', 'EC']
    values = {}
    for param in params:
        try:
            values[param] = float(lab_value_row[param])
        except (ValueError, TypeError, KeyError):
            values[param] = np.nan  # Handle missing or invalid data
    
    # Compute ratings
    ratings = {}
    for param in params:
        ratings[param] = get_rating(param, values[param])
    
    # Get the index of the 'Lab(Rating)' row
    lab_rating_idx = soil_rows[soil_rows['SensorID'] == 'Lab(Rating)'].index
    if lab_rating_idx.empty:
        continue  # Skip if 'Lab(Rating)' row is not found
    
    lab_rating_idx = lab_rating_idx[0]
    
    # Update the 'Lab(Rating)' row with computed ratings
    for param in params:
        df.at[lab_rating_idx, param] = ratings[param]

# Write the updated DataFrame back to the CSV file
df.to_csv(output_path, index=False, header=False)

print(f"Ratings have been added to '{output_path}'.")