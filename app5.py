import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import glob
import os

# Get list of combined CSV files
combined_csv_files = glob.glob('combined*.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Desired SensorIDs order, including 'Lab(Rating)'
desired_sensor_ids = ['Lab(Rating)', 'Lab(Value)', '1', '2', '3', '4', 'phProbe']

# Prepare a dictionary to store DataFrames and a list for Divs
dataframes = {}
graph_divs = []

# Function to prepare DataFrames
def prepare_dataframe(df):
    # Convert columns to numeric, replace NaNs with -1
    numeric_cols = ['EC', 'K', 'N', 'P', 'PH']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['SensorID'] = df['SensorID'].astype(str)
    df['soil_type'] = df['soil_type'].astype(str)
    # Replace NaNs with -1
    df.fillna(-1, inplace=True)
    return df

# Loop over each combined CSV file
for csv_file in combined_csv_files:
    # Extract the suffix for labeling
    base_filename = os.path.basename(csv_file)
    if base_filename == 'combined.csv':
        suffix = 'BASELINE'
    else:
        # Extract the suffix after 'combined_' and before '.csv'
        suffix = base_filename.replace('combined_', '').replace('.csv', '')
    
    # Read and prepare the DataFrame
    df = pd.read_csv(csv_file)
    df = prepare_dataframe(df)
    dataframes[suffix] = df  # Store DataFrame for callbacks
    
    # Create unique IDs for components
    graph_id = f'parameter-graph-{suffix}'
    dropdown_id = f'parameter-dropdown-{suffix}'
    
    # Create a Div for each dataset
    graph_div = html.Div([
        html.H2(f'Data: {suffix}', style={'text-align': 'center'}),
        dcc.Graph(id=graph_id),
        html.Div([
            html.Label('Select Parameter:', style={'fontSize': 20}),
            dcc.Dropdown(
                id=dropdown_id,
                options=[{'label': param, 'value': param} for param in ['EC', 'K', 'N', 'P', 'PH']],
                value='PH',  # Default parameter
                clearable=False,
                style={'width': '300px', 'margin': 'auto', 'fontSize': 18}
            )
        ], style={'text-align': 'center', 'padding': '20px'})
    ], style={'margin-bottom': '50px'})
    
    graph_divs.append(graph_div)

# Set the app layout to include all graph Divs
app.layout = html.Div([
    html.H1('Soil Parameter Visualization', style={'text-align': 'center'}),
    *graph_divs  # Unpack the list of Divs
])

# Function to create the heatmap
def create_heatmap(df, selected_parameter):
    # Filter the DataFrame for desired SensorIDs
    df_filtered = df[df['SensorID'].isin(desired_sensor_ids)]
    
    # Pivot the data
    df_pivot = df_filtered.pivot_table(
        index='SensorID',
        columns='soil_type',
        values=selected_parameter
    )
    
    # Replace -1 with NaN for better handling
    df_pivot.replace(-1, np.nan, inplace=True)
    
    # Reindex the rows to match desired SensorID order
    df_pivot = df_pivot.reindex(desired_sensor_ids)
    
    # Get all soil types
    all_soil_types = df_pivot.columns
    
    # Get Lab(Rating) and Lab(Value) per soil_type
    lab_rating = df_pivot.loc['Lab(Rating)']
    lab_values = df_pivot.loc['Lab(Value)']
    
    # Ensure Lab(Rating) and Lab(Value) are numeric
    lab_rating_numeric = pd.to_numeric(lab_rating, errors='coerce')
    lab_values_numeric = pd.to_numeric(lab_values, errors='coerce')
    
    # Create a DataFrame for sorting, include all soil_types
    lab_df = pd.DataFrame({
        'Lab_Rating': lab_rating_numeric,
        'Lab_Value': lab_values_numeric
    }, index=all_soil_types)
    
    # Sort soil_types based on Lab_Rating and Lab_Value, NaNs are placed at the end
    lab_df_sorted = lab_df.sort_values(by=['Lab_Rating', 'Lab_Value'], na_position='last')
    
    # Get the ordered list of soil_types
    ordered_soil_types = lab_df_sorted.index.tolist()
    
    # Reindex the columns (soil_types) based on sorted Lab(Rating) and Lab(Value)
    df_pivot = df_pivot[ordered_soil_types]
    
    # Normalize data per row
    def normalize_row(x):
        min_val = x.min()
        max_val = x.max()
        if pd.isnull(min_val) or pd.isnull(max_val) or min_val == max_val:
            return x * np.nan  # Return NaN if min and max are NaN or equal
        else:
            return (x - min_val) / (max_val - min_val)
    
    df_normalized = df_pivot.apply(normalize_row, axis=1)
    
    # Prepare text data
    text_data = df_pivot.values.copy()
    text_data = np.where(np.isnan(text_data), '', np.round(text_data, 2).astype(str))
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df_normalized.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        text=text_data,
        texttemplate="%{text}",
        colorscale='Viridis',
        colorbar=dict(title="Normalized Value"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'{selected_parameter} Values Across Soil Types and Sensors',
        xaxis_title='Soil Type',
        yaxis_title='SensorID',
        height=600
    )
    
    return fig

# Function to generate callbacks dynamically
def generate_callback(graph_id, dropdown_id, df):
    @app.callback(
        Output(graph_id, 'figure'),
        Input(dropdown_id, 'value')
    )
    def update_graph(selected_parameter):
        return create_heatmap(df, selected_parameter)
    return update_graph

# Generate callbacks for each dataset
for suffix, df in dataframes.items():
    graph_id = f'parameter-graph-{suffix}'
    dropdown_id = f'parameter-dropdown-{suffix}'
    generate_callback(graph_id, dropdown_id, df)

if __name__ == '__main__':
    app.run_server(debug=True)
