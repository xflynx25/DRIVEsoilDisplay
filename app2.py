import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Replace 'combined.csv' with the path to your actual CSV file
csv_file_path = 'combined.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Separate sensor data and lab data based on `SensorID`
sensor_data = df[df['SensorID'].str.isnumeric()]  # Rows where SensorID is numeric
lab_data = df[~df['SensorID'].str.isnumeric()]   # Rows where SensorID is not numeric

# Convert to DataFrames
df_sensor = pd.DataFrame(sensor_data)
df_lab = pd.DataFrame(lab_data)

# Convert metric columns to numeric
metrics = ['EC', 'K', 'N', 'P', 'PH']
for m in metrics:
    df_sensor[m] = pd.to_numeric(df_sensor[m], errors='coerce')
    df_lab[m] = pd.to_numeric(df_lab[m], errors='coerce')

# Extract lab values and ratings
lab_values_df = df_lab[df_lab['SensorID'] == 'Lab(Value)']
lab_ratings_df = df_lab[df_lab['SensorID'] == 'Lab(Rating)']
ph_probe_df = df_lab[df_lab['SensorID'] == 'phProbe']


# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Metrics for selection
metrics = ['EC', 'K', 'N', 'P', 'PH']

app.layout = html.Div([
    html.H1("Soil Metric Visualization"),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[{'label': m, 'value': m} for m in metrics],
        value='EC',
        clearable=False,
        style={'width': '50%', 'margin': '0 auto'}
    ),
    dcc.Graph(id='heatmap')
])


@app.callback(
    Output('heatmap', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_heatmap(selected_metric):
    # Create pivot table from sensor data
    pivot_df = df_sensor.pivot(index='SensorID', columns='soil_type', values=selected_metric)
    
    # Get lab values and ratings
    lab_values = lab_values_df.set_index('soil_type')[selected_metric]
    lab_ratings = lab_ratings_df.set_index('soil_type')[selected_metric]
    
    # Map and normalize lab ratings
    rating_mapping = {'vL': 1, 'L': 2, 'M': 3, 'H': 4, 'vH': 5}
    lab_ratings = lab_ratings.fillna('M')  # Default to 'M' if ratings are missing
    lab_ratings_mapped = lab_ratings.map(rating_mapping).fillna(0)
    max_rating = max(rating_mapping.values())
    normalized_lab_row = lab_ratings_mapped.reindex(pivot_df.columns).fillna(0) / max_rating
    
    # Add lab values as a row
    pivot_df.loc['Lab'] = lab_values
    
    # Add phProbe data if selected_metric is 'PH'
    if selected_metric == 'PH':
        ph_probe_values = ph_probe_df.set_index('soil_type')['PH']
        pivot_df.loc['phProbe'] = ph_probe_values
    else:
        pivot_df.loc['phProbe'] = np.nan  # Only PH metric has phProbe data
    
    # Sort soil_types based on lab values
    sorted_soil_types = lab_values.sort_values().index.tolist()
    pivot_df = pivot_df[sorted_soil_types]
    
    # Reorder the index to have 'Lab' and 'phProbe' at the top
    desired_order = ['Lab', 'phProbe'] + [sid for sid in pivot_df.index if sid not in ['Lab', 'phProbe']]
    pivot_df = pivot_df.reindex(desired_order)
    
    # Prepare data for heatmap
    z = []
    text = []
    for idx, row in pivot_df.iterrows():
        row_values = row.values.astype(float)
        if idx == 'Lab':
            z.append(normalized_lab_row)  # Use normalized lab ratings for color
            text.append(['{:.2f}'.format(val) if not np.isnan(val) else '' for val in row.values])
        else:
            mask = np.isfinite(row_values)
            normalized_row = (row_values - row_values[mask].min()) / (row_values[mask].max() - row_values[mask].min()) if np.any(mask) else np.zeros_like(row_values)
            z.append(normalized_row)
            text.append(['{:.2f}'.format(val) if not np.isnan(val) else '' for val in row_values])

    # Define the y-axis labels
    y_labels = pivot_df.index.tolist()

    # Create heatmap
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,  # All rows combined
        x=pivot_df.columns,
        y=y_labels,  # Match exactly with pivot_df.index
        text=text,
        hoverinfo='text',
        colorscale='RdYlBu_r',
        showscale=True,
        xgap=1,
        ygap=1,
        zmin=0,
        zmax=1,
        colorbar=dict(title='Normalized Value')
    ))

    # Add annotations for all rows
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            if not np.isnan(val):
                annotations.append(
                    dict(
                        x=pivot_df.columns[j],
                        y=y_labels[i],  # Match the heatmap's row labels
                        text=text[i][j],
                        showarrow=False,
                        font=dict(color='white'),  # Set font color to white
                        align='center',
                        valign='middle'
                    )
                )

    # Update layout
    fig.update_layout(
        title=f"Heatmap of {selected_metric}",
        xaxis_title="Soil Type",
        yaxis_title="Sensor ID",
        annotations=annotations,
        yaxis=dict(autorange="reversed"),  # Reverse y-axis to have 'Lab' at the top
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='lightgray'
    )
    
    return fig




if __name__ == '__main__':
    app.run_server(debug=True)
