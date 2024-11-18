import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Load your data
df_combined = pd.read_csv('combined.csv')

# Convert columns to numeric, replace NaNs with -1
numeric_cols = ['EC', 'K', 'N', 'P', 'PH']
for col in numeric_cols:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

df_combined['SensorID'] = df_combined['SensorID'].astype(str)
df_combined['soil_type'] = df_combined['soil_type'].astype(str)

# Replace NaNs with -1
df_combined.fillna(-1, inplace=True)

# Desired SensorIDs order
desired_sensor_ids = ['Lab(Value)', '1', '2', '3', '4', 'phProbe']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Soil Parameter Visualization', style={'text-align': 'center'}),
    dcc.Graph(id='parameter-graph'),
    html.Div([
        html.Label('Select Parameter:', style={'fontSize': 20}),
        dcc.Dropdown(
            id='parameter-dropdown',
            options=[
                {'label': param, 'value': param} for param in ['EC', 'K', 'N', 'P', 'PH']
            ],
            value='PH',  # Default parameter set to 'PH'
            clearable=False,
            style={'width': '300px', 'margin': 'auto', 'fontSize': 18}
        )
    ], style={'width': '100%', 'text-align': 'center', 'padding': '20px'})
])

@app.callback(
    Output('parameter-graph', 'figure'),
    Input('parameter-dropdown', 'value')
)
def update_graph(selected_parameter):
    # Filter the DataFrame for desired SensorIDs
    df_filtered = df_combined[df_combined['SensorID'].isin(desired_sensor_ids)]

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

    # Get Lab(Value) per soil_type
    lab_values = df_pivot.loc['Lab(Value)'].dropna()

    # Sort soil_types based on Lab(Value) for the selected parameter
    lab_values_sorted = lab_values.sort_values()
    ordered_soil_types = lab_values_sorted.index.tolist()

    # Reindex the columns (soil_types) based on sorted Lab(Value)
    df_pivot = df_pivot[ordered_soil_types]

    # Normalize data per row
    df_normalized = df_pivot.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

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

if __name__ == '__main__':
    app.run_server(debug=True)
