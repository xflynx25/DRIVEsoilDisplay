import dash
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
from plotly.subplots import make_subplots

from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.express as px


app = dash.Dash(__name__)
server = app.server

pd.set_option('future.no_silent_downcasting', True)

# Define global variables
metrics = ['Temp', 'Humid', 'PH', 'EC', 'N', 'P', 'K', 'Latitude', 'Longitude', 'Satellites', 'HDOP']

def read_and_process_data():
    # Specify the directory containing the CSV files
    data_dir = '/Users/jflyn/Documents/projects/DRIVEsoilDisplay/MeasurementCSVs'  # Replace with your directory
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    # Initialize a list to hold DataFrames
    dfs = []

    for file in csv_files:
        # Extract the filename without extension to use as soil_type
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

    combined_df.replace(-1, np.nan, inplace=True)
    combined_df.replace("Invalid", np.nan, inplace=True)

    # Retain old downcasting behavior
    combined_df = combined_df.infer_objects(copy=False)

    return combined_df

def generate_statistics(df):
    # Calculate statistics
    grouped = df.groupby(['soil_type', 'SensorID'])
    stats_df = grouped[metrics].agg(['mean']).reset_index()
    stats_df.columns = [' '.join(col).strip() if col[1] else col[0] for col in stats_df.columns.values]
    return stats_df
 
def create_time_series(df, metric):
    import plotly.graph_objects as go

    # Filter data
    plot_df = df.dropna(subset=[metric, 'Timestamp', 'SensorID', 'soil_type']).copy()
    plot_df = plot_df.sort_values('Timestamp')

    # Convert Timestamp to seconds
    plot_df['TimeSeconds'] = plot_df['Timestamp'] / 1000

    # Ensure SensorID is a string
    plot_df['SensorID'] = plot_df['SensorID'].astype(str)

    # Get unique soil_types and SensorIDs
    soil_types = plot_df['soil_type'].unique()
    sensor_ids = plot_df['SensorID'].unique()

    # Assign colors to soil_types
    color_map = px.colors.qualitative.Plotly
    soil_type_colors = {soil: color_map[i % len(color_map)] for i, soil in enumerate(soil_types)}

    # Assign symbols to SensorIDs
    symbol_list = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']
    sensor_symbols = {sensor: symbol_list[i % len(symbol_list)] for i, sensor in enumerate(sensor_ids)}

    # Create the figure
    fig = go.Figure()

    # Add dummy traces for the soil_type legend
    for soil in soil_types:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                color=soil_type_colors[soil]
            ),
            legendgroup='Soil Type',
            legendgrouptitle_text='Soil Type',
            showlegend=True,
            name=soil
        ))

    # Add dummy traces for the SensorID legend
    for sensor in sensor_ids:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                color='black',
                symbol=sensor_symbols[sensor]
            ),
            legendgroup='Sensor ID',
            legendgrouptitle_text='Sensor ID',
            showlegend=True,
            name=f'Sensor {sensor}'
        ))

    # Now, add the actual data traces
    # Group the data by soil_type and SensorID
    grouped = plot_df.groupby(['soil_type', 'SensorID'])

    for (soil, sensor), group in grouped:
        fig.add_trace(go.Scatter(
            x=group['TimeSeconds'],
            y=group[metric],
            mode='lines+markers',
            marker=dict(
                color=soil_type_colors[soil],
                symbol=sensor_symbols[sensor]
            ),
            line=dict(color=soil_type_colors[soil]),
            legendgroup='Data',
            showlegend=False  # Do not show in legend to avoid clutter
        ))

    # Update layout
    fig.update_layout(
        xaxis_title='Time (seconds)',
        yaxis_title=metric,
        title=f'{metric} over Time',
        legend_title_text='',  # We have custom legend group titles
    )

    return fig



def create_boxplot(df, metric, soil_types):
    # Filter data
    boxplot_df = df.dropna(subset=[metric, 'soil_type', 'SensorID'])
    
    # Convert 'SensorID' and 'soil_type' to string to ensure categorical treatment
    boxplot_df['SensorID'] = boxplot_df['SensorID'].astype(str)
    boxplot_df['soil_type'] = pd.Categorical(boxplot_df['soil_type'], categories=soil_types, ordered=True)

    fig = px.box(
        boxplot_df,
        x='soil_type',
        y=metric,
        color='SensorID',
        title=f'Boxplot of {metric} by Soil Type and SensorID',
        category_orders={'soil_type': soil_types}
    )
    fig.update_layout(
        xaxis_title='Soil Type',
        yaxis_title=metric,
        legend_title_text='SensorID',
        title=''
    )
    return fig

def create_mean_barplot(stats_df, metric, soil_types):
    # Filter stats_df for the current metric
    barplot_df = stats_df[[col for col in stats_df.columns if metric in col or col in ['soil_type', 'SensorID']]]

    # Melt the DataFrame
    barplot_df = pd.melt(barplot_df, id_vars=['soil_type', 'SensorID'], var_name='metric', value_name='mean')
    barplot_df = barplot_df[barplot_df['metric'] == f'{metric} mean']

    # Convert 'SensorID' and 'soil_type' to string to ensure discrete colors and grouping
    barplot_df['SensorID'] = barplot_df['SensorID'].astype(str)
    barplot_df['soil_type'] = pd.Categorical(barplot_df['soil_type'], categories=soil_types, ordered=True)

    fig = px.bar(
        barplot_df,
        x='soil_type',
        y='mean',
        color='SensorID',
        barmode='group',  # Ensure bars are grouped
        title=f'Mean {metric} by Soil Type and SensorID',
        category_orders={'soil_type': soil_types}
    )
    fig.update_layout(
        xaxis_title='Soil Type',
        yaxis_title=f'Mean {metric}',
        legend_title_text='SensorID',
        title=''
    )
    return fig

# Assuming `read_and_process_data()` and other existing functions are unchanged

# Add metrics for the new visualization
additional_metrics = ['PH', 'N', 'P', 'K', 'EC']

# Extend layout
app.layout = html.Div([
    html.Div(
        html.H1(
            "DHI-DRIVE RealTime Soil Monitoring",
            style={
                'textAlign': 'center',
                'color': 'white',
                'padding': '20px',
                'margin': '0'
            }
        ),
        style={
            'background': 'linear-gradient(to right, orange , yellow)',
            'borderBottom': '2px solid #e0e0e0'
        }
    ),
    # Existing metric display and buttons
    html.Div([
        html.H2(id='current-metric', style={'textAlign': 'center', 'marginTop': '20px'}),
        html.Div([
            html.Button('Previous Metric', id='prev-button', n_clicks=0, style={
                'marginRight': '10px',
                'padding': '10px 20px',
                'fontSize': '16px',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }),
            html.Button('Next Metric', id='next-button', n_clicks=0, style={
                'padding': '10px 20px',
                'fontSize': '16px',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer'
            })
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
    dcc.Graph(id='main-graph'),
    html.Div([
        html.Label("Select Metric for 2D Table:"),
        dcc.Dropdown(
            id='table-metric-dropdown',
            options=[{'label': m, 'value': m} for m in additional_metrics],
            value=additional_metrics[0],
            style={'width': '50%', 'margin': '10px auto'}
        )
    ], style={'textAlign': 'center'}),
    html.Div(id='metric-table-container', style={'margin': '20px'})
])

# Callback for 2D table visualization
@app.callback(
    Output('metric-table-container', 'children'),
    [Input('table-metric-dropdown', 'value')]
)
def update_table(selected_metric):
    # Read the data
    df = read_and_process_data()
    
    # Pivot the data to create a 2D table
    pivot_df = df.pivot(index='SensorID', columns='soil_type', values=selected_metric)
    
    # Sort the columns based on the "Lab(Value)" row
    lab_values = df[df['SensorID'] == 'Lab(Value)'].set_index('soil_type')[selected_metric]
    sorted_columns = lab_values.sort_values().index
    pivot_df = pivot_df[sorted_columns]
    
    # Color map for the "Lab(Rating)" row
    lab_ratings = df[df['SensorID'] == 'Lab(Rating)'].set_index('soil_type')[selected_metric]
    rating_colors = lab_ratings.map({
        'vL': 'lightblue',  # Very Low
        'L': 'blue',        # Low
        'M': 'green',       # Medium
        'H': 'yellow',      # High
        'vH': 'red',        # Very High
        np.nan: 'gray'      # NaN
    })
    
    # Apply heatmap coloring for sensor rows
    def color_scale(row):
        min_val, max_val = row.min(), row.max()
        return row.apply(
            lambda x: f'rgba(255, 0, 0, {((x - min_val) / (max_val - min_val)) if pd.notna(x) else 0.1})'
        )

    row_styles = pivot_df.apply(color_scale, axis=1)
    
    # Create the table with style
    table_data = pivot_df.reset_index()
    table = dash_table.DataTable(
        data=table_data.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in table_data.columns],
        style_data_conditional=[
            # Style the "Lab(Value)" row
            *[
                {
                    'if': {'filter_query': f'{{SensorID}} = "Lab(Value)"', 'column_id': soil},
                    'backgroundColor': rating_colors[soil],
                    'color': 'black'
                } for soil in rating_colors.index
            ],
            # Style the other rows with heatmap
            *[
                {
                    'if': {'filter_query': f'{{SensorID}} = {sensor}', 'column_id': soil},
                    'backgroundColor': row_styles.loc[sensor, soil],
                    'color': 'black'
                } for sensor in row_styles.index for soil in row_styles.columns
            ]
        ]
    )

    return table

if __name__ == '__main__':
    app.run_server(debug=True)
