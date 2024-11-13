import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
from plotly.subplots import make_subplots

app = dash.Dash(__name__)
server = app.server

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
    # Current Metric Display and Buttons
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
    # Dropdown Menu for Metric Selection
    html.Div([
        dcc.Dropdown(
            id='metric-dropdown',
            options=[{'label': m, 'value': idx} for idx, m in enumerate(metrics)],
            value=0,
            clearable=False,
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'padding': '20px'}),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    ),
    dcc.Store(id='metric-index', data=0)  # Store the current metric index
])

@app.callback(
    Output('metric-index', 'data'),
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks'),
     Input('metric-dropdown', 'value')],
    [State('metric-index', 'data')]
)


def update_metric_index(prev_clicks, next_clicks, dropdown_value, current_index):
    ctx = dash.callback_context

    if not ctx.triggered:
        return current_index
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'prev-button':
        current_index = (current_index - 1) % len(metrics)
    elif button_id == 'next-button':
        current_index = (current_index + 1) % len(metrics)
    elif button_id == 'metric-dropdown':
        current_index = dropdown_value

    return current_index

# Synchronize the dropdown with the current metric index
@app.callback(
    Output('metric-dropdown', 'value'),
    [Input('metric-index', 'data')]
)
def update_dropdown(metric_index):
    return metric_index

@app.callback(
    Output('current-metric', 'children'),
    [Input('metric-index', 'data')]
)
def update_current_metric(metric_index):
    metric = metrics[metric_index]
    return f"Current Metric: {metric}"

@app.callback(
    Output('main-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('metric-index', 'data')]
)
def update_graph(n_intervals, metric_index):
    df = read_and_process_data()
    stats_df = generate_statistics(df)
    metric = metrics[metric_index]

    # Get the list of soil_types in sorted order
    soil_types = sorted(df['soil_type'].unique())

    # Create a subplot with all three plots
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Time Series", "Boxplot", "Mean Value"))

    # Time Series
    time_series_fig = create_time_series(df, metric)
    for trace in time_series_fig['data']:
        fig.add_trace(trace, row=1, col=1)
    fig.update_xaxes(title_text='Time (seconds)', row=1, col=1)
    fig.update_yaxes(title_text=metric, row=1, col=1)

    # Boxplot
    boxplot_fig = create_boxplot(df, metric, soil_types)
    for trace in boxplot_fig['data']:
        fig.add_trace(trace, row=1, col=2)
    fig.update_xaxes(title_text='Soil Type', row=1, col=2)
    fig.update_yaxes(title_text=metric, row=1, col=2)

    # Mean Barplot
    mean_barplot_fig = create_mean_barplot(stats_df, metric, soil_types)
    for trace in mean_barplot_fig['data']:
        fig.add_trace(trace, row=1, col=3)
    fig.update_xaxes(title_text='Soil Type', row=1, col=3)
    fig.update_yaxes(title_text=f'Mean {metric}', row=1, col=3)

    fig.update_layout(
        height=600,
        width=1800,
        title_text=f"Visualization for {metric}",
        showlegend=True,
        legend_title_text='Legend',
        legend=dict(x=1.05, y=1)
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
