import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
from datetime import timedelta
import xarray as xr

# Load the data and create the first 3D scatter plot
combined_df = pd.read_csv('Semester_2_2023/Lidar Website/combined_df.csv')
smaller_by_100_df = pd.read_csv('Semester_2_2023/Lidar Website/smaller_by_100_df.csv')
df = smaller_by_100_df

df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
min_date = df['date'].min()
max_date = df['date'].max()


def create_scatter3d_fig(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=df, number_of_rows = 10000):
    #Include the following line once the OBS is added to the dataframe
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])] #This line of code repeats itself in most functions.
    fig3d = px.scatter_3d(filtered_df.iloc[:number_of_rows], x='x', y='y', z='z', color='radial_velocity',
                          color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scan')
    fig3d.update_layout(autosize=True, scene=dict(aspectmode='data')) #Autosizing the plot
    fig3d.update_traces(marker={'size': 2})#, hoverinfo='text', hovertext=[...]) #Adding hovertext unsure if the last part is desirable
    return fig3d

def create_scatter3d_animated_fig(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=df):
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]
    fig3d_animated = px.scatter_3d(filtered_df, x='x', y='y', z='z', color='radial_velocity',
                                   color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scan Animated',
                                   animation_frame='time_step')
    fig3d_animated.update_layout(autosize=True, scene=dict(
        xaxis=dict(range=[-1000, 1000]),
        yaxis=dict(range=[-1000, 1000]),
        zaxis=dict(range=[0, 200]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.1)
    ))
    fig3d_animated.update_traces(marker={'size': 2})
    return fig3d_animated

def aggregate_max_velocity(dataframe, OBS_range):
    #This is a function that aggregates the maximum radial velocity for each time step for a time series
    # Filter the dataframe based on OBS range
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]

    # Group by time and calculate the max radial velocity
    aggregated_df = filtered_df.groupby('time_step')['radial_velocity'].max().abs().reset_index()

    return aggregated_df

def create_time_series_fig(OBS_range, dataframe=smaller_by_100_df):
    aggregated_df = aggregate_max_velocity(dataframe, OBS_range)
    fig = px.line(aggregated_df, x='time_step', y='radial_velocity', title='Max Absolute Radial Velocity Over Time')
    return fig

# Create the heatmap figure
def create_heat_map(color_scheme='Plasma', dataframe=combined_df, OBS_range=(0.019, 2)):
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]
    heatmap_df = filtered_df.pivot(index="distance", columns="time", values="radial_velocity")
    fig = px.imshow(heatmap_df, 
                    labels=dict(x="time", y="distance", color="radial_velocity"), 
                    origin='lower', title='Daily Vertical Stare',
                    color_continuous_scale = color_scheme)
    return fig

def create_heat_map_fake_data(color_scheme='Plasma'): #Fix the colors!
    airtemps = xr.tutorial.open_dataset('air_temperature').air.sel(lon=250.0)
    fig = px.imshow(airtemps.T, color_continuous_scale=color_scheme, origin='lower', title='Daily Vertical Stare')
    return fig

# Initialize the Dash app
app = Dash(__name__)

# Define the layout of the Dash app using HTML Divs for positioning
app.layout = html.Div([

    # Main content area for graphs
        html.Div([
            html.H3("Wind Dashboard", style={'textAlign': 'center'}),
            html.Div([dcc.Loading(dcc.Graph(figure=create_scatter3d_animated_fig(), id='scatter3d-animated-plot'))], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Loading(dcc.Graph(figure=create_scatter3d_fig(), id='scatter3d-plot'))], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Loading(dcc.Graph(figure=create_heat_map(), id='heatmap-plot'))]),
            dcc.Graph(id='time-series-plot') 
        ], style={'flex': 1}),

        # Sidebar for the dropdown
        html.Div([
            html.H3("Controls"),
            html.H5("Color Scheme"),
            dcc.Dropdown(
                id='color-scheme-dropdown',
                options=[
                    {'label': 'Red-Blue', 'value': 'RdBu'},
                    {'label': 'Deep', 'value': 'Deep'},
                    {'label': 'Inferno', 'value': 'Inferno'},
                    {'label': 'Turbo', 'value': 'Turbo'},
                    {'label': 'Picnic', 'value': 'Picnic'},
                    {'label': 'Spectral', 'value': 'Spectral'},
                    {'label': 'Balance', 'value': 'Balance'},
                    {'label': 'Tropic', 'value': 'Tropic'},
                    {'label': 'Electric', 'value': 'Electric'},
                    {'label': 'Dense', 'value': 'Dense'},
                    {'label': 'Gray', 'value': 'Gray'},
                    {'label': 'Plasma', 'value': 'Plasma'},
                    {'label': 'Viridis', 'value': 'Viridis'},
                ],
                value='Inferno'  # Default value
            ),
            html.H5("OBS signal Range"),
                dcc.RangeSlider(
                    id='OBS-range-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=[0.1, 1],  # Default range
                    marks={i: '{:.1f}'.format(i) for i in np.arange(0, 1.1, 0.2)},
            ),
            html.H5("Color Range"),
                dcc.RangeSlider(
                    id='color-range-slider',
                    min=-30,
                    max=30,
                    step=1,
                    value=[-5, 5],  # Default color range
                    marks={i: str(i) for i in range(-30, 31, 10)},
            ),
            html.H5("Date Range"),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=min_date,
                    end_date=max_date,
            ),
        ], style={'width': '20%', 'padding': '20px', 'backgroundColor': '#f2f2f2'})
    ], style={'display': 'flex'})

# Callback for updating the plots based on selected color scheme, OBS range, and color range
@app.callback(
    [Output('scatter3d-plot', 'figure'),
     Output('scatter3d-animated-plot', 'figure'),
     Output('time-series-plot', 'figure'),
     Output('heatmap-plot', 'figure')],
    [Input('color-scheme-dropdown', 'value'),
     Input('OBS-range-slider', 'value'),
     Input('color-range-slider', 'value')]  # Include the color range slider input
)
def update_plots(color_scheme, OBS_range, color_range):
    scatter3d_fig = create_scatter3d_fig(color_scheme, OBS_range, color_range=color_range)
    scatter3d_animated_fig = create_scatter3d_animated_fig(color_scheme, OBS_range, color_range=color_range)
    time_series_fig = create_time_series_fig(OBS_range)
    heatmap_fig = create_heat_map(color_scheme)
    return scatter3d_fig, scatter3d_animated_fig, time_series_fig, heatmap_fig





# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
