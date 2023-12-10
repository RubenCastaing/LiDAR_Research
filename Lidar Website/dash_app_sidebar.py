import pandas as pd
import plotly.express as px
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
from datetime import timedelta
import xarray as xr

pio.templates.default = 'plotly_white'

# Load the data and create the first 3D scatter plot
combined_df = pd.read_csv('LiDAR_Research/Lidar Website/combined_df.csv')
smaller_by_100_df = pd.read_csv('LiDAR_Research/Lidar Website/smaller_by_100_df.csv')
big_df = pd.read_csv('LiDAR_Research/Lidar Website/big_df.csv')
df = combined_df

df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
min_date = df['date'].min()
max_date = df['date'].max()


def create_scatter3d_fig(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=df, number_of_rows = 10000):
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])] #This line of code repeats itself in most functions.
    fig3d = px.scatter_3d(filtered_df.iloc[:number_of_rows], x='x', y='y', z='z', color='radial_velocity',
                          color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scan')
    fig3d.update_layout(autosize=True, scene=dict(aspectmode='data', xaxis_title='East (m)',  yaxis_title='North (m)', zaxis_title='Height (m)'), coloraxis_colorbar=dict(title='Radial Velocity')) #Autosizing the plot
    fig3d.update_traces(marker={'size': 2})#, hoverinfo='text', hovertext=[...]) #Adding hovertext unsure if the last part is desirable
    return fig3d

def create_scatter3d_animated_fig(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=big_df):
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]
    fig3d_animated = px.scatter_3d(filtered_df, x='x', y='y', z='z', color='radial_velocity',
                                   color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scan Animated',
                                   animation_frame='time_step')
    fig3d_animated.update_layout(autosize=True, coloraxis_colorbar=dict(title='Radial Velocity'), scene=dict(xaxis_title='East (m)',  yaxis_title='North (m)', zaxis_title='Height (m)',
        xaxis=dict(range=[-1000, 1000]), yaxis=dict(range=[-1000, 1000]), zaxis=dict(range=[0, 200]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.1)
    ))
    fig3d_animated.update_traces(marker={'size': 2})
    return fig3d_animated

def aggregate_max_velocity(dataframe, OBS_range):
    #This is a function that aggregates the maximum absoltute radial velocity for each time step for a time series
    # Filter the dataframe based on OBS range
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]

    # Group by time and calculate the max radial velocity
    aggregated_df = filtered_df.groupby('time_step')['radial_velocity'].max().abs().reset_index()

    return aggregated_df

def create_time_series_fig(OBS_range, dataframe=big_df):
    aggregated_df = aggregate_max_velocity(dataframe, OBS_range)
    fig = px.line(aggregated_df, x='time_step', y='radial_velocity', title='Wind Velocity Over Time')
    fig.update_layout(xaxis_title='Time step', yaxis_title='Max Absolute Radial Velocity')
    return fig

# Create the heatmap figure
#This creates the varience in funtion which will be extra slow
def create_heat_map(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=big_df, attribute='radial_velocity'):
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]

    if attribute == 'variance':
        # Calculate the variance and explicitly name the column
        aggregated_df = filtered_df.groupby(['distance', 'time'])['radial_velocity'].var().reset_index(name='variance')
        attribute = 'variance'  # Set the attribute to the new column name
    else:
        aggregated_df = filtered_df.groupby(['distance', 'time'])['radial_velocity'].mean().reset_index(name='radial_velocity')

    heatmap_df = aggregated_df.pivot(index="distance", columns="time", values=attribute)

    fig = px.imshow(heatmap_df, 
                    labels=dict(x="Time", y="Distance", color=attribute.title()), 
                    origin='lower', title=f'Daily Vertical Stare - {attribute.title()}',
                    color_continuous_scale=color_scheme, range_color=color_range)
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
                    marks = {i: '{:.1f}'.format(i) for i in [x * 0.2 for x in range(6)]},
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
            html.H5("Varience or Velocity"),
                dcc.Dropdown(
                    id='attribute-dropdown',
                        options=[
                            {'label': 'Radial Velocity', 'value': 'radial_velocity'},
                            {'label': 'Variance', 'value': 'variance'}],
                        value='radial_velocity'  # Default value
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
     Input('color-range-slider', 'value'),
     Input('attribute-dropdown', 'value')]  # Add the new attribute dropdown input
)
def update_plots(color_scheme, OBS_range, color_range, selected_attribute):
    scatter3d_fig = create_scatter3d_fig(color_scheme, OBS_range, color_range=color_range)
    scatter3d_animated_fig = create_scatter3d_animated_fig(color_scheme, OBS_range, color_range=color_range)
    time_series_fig = create_time_series_fig(OBS_range)
    heatmap_fig = create_heat_map(color_scheme, OBS_range, color_range, attribute=selected_attribute)  # Pass the selected attribute
    return scatter3d_fig, scatter3d_animated_fig, time_series_fig, heatmap_fig



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
