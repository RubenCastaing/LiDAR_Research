#This code creates a dashboard using plotly and dash.
#It firstly creates the plots, then places them on a dashboard and
#Then uses callbacks to update the graphs from user input. 

import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import os #Is needed for hosting online.
from Data_Wrangling import wrangle_folder
import base64
import io
from file_upload import parse_contents
import json
import numpy as np
from scipy.interpolate import griddata
import logging
from logging import NullHandler

#Getting a basic templete.
pio.templates.default = 'plotly_white'
empty_df = pd.read_csv(r'C:\Users\casta\OneDrive\Desktop\Rubens_Stuff\Semester_2_2023\LiDAR_Research_main\Lidar_Website\empty_df.csv')
polar_df = pd.read_csv(r'C:\Users\casta\OneDrive\Desktop\Rubens_Stuff\Semester_2_2023\LiDAR_Research_main\Lidar_Website\polar2_df.csv')
df = polar_df

#df = wrangle_folder(folder_path) #This breaks with parellel processing. Run it in __main__.
#I'm using a global varble to ensure I don't get parellel processing issues in windows.
#This will get updated as the plots update.
main_df = None

#Calculating time values for the time series.
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
min_date = df['date'].min()
max_date = df['date'].max()

#This part of the code creates the different plots in plotly.
#Note each function calculates obs_signal. This is extra slow.

def create_3d_mesh_plot(color_scheme='Plasma', color_range=[-20, 20], dataframe =df):
    # Extract data
    x = dataframe['x'].values
    y = dataframe['y'].values
    z = dataframe['z'].values
    radial_velocity = dataframe['radial_velocity'].values

    # Check if there are enough points for interpolation
    if len(np.unique(x)) < 4 or len(np.unique(y)) < 4:
        # Return a placeholder plot or a message
        return go.Figure()

    # Interpolate data
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    grid_radial_velocity = griddata((x, y), radial_velocity, (grid_x, grid_y), method='linear')


    # Determine the maximum range among x, y, z
    max_range = np.array([x.ptp(), y.ptp(), z.ptp()]).max() / 2.0

    # Calculate the center point for each axis
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    
    # Creating the plot with the updated color scale
    fig = go.Figure(data=[
        go.Surface(
            z=grid_z, x=grid_x, y=grid_y, 
            surfacecolor=grid_radial_velocity, 
            cmin=color_range[0], 
            cmax=color_range[1],
            colorscale=color_scheme  # Updated color scale
        )
    ])

    # Update layout for consistent axis scale
    fig.update_layout(
        title='3D Mesh Plot with Radial Velocity Coloring',
        autosize=False,
        width=600,
        height=600,
        scene=dict(
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range], autorange=False),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range], autorange=False),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range], autorange=False),
            aspectmode='manual'
        )
    )
    return fig

def create_animated_3d_mesh_plot(color_scheme='RdBu', color_range=[-5, 5], df = df):
    # Ensure 'time' is a datetime column
    df['time'] = pd.to_datetime(df['time'])

    # Unique time steps
    time_steps = df['time_step'].unique()
    
    # Initial empty figure
    fig = go.Figure()

    # Create a frame for each time step
    frames = []
    for time_step in time_steps:
        df_filtered = df[df['time_step'] == time_step].copy()
        # Extract the first time for the current time step
        first_time = df_filtered['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        x = df_filtered['x'].values
        y = df_filtered['y'].values
        z = df_filtered['z'].values
        radial_velocity = df_filtered['radial_velocity'].values

        # Grid for interpolation
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
        grid_radial_velocity = griddata((x, y), radial_velocity, (grid_x, grid_y), method='linear')

        # Frame title with subtitle for each frame
        frame_title = f'Interpolated mesh plot <br><sub>Time Step: {time_step}, Time: {first_time}</sub>'
        frame = go.Frame(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, surfacecolor=grid_radial_velocity, colorscale=color_scheme, cmin=color_range[0], cmax=color_range[1])],
                        name=str(time_step),
                        layout=go.Layout(title_text=frame_title))
        frames.append(frame)

    fig.frames = frames
    
    # Play and Pause Buttons
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 300}}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}]
                    }
                ],
                "direction": "left",
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
                "pad": {"r": 10, "t": 87},
            }
        ]
    )

        # Add initial data to display before animation starts
    if len(time_steps) > 0:
        initial_time_step = time_steps[0]
        df_initial = df[df['time_step'] == initial_time_step]
        first_time_initial = df_initial['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        x = df_initial['x'].values
        y = df_initial['y'].values
        z = df_initial['z'].values
        radial_velocity = df_initial['radial_velocity'].values

        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
        grid_radial_velocity = griddata((x, y), radial_velocity, (grid_x, grid_y), method='linear')

        # Set the initial data
        fig.add_trace(go.Surface(z=grid_z, x=grid_x, y=grid_y, surfacecolor=grid_radial_velocity, colorscale=color_scheme, cmin=color_range[0], cmax=color_range[1]))

        # Initial title with subtitle
        initial_title = f'Interpolated mesh plot <br><sub>Time Step: {initial_time_step}, Time: {first_time_initial}</sub>'

        # Apply the initial title to the layout
        fig.update_layout(
            title=initial_title,
            scene=dict(zaxis=dict(range=[min(z), max(z)]), yaxis=dict(range=[min(y), max(y)]), xaxis=dict(range=[min(x), max(x)])),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=0.5)
        )


    return fig

#Scatter3d is a less useful plot as lacks animation.
def create_scatter3d_fig(color_scheme='Plasma', color_range=[-20, 20], dataframe=df, number_of_rows = 10000): 
    fig3d = px.scatter_3d(dataframe.iloc[:number_of_rows], x='x', y='y', z='z', color='radial_velocity',
                          color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scan All Data <br><sup>Wind goes from negitive to positive</sup>')
    fig3d.update_layout(autosize=True, scene=dict(aspectmode='data', xaxis_title="East", yaxis_title="North", zaxis_title="Altitude")) #Autosizing the plot and naming the axis
    fig3d.update_traces(marker={'size': 3})#, hoverinfo='text', hovertext=[...]) #Adding hovertext unsure if the last part is desirable
    return fig3d

#This plot animated Cartesion LiDAR data across time.
def create_scatter3d_animated_fig(color_scheme='Plasma', color_range=[-20, 20], dataframe=df):
    # Creating the plot
    fig3d_animated = px.scatter_3d(dataframe, x='x', y='y', z='z', color='radial_velocity',
                                   color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scatter plot <br><sup>Wind goes from negitive to positive</sup>',
                                   animation_frame='time_step')
    # Changing the plot size and labels
    fig3d_animated.update_layout(autosize=True, 
                                 scene=dict(xaxis=dict(range=[-1000, 1000]), 
                                            yaxis=dict(range=[-1000, 1000]), 
                                            zaxis=dict(range=[0, 200]),
                                            aspectmode='manual', 
                                            aspectratio=dict(x=1, y=1, z=0.1),
                                            xaxis_title="East", 
                                            yaxis_title="North", 
                                            zaxis_title="Altitude"),
                                 coloraxis_showscale=False)  # Hide the color scale for the entire figure

    # Updating marker size
    fig3d_animated.update_traces(marker=dict(size=3))

    return fig3d_animated



def aggregate_max_velocity(dataframe, Time_type):
    #Aggregating maximum absoltute radial velocity for each time step for the time series
    
    # Group by time and calculate the max radial velocity
    aggregated_df = dataframe.groupby(Time_type)['radial_velocity'].max().abs().reset_index() #Instead of the max, I want the average of the top 50
    
    idx = dataframe.groupby(Time_type)['radial_velocity'].idxmax()
    max_wind_df = dataframe.loc[idx]
    
     # Define compass directions
    compass_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    compass_bins = np.linspace(0, 360, len(compass_labels) + 1, endpoint=True)
    
    # Map the azimuth to compass labels
    max_wind_df['compass_direction'] = pd.cut(max_wind_df['azimuth'], bins=compass_bins, labels=compass_labels, right=False, ordered=False)
    
    # Select only the required columns
    max_wind_df = max_wind_df[[Time_type, 'radial_velocity', 'compass_direction']]

    return aggregated_df

#The time series can take time or time_step in Time_type.
def create_time_series_fig(Time_type = 'time_step', dataframe=df):  #Note this loads a different dataframe.
    #Calculating max velocity.
    aggregated_df = aggregate_max_velocity(dataframe, Time_type)
    #Plotting the line
    fig = px.line(aggregated_df, x= Time_type, y='radial_velocity', title='Wind Velocity Over Time <br><sup>The max velocity for each scan</sup>')
    fig.update_layout(xaxis_title='Time step', yaxis_title='Max Absolute Radial Velocity')
    return fig

def create_wind_direction_time_series(dataframe):
    # Filter for positive radial velocities
    positive_wind_df = dataframe[dataframe['radial_velocity'] > 0]

    # Sort the dataframe by time_step and radial_velocity in descending order
    sorted_df = positive_wind_df.sort_values(by=['time_step', 'radial_velocity'], ascending=[True, False])

    # Create a copy of the sorted dataframe before dropping duplicates
    max_wind_df = sorted_df.copy().drop_duplicates(subset='time_step')

    # Define compass directions
    compass_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    compass_bins = np.linspace(0, 360, len(compass_labels) + 1, endpoint=True)

    # Map the azimuth to compass labels
    max_wind_df['compass_direction'] = pd.cut(max_wind_df['azimuth'], bins=compass_bins, labels=compass_labels, right=False, ordered=False)

    # Create the line plot with compass directions
    fig = px.line(max_wind_df, x='time_step', y='compass_direction', title='Dominant Wind Direction Over Time')
    fig.update_layout(xaxis_title='Scan Number', yaxis_title='Wind Direction', yaxis={'type':'category'})

    return fig


# Create the heatmap figure
#I need to apply the color range to the range color part of px
def create_heat_map(color_scheme='Plasma', color_range=[-20, 20], dataframe=df):
    # Getting the mean radial velocity for any distance and time.
    # This prevents the code from breaking with duplicate values.
    aggregated_df = dataframe.groupby(['distance', 'time'])['radial_velocity'].mean().reset_index()
    # Plotting the heatmap
    heatmap_df = aggregated_df.pivot(index="distance", columns="time", values="radial_velocity")
    fig = px.imshow(heatmap_df,
                    labels=dict(x="Time", y="Distance", color="Radial Velocity"),
                    origin='lower',
                    title='Wind Speed Heatmap<br><sub>Data below 90m is excluded due to inaccuracy.</sub>',
                    color_continuous_scale=color_scheme,
                    range_color=color_range)
    # Update layout to adjust the title
    fig.update_layout(
        title=dict(text='Wind Speed Heatmap<br><sub>Data below 90m is excluded due to inaccuracy.</sub>')
    )
    return fig


# Initialize the Dash app.
app = Dash('Wind LiDAR Analytics')

# Configure Flask logger to ignore messages so exe files work.
log = logging.getLogger('werkzeug')
log.disabled = True

# Define the layout of the Dash app using HTML Divs for positioning
app.layout = html.Div([
    # Main content area for graphs
        html.Div([
            html.H3("Wind LiDAR Dashboard", style={'textAlign': 'center'}),
            html.Div([dcc.Loading(dcc.Graph(figure=create_scatter3d_animated_fig(), id='scatter3d-animated-plot'))], style={'width': '50%', 'display': 'inline-block'}),
            #html.Div([dcc.Loading(dcc.Graph(figure=create_scatter3d_fig(), id='scatter3d-plot'))], style={'width': '50%', 'display': 'inline-block'}),
            #html.Div([dcc.Loading(dcc.Graph(figure=create_3d_mesh_plot(), id='mesh-plot'))], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Loading(dcc.Graph(figure=create_animated_3d_mesh_plot(), id='animated_mesh_plot'))], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Loading(dcc.Graph(figure=create_heat_map(), id='heatmap-plot'))]),
            dcc.Graph(id='time-series-plot'),
            html.Div([dcc.Loading(dcc.Graph(id='wind-direction-time-series-plot'))]),

        ], style={'flex': 1}),

        # An empty div for storing data from callbacks
        html.Div(id='uploaded-data', style={'display': 'none'}),
        
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
                value='RdBu'  # Default value
            ),
            html.H3(''), #Space in the panel
            html.H5("OBS signal Cutoff"),
                dcc.RangeSlider(
                    id='OBS-range-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=[0.1, 1],  # Default range
                    marks = {i: '{:.1f}'.format(i) for i in [x * 0.2 for x in range(6)]},
            ),
                
            html.H3(''),
            html.Div([
            html.H5("Wind velocity range"),
            dcc.Slider(
                id='symmetrical-velocity-slider',
                min=0,
                max=30,
                step=0.5,
                value=5,  # Default value
                marks={i: str(i) for i in range(0, 31, 5)}
            ),
        
            # Hidden Div to store the symmetrical range values or display them
            html.Div(id='symmetrical-velocity-output')
            ]),

            html.H3(''),
            html.H5('Upload Hpl, CSV or NC files.'),
            
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            
        ], style={'width': '20%', 'padding': '20px', 'backgroundColor': '#f2f2f2'})
    ], style={'display': 'flex'})	

@app.callback(
    Output('uploaded-data', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)

def update_output(contents, filename):
    if contents is not None:
        content = contents[0]
        name = filename[0] if isinstance(filename, list) else filename
        df = parse_contents(content, name)
        return df
    return None

# Callback for updating the plots based on selected color scheme, OBS range, and color range
@app.callback(
    [Output('scatter3d-animated-plot', 'figure'),
     Output('time-series-plot', 'figure'),
     Output('heatmap-plot', 'figure'),
     Output('wind-direction-time-series-plot', 'figure'),
     Output('animated_mesh_plot', 'figure')],
    [Input('color-scheme-dropdown', 'value'),
     Input('OBS-range-slider', 'value'),
     Input('symmetrical-velocity-slider', 'value'), 
     Input('uploaded-data', 'children')]
)

#This here is the current bug. The user data is loaded as a dictionary instead of a CSV so cannot upload.
#I need to google how to convert the dict to a csv OR
#I need to work with the filepath and send it to the wrangling function.
def update_plots(color_scheme, OBS_range, color_value, uploaded_data):
    global main_df
    color_range = [-color_value, color_value]
    
    if uploaded_data is not None:
        actual_data = uploaded_data['props']['children'][1]['props']['data'] #finding the data in the JSON
        uploaded_dataframe = pd.DataFrame(actual_data)
        main_df = uploaded_dataframe
        
    else:
        #If there is no uploaded data, use the set dataframe from the top of the code.
        main_df = df
        
        #This is for loading the data from a folder.
        #folder_path = 'Donqgis_data/nc_files_bottlelake_not_stares'
        #main_df = wrangle_folder(folder_path)

    #Removing noise
    #If this breaks, it is because the data is not in a data frame so cannot find obs singnal 
    filtered_df = main_df[
    (main_df['obs_signal'] >= OBS_range[0]) & 
    (main_df['obs_signal'] <= OBS_range[1]) &
    (main_df['distance'] >= 90)
    ]
    
    scatter3d_animated_fig = create_scatter3d_animated_fig(color_scheme, color_range=color_range, dataframe=filtered_df)
    time_series_fig = create_time_series_fig(dataframe=filtered_df)
    heatmap_fig = create_heat_map(color_scheme, color_range=color_range, dataframe=filtered_df)
    wind_direction_time_series_fig = create_wind_direction_time_series(dataframe=filtered_df)
    animated_mesh_plot_fig = create_animated_3d_mesh_plot(color_scheme, color_range=color_range, df=filtered_df)
    return scatter3d_animated_fig, time_series_fig, heatmap_fig, wind_direction_time_series_fig, animated_mesh_plot_fig

# Run the app

#Use this if hosting online
#if __name__ == '__main__':
#    port = int(os.environ.get("PORT", 80))
#    app.run_server(debug=True, host='0.0.0.0', port=port)

#Use this for local hosting
if __name__ == '__main__':
    app.run_server(debug=False)