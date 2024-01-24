#This code creates a dashboard using plotly and dash.
#It firstly creates the plots, then places them on a dashboard and
#Then uses callbacks to update the graphs from user input. 

import pandas as pd
import plotly.express as px
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import os #Is needed for hosting online.
from Data_Wrangling import wrangle_folder
import base64
import io
from file_upload import parse_contents
import json

#Getting a basic templete.
pio.templates.default = 'plotly_white'
empty_df = pd.read_csv('empty_df.csv')
combined_df = pd.read_csv('combined_df.csv')
df = empty_df

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
#Scatter3d is a less useful plot as lacks animation.
def create_scatter3d_fig(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=df, number_of_rows = 10000): 
    fig3d = px.scatter_3d(dataframe.iloc[:number_of_rows], x='x', y='y', z='z', color='radial_velocity',
                          color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scan All Data <br><sup>Wind goes from negitive to positive</sup>')
    fig3d.update_layout(autosize=True, scene=dict(aspectmode='data', xaxis_title="East", yaxis_title="North", zaxis_title="Altitude")) #Autosizing the plot and naming the axis
    fig3d.update_traces(marker={'size': 2})#, hoverinfo='text', hovertext=[...]) #Adding hovertext unsure if the last part is desirable
    return fig3d

#This plot animated Cartesion LiDAR data across time.
def create_scatter3d_animated_fig(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=df):
    #Creating the plot
    fig3d_animated = px.scatter_3d(dataframe, x='x', y='y', z='z', color='radial_velocity',
                                   color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scan Animated',
                                   animation_frame='time_step')
    #Changing the plot size and labels.
    fig3d_animated.update_layout(autosize=True, scene=dict(xaxis=dict(range=[-1000, 1000]), yaxis=dict(range=[-1000, 1000]), zaxis=dict(range=[0, 200]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.1),
        xaxis_title="East", yaxis_title="North", zaxis_title="Altitude",
    ))
    #Making the text larger
    fig3d_animated.update_traces(marker={'size': 2})
    return fig3d_animated

def aggregate_max_velocity(dataframe, OBS_range, Time_type):
    #Aggregating maximum absoltute radial velocity for each time step for the time series
    
    # Group by time and calculate the max radial velocity
    aggregated_df = dataframe.groupby(Time_type)['radial_velocity'].max().abs().reset_index() #Instead of the max, I want the average of the top 50

    return aggregated_df

#The time series can take time or time_step in Time_type.
def create_time_series_fig(OBS_range, Time_type = 'time_step', dataframe=df):  #Note this loads a different dataframe.
    #Calculating max velocity.
    aggregated_df = aggregate_max_velocity(dataframe, OBS_range, Time_type)
    #Plotting the line
    fig = px.line(aggregated_df, x= Time_type, y='radial_velocity', title='Wind Velocity Over Time <br><sup>The max velocity for each scan</sup>')
    fig.update_layout(xaxis_title='Time step', yaxis_title='Max Absolute Radial Velocity')
    return fig

# Create the heatmap figure
#I need to apply the color range to the range color part of px
def create_heat_map(color_scheme='Plasma', OBS_range=(0.019, 2), color_range = [-20,20], dataframe=df):
    #Getting the mean radial velocity for any distance and time.
    #This prevents the code breaking with duplicate values.
    aggregated_df = dataframe.groupby(['distance', 'time'])['radial_velocity'].mean().reset_index()    # Aggregate the data
    #Plotting the heatmap
    heatmap_df = aggregated_df.pivot(index="distance", columns="time", values="radial_velocity")
    fig = px.imshow(heatmap_df, 
                    labels=dict(x="Time", y="Distance", color="Radial Velocity"), 
                    origin='lower', title='Vertical Stare',
                    color_continuous_scale = color_scheme, range_color= color_range)
    return fig

#Create the wind direction graph.
def create_wind_direction_graph():
    pass

# Initialize the Dash app. Unsure what this does.
app = Dash('Wind LiDAR Analytics')

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
                value='Inferno'  # Default value
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
            html.H3(''), #Adding a gap. This is not likely to work well for mobile.
            html.H5("Wind range"),
                dcc.RangeSlider(
                    id='color-range-slider',
                    min=-50,
                    max=50,
                    step=0.5,
                    value=[-10, 10],  # Default color range
                    marks={i: str(i) for i in range(-50, 51, 10)},
            ),

            html.H3(''),
            html.H5('Upload Data (Only working for CSVs)'),
            
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
    [Output('scatter3d-plot', 'figure'),
     Output('scatter3d-animated-plot', 'figure'),
     Output('time-series-plot', 'figure'),
     Output('heatmap-plot', 'figure')],
    [Input('color-scheme-dropdown', 'value'),
     Input('OBS-range-slider', 'value'),
     Input('color-range-slider', 'value'),
     Input('uploaded-data', 'children')]
)

#This here is the current bug. The user data is loaded as a dictionary instead of a CSV so cannot upload.
#I need to google how to convert the dict to a csv OR
#I need to work with the filepath and send it to the wrangling function.
def update_plots(color_scheme, OBS_range, color_range, uploaded_data):
    
    if uploaded_data is not None:
        actual_data = uploaded_data['props']['children'][1]['props']['data']
        uploaded_dataframe = pd.DataFrame(actual_data)
        main_df = uploaded_dataframe
        print('This is the uploaded dataframe:')
        print(uploaded_dataframe)
        
    else:
        #If there is no uploaded data, use the combined dataframe.
        main_df = combined_df
        
        #This is for loading the data from a folder.
        #folder_path = 'Donqgis_data/netcdf/20230428/User2_252_20230428_124459'
        #main_df = wrangle_folder(folder_path)

    #Removing noise
    #If this breaks, it is because the data is not in a data frame so cannot find obs singnal 
    filtered_df = main_df[(main_df['obs_signal'] >= OBS_range[0]) & (main_df['obs_signal'] <= OBS_range[1])]
    
    scatter3d_fig = create_scatter3d_fig(color_scheme, OBS_range, color_range=color_range, dataframe=filtered_df)
    scatter3d_animated_fig = create_scatter3d_animated_fig(color_scheme, OBS_range, color_range=color_range, dataframe=filtered_df)
    time_series_fig = create_time_series_fig(OBS_range, dataframe=filtered_df)
    heatmap_fig = create_heat_map(color_scheme, OBS_range, color_range=color_range, dataframe=filtered_df)
    return scatter3d_fig, scatter3d_animated_fig, time_series_fig, heatmap_fig

# Run the app

#Use this if hosting online
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 80))
    app.run_server(debug=True, host='0.0.0.0', port=port)

#Use this for local hosting
#if __name__ == '__main__':
#    app.run_server(debug=True)