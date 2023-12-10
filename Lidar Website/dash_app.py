#This code creates a dashboard using plotly and dash.
#It firstly creates the plots, then places them on a dashboard and
#Then uses callbacks to update the graphs from user input. 

import pandas as pd
import plotly.express as px
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import os

#Getting a basic templete.
pio.templates.default = 'plotly_white'

# Loading the data
#Combined_df has about 6 hours of data. 
combined_df = pd.read_csv('combined_df.csv')
#small_by_100_df is every 100 point. This makes it fast.
smaller_by_100_df = pd.read_csv('smaller_by_100_df.csv')
#big_df = pd.read_csv('Semester_2_2023/Lidar Website/big_df.csv')
big_df = combined_df #Get rid of this workaround

df = smaller_by_100_df

#Calculating time values for the time series.
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
min_date = df['date'].min()
max_date = df['date'].max()

#This part of the code creates the different plots in plotly.
#Note each function calculates obs_singnal. This is extra slow.
#Scatter3d is a less useful plot as lacks animation.
def create_scatter3d_fig(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=df, number_of_rows = 10000):
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])] #This line of code repeats itself in most functions.
    fig3d = px.scatter_3d(filtered_df.iloc[:number_of_rows], x='x', y='y', z='z', color='radial_velocity',
                          color_continuous_scale=color_scheme, range_color=color_range, title='Wind Lidar Scan')
    fig3d.update_layout(autosize=True, scene=dict(aspectmode='data', xaxis_title="East", yaxis_title="North", zaxis_title="Altitude")) #Autosizing the plot and naming the axis
    fig3d.update_traces(marker={'size': 2})#, hoverinfo='text', hovertext=[...]) #Adding hovertext unsure if the last part is desirable
    return fig3d

#This plot animated Cartesion LiDAR data across time.
def create_scatter3d_animated_fig(color_scheme='Plasma', OBS_range=(0.019, 2), color_range=[-20, 20], dataframe=df):
    #Removing noise
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]
    #Creating the plot
    fig3d_animated = px.scatter_3d(filtered_df, x='x', y='y', z='z', color='radial_velocity',
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
    # Filter the dataframe based on OBS range
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]

    # Group by time and calculate the max radial velocity
    aggregated_df = filtered_df.groupby(Time_type)['radial_velocity'].max().abs().reset_index() #Instead of the max, I want the average of the top 50

    return aggregated_df

#The time series can take time or time_step in Time_type.
def create_time_series_fig(OBS_range, Time_type = 'time', dataframe=smaller_by_100_df):  #Note this loads a different dataframe.
    #Calculating max velocity.
    aggregated_df = aggregate_max_velocity(dataframe, OBS_range, Time_type)
    #Plotting the line
    fig = px.line(aggregated_df, x= Time_type, y='radial_velocity', title='Wind Velocity Over Time')
    fig.update_layout(xaxis_title='Time step', yaxis_title='Max Absolute Radial Velocity')
    return fig

# Create the heatmap figure
def create_heat_map(color_scheme='Plasma', OBS_range=(0.019, 2), dataframe=big_df):
    #Removing noise
    filtered_df = dataframe[(dataframe['obs_signal'] >= OBS_range[0]) & (dataframe['obs_signal'] <= OBS_range[1])]
    #Getting the mean radial velocity for any distance and time.
    #This prevents the code breaking with duplicate values.
    aggregated_df = filtered_df.groupby(['distance', 'time'])['radial_velocity'].mean().reset_index()    # Aggregate the data
    #Plotting the heatmap
    heatmap_df = aggregated_df.pivot(index="distance", columns="time", values="radial_velocity")
    fig = px.imshow(heatmap_df, 
                    labels=dict(x="time", y="distance", color="radial_velocity"), 
                    origin='lower', title='Daily Vertical Stare',
                    color_continuous_scale = color_scheme)
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
            
            html.H5("Time series measurement"),
                dcc.Dropdown(
                    id='Time-measurement-dropdown',
                options=[
                    {'label': 'Time Step', 'value': 'time_step'},
                    {'label': 'Time', 'value': 'time'},
                ],
                value='time_step', #Default value
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
     Input('Time-measurement-dropdown', 'value'),
     ]
)
def update_plots(color_scheme, OBS_range, color_range, Time_measurement):
    scatter3d_fig = create_scatter3d_fig(color_scheme, OBS_range, color_range=color_range)
    scatter3d_animated_fig = create_scatter3d_animated_fig(color_scheme, OBS_range, color_range=color_range)
    time_series_fig = create_time_series_fig(OBS_range, Time_measurement)
    heatmap_fig = create_heat_map(color_scheme, OBS_range)
    return scatter3d_fig, scatter3d_animated_fig, time_series_fig, heatmap_fig


# Run the app

#Use this if hosting online
#if __name__ == '__main__':
#    port = int(os.environ.get("PORT", 80))
#    app.run_server(debug=True, host='0.0.0.0', port=port)

#Use this for local hosting
if __name__ == '__main__':
    app.run_server(debug=True)