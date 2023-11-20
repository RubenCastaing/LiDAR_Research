import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

df = pd.read_csv('Semester_2_2023/Lidar Website/combined_df.csv')

# Pivot the DataFrame for the heatmap
heatmap_df = df.pivot(index="distance", columns="time", values="radial_velocity")


# Create the heatmap figure
def create_heatmap():
    fig = px.imshow(heatmap_df, 
                    labels=dict(x="Time", y="Distance", color="Radial Velocity"), 
                    origin='lower')
    return fig

# Initialize the Dash app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    dcc.Graph(figure=create_heatmap())
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
