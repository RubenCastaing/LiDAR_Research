import pandas as pd
import plotly.express as px

df = pd.read_csv('Semester_2_2023/Lidar Website/combined_df.csv')

first_1000_rows = df.iloc[:10000]

fig = px.scatter_3d(first_1000_rows, x='x', y='y', z='z', color='radial_velocity', title='3D Scatter Plot')

fig.update_layout(scene=dict(
    aspectmode='data'
))

fig.update_traces(marker={'size': 2})

fig.show()
fig.write_html('Semester_2_2023/Lidar Website/3d_scatter_plot2.html')