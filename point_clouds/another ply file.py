import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import seaborn as sns
import open3d as o3d

filename = r"/csse/users/rca106/Desktop/Semester 2 2023/Hpl and NC files/Library_Lidar_Data_samples/Library_2.nc"
dataset_xr = xr.open_dataset(filename)

# Convert variables into pandas dataframes
df = dataset_xr.to_dataframe()

# Extract values for calculation
distance = df.index.get_level_values('distance')
azimuth = df['azimuth']
elevation = df['elevation']
pitch = df['pitch']
roll = df['roll']

# Convert azimuth and elevation from degrees to radians
azimuth_rad = np.deg2rad(azimuth)
elevation_pitch_rad = np.deg2rad(elevation-pitch)
elevation_pitch_deg = elevation-pitch

# Spherical to Cartesian conversion
df['x'] = distance * np.cos(elevation_pitch_deg) * np.sin(azimuth_rad) 
df['y'] = distance * np.cos(elevation_pitch_deg) * np.cos(azimuth_rad)
df['z'] = distance * np.sin(elevation_pitch_deg)

# Assume that 'df' is your DataFrame and it has columns 'x', 'y', 'z', and 'radial_velocity'
df['color'] = (df['radial_velocity'] - df['radial_velocity'].min()) / (df['radial_velocity'].max() - df['radial_velocity'].min())
df['color'] = df['radial_velocity']

# Reset 'time' index level to a column
df = df.reset_index(level='time')

# Normalize radial_velocity for color mapping
df['color'] = (df['radial_velocity'] - df['radial_velocity'].min()) / (df['radial_velocity'].max() - df['radial_velocity'].min())

# Convert the 'time' column into the number of seconds since the first timestamp in the data
df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()

# Get unique time values
times = df['time'].unique()

# Create point cloud
pts = df[['x', 'y', 'z']].values
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
#o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=True)
#pcd = o3d.io.read_point_cloud('my_pts.ply')

# visualization
o3d.visualization.draw_geometries([pcd])
