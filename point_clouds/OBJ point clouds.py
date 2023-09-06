import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import open3d as o3d
from numba import cuda


print('starting')
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
df['color'] = df['radial_velocity']

# Convert radial velocity to a color (you can modify this to your liking)
max_velocity = df['radial_velocity'].max()
min_velocity = df['radial_velocity'].min()

def velocity_to_color(velocity):
    # Convert the velocity to a value between 0 and 1
    normalized_velocity = (velocity - min_velocity) / (max_velocity - min_velocity)
    
    # Convert the value to a color where blue is -1 and red is 1
    blue = max(0, 1 - 2 * normalized_velocity)
    red = max(0, 2 * normalized_velocity - 1)
    
    return [red, 0, blue]

df['color'] = df['radial_velocity'].apply(velocity_to_color)

# Create point cloud with colors
pts = df[['x', 'y', 'z']].values
colors = np.stack(df['color'].values)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Convert point cloud to mesh
mesh = o3d.geometry.TriangleMesh()
@cuda.jit
for i, point in enumerate(pcd.points):
    # Create a very small triangle for each point
    mesh.vertices.append(point)
    mesh.vertices.append(point + [0, 0.0001, 0])
    mesh.vertices.append(point + [0, 0, 0.0001])
    mesh.triangles.append([3*i, 3*i+1, 3*i+2])
    color = pcd.colors[i]
    for _ in range(3):
        mesh.vertex_colors.append(color)

# Save as .obj file
o3d.io.write_triangle_mesh("my_pts.obj", mesh)

# Visualization
o3d.visualization.draw_geometries([pcd])
print('finished')