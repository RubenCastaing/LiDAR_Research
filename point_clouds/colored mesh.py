import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import seaborn as sns
import open3d as o3d
import scipy.spatial as spatial

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
intensity = df['intensity']

# Calculate obs_signal based on intensity
intensity = df['intensity']
obs_signal = abs(intensity - 1)

# Add obs_signal to DataFrame
df['obs_signal'] = obs_signal

# Filter rows where obs_signal is less than 0.01 or greater than 1
df = df[df['obs_signal'] >= 0.019]
df = df[df['obs_signal'] <= 1]

# Further processing, like removing points where elevation is not equal to 2
df = df[df['elevation'] == 2]

df = df[(df['azimuth'] < 130) | (df['azimuth'] > 210)]

# Convert azimuth and elevation from degrees to radians
azimuth_rad = np.deg2rad(azimuth)
elevation_pitch_rad = np.deg2rad(elevation)#-pitch)
elevation_pitch_deg = elevation#-pitch #mixing it with pitch makes it look weird

# Spherical to Cartesian conversion
#Jaiwei's calculations
#df['x'] = distance * np.cos(elevation_pitch_deg) * np.sin(azimuth_rad)
#df['y'] = distance * np.cos(elevation_pitch_deg) * np.cos(azimuth_rad)
#df['z'] = distance * np.sin(elevation_pitch_deg)


#My calcluations
df['x'] = distance * np.sin(elevation_pitch_deg) * np.cos(azimuth_rad)
df['y'] = distance * np.sin(elevation_pitch_deg) * np.sin(azimuth_rad)
df['z'] = distance * np.cos(elevation_pitch_deg) * -1 #negative to make it look right. I don't know why.


# Reset 'time' index level to a column
df = df.reset_index(level='time')

# Convert the 'time' column into the number of seconds since the first timestamp in the data
df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()

# Get unique time values
times = df['time'].unique()

# Convert radial_velocity values to RGB using a blue-to-red colormap.
#spectral and set1 are also good for seeing different things in the data.
colors = plt.cm.seismic((df['radial_velocity'] - df['radial_velocity'].min()) / (df['radial_velocity'].max() - df['radial_velocity'].min()))[:, :3]

# Create point cloud
pts = df[['x', 'y', 'z']].values
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

#Code for a point cloud
# Create a coordinate frame
#frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])

# Write point cloud to PLY (optional)
#o3d.io.write_point_cloud("my_pts_colored.ply", pcd, write_ascii=True)

# Visualization
#o3d.visualization.draw_geometries([pcd])


# Assume `pcd` is your point cloud
# Estimate normals (necessary for Poisson reconstruction)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
o3d.visualization.draw_geometries([pcd])


# Orient the normals
pcd.orient_normals_consistent_tangent_plane(100)

# Perform Poisson reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# You can also crop the mesh using the density (optional)
mesh.remove_vertices_by_mask(densities < np.max(densities) * 0.1)


#I will need to think about what the size will ought to be!
# Create a coordinate frame
#frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])

# Visualization
o3d.visualization.draw_geometries([mesh])

# Optionally, you can save the mesh as a file
o3d.io.write_triangle_mesh("/csse/users/rca106/Desktop/Semester 2 2023/Points_files/new_mesh.ply", mesh, write_ascii=True)
print('done')

