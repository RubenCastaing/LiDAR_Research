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

# Before filtering, keep track of the original number of rows for each distance
original_counts = df.index.get_level_values('distance').value_counts()

# Extract values for calculation
intensity = df['intensity']
distance = df.index.get_level_values('distance')
obs_signal = abs(intensity - 1)

# Add obs_signal to DataFrame
df['obs_signal'] = obs_signal

# Filter rows where obs_signal is less than 0.01 or greater than 1
df = df[df['obs_signal'] >= 0.019]
df = df[df['obs_signal'] <= 1]

# Calculate the mode of the 'elevation' column
mode_elevation = df['elevation'].mode()[0]

# Keep only the rows where 'elevation' is equal to the mode
df = df[df['elevation'] == mode_elevation]

df = df[(df['azimuth'] < 130) | (df['azimuth'] > 210)]

# After filtering, count the number of rows for each distance again
filtered_counts = df.index.get_level_values('distance').value_counts()

# Identify distances where more than 60% of values have been removed
to_remove = []
for distance, original_count in original_counts.items():
    filtered_count = filtered_counts.get(distance, 0)  # Get count, or 0 if distance is completely removed
    removed_count = original_count - filtered_count
    if removed_count / original_count > 0.6:
        to_remove.append(distance)

# Remove all rows corresponding to the identified distances
df = df[~df.index.get_level_values('distance').isin(to_remove)]

#DELTETE THIS REPEATING CODE!!!!
distance = df.index.get_level_values('distance')
azimuth = df['azimuth']
elevation = df['elevation']
pitch = df['pitch']
roll = df['roll']
intensity = df['intensity']

# Convert azimuth and elevation from degrees to radians
azimuth_rad = np.deg2rad(df['azimuth'])
elevation_rad = np.deg2rad(df['elevation'])#-pitch)
elevation_deg = df['elevation']#-pitch #mixing it with pitch makes it look weird

#My calcluations
#This can be done in parelell.
df['x'] = distance * np.cos(elevation_rad) * np.sin(azimuth_rad) 
df['y'] = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
df['z'] = distance * np.sin(elevation_rad) 


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

# Assume `pcd` is your point cloud
# Estimate normals (necessary for Poisson reconstruction)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
o3d.visualization.draw_geometries([pcd])


# Orient the normals
pcd.orient_normals_consistent_tangent_plane(100)

# Perform Poisson reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# You can also crop the mesh using the density (optional)
mesh.remove_vertices_by_mask(densities < np.max(densities) * 0.85)


#I will need to think about what the size will ought to be!
# Create a coordinate frame
#frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])

# Visualization
o3d.visualization.draw_geometries([mesh])

# Optionally, you can save the mesh as a file
o3d.io.write_triangle_mesh("/csse/users/rca106/Desktop/Semester 2 2023/Points_files/flat_mesh.ply", mesh, write_ascii=True)
print('done')

