import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

print('starting')

filename = r"/csse/users/rca106/Desktop/Semester 2 2023/Hpl and NC files/Library_Lidar_Data_samples/Library_2.nc"
Americas_cup_nc_file = nc.Dataset(filename, 'r', format='NETCDF4')
# Load NetCDF file
print(Americas_cup_nc_file.variables.keys())

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
elevation_rad = np.deg2rad(elevation)
elevation_deg = elevation

print(df)

# Spherical to Cartesian conversion
# I think this may be wrong is pitch is not used to calculate y.
# Make sure to correct this.
df['x'] = distance * np.cos(elevation_rad) * np.sin(azimuth_rad) 
df['y'] = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
df['z'] = distance * np.sin(elevation_rad) 

#Old version
#df['x'] = distance * np.sin(elevation_pitch_deg) * np.cos(azimuth_rad) 
#df['y'] = distance * np.sin(elevation_pitch_deg) * np.sin(azimuth_rad)
#df['z'] = distance * np.cos(elevation_pitch_deg)

'''
z = np.sin(np.deg2rad(ds_in.elevation))*ds_in.distance
    x = np.sin(np.deg2rad(ds_in.azimuth)) * \
        (np.cos(np.deg2rad(ds_in.elevation))*ds_in.distance)
    y = np.cos(np.deg2rad(ds_in.azimuth)) * \
        (np.cos(np.deg2rad(ds_in.elevation))*ds_in.distance)'''



#df.to_csv("OneDrive\Desktop\Ruben's_Stuff\.Semester 1 2023\Data 309\inital data exploration\Cartesion_coordinates.csv")



# Assume that 'df' is your DataFrame and it has columns 'x', 'y', 'z', and 'radial_velocity'

# Normalize radial_velocity for color mapping
df['color'] = (df['radial_velocity'] - df['radial_velocity'].min()) / (df['radial_velocity'].max() - df['radial_velocity'].min())

#Add this to remove the normalising for color.
df['color'] = df['radial_velocity']


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

#Consider using a different color map
scatter = ax.scatter(df['x'], df['y'], df['z'], c=df['color'], cmap='viridis')

# Add colorbar to the plot
colorbar = plt.colorbar(scatter)
colorbar.set_label('Radial Velocity')

#What does X and Y truly mean here? I assume X points North-South and Y points east west.
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.axis('equal')

plt.show()
