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
elevation_pitch_rad = np.deg2rad(elevation-pitch)
elevation_pitch_deg = elevation-pitch

print(df)

# Spherical to Cartesian conversion
# I think this may be wrong is pitch is not used to calculate y.
# Make sure to correct this.
df['x'] = distance * np.cos(elevation_pitch_deg) * np.sin(azimuth_rad) 
df['y'] = distance * np.cos(elevation_pitch_deg) * np.cos(azimuth_rad)
df['z'] = distance * np.sin(elevation_pitch_deg) #I don't think z should be effected by the azimuth. #I think I need to use sin.

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

#plt.show()



# Assume that 'df' is your DataFrame and it has columns 'x', 'y', 'z', 'radial_velocity' and 'time'

# Reset 'time' index level to a column
df = df.reset_index(level='time')

# Normalize radial_velocity for color mapping
df['color'] = (df['radial_velocity'] - df['radial_velocity'].min()) / (df['radial_velocity'].max() - df['radial_velocity'].min())

# Convert the 'time' column into the number of seconds since the first timestamp in the data
df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Get unique time values
times = df['time'].unique()

# Initialization function for the scatter plot
def init():
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig,

# Update function for each frame of the animation
# Update function for each frame of the animation
def update(i):
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([df['x'].min(), df['x'].max()])  # Set limits for X axis
    ax.set_ylim([df['y'].min(), df['y'].max()])  # Set limits for Y axis
    ax.set_zlim([df['z'].min(), df['z'].max()])  # Set limits for Z axis
    current_time = times[i]
    data = df[df['time'] == current_time]
    scatter = ax.scatter(data['x'], data['y'], data['z'], c=data['color'], cmap='viridis')
    ax.set_title('Time: ' + str(current_time))
    return fig,


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False)

plt.show()