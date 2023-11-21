import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.widgets import RangeSlider, Slider
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime


def parallel_visualization(files_to_visualize):
    # Create a pool of worker processes
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(visualize_single_file, files_to_visualize))
    return results

# Function to visualize a single file
def visualize_single_file(filename):

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

    #This is only needed for the Uni data
    #df = df[(df['azimuth'] < 130) | (df['azimuth'] > 210)]

    # After filtering, count the number of rows for each distance again
    filtered_counts = df.index.get_level_values('distance').value_counts()

    # Store the first distance to remove
    first_to_remove = None

    # Loop through each distance and its original count
    for distance, original_count in original_counts.items():
        filtered_count = filtered_counts.get(distance, 0)  # Get count, or 0 if distance is completely removed
        removed_count = original_count - filtered_count

        # Check if more than 70% of values are removed for this distance
        if removed_count / original_count > 0.3:
            first_to_remove = 1500 #This distance needs to found automatically.
            break  # Stop when we find the first such distance

    # If we found such a distance, remove it and all distances that come after it
    if first_to_remove is not None:
        df = df[df.index.get_level_values('distance') < first_to_remove]

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
    df = df.reset_index(level='distance')

    df = df.drop(columns=['beta', 'intensity', 'azimuth', 'elevation', 'pitch', 'roll'])

    #round x, y, z and obs_signal to 4 decimal places
    df = df.round({'x': 4, 'y': 4, 'z': 4, 'obs_signal': 4})

    print (df)
    return df


folder_path = '/csse/users/rca106/Desktop/Semester_2_2023/Hpl and NC files/Bottlelake_Forest'
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
file_names = sorted(file_names)[:100]  # Limiting to first 100 files, adjust this if

files_to_visualize = [os.path.join(folder_path, filename) for filename in file_names]

# Call the function to perform parallel visualization
results = parallel_visualization(files_to_visualize)

# Initialize an empty DataFrame to store combined data
combined_df = pd.DataFrame()

# Combine results into a single DataFrame
time_step = 0
for df in results:
    df['time_step'] = time_step
    combined_df = pd.concat([combined_df, df])
    time_step += 1

# Create a custom colormap
colors = ["red", "white", "blue"]
cmap = ListedColormap(colors)

# Create the figure and the 3D axis
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create an empty scatter plot with the 'seismic' colormap
sc = ax.scatter([], [], [], c=[], cmap='seismic', norm=Normalize(vmin=-5, vmax=5))

# Set axis limits
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(-1000, 1000)

clip_range = [-2, 2]  # initial clipping range

def init():
    sc._offsets3d = ([], [], [])
    sc.set_array([])
    return sc,

cbar = None


def update(frame):
    global cbar
    
    time_step = frame
    current_df = combined_df[combined_df['time_step'] == time_step]
    x = current_df['x']
    y = current_df['y']
    z = current_df['z']
    color_data = current_df['radial_velocity']

    current_time = current_df['time'].unique()
    current_time = pd.to_datetime(current_time[0]).strftime('%Y-%m-%d %H:%M:%S')  # Convert to readable form
    
    # Clip based on the value from your slider 
    slider_val = slider.val  
    filtered_df = current_df[(current_df['radial_velocity'] <= slider_val) & (current_df['radial_velocity'] >= -slider_val)]

    x = filtered_df['x']
    y = filtered_df['y']
    z = filtered_df['z']
    color_data = filtered_df['radial_velocity']
    
    # Create a custom color normalization based on the slider value
    norm = Normalize(vmin=-slider_val, vmax=slider_val)
    
    # Remove the previous color bar
    if cbar:
        cbar.remove()
    
    # Add the new color bar
    cbar = plt.colorbar(sc, ax=ax, norm=norm, cmap='seismic')
    cbar.set_label('Radial Velocity')
    
    # Update scatter plot
    sc._offsets3d = (x, y, z)
    sc.set_array(color_data)
    sc.set_norm(norm)
    
    # Set the title with the time
    ax.set_title(f'Radial wind velocity over Bottlelake forest\n {current_time}', fontsize=16)

    return sc,

# Add an axes for the slider
slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(slider_ax, 'Clip', 0.1, 50.0, valinit=50)

# Update the plot when the slider changes
def slider_update(val):
    update(int(val))
slider.on_changed(slider_update)

def update_clip(val):
    global clip_range
    clip_range = val

slider.on_changed(update_clip)

ani = FuncAnimation(fig, update, frames=range(0, 100), init_func=init, blit=False, interval = 300)

print(len(combined_df))

plt.show()

#only get the first 10000 rows of combined_df
combined_df = combined_df.iloc[:500000]

combined_df.to_csv('Semester_2_2023/Lidar Website/big_df.csv')