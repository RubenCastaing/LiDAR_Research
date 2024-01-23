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

#Note the first to remove is set to 2500. This needs to be found automatically.

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
            first_to_remove = 2500 #This distance needs to found automatically.
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

def wrangle_folder(folder_path):
    #Load each file path in the fodler into a list
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_names = sorted(file_names)[:10000]  # Limiting to first 10000 files, adjust as needed
    files_to_visualize = [os.path.join(folder_path, filename) for filename in file_names]
    
    #Process each file in parallel
    results = parallel_visualization(files_to_visualize)

    # Initialize an empty DataFrame to store combined data
    combined_df = pd.DataFrame()

    # Combine results into a single DataFrame
    time_step = 0
    for df in results:
        df['time_step'] = time_step
        combined_df = pd.concat([combined_df, df])
        time_step += 1
    
    return combined_df
