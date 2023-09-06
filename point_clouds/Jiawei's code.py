import netCDF4 as nc
import xarray as xr

filename = r"/csse/users/rca106/Desktop/Semester 2 2023/Hpl and NC files/Library_Lidar_Data_samples/Library_2.nc"
Americas_cup_nc_file = nc.Dataset(filename, 'r', format='NETCDF4')

var_input='radial_velocity'

def lidar_3d_plot(file_input, var_input, idx_min=0, idx_max=5000, vmin=0, vmax=30):
    # Check if the file_input is an xarray Dataset, or else open it as one
    if file_input == xr.core.dataset.Dataset:
        ds_input = file_input
    else:
        ds_input = xr.open_dataset(file_input)

    # Ensure there are more than 2 timesteps to plot
    if ds_input.time.shape[0] > 2:

        # Restrict the range of distance indices if provided and if they are valid
        if (idx_min < idx_max) & (idx_min >= 0):
            ds_input = ds_input.isel(distance=range(int(idx_min), min(
                int(idx_max), int(ds_input.distance.shape[0]))))

        # Calculate the x, y, z coordinates using a separate function 'calculate_3d_coord'
        x, y, z = calculate_3d_coord(ds_input)

        # Determine color mapping attributes with another function 'calc_color_map'
        color_min, color_max, cmap = calc_color_map(ds_input[var_input], vmin, vmax)

        # Create a Plotly figure with a 3D surface plot using the coordinates and color mapping
        fig = go.Figure(data=[go.Surface(x=x.values, y=y.values, z=z.values, cmin=color_min,
                        cmax=color_max, surfacecolor=ds_input[var_input].values, colorscale=cmap)])
    else:
        # If there are not enough timesteps, create an empty figure
        fig = go.Figure()

    # Set layout properties including title, size, and margins
    fig.update_layout(title='3D LiDAR Scan ' + str(ds_input[var_input].time[0].values)[
                      :16], autosize=True, width=500, height=400, margin=dict(l=10, r=10, b=10, t=30))

    # Configure the Plotly pane, disabling scrollZoom
    config = dict({'scrollZoom': False})
    lidar_3d_pane = pn.pane.Plotly(fig, config=config)

    # If ds_input was opened in this function, close it
    if file_input != xr.core.dataset.Dataset:
        ds_input.close()

    # Return the Plotly pane to be displayed
    return lidar_3d_pane

lidar_3d_plot(Americas_cup_nc_file, var_input, idx_min=0, idx_max=5000, vmin=0, vmax=30)