from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import base64
import io
import pandas as pd
from Data_Wrangling import visualize_single_file
import os
import glob
import tempfile
import plotly.express as px
from HPL_to_NC import *
import shutil
import zipfile

def process_zip_contents(contents, filename):
    # Decode the base64 contents
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Use a temporary file to handle the zip file
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, filename)
        with open(zip_path, 'wb') as tmp_zip:
            tmp_zip.write(decoded)
        
        with zipfile.ZipFile(zip_path, 'r') as zipped_file:
            zipped_file.extractall(tmpdirname)
            
            # Process extracted files
            dataframes = []  # List to hold dataframes of extracted files
            for file_name in zipped_file.namelist():
                file_path = os.path.join(tmpdirname, file_name)
                # Assuming CSV for simplicity; adjust as needed for other file types
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                # Add conditions for other file types
                
            # Combine dataframes or handle them as needed
            # For this example, let's just return a simple component indicating success
            return html.Div([
                html.H5(f"Processed zip file: {filename}"),
                html.P(f"Contained files: {', '.join(zipped_file.namelist())}"),
                # You could also display dataframes or summaries here
            ])

# This function would then be called by parse_contents when a zip file is detected



def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'zip' in filename:
        # If the file is a zip, process it accordingly
            return process_zip_contents(contents, filename)
        elif 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'nc' in filename:
            # Temporarily save the uploaded netCDF file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
                tmp.write(decoded)
                tmp_path = tmp.name
            
            # Process the file
            df = visualize_single_file(tmp_path)

            # Clean up the temporary file
            os.remove(tmp_path)
        elif 'hpl' in filename:
            print('reading a hpl file')
            df_list = []
            # Temporarily save the uploaded HPL file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.hpl') as tmp:
                tmp.write(decoded)
                tmp_path = tmp.name
            
            # Convert the HPL file to NetCDF
            convert_hpl_to_ncfiles(tmp_path)

            # Find the subfolder within the initial directory
            for directory in os.scandir('Temp_nc_files'):
                if directory.is_dir():  # Check if it is a directory
                    subfolder_path = directory.path
                    break

            # Ensure a subfolder was found
            if subfolder_path is not None:
                # Iterate over all entries in the subfolder
                for entry in os.scandir(subfolder_path):
                    if entry.is_file() and entry.name.endswith('.nc'):  # Check if it is a file
                        df_temp = visualize_single_file(entry.path)  # Process each NC file into a DataFrame
                        df_list.append(df_temp)
                        print(len(df_temp))
            
            # Cleaning up temporary NC files
            for directory in os.scandir('Temp_nc_files'):
                if directory.is_dir():  # Check if it is a directory
                    shutil.rmtree(directory) # Delete subfolder
                else:
                    os.remove(directory) # Delete file
            
            # Clean up the temporary HPL file
            os.remove(tmp_path)

            # Add a time step column to each DataFrame before concatenation
            for i, df in enumerate(df_list):
                df['time_step'] = i  # Assign a unique time step value for each DataFrame

            # Combine all DataFrames into a single DataFrame, preserving the time step
            df = pd.concat(df_list, ignore_index=True)
            
        else:
            return html.Div(['Unsupported file type: {}'.format(filename)])

    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])

    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),
    ])

