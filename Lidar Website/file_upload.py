from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import base64
import io
import pandas as pd
from Data_Wrangling import visualize_single_file
import os
import tempfile
import plotly.express as px

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
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

    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])

    #This is returning the netcdf file as a dataframe in pandas.
    #return df

    #This returns the whole table
    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),
    ])


if __name__ == '__main__':
    df = pd.read_csv('smaller_by_100_df.csv')

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
    html.Div([dcc.Loading(dcc.Graph(id='scatter-plot'))], style={'width': '50%', 'display': 'inline-block'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-data-upload'),
    ])

    @callback(Output('scatter-plot', 'figure'),
            [Input('upload-data', 'contents')],
            [State('upload-data', 'filename')])
    def update_plots(list_of_contents, list_of_names):
        if list_of_contents is not None:
            # Use the first file for simplicity
            content = list_of_contents[0]
            name = list_of_names[0]
            
            df = parse_contents(content, name)  # Assuming this returns a DataFrame

            # Create a simple scatter plot
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1])  # Modify as needed
            return fig
        else:
            return px.scatter()

    @callback(Output('output-data-upload', 'children'),
            [Input('upload-data', 'contents')],
            [State('upload-data', 'filename')])
    def update_output(list_of_contents, list_of_names):
        if list_of_contents is not None:
            children = [
                parse_contents(c, n) for c, n in
                zip(list_of_contents, list_of_names)]
            return children
    
    app.run(debug=True)
