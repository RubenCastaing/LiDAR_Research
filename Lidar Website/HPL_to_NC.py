import numpy as np
import pandas as pd
import xarray as xr
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import datetime
from pathlib import Path

#Functions for converting hpl to nc, written by Jiwei

def hpl_header(file_path,header_n=17):
    #import hpl files into intercal storage
    #header_n is the length of header
    with open(file_path, 'r') as text_file:
        lines=text_file.readlines()

    #write lines into Dictionary
    data_temp=dict()
    data_temp['header_n'] = header_n
    data_temp['filename']=lines[0].split()[-1]
    data_temp['system_id']=int(lines[1].split()[-1])
    data_temp['number_of_gates']=int(lines[2].split()[-1])
    data_temp['range_gate_length_m']=float(lines[3].split()[-1])
    data_temp['gate_length_pts']=int(lines[4].split()[-1])
    data_temp['pulses_per_ray']=int(lines[5].split()[-1])
    data_temp['number_of_waypoints_in_file']=int(lines[6].split()[-1])
    rays_n=(len(lines)-header_n)/(data_temp['number_of_gates']+1)

    '''
    number of lines does not match expected format if the number of range gates
    was changed in the measuring period of the data file (especially possible for stare data)
    '''
    if not rays_n.is_integer():
        #print('Number of lines does not match expected format')
        return np.nan

    data_temp['no_of_rays_in_file']=int(rays_n)
    data_temp['scan_type']=''.join(lines[7].split()[2:])
    data_temp['focus_range']=lines[8].split()[-1]
    data_temp['start_time']=pd.to_datetime(' '.join(lines[9].split()[-2:]))
    data_temp['resolution']=('%s %s' % (lines[10].split()[-1],'m s-1'))
    data_temp['range_gates']=np.arange(0,data_temp['number_of_gates'])

    if data_temp['scan_type'].endswith("-overlapping"):
        # print("Gateoverlap turned on.")
        data_temp['center_of_gates']=data_temp['range_gate_length_m']/2 \
        + (data_temp['range_gates']*data_temp['range_gate_length_m']/data_temp['gate_length_pts'])
    elif data_temp['scan_type'].endswith("-nonoverlapping"):
        # print("Gateoverlap turned off.")
        data_temp['center_of_gates']=(data_temp['range_gates']+0.5)*data_temp['range_gate_length_m']
    else:
        sys.exit("New string in the line Range of measurement (center of gate). Need to check!")
    return data_temp

def convert_hpl_to_ncfiles(file_path,header_n=17,ray_min=5,subfolder=True,output_path='Temp_nc_files'):
    #convert an hpl file into nc files and save them if save_file=True
    #ray_min is the minimum rays needed in order to be considered as a scan block.
    file_path = Path(file_path).absolute()

    if not file_path.name.endswith(".hpl"):
        sys.exit("Wrong file. No hpl file was found.")

    nc_folder_path = file_path.parent / file_path.stem
    # if no specified output, then use the hpl file location as the default output location
    if output_path == None:
        file_path_prefix = nc_folder_path
    else:
        file_path_prefix = Path(output_path).absolute() / nc_folder_path.stem

    #if subfolder is True, save the nc file into a subfolder named as the hpl file name
    if  subfolder == True:
        if file_path_prefix.exists():
            # should work on this in the next version to avoid overwrite nc files if they are already there.
            print("Folder exists. Should work on this in the next version to avoid overwrite nc files if they are already there.")
        else:
            file_path_prefix.mkdir(parents=True, exist_ok=False)

    #file_header = hpl_header(file_path,header_n=header_n)
    block_info,file_header=get_data_block_headers(file_path,header_n=header_n)
    scan_block_idx_array=get_scan_blocks(block_info)

    with open(file_path, 'r') as text_file:
        lines=text_file.readlines()
        if scan_block_idx_array.shape[0] == 0:
            scanblock_to_nc(lines,file_header,block_info,0,block_info['block_header_line_num'].shape[0],file_path_prefix=file_path_prefix)
        for i in range(0,scan_block_idx_array.shape[0]):
            if subfolder == True:
                file_path_prefix_tmp = file_path_prefix / (file_path_prefix.name +"_{}_".format(i))
            else:
                file_path_prefix_tmp = file_path_prefix.parent / (file_path_prefix.name +"_{}_".format(i))
            if scan_block_idx_array.shape[0] == 1:
                scanblock_to_nc(lines,file_header,block_info,scan_block_idx_array[i],block_info['block_header_line_num'].shape[0],file_path_prefix=file_path_prefix_tmp)
            elif i < (scan_block_idx_array.shape[0]-1) and (scan_block_idx_array[i+1] - scan_block_idx_array[i]>ray_min):
                scanblock_to_nc(lines,file_header,block_info,scan_block_idx_array[i],scan_block_idx_array[i+1],file_path_prefix=file_path_prefix_tmp)
            elif i == (scan_block_idx_array.shape[0]-1) and ((block_info['block_header_line_num'].shape[0]-1-scan_block_idx_array[i])>ray_min):
                scanblock_to_nc(lines,file_header,block_info,scan_block_idx_array[i],block_info['block_header_line_num'].shape[0],file_path_prefix=file_path_prefix_tmp)

def scanblock_to_nc(lines,file_header,block_info_dict,scan_block_start_index,scan_block_end_index,file_path_prefix,save_file=True):
    #get one scan block from the hpl file to into an nc file
    file_path_prefix = Path(file_path_prefix).absolute()
    gates_num = file_header['number_of_gates']
    ray_num = scan_block_end_index - scan_block_start_index
    header_n = file_header['header_n']
    data_temp = dict()

    #initialize
    data_temp['radial_velocity'] = np.full([ray_num,gates_num],np.nan) #m s-1
    data_temp['intensity'] = np.full([ray_num,gates_num],np.nan) #SNR+1
    data_temp['beta'] = np.full([ray_num,gates_num],np.nan) #m-1 sr-1
    data_temp['elevation'] = np.full(ray_num,np.nan) #degrees
    data_temp['azimuth'] = np.full(ray_num,np.nan) #degrees
    data_temp['time'] = np.full(ray_num,block_info_dict['time'][0])
    data_temp['pitch'] = np.full(ray_num,np.nan) #degrees
    data_temp['roll'] = np.full(ray_num,np.nan) #degrees

    for i in range(0,ray_num): #loop rays
        lines_temp = lines[header_n+(scan_block_start_index+i)*(gates_num+1)+1:header_n+(scan_block_start_index+i)*(gates_num+1)+gates_num+1]
        # header_temp = np.asarray(lines[header_n+(i*gates_num)+i].split(),dtype=float)
        data_temp['time'][i] = block_info_dict['time'][scan_block_start_index+i]
        data_temp['azimuth'][i] = block_info_dict['azimuth'][scan_block_start_index+i]
        data_temp['elevation'][i] = block_info_dict['elevation'][scan_block_start_index+i]
        data_temp['pitch'][i] = block_info_dict['pitch'][scan_block_start_index+i]
        data_temp['roll'][i] = block_info_dict['roll'][scan_block_start_index+i]
        for j in range(0,gates_num): #loop range gates
            line_temp=np.asarray(lines_temp[j].split(),dtype=float)
            data_temp['radial_velocity'][i,j] = line_temp[1]
            data_temp['intensity'][i,j] = line_temp[2]
            data_temp['beta'][i,j] = line_temp[3]
            if line_temp.size>4:
                data_temp['spectral_width'][i,j] = line_temp[4]

    ds = xr.Dataset(coords={'time':data_temp['time'],
                            'azimuth':(('time',),data_temp['azimuth']),
                            'elevation':(('time',),data_temp['elevation']),
                            'distance':file_header['center_of_gates'],
                            'pitch':(('time',),data_temp['pitch']),
                            'roll':(('time',),data_temp['roll']),
                            },
                    data_vars={'radial_velocity':(['time','distance'],
                                                  data_temp['radial_velocity']),
                               'beta': (('time','distance'),
                                        data_temp['beta']),
                               'intensity': (( 'time','distance'),
                                             data_temp['intensity'])
                              }
                   )
    if save_file == True:
        ds.to_netcdf(file_path_prefix.parent / (file_path_prefix.name + "{}-{}.nc".format(scan_block_start_index,scan_block_end_index)))
        ds.close()
        return
    else:
        return ds

def get_data_block_headers(file_path,header_n=17):
    #import hpl files into intercal storage
    file_header = hpl_header(file_path,header_n=header_n)
    with open(file_path, 'r') as text_file:
        lines=text_file.readlines()
    block_info = dict()
    block_info['block_header_line_num'] = np.array([17+n*(file_header["number_of_gates"]+1) \
                                                    for n in range(int(file_header['no_of_rays_in_file']))])
    block_info['block_header_line'] = np.array([lines[i] for i in block_info['block_header_line_num']])
    block_info['azimuth']= np.array([float(block_info['block_header_line'][i].split()[1]) for i in range(len(block_info['block_header_line']))])
    block_info['elevation'] = np.array([float(block_info['block_header_line'][i].split()[2]) for i in range(len(block_info['block_header_line']))])


    #use start time from file_header to get time info for each ray
    block_info['time'] = [float(block_info['block_header_line'][i].split()[0]) for i in range(len(block_info['block_header_line']))]

    block_info['time']=pd.to_timedelta([datetime.timedelta(seconds=x*60*60.0) for x in block_info['time']])

    block_info['time'] = np.array(file_header['start_time'].floor('D') + block_info['time'])

    block_info['pitch'] = np.array([float(block_info['block_header_line'][i].split()[3]) for i in range(len(block_info['block_header_line']))])
    block_info['roll'] = np.array([float(block_info['block_header_line'][i].split()[4]) for i in range(len(block_info['block_header_line']))])
    return block_info,file_header

def get_scan_blocks(block_info_dict,interval_min=0.01):
    # This function get the scan start block number index by looking at none move rays.
    #interval_min is the minimal interval of the movement, smaller than which will be considered no move and will split the scans.
    split_indicate_list = []
    for i in range(len(block_info_dict['azimuth'])-1):
        if (np.abs(block_info_dict['azimuth'][i+1] - block_info_dict['azimuth'][i]) < interval_min) and \
           (np.abs(block_info_dict['elevation'][i+1] - block_info_dict['elevation'][i]) < interval_min):
            split_indicate_list.append(1)
        else:
            split_indicate_list.append(0)
    block_index_array=np.where(np.array(split_indicate_list)==1)[0] #np.where will have two output here so need to use [0] to remove the second empty output.
    if 0 not in block_index_array:
        block_index_array=np.insert(block_index_array,0,0)
    block_index_array = remove_continuous(block_index_array)
    block_index_array = split_full_circles(block_info_dict,block_index_array)
    return block_index_array

def remove_continuous(array):
    # initialize an empty list to store the result
    result = []
    # loop through the array
    for i in range(len(array)):
    # if the current element is the first one or not continuous with the previous one
    # if use array[i] - array[i-1] < 3 below instead, it will also skip elements with differences smaller than 3.
        if i == 0 or array[i] - array[i-1] !=1:
        #append it to the result list
            result.append(array[i])
    # return the result
    return np.array(result)

def split_full_circles(block_info_dict,block_index_array):
    #This should split the full circle scans into individual circle
    for i in range(block_index_array.shape[0]):
        if i < (block_index_array.shape[0]-1):
            for j in range(block_index_array[i],block_index_array[i+1]):
                if np.abs(block_info_dict['azimuth'][j] - block_info_dict['azimuth'][i]) >=360:
                    block_index_array=np.insert(block_index_array,i+1,j)
        elif block_index_array[i] < block_info_dict['azimuth'].shape[0]:
            for j in range(block_index_array[i],block_info_dict['azimuth'].shape[0]):
                if np.abs(block_info_dict['azimuth'][j] - block_info_dict['azimuth'][i]) >=360:
                    block_index_array=np.insert(block_index_array, i+1,j)
    return block_index_array