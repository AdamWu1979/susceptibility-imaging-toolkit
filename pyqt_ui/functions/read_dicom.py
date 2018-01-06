from os import listdir
from os.path import isfile, join

import dicom
import numpy as np

def is_dcm(file_name):
    return len(file_name) >= 4 and (file_name.lower().endswith('.dcm') or
                                    file_name.lower().endswith('.ima') or
                                    file_name.lower().endswith('.img'))
def read_dcm(file_path, file_name):
    return dicom.read_file(file_path + '\\' + file_name)

def read_dicom(file_path):
    """read_DICOM_HW reads MRI dicom files from Siemens, GE, or Philips scanners
    
    Return values:
        raw_data: 512 x 512 x 1 x 2
        voxel_size: 3x1 col vector
           ex: .4688 .4688 2.0000
        matrix_size: 1x3 row vector
           ex: 512 512 1
        center_frequency: int
           ex: 127782747
        delta_echo_time: 1x1
           ex: 0.0036
        echo_time: 2x1
           ex: 0.0045 0.0081
        field_direction: 3x1
           ex: NaN NaN NaN
        field_strength: int
           ex: 3
    
    Sample usage:
    >>> file_path = 'C:\\Users\\Steven\\Desktop\\College\\Freshman\\QSM_py\\test_data_GE'
    >>> raw_data, voxel_size, matrix_size, CF, delta_TE, TE, B0_dir, B0 = read_DICOM_HW(file_path)
    
    Steven Cao
    University of California, Berkeley
    Updated 10 October 2017
    """
    file_list = listdir(file_path)
    file_list = [file for file in file_list if is_dcm(file)]
    assert len(file_list) > 0, 'No dicom files in {0}'.format(file_path)
    info = read_dcm(file_path, file_list[0])

    matrix_size = np.zeros(3, dtype=int)
    voxel_size = np.zeros(3, dtype=float)

    #Manufacturer: SIE, GE, or Ph
    #All files must have same manufacturer and field strength
    manufacturer = info.Manufacturer
    field_strength = int(info.MagneticFieldStrength)

    #searching for min and max slice location:

    min_slice_location = np.float(info.SliceLocation)
    min_position = np.array(info.ImagePositionPatient).astype(float)
    max_slice_location = np.float(info.SliceLocation)
    max_position = np.array(info.ImagePositionPatient).astype(float)
    echo_numbers = int(info.EchoNumbers)
    for file_name in file_list[1:]:
        file = read_dcm(file_path, file_name)
        if np.float(file.SliceLocation) < min_slice_location:
            min_slice_location = np.float(file.SliceLocation)
            min_position = np.array(file.ImagePositionPatient).astype(float)
        elif np.float(file.SliceLocation) > max_slice_location:
            max_slice_location = np.float(file.SliceLocation)
            max_position = np.array(file.ImagePositionPatient).astype(float)
        if int(file.EchoNumbers) > echo_numbers:
            echo_numbers = int(file.EchoNumbers)

    #set the sizes and other image characteristics:
    voxel_size = np.array([info.PixelSpacing[0],
                  info.PixelSpacing[1],
                  info.SliceThickness]).astype(float)

    matrix_size = np.array([info.Rows,
                   info.Columns,
                   round(np.linalg.norm(max_position - min_position)/voxel_size[2]) + 1]).astype(int)

    center_frequency = info.ImagingFrequency * 1e6

    affine_2d = np.reshape(info.ImageOrientationPatient, (3,2)).astype(float)
    affine_3d = np.concatenate((affine_2d,
                                np.array([[element] for element in (max_position - min_position) / ((matrix_size[2] - 1) * voxel_size[2] - 1)])),
                               axis = 1)
    field_direction = np.linalg.lstsq(affine_3d, [0, 0, 1])[0]

    #image processing:
    if manufacturer.startswith('GE'):
        image_magnitude = np.zeros(np.append(matrix_size, echo_numbers))
        image_phase = np.zeros(np.append(matrix_size, echo_numbers))
        echo_time = np.zeros(echo_numbers)
        #for file in files:
        for file_name in file_list:
            file = read_dcm(file_path, file_name)
            
            slice_number = int(np.sqrt(sum(np.square(np.array(file.ImagePositionPatient).astype(float)- min_position)))
                               /voxel_size[2])
            file_echo_numbers = int(file.EchoNumbers) - 1
            if echo_time[file_echo_numbers] == 0:
                echo_time[file_echo_numbers] = np.float(file.EchoTime) * 1e-3
            if int(file.InstanceNumber) % 2 == 1:
                image_magnitude[:,:,slice_number,file_echo_numbers] = list(file.pixel_array)
            else:
                image_phase[:,:,slice_number,file_echo_numbers] = list(file.pixel_array)

        max_v, min_v = np.max(image_phase[:,:,:,0]), np.min(image_phase[:,:,:,0])
        image_phase = 2 * np.pi * image_phase / (max_v - min_v)

        print('from GE MRI SCANNER')
    elif manufacturer.startswith('SIE'):
        #num_channels = 1
        image_magnitude = np.zeros(np.append(matrix_size, echo_numbers))
        image_phase = np.zeros(np.append(matrix_size, echo_numbers))
        echo_time = np.zeros(echo_numbers)
        #for file in files:
        for file_name in file_list:
            file = read_dcm(file_path, file_name)
            
            slice_number = int(np.sqrt(sum(np.square(np.array(file.ImagePositionPatient).astype(float)- min_position)))
                               /voxel_size[2])
            file_echo_numbers = int(file.EchoNumbers) - 1
            #channel = 0
            if echo_time[file_echo_numbers] == 0:
                echo_time[file_echo_numbers] = np.float(file.EchoTime) * 1e-3
            if 'M' in file.ImageType or 'm' in file.ImageType:
                image_magnitude[:,:,slice_number,file_echo_numbers] = list(file.pixel_array)
            elif 'P' in file.ImageType or 'p' in file.ImageType:
                image_phase[:,:,slice_number,file_echo_numbers] = ((list(file.pixel_array) * np.float(file.RescaleSlope) + np.float(file.RescaleIntercept))
                                                                   /(np.float(file.LargestImagePixelValue) * np.pi))
        
        print('from SIEMENS MRI SCANNER')
    elif manufacturer.startswith('Ph'):
        image_magnitude = np.zeros(np.append(matrix_size, echo_numbers))
        image_phase = np.zeros(np.append(matrix_size, echo_numbers))
        echo_time = np.zeros(echo_numbers)
        #for file in files:
        for file_name in file_list:
            file = read_dcm(file_path, file_name)
            
            slice_number = int(np.sqrt(sum(np.square(np.array(file.ImagePositionPatient).astype(float)- min_position)))
                               /voxel_size[2])
            file_echo_numbers = int(file.EchoNumbers) - 1
            if echo_time[file_echo_numbers] == 0:
                echo_time[file_echo_numbers] = np.float(file.EchoTime) * 1e-3
            if 'M' in file.ImageType or 'm' in file.ImageType:
                image_magnitude[:,:,slice_number,file_echo_numbers] = list(file.pixel_array)
            elif 'P' in file.ImageType or 'p' in file.ImageType:
                image_phase[:,:,slice_number,file_echo_numbers] = list(file.pixel_array)

        max_v, min_v = np.max(image_phase[:,:,:,0]), np.min(image_phase[:,:,:,0])
        image_phase = 2 * np.pi * image_phase / (max_v - min_v)
        
        print('from PHILIPS MRI SCANNER')
        
    raw_data = image_magnitude * np.exp(-1j * image_phase)

    if len(echo_time) == 1:
        delta_echo_time = echo_time
    else:
        delta_echo_time = echo_time[1] - echo_time[0]

    return raw_data, voxel_size, matrix_size, center_frequency, delta_echo_time, echo_time, field_direction, field_strength
