from os import listdir
from os.path import join

from pydicom import dcmread
import numpy as np
from np.linalg import norm

def is_dcm(file_name):
    return (file_name.lower().endswith('.dcm') or
            file_name.lower().endswith('.ima') or
            file_name.lower().endswith('.img'))
    
def read_dicom(folder):
    """read_dicom reads MRI dicom files from Siemens, GE, or Philips scanners.
    
    Return values:
        raw_data: complex-valued, H x W x D x E
        voxel_size: resolution of scan in mm/voxel (ex: .4688 .4688 2.0000)
        CF: center frequency in Hz (ex: 127782747)
        delta_TE: difference between echoes in ms (ex: 3.6)
        TE: times in ms of echoes (ex: 4.5 8.1 ...)
        B0_dir: direction of B0 (ex: 0 0 1)
        B0: strength of B0 in Tesla (ex: 3)
    
    Sample usage:
    >>> file_path = '.../test_data_GE'
    >>> raw_data, params = read_DICOM_HW(file_path)
    
    Steven Cao, Hongjiang Wei, Chunlei Liu
    University of California, Berkeley
    """
    files = [join(folder, f) for f in listdir(folder) if is_dcm(f)]
    assert len(files) > 0, 'No dicom files in {0}'.format(folder)
    info = dcmread(files[0])
    maker = info.Manufacturer
    print('Reading', len(files), maker, 'dicom files...')
    
    # Find min and max slice location, number of echoes
    min_slice = np.float(info.SliceLocation)
    max_slice = np.float(info.SliceLocation)
    min_pos = np.array(info.ImagePositionPatient)
    max_pos = np.array(info.ImagePositionPatient)
    max_echo = int(info.EchoNumbers)
    for f in files[1:]:
        file = dcmread(f)
        slice_loc = np.float(info.SliceLocation)
        echo = int(info.EchoNumbers)
        if slice_loc < min_slice:
            min_slice = slice_loc
            min_pos = np.array(file.ImagePositionPatient)
        if slice_loc > max_slice:
            max_slice = slice_loc
            max_pos = np.array(file.ImagePositionPatient)
        if echo > max_echo:
            max_echo = echo
    
    voxel_size = np.array([info.PixelSpacing[0], info.PixelSpacing[1],
                           info.SliceThickness])
    slices = norm(max_pos - min_pos) // voxel_size[2]
    
    # Fill mag, phase, and TE arrays
    mag = np.zeros((info.Rows, info.Columns, slices))
    phase = np.zeros((info.Rows, info.Columns, slices))
    TE = np.zeros(max_echo)
    for f in files:
        file = dcmread(f)
        slice_num = (norm(np.array(file.ImagePositionPatient) - min_pos) //
                     voxel_size[2])
        echo = int(file.EchoNumbers) - 1
        TE[echo] = float(file.EchoTime)
        if maker.startswith('GE'):
            if int(file.InstanceNumber) % 2 == 1:
                mag[:,:,slice_num,echo] = file.pixel_array
            else:
                phase[:,:,slice_num,echo] = file.pixel_array
        elif maker.startswith('Ph'):
            if 'm' in file.ImageType or 'M' in file.ImageType:
                mag[:,:,slice_num,echo] = file.pixel_array
            elif 'p' in file.ImageType or 'P' in file.ImageType:
                phase[:,:,slice_num,echo] = file.pixel_array
        elif maker.startswith('SIE'):
            if 'm' in file.ImageType or 'M' in file.ImageType:
                mag[:,:,slice_num,echo] = file.pixel_array
            elif 'p' in file.ImageType or 'P' in file.ImageType:
                phase[:,:,slice_num,echo] = ((file.pixel_array * 
                     np.float(file.RescaleSlope) + 
                     np.float(file.RescaleIntercept)) / 
                     (np.float(file.LargestImagePixelValue) * np.pi))
    if maker.startswith('GE') or maker.startswith('Ph'):
        phase = 2 * np.pi * phase / (np.max(phase) - np.min(phase))
    
    # Acq params
    CF = info.ImagingFrequency * 1e6
    if len(TE) == 1:
        delta_TE = TE
    else:
        delta_TE = TE[1] - TE[0]
    affine_2d = np.array(info.ImageOrientationPatient).reshape(3,2)
    z = (max_pos - min_pos) / ((slices - 1) * voxel_size[2] - 1)
    affine_3d = np.concatenate((affine_2d, z), axis = 1)
    B0_dir = np.linalg.lstsq(affine_3d, [0, 0, 1])[0]
    B0 = int(info.MagneticFieldStrength)
    params = {'voxel_size': voxel_size, 'CF': CF, 'delta_TE': delta_TE, 
              'TE': TE, 'B0_dir': B0_dir, 'B0': B0}
    return mag * np.exp(-1j * phase), params