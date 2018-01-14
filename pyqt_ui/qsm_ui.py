import sys
import time
from math import ceil

from PyQt5 import QtCore, QtWidgets

import numpy as np
import scipy.io as scio
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector

from qsm_window import Ui_MainWindow

from functions.read_dicom import read_dicom
from functions.create_mask import create_mask
from functions.v_sharp import v_sharp
from functions.laplacian_unwrap import laplacian_unwrap
from functions.qsm_star import qsm_star
from functions.qsm_ics import qsm_ics
from functions.parser import parse_int, parse_float, parse_array

"""
UI for the Susceptibility Imaging Toolkit (SITK)
University of California, Berkeley

"""

#sys.stderr = sys.stdout

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        """Initializes the main window for the Susceptibility Imaging Toolkit UI
        1. Connects buttons, sliders, and other objects to their respective actions
        2. Initializes window variables that hold data
        3. Initializes the two canvases with a placeholder image
        4. Sets minimum values for the sliders (0)
           The maximums are set when data is viewed
        (5. Initializes the adjust contrast histogram)
        """
        super(MainWindow, self).__init__()
        self.setupUi(self)

        #Connect actions
        self.actionRead_Dicom.triggered.connect(self.on_actionRead_Dicom)
        self.actionAbout.triggered.connect(self.on_actionAbout)
        self.actionOpen.triggered.connect(self.on_actionOpen)#checkable
        self.actionSave.triggered.connect(self.on_actionSave)#checkable
        self.actionView.triggered.connect(self.on_actionView)#checkable
        self.actionSave_All.triggered.connect(self.on_actionSave_All)
        self.actionAxial.triggered.connect(self.on_actionAxial)#checkable
        self.actionSaggital.triggered.connect(self.on_actionSaggital)#checkable
        self.actionCoronal.triggered.connect(self.on_actionCoronal)#checkable
        self.actionAll_Orientations.triggered.connect(self.on_actionAll_Orientations)#checkable
        self.actionZoom_In.triggered.connect(self.on_actionZoom_In)#checkable
        self.actionZoom_Out.triggered.connect(self.on_actionZoom_Out)#checkable
        
        #Connect main page actions
        self.pushButton_create_mask.clicked.connect(self.on_create_mask)
        self.pushButton_unwrap_phase.clicked.connect(self.on_unwrap_phase)
        self.pushButton_v_sharp.clicked.connect(self.on_v_sharp)
        self.pushButton_qsm_star.clicked.connect(self.on_qsm_star)
        self.pushButton_qsm_ics.clicked.connect(self.on_qsm_ics)
        self.treeWidget_data.clicked.connect(self.on_data_clicked)
        self.treeWidget_parameters.itemChanged.connect(self.on_parameters_changed)

        #Connect all-orientation-view actions
        self.spinBox_all_x.valueChanged.connect(self.on_spinBox_all_x_changed)
        self.spinBox_all_y.valueChanged.connect(self.on_spinBox_all_y_changed)
        self.spinBox_all_z.valueChanged.connect(self.on_spinBox_all_z_changed)
        self.spinBox_all_v.valueChanged.connect(self.on_spinBox_all_v_changed)
        
        #Connect single-view actions
        self.spinBox_single_slice_left.valueChanged.connect(
            lambda: (self.on_slice_changed_left(self.spinBox_single_slice_left.value()) or
                     self.view_image()))
        self.horizontalScrollBar_single_slice_left.valueChanged.connect(
            lambda: (self.on_slice_changed_left(self.horizontalScrollBar_single_slice_left.value()) or
                     self.view_image()))
        self.spinBox_single_slice_right.valueChanged.connect(
            lambda: (self.on_slice_changed_right(self.spinBox_single_slice_right.value()) or
                     self.view_image()))
        self.horizontalScrollBar_single_slice_right.valueChanged.connect(
            lambda: (self.on_slice_changed_right(self.horizontalScrollBar_single_slice_right.value()) or
                     self.view_image()))
        self.horizontalScrollBar_single_slice_both.valueChanged.connect(
            lambda: (self.on_slice_changed_right(self.horizontalScrollBar_single_slice_both.value()) or
                     self.on_slice_changed_left(self.horizontalScrollBar_single_slice_both.value()) or
                     self.view_image()))
        self.spinBox_single_v.valueChanged.connect(self.on_spinBox_single_v_changed)
        
        #Initialize all parameters and variables
        ### Data
        self.magnitude = None
        self.phase = None
        self.brain_mask = None
        self.unwrapped = None
        self.tissue_phase = None
        self.susceptibility = None

        ### Parameters
        self.parameters = {}
        self.on_parameters_changed()

        ### Zoom
        #connect key press and release in canvases to rectangle draw, then save coordinates (like a crop)
        self.zoom_coords = {'axial':((None, None), (None, None)),
                            'coronal':((None, None), (None, None)),
                            'saggital':((None, None), (None, None)),
                            'left':((None, None), (None, None)),
                            'right':((None, None), (None, None))}
    
        ### Image display
        frame_dimensions = (741, 481)
        dpi = 1
        self.current_image = np.arange(64).reshape([4,4,4,1])
        self.current_image_left = self.current_image
        self.current_image_right = self.current_image
        
        self.frame_all_figure = Figure(frame_dimensions, dpi = dpi)
        self.frame_all_canvas = FigureCanvas(self.frame_all_figure)
        self.frame_all_canvas.setParent(self.frame_all)
        self.frame_all_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        #connect keypressing to rectangle draw for zoom
        self.frame_all_axial= self.frame_all_figure.add_subplot(131)
        self.frame_all_saggital= self.frame_all_figure.add_subplot(132)
        self.frame_all_coronal= self.frame_all_figure.add_subplot(133)
        self.frame_all_axial.set_axis_off()
        self.frame_all_saggital.set_axis_off()
        self.frame_all_coronal.set_axis_off()
        self.frame_all_figure.subplots_adjust(top = 0.999,
                                              bottom = 0.001,
                                              left = 0.001,
                                              right = 0.999,
                                              wspace = 0.005,
                                              hspace = 0.005)
        self.frame_all_axial_image = self.frame_all_axial.imshow(self.current_image[:,:,0,0],
                                                                 interpolation = 'nearest',
                                                                 cmap = 'gray',
                                                                 origin = 'lower')
        self.frame_all_saggital_image = self.frame_all_saggital.imshow(self.current_image[0,:,:,0],
                                                                       interpolation = 'nearest',
                                                                       cmap = 'gray',
                                                                       origin = 'lower')
        self.frame_all_coronal_image = self.frame_all_coronal.imshow(self.current_image[:,0,:,0],
                                                                     interpolation = 'nearest',
                                                                     cmap = 'gray',
                                                                     origin = 'lower')
        self.frame_all_canvas.draw()
        
        self.frame_single_figure = Figure(frame_dimensions, dpi = dpi)
        self.frame_single_canvas = FigureCanvas(self.frame_single_figure)
        self.frame_single_canvas.setParent(self.frame_single)
        self.frame_single_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        self.frame_single_left= self.frame_single_figure.add_subplot(121)
        self.frame_single_right= self.frame_single_figure.add_subplot(122)
        self.frame_single_left.set_axis_off()
        self.frame_single_right.set_axis_off()
        self.frame_single_figure.subplots_adjust(top = 0.999,
                                                 bottom = 0.001,
                                                 left = 0.001,
                                                 right = 0.999,
                                                 wspace = 0.005,
                                                 hspace = 0.005)
        self.frame_single_left_image = self.frame_single_left.imshow(self.current_image_left[:,:,0,0],
                                                                     interpolation = 'nearest',
                                                                     cmap = 'gray',
                                                                     origin = 'lower')
        self.frame_single_right_image = self.frame_single_right.imshow(self.current_image_left[:,:,0,0],
                                                                       interpolation = 'nearest',
                                                                       cmap = 'gray',
                                                                       origin = 'lower')
        self.frame_single_canvas.draw()

        #Zoom selectors
        self.axial_selector = RectangleSelector(self.frame_all_axial,
                                                lambda a,b: self.on_select(a,b,'axial'))
        self.coronal_selector = RectangleSelector(self.frame_all_coronal,
                                                 lambda a,b: self.on_select(a,b,'coronal'))
        self.saggital_selector = RectangleSelector(self.frame_all_saggital,
                                                   lambda a,b: self.on_select(a,b,'saggital'))
        self.left_selector = RectangleSelector(self.frame_single_left,
                                               lambda a,b: self.on_select(a,b,'left'))
        self.right_selector = RectangleSelector(self.frame_single_right,
                                                lambda a,b: self.on_select(a,b,'right'))
        self.selectors = [self.axial_selector,
                          self.coronal_selector,
                          self.saggital_selector,
                          self.left_selector,
                          self.right_selector]
        for selector in self.selectors:
            selector.set_active(False)
        
        #Set minimums
        self.spinBox_all_x.setMinimum(0)
        self.spinBox_all_y.setMinimum(0)
        self.spinBox_all_z.setMinimum(0)
        self.spinBox_all_v.setMinimum(0)
        self.spinBox_single_slice_left.setMinimum(0)
        self.spinBox_single_slice_right.setMinimum(0)
        self.spinBox_single_v.setMinimum(0)

        ###Start with 3-orientation view, open action
        self.on_actionAll_Orientations()
        self.on_actionOpen()

        self.statusBar().showMessage('Ready')

    ### Misc actions
    def on_actionAbout(self):
        """Displays an about box with information about the software.
        """
        QtWidgets.QMessageBox.about(self, "About",
        """
        Susceptibility Imaging Toolkit in Python for MRI analysis.
        This program can be used and modified for non-commerical use.
        Steven Cao, Hongjiang Wei, Chunlei Liu, University of California, Berkeley.
        Wei Li, University of Texas Health Science Center at San Antonio.
        Contact: stevencao@berkeley.edu.
        """
        )

    ### Open/Save actions
    def on_actionRead_Dicom(self):
        """Reads a set of dicom files from a folder where each file is one slice of the patient.
        Formatting information can be found in the read_dicom function.
        """
        self.statusBar().showMessage('Dicom read successfully')

    def on_actionOpen(self):
        """Checks the open action so that clicking the data treeWidget opens mat files.
        Open, Save, and View are the three options for treeWidget actions.
        Exactly one of the three must be selected at all times.
        """
        self.actionOpen.setChecked(True)
        self.actionSave.setChecked(False)
        self.actionView.setChecked(False)
        self.statusBar().showMessage('Open action checked')

    def on_actionSave(self):
        """Checks the save action so that clicking the data treeWidget saves data into a mat file.
        Open, Save, and View are the three options for treeWidget actions.
        Exactly one of the three must be selected at all times.
        """
        self.actionOpen.setChecked(False)
        self.actionSave.setChecked(True)
        self.actionView.setChecked(False)
        self.statusBar().showMessage('Save action checked')

    def on_actionView(self):
        """Checks the view action so that clicking the data treeWidget views data.
        Open, Save, and View are the three options for treeWidget actions.
        Exactly one of the three must be selected at all times.
        """
        self.actionOpen.setChecked(False)
        self.actionSave.setChecked(False)
        self.actionView.setChecked(True)
        self.statusBar().showMessage('View action checked')

    def on_actionSave_All(self):
        """
        Saves all data into a single mat so that open all (not added yet) loads an entire session.
        """
        self.statusBar().showMessage('Files saved successfully')

    def remove_nests(self, val):
        """
        If val is a length one array, it returns remove_nests(val[0]).
        Therefore, if val is a nested array, it removes the nested structure until
        val is either a value or a long array.
        """
        if hasattr(val, '__len__') and len(val) == 1:
            return self.remove_nests(val[0])
        return val
    
    def on_data_clicked(self):
        """Handles when treeWidget_data is clicked.
        Either opens, saves, or views based on which action is checked.
        """
        #rows:
        #0 - magnitude
        #1 - phase
        #2 - parameters
        #3 - mask
        #4 - unwrapped
        #5 - tissue phase
        #6 - susceptibility
        row = self.treeWidget_data.selectionModel().currentIndex().row()
        if self.actionOpen.isChecked():
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '')[0]
            if fname == '':
                self.statusBar().showMessage('File open canceled')
                return
            file = scio.loadmat(fname)
            if row == 2:
                for key in file:
                    if not key.startswith('__'):
                        self.parameters[key] = self.remove_nests(file[key])
                self.treeWidget_data.invisibleRootItem().child(row).setText(1, 'Loaded')
                self.update_treeWidget_parameters()
            else:
                for key in file:
                    if not key.startswith('__'):
                        val = file[key]
                if val is None:
                    self.statusBar().showMessage('No data found in file')
                    return
                
                self.statusBar().showMessage('Loading file...')
                if len(val.shape) == 3:
                    val = val.reshape((val.shape[0],val.shape[1],val.shape[2],1))

                if row == 0:
                    self.magnitude = val
                elif row == 1:
                    self.phase = val
                elif row == 3:
                    self.brain_mask = val
                elif row == 4:
                    self.unwrapped = val
                elif row == 5:
                    self.tissue_phase = val
                elif row == 6:
                    self.susceptibility = val
                else:
                    self.statusBar().showMessage('Row '+row+' out of bounds')
                    return
                
                self.treeWidget_data.invisibleRootItem().child(row).setText(1, str(val.shape))
                self.view_image(val, new_image = True)

                self.statusBar().showMessage('File loaded successfully')

        elif self.actionSave.isChecked():
            fname = QtWidgets.QFileDialog.getSaveFileName(self,"Save File","data.mat",filter ="MATLAB Data (*.mat)")[0]
            if fname == '':
                self.statusBar().showMessage('Save canceled')
                return
            mdict = {}
            if row == 0:
                mdict['magnitude'] = self.magnitude
            elif row == 1:
                mdict['phase'] = self.phase
            elif row == 2:
                for key in self.parameters:
                    mdict[key] = self.parameters[key]
            elif row == 3:
                mdict['brain_mask'] = self.brain_mask
            elif row == 4:
                mdict['unwrapped'] = self.unwrapped
            elif row == 5:
                mdict['tissue_phase'] = self.tissue_phase
            elif row == 6:
                mdict['susceptibility'] = self.susceptibility
            else:
                self.statusBar().showMessage('Row '+row+' out of bounds')
                return
            scio.savemat(fname, mdict)
        elif self.actionView.isChecked():
            if row == 0:
                self.view_image(self.magnitude, new_image = True)
            elif row == 1:
                self.view_image(self.phase, new_image = True)
            elif row == 2:
                self.toolBox_main.setCurrentIndex(1)
            elif row == 3:
                self.view_image(self.brain_mask, new_image = True)
            elif row == 4:
                self.view_image(self.unwrapped, new_image = True)
            elif row == 5:
                self.view_image(self.tissue_phase, new_image = True)
            elif row == 6:
                self.view_image(self.susceptibility, new_image = True)
            else:
                self.statusBar().showMessage('Row '+row+' out of bounds')
                return
        else:
            self.statusBar().showMessage('Error: No tree action checked')
            return
    
    def on_parameters_changed(self):
        """Handles when treeWidget_parameters is changed.
        Parses the text in the treeWidget and updates the parameters dict.
        """
        try:
            self.parameters['TE'] = parse_int(self.treeWidget_parameters.topLevelItem(0).child(0).text(1))
            self.parameters['voxelsize'] = parse_array(self.treeWidget_parameters.topLevelItem(0).child(1).text(1), np.float)
            self.parameters['B0'] = parse_int(self.treeWidget_parameters.topLevelItem(0).child(2).text(1))
            self.parameters['B0_dir'] = parse_array(self.treeWidget_parameters.topLevelItem(0).child(3).text(1), int)

            self.parameters['v_sharp_r'] = parse_int(self.treeWidget_parameters.topLevelItem(1).child(0).text(1))

            self.parameters['qsm_star_tau'] = parse_float(self.treeWidget_parameters.topLevelItem(2).child(0).text(1))

            self.parameters['qsm_ics_alpha'] = parse_float(self.treeWidget_parameters.topLevelItem(3).child(0).text(1))
            self.parameters['qsm_ics_beta'] = parse_float(self.treeWidget_parameters.topLevelItem(3).child(1).text(1))
            self.parameters['qsm_ics_max_iterations'] = parse_float(self.treeWidget_parameters.topLevelItem(3).child(2).text(1))
            self.parameters['qsm_ics_tolerance'] = parse_float(self.treeWidget_parameters.topLevelItem(3).child(3).text(1))
            self.statusBar().showMessage('Parameters changed successfully')
        except:
            self.update_treeWidget_parameters()
            self.statusBar().showMessage('Error: Text parsing failed')

    def update_treeWidget_parameters(self):
        """When the parameters dict is changed directly (like through loading params.mat),
        this function updates the text in treeWidget_parameters.
        """
        self.treeWidget_parameters.blockSignals(True)
        
        self.treeWidget_parameters.topLevelItem(0).child(0).setText(1, str(self.parameters['TE']))
        self.treeWidget_parameters.topLevelItem(0).child(1).setText(1, str(self.parameters['voxelsize']))
        self.treeWidget_parameters.topLevelItem(0).child(2).setText(1, str(self.parameters['B0']))
        self.treeWidget_parameters.topLevelItem(0).child(3).setText(1, str(self.parameters['B0_dir']))

        self.treeWidget_parameters.topLevelItem(1).child(0).setText(1, str(self.parameters['v_sharp_r']))

        self.treeWidget_parameters.topLevelItem(2).child(0).setText(1, str(self.parameters['qsm_star_tau']))

        self.treeWidget_parameters.topLevelItem(3).child(0).setText(1, str(self.parameters['qsm_ics_alpha']))
        self.treeWidget_parameters.topLevelItem(3).child(1).setText(1, str(self.parameters['qsm_ics_beta']))
        self.treeWidget_parameters.topLevelItem(3).child(2).setText(1, str(self.parameters['qsm_ics_max_iterations']))
        self.treeWidget_parameters.topLevelItem(3).child(3).setText(1, str(self.parameters['qsm_ics_tolerance']))

        self.statusBar().showMessage('Parameters loaded successfully')
        self.treeWidget_parameters.blockSignals(False)
        
    ### Orientation/Drawing actions
    def zoomed(self, image, key, orientation, xyz, v):
        """Helper for view_image:
        Takes in an image, a key representing which frame the image is in,
        the orientation (axial, saggital, or coronal), an xyz coordinate, and a v coordinate
        Returns the image zoomed in and sliced properly and in the proper orientation.
        """
        if orientation == 'axial':
            return image[self.zoom_coords[key][0][0]:
                         self.zoom_coords[key][1][0],
                         self.zoom_coords[key][0][1]:
                         self.zoom_coords[key][1][1],
                         xyz,v]
        elif orientation == 'saggital':
            return np.flipud(image[xyz,
                                   self.zoom_coords[key][0][1]:
                                   self.zoom_coords[key][1][1],
                                   self.zoom_coords[key][0][0]:
                                   self.zoom_coords[key][1][0],
                                   v].T)
        else:
            return np.flipud(image[self.zoom_coords[key][0][1]:
                                   self.zoom_coords[key][1][1],
                                   xyz,
                                   self.zoom_coords[key][0][0]:
                                   self.zoom_coords[key][1][0],
                                   v].T)
        
    def view_image(self, image = None, new_image = False):
        """Re-draws the canvas based on current settings:
        - Which view (Coronal, Axial, Saggital, All 3)
        - Which slice to show (found in spin boxes)
        Sets the maximums for the spin boxes and sliders based on image size.
        Sets the plot's max and min for contrast purposes.
        """
        if self.actionAll_Orientations.isChecked():
            if image is None:
                image = self.current_image
            self.current_image = image
            self.stackedWidget_image.setCurrentIndex(0)
            self.spinBox_all_x.setMaximum(image.shape[0] - 1)
            self.spinBox_all_y.setMaximum(image.shape[1] - 1)
            self.spinBox_all_z.setMaximum(image.shape[2] - 1)
            self.spinBox_all_v.setMaximum(image.shape[3] - 1)
            
            x, y, z, v = self.spinBox_all_x.value(), self.spinBox_all_y.value(), self.spinBox_all_z.value(), self.spinBox_all_v.value()

            if new_image:
                self.frame_all_axial_image.remove()
                self.frame_all_saggital_image.remove()
                self.frame_all_coronal_image.remove()
                
                self.frame_all_axial_image = self.frame_all_axial.imshow(self.zoomed(image,'axial','axial',z,v),
                                                                         interpolation = 'nearest',
                                                                         cmap = 'gray',
                                                                         origin = 'lower')
                self.frame_all_saggital_image = self.frame_all_saggital.imshow(self.zoomed(image,'saggital','saggital',x,v),
                                                                               interpolation = 'nearest',
                                                                               cmap = 'gray',
                                                                               origin = 'lower')
                self.frame_all_coronal_image = self.frame_all_coronal.imshow(self.zoomed(image,'coronal','coronal',y,v),
                                                                             interpolation = 'nearest',
                                                                             cmap = 'gray',
                                                                             origin = 'lower')
            else:
                self.frame_all_axial_image.set_data(self.zoomed(image,'axial','axial',z,v))
                self.frame_all_saggital_image.set_data(self.zoomed(image,'saggital','saggital',x,v))
                self.frame_all_coronal_image.set_data(self.zoomed(image,'coronal','coronal',y,v))

            self.frame_all_axial_image.set_clim(np.min(image), np.max(image))
            self.frame_all_saggital_image.set_clim(np.min(image), np.max(image))
            self.frame_all_coronal_image.set_clim(np.min(image), np.max(image))

            self.frame_all_canvas.draw()
        else:
            self.stackedWidget_image.setCurrentIndex(1)
            if self.radioButton_data_left.isChecked():
                if image is None:
                    image = self.current_image_left
                self.current_image_left = image
            else:
                if image is None:
                    image = self.current_image_right
                self.current_image_right = image
            if self.actionAxial.isChecked():
                max_slice_left = self.current_image_left.shape[2] - 1
                max_slice_right = self.current_image_right.shape[2] - 1
                
                self.spinBox_single_slice_left.setMaximum(max_slice_left)
                self.horizontalScrollBar_single_slice_left.setRange(0, max_slice_left)
                self.spinBox_single_slice_right.setMaximum(max_slice_right)
                self.horizontalScrollBar_single_slice_right.setRange(0, max_slice_right)
                self.horizontalScrollBar_single_slice_both.setRange(0, max(max_slice_left, max_slice_right))
                
                self.spinBox_single_v.setMaximum(max(self.current_image_left.shape[3] - 1,
                                                     self.current_image_right.shape[3] - 1))

                #v_left and v_right are the same unless the two images have different sizes in the v dimension.
                z_left = self.spinBox_single_slice_left.value()
                v_left = min(self.spinBox_single_v.value(), self.current_image_left.shape[3] - 1)
                z_right = self.spinBox_single_slice_right.value()
                v_right = min(self.spinBox_single_v.value(), self.current_image_right.shape[3] - 1)

                if new_image:
                    self.frame_single_left_image.remove()
                    self.frame_single_right_image.remove()
                    
                    self.frame_single_left_image = self.frame_single_left.imshow(self.zoomed(self.current_image_left,
                                                                                             'left','axial',z_left,v_left),
                                                                                 interpolation = 'nearest',
                                                                                 cmap = 'gray',
                                                                                 origin = 'lower')                
                    self.frame_single_right_image = self.frame_single_right.imshow(self.zoomed(self.current_image_right,
                                                                                               'right','axial',z_right,v_right),
                                                                                   interpolation = 'nearest',
                                                                                   cmap = 'gray',
                                                                                   origin = 'lower')
                else:
                    self.frame_single_left_image.set_data(self.zoomed(self.current_image_left,'left','axial',z_left,v_left))                
                    self.frame_single_right_image.set_data(self.zoomed(self.current_image_right,'right','axial',z_right,v_right))
                
                self.frame_single_left_image.set_clim(np.min(self.current_image_left), np.max(self.current_image_left))
                self.frame_single_right_image.set_clim(np.min(self.current_image_right), np.max(self.current_image_right))
            elif self.actionSaggital.isChecked():
                max_slice_left = self.current_image_left.shape[0] - 1
                max_slice_right = self.current_image_right.shape[0] - 1
                
                self.spinBox_single_slice_left.setMaximum(max_slice_left)
                self.horizontalScrollBar_single_slice_left.setRange(0, max_slice_left)
                self.spinBox_single_slice_right.setMaximum(max_slice_right)
                self.horizontalScrollBar_single_slice_right.setRange(0, max_slice_right)
                self.horizontalScrollBar_single_slice_both.setRange(0, max(max_slice_left, max_slice_right))
                
                self.spinBox_single_v.setMaximum(max(self.current_image_left.shape[3] - 1,
                                                     self.current_image_right.shape[3] - 1))
                
                x_left = self.spinBox_single_slice_left.value()
                v_left = min(self.spinBox_single_v.value(), self.current_image_left.shape[3] - 1)
                x_right = self.spinBox_single_slice_right.value()
                v_right = min(self.spinBox_single_v.value(), self.current_image_right.shape[3] - 1)

                if new_image:
                    self.frame_single_left_image.remove()
                    self.frame_single_right_image.remove()
                    
                    self.frame_single_left_image = self.frame_single_left.imshow(self.zoomed(self.current_image_left,
                                                                                             'left','saggital',x_left,v_left),
                                                                                 interpolation = 'nearest',
                                                                                 cmap = 'gray',
                                                                                 origin = 'lower')
                    self.frame_single_right_image = self.frame_single_right.imshow(self.zoomed(self.current_image_right,
                                                                                               'right','saggital',x_right,v_right),
                                                                                   interpolation = 'nearest',
                                                                                   cmap = 'gray',
                                                                                   origin = 'lower')
                else:
                    self.frame_single_left_image.set_data(self.zoomed(self.current_image_left,'left','saggital',x_left,v_left))
                    self.frame_single_right_image.set_data(self.zoomed(self.current_image_right,'right','saggital',x_right,v_right))

                self.frame_single_left_image.set_clim(np.min(self.current_image_left), np.max(self.current_image_left))
                self.frame_single_right_image.set_clim(np.min(self.current_image_right), np.max(self.current_image_right))
                                                          
            elif self.actionCoronal.isChecked():
                max_slice_left = self.current_image_left.shape[1] - 1
                max_slice_right = self.current_image_right.shape[1] - 1
                
                self.spinBox_single_slice_left.setMaximum(max_slice_left)
                self.horizontalScrollBar_single_slice_left.setRange(0, max_slice_left)
                self.spinBox_single_slice_right.setMaximum(max_slice_right)
                self.horizontalScrollBar_single_slice_right.setRange(0, max_slice_right)
                self.horizontalScrollBar_single_slice_both.setRange(0, max(max_slice_left, max_slice_right))
                
                self.spinBox_single_v.setMaximum(max(self.current_image_left.shape[3] - 1,
                                                     self.current_image_right.shape[3] - 1))
                
                y_left = self.spinBox_single_slice_left.value()
                v_left = min(self.spinBox_single_v.value(), self.current_image_left.shape[3] - 1)
                y_right = self.spinBox_single_slice_right.value()
                v_right = min(self.spinBox_single_v.value(), self.current_image_right.shape[3] - 1)

                if new_image:
                    self.frame_single_left_image.remove()
                    self.frame_single_right_image.remove()
                    
                    self.frame_single_left_image = self.frame_single_left.imshow(self.zoomed(self.current_image_left,'left','coronal',y_left,v_left),
                                                                                 interpolation = 'nearest',
                                                                                 cmap = 'gray',
                                                                                 origin = 'lower')
                    self.frame_single_right_image = self.frame_single_right.imshow(self.zoomed(self.current_image_right,'right','coronal',y_right,v_right),
                                                                                  interpolation = 'nearest',
                                                                                  cmap = 'gray',
                                                                                  origin = 'lower')
                else:
                    self.frame_single_left_image.set_data(self.zoomed(self.current_image_left,'left','coronal',y_left,v_left))
                    self.frame_single_right_image.set_data(self.zoomed(self.current_image_right,'right','coronal',y_right,v_right))

                self.frame_single_left_image.set_clim(np.min(self.current_image_left), np.max(self.current_image_left))
                self.frame_single_right_image.set_clim(np.min(self.current_image_right), np.max(self.current_image_right))
            else:
                self.statusBar().showMessage('No orientation checked')
                return
            self.frame_single_canvas.draw()
        
    def on_actionAxial(self):
        """Checks the axial view for view_image.
        Exactly one of Axial, Saggital, Coronal, and All_Orientations is checked at all times.
        """
        self.actionAxial.setChecked(True)
        self.actionSaggital.setChecked(False)
        self.actionCoronal.setChecked(False)
        self.actionAll_Orientations.setChecked(False)

        self.on_actionZoom_Out()

        self.view_image(new_image = True)
        self.statusBar().showMessage('Axial orientation checked')

    def on_actionSaggital(self):
        """Checks the saggital view for view_image.
        Exactly one of Axial, Saggital, Coronal, and All_Orientations is checked at all times.
        """
        self.actionAxial.setChecked(False)
        self.actionSaggital.setChecked(True)
        self.actionCoronal.setChecked(False)
        self.actionAll_Orientations.setChecked(False)

        self.on_actionZoom_Out()

        self.view_image(new_image = True)
        self.statusBar().showMessage('Saggital orientation checked')

    def on_actionCoronal(self):
        """Checks the coronal view for view_image.
        Exactly one of Axial, Saggital, Coronal, and All_Orientations is checked at all times.
        """
        self.actionAxial.setChecked(False)
        self.actionSaggital.setChecked(False)
        self.actionCoronal.setChecked(True)
        self.actionAll_Orientations.setChecked(False)

        self.on_actionZoom_Out()

        self.view_image(new_image = True)
        self.statusBar().showMessage('Coronal orientation checked')

    def on_slice_changed_left(self, value):
        """Called when the left slice value is changed.
        The slice value can be changed by either the left spin box, the left scroll bar, or the unified scroll bar.
        This function updates the values so that they are consistent with one another.
        __init__ relies on the fact that this function returns None because it combines this function
        with view_image in a lambda statement using or: (self.on_slice_changed_left(value) or self.view_image())
        """
        self.spinBox_single_slice_left.blockSignals(True)
        self.horizontalScrollBar_single_slice_left.blockSignals(True)
        
        self.spinBox_single_slice_left.setValue(value)
        self.horizontalScrollBar_single_slice_left.setValue(value)

        self.spinBox_single_slice_left.blockSignals(False)
        self.horizontalScrollBar_single_slice_left.blockSignals(False)

    def on_slice_changed_right(self, value):
        """Called when the right slice value is changed.
        The slice value can be changed by either the right spin box, the right scroll bar, or the unified scroll bar.
        This function updates the values so that they are consistent with one another.
        __init__ relies on the fact that this function returns None because it combines this function
        with view_image in a lambda statement using or: (self.on_slice_changed_right(value) or self.view_image())
        """
        self.spinBox_single_slice_right.blockSignals(True)
        self.horizontalScrollBar_single_slice_right.blockSignals(True)
        
        self.spinBox_single_slice_right.setValue(value)
        self.horizontalScrollBar_single_slice_right.setValue(value)

        self.spinBox_single_slice_right.blockSignals(False)
        self.horizontalScrollBar_single_slice_right.blockSignals(False)

    def on_spinBox_single_v_changed(self):
        """Called when the v value is changed.
        Currently the left and right sides share the same v value.
        """
        value = self.spinBox_single_v.value()
        self.view_image()

    def on_actionAll_Orientations(self):
        """Checks the 3-orientation view for view_image.
        Exactly one of Axial, Saggital, Coronal, and All_Orientations is checked at all times.
        """
        self.actionAxial.setChecked(False)
        self.actionSaggital.setChecked(False)
        self.actionCoronal.setChecked(False)
        self.actionAll_Orientations.setChecked(True)

        self.view_image(new_image = True)
        self.statusBar().showMessage('3-orientation view checked')

    def on_spinBox_all_x_changed(self):
        """Changes the x value for the 3-orientation view for the saggital image.
        """
        value = self.spinBox_all_x.value()
        self.view_image()
        
    def on_spinBox_all_y_changed(self):
        """Changes the y value for the 3-orientation view for the coronal image.
        """
        value = self.spinBox_all_y.value()
        self.view_image()
        
    def on_spinBox_all_z_changed(self):
        """Changes the z value for the 3-orientation view for the axial image.
        """
        value = self.spinBox_all_z.value()
        self.view_image()

    def on_spinBox_all_v_changed(self):
        """Changes the v value for the 3-orientation view for all three views.
        """
        value = self.spinBox_all_v.value()
        self.view_image()

    ### Zoom actions
    def on_actionZoom_In(self):
        if self.actionZoom_In.isChecked():
            for selector in self.selectors:
                selector.set_active(True)
            self.statusBar().showMessage('Zoom in checked')
        else:
            for selector in self.selectors:
                selector.set_active(False)
            self.statusBar().showMessage('Zoom in unchecked')

    def on_actionZoom_Out(self):
        if self.actionAll_Orientations.isChecked():
            self.zoom_coords['axial'] = ((None, None), (None, None))
            self.zoom_coords['coronal'] = ((None, None), (None, None))
            self.zoom_coords['saggital'] = ((None, None), (None, None))
        else:
            if self.radioButton_data_left.isChecked():
                self.zoom_coords['left'] = ((None, None), (None, None))
            else:
                self.zoom_coords['right'] = ((None, None), (None, None))
        self.view_image(new_image = True)
        self.statusBar().showMessage('Zoomed out successfully')

    def on_select(self, eclick, erelease, key):
        if self.zoom_coords[key][0][0] is None:
            self.zoom_coords[key] = [[ceil(eclick.ydata), ceil(eclick.xdata)],
                                     [int(erelease.ydata), int(erelease.xdata)]]
        else:
            self.zoom_coords[key][1][0] = self.zoom_coords[key][0][0] + int(erelease.ydata)
            self.zoom_coords[key][1][1] = self.zoom_coords[key][0][1] + int(erelease.xdata)
            
            self.zoom_coords[key][0][0] += ceil(eclick.ydata)
            self.zoom_coords[key][0][1] += ceil(eclick.xdata)

        #print(self.zoom_coords[key])
        self.view_image(new_image = True)
        
    ### Functions
    def on_create_mask(self):
        """Creates a mask, stores it in self.brain_mask, and updates the size in treeWidget_data.
        """
        if self.magnitude is None:
            self.statusBar().showMessage('Error: no magnitude data found')
            return
        self.statusBar().showMessage('Creating mask...')
        self.brain_mask = create_mask(self.magnitude)
        self.treeWidget_data.invisibleRootItem().child(3).setText(1, str(self.brain_mask.shape))
        self.statusBar().showMessage('Mask created successfully')

    def on_unwrap_phase(self):
        if self.phase is None:
            self.statusBar().showMessage('Error: no phase data found')
            return
        self.statusBar().showMessage('Unwrapping phase...')
        self.unwrapped, _ = laplacian_unwrap(self.phase, self.parameters['voxelsize'])
        self.treeWidget_data.invisibleRootItem().child(4).setText(1, str(self.unwrapped.shape))
        self.statusBar().showMessage('Phase unwrapped successfully')

    def on_v_sharp(self):
        if self.unwrapped is None:
            self.on_unwrap_phase()
        if self.brain_mask is None:
            self.on_create_mask()
        if self.unwrapped is None or self.brain_mask is None:
            return
        self.statusBar().showMessage('Removing background phase...')
        self.tissue_phase, self.brain_mask = v_sharp(self.unwrapped, self.brain_mask,
                                                     voxel_size = self.parameters['voxelsize'],
                                                     smv_size = self.parameters['v_sharp_r'])
        self.treeWidget_data.invisibleRootItem().child(5).setText(1, str(self.tissue_phase.shape))
        self.statusBar().showMessage('Background phase removed successfully')

    def on_qsm_star(self):
        if self.tissue_phase is None:
            self.on_v_sharp()
        if self.tissue_phase is None:
            return   
        self.statusBar().showMessage('Solving for susceptibility with QSM-STAR...')
        self.susceptibility = qsm_star(self.tissue_phase, self.brain_mask,
                                       voxel_size = self.parameters['voxelsize'],
                                       B0_dir = self.parameters['B0_dir'],
                                       B0 = self.parameters['B0'],
                                       TE = self.parameters['TE'],
                                       tau = self.parameters['qsm_star_tau'])
        self.treeWidget_data.invisibleRootItem().child(6).setText(1, str(self.susceptibility.shape))
        self.statusBar().showMessage('Susceptibility calculated successfully')

    def on_qsm_ics(self):
        if self.magnitude is None:
            self.statusBar().showMessage('Error: no magnitude data found')
            return
        if self.tissue_phase is None:
            self.on_v_sharp()
        if self.tissue_phase is None:
            return
        self.statusBar().showMessage('Solving for susceptibility with QSM-iCS...')
        self.susceptibility = qsm_ics(self.tissue_phase, self.brain_mask, self.magnitude,
                                      voxel_size = self.parameters['voxelsize'],
                                      B0_dir = self.parameters['B0_dir'],
                                      B0 = self.parameters['B0'],
                                      TE = self.parameters['TE'],
                                      alpha = self.parameters['qsm_ics_alpha'],
                                      beta = self.parameters['qsm_ics_beta'],
                                      max_iter = self.parameters['qsm_ics_max_iterations'],
                                      tol_update = self.parameters['qsm_ics_tolerance'])
        self.treeWidget_data.invisibleRootItem().child(6).setText(1, str(self.susceptibility.shape))
        self.statusBar().showMessage('Susceptibility calculated successfully')

def main(): 
    a = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(a.exec_())
    
if __name__ == "__main__":
    main()    
