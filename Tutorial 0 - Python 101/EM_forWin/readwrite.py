# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile
from os import getcwd, mkdir, path
import matplotlib
# matplotlib.rcParams["backend"] = "Agg"
import hyperspy.api as hs

from EM import functions


def saveTiff(array, filename, bits=16):
    t = 'uint{}'.format(bits)

    imarr = ((2**bits-1)*functions.scale(array)).astype(t)

    if len(np.shape(imarr))>2:
        imarr = np.swapaxes(imarr,0,2)
    tifffile.imsave(filename+'.tif', imarr)


def readTiffStack(filename):
    image = tifffile.imread(filename)
    data = np.array(image)
    if len(np.shape(data))==3:
        data = np.swapaxes(data, 0, 2)
    return data

def makeNewDir(name):
    newdir = getcwd()+'/{}/'.format(name)

    if path.exists(newdir):
        print('Directory exists.')
    else:
        mkdir(newdir)

    return newdir


def read(self, filename):

    if filename.endswith('.tif'):
        self.data = readTiffStack(filename)
        
        if self.data.ndim == 3:
            self.x, self.y, self.z = np.shape(self.data)
            self.is3D = True
        else:
            self.x, self.y = np.shape(self.data)
            self.is3D = False



    else:
        hyper = hs.load(filename)

        if (hyper.data).ndim==3:
            self.data = np.swapaxes(hyper.data, 0, 2)
            self.x, self.y, self.z = np.shape(self.data)
            self.is3D = True
        else:
            self.data = hyper.data
            self.x, self.y = np.shape(self.data)
            self.is3D = False

    self.info = {}
    self.original_data = self.data


    if filename.endswith('.dm3'):
        self.info['pixelSize'] = hyper.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Scale
        self.info['unit'] = hyper.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Units
        self.info['camera'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Acquisition.Device.Name
        self.info['date'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.DataBar.Acquisition_Date
        self.info['time'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.DataBar.Acquisition_Time
        self.info['exposureTime'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.DataBar.Exposure_Time_s
        self.info['magnification'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Indicated_Magnification

        if hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Illumination_Mode == "u'STEM'":
            self.info['cameraLength'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.STEM_Camera_Length      

        self.info['voltage'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Voltage

    if filename.endswith('.ser'):
        index = int(filename.rstrip('.ser').split('_')[-1])
        print(filename.split('.')[0].rstrip('_{}'.format(index))+'.emi')
        if path.exists(filename.split('.')[0].rstrip('_{}'.format(index))+'.emi'):
            try:
                print("getting metadata from corresponding emi file")
                hyper = hs.load(filename.split('.')[0].rstrip('_{}'.format(index))+'.emi')[index-1]
                mode = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Mode.split(' ')[1]
                if mode=='STEM':
                    self.info['dwellTime'] = float(hyper.original_metadata.ObjectInfo.AcquireInfo.DwellTimePath)
                    self.info['frameTime'] = float(hyper.original_metadata.ObjectInfo.AcquireInfo.FrameTime)
                    self.info['cameraLength'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Camera_length_m #meters
                    self.info['scanRotation'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Stem_rotation_deg
                elif mode=='TEM':
                    self.info['camera'] = hyper.original_metadata.ObjectInfo.AcquireInfo.CameraNamePath
                    self.info['exposureTime']= hyper.original_metadata.ObjectInfo.AcquireInfo.DwellTimePath

                self.info['date'] = hyper.original_metadata.ObjectInfo.AcquireDate
                self.info['time'] = hyper.original_metadata.ObjectInfo.AcquireDate.split(' ')[3]
                self.info['magnification'] = hyper.original_metadata.ObjectInfo.AcquireInfo.Magnification
                self.info['voltage'] = hyper.original_metadata.ObjectInfo.ExperimentalConditions.MicroscopeConditions.AcceleratingVoltage
                self.info['C2'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.C2_Aperture_um #um
                self.info['defocus'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Defocus_um #um
                self.info['illuminatedArea'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Illuminated_Area_Diameter_um
                try:
                    self.info['objectiveAperture'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.OBJ_Aperture_um
                except:
                    self.info['objectiveAperture'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.OBJ_Aperture

                self.info['spotSize'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Spot_size
                self.info['stageA'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Stage_A_deg
                self.info['stageB'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Stage_B_deg
                self.info['stageX'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Stage_X_um
                self.info['stageY'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Stage_Y_um
                self.info['stageZ'] = hyper.original_metadata.ObjectInfo.ExperimentalDescription.Stage_Z_um
            except:
                print("emi metadata doesn't make sense, using ser metadata")
        else:
            print("no emi fouund at {}".format(filename.rstrip('{}_.ser'.format(index))+'.emi'))

        self.info['pixelSize'] = hyper.original_metadata.ser_header_parameters.CalibrationDeltaX * 10**9
        self.info['unit'] = 'nm'
        
    if filename.endswith('.dm4'):
        try: #get metadata
            self.info['pixelSize'] = hyper.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Scale
            self.info['unit'] = hyper.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Units
            self.info['camera'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Acquisition.Device.Name
            self.info['date'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.DataBar.Acquisition_Date
            self.info['time'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.DataBar.Acquisition_Time
            self.info['exposureTime'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.DataBar.Exposure_Time_s

            self.info['magnification'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Indicated_Magnification
            if hyper.original_metadata.ImageList.TagGroup0.ImageTags.EELS_Spectrometer.Slit_inserted==1:
                self.info['slitWidth'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.EELS_Spectrometer.Slit_width_eV
            self.info['voltage'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Voltage
            self.info['stageX'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Stage_Position.Stage_X
            self.info['stageY'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Stage_Position.Stage_Y
            self.info['stageZ']= hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Stage_Position.Stage_Z
            self.info['stageA'] = hyper.original_metadata.ImageList.TagGroup0.ImageTags.Microscope_Info.Stage_Position.Stage_Alpha
        except:
            "Some part of metadata failed. Check self.info"
        
    return


def scaleBar(data, pixelsize, unit):
    if data.ndim == 3:
        data = data[:,:,0]
    y, x = (np.shape(data))
    target = x/10*pixelsize
    try:
        if unit.encode('utf-8') in ['micron', 'um', 'µm', "u'\xb5m'"]:
            unit = 'µm'.decode('utf-8')
            lengths = [0.5, 1, 2, 5]
        elif unit in ['nm', 'nanometer']:
            unit = 'nm'
            lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        else:
            print("I don't know this unit")
            unit = 'pixels'
            lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    except:
        print("I don't know this unit")
        unit = 'pixels'
        lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    optimize = []
    for i in range(len(lengths)):
        optimize.append(abs(lengths[i]-target))
    scalebar = lengths[optimize.index(min(optimize))]
    scalebarpix = scalebar//pixelsize
    scalebarx = int(0.9*x)-scalebarpix//2
    scalebary = int(0.9*y)

    data = 255*functions.scale(data)
    hist, bins = np.histogram(data, bins=255)
    histmax = bins[np.argmax(hist)]
    im = Image.fromarray(data)


    if histmax>np.mean(hist):
        color = np.min(data)
    else:
        color = np.max(data)
    draw = ImageDraw.Draw(im)
    fontpath = "/Library/Fonts/Verdana.ttf"
    font = ImageFont.truetype(fontpath, int(2*y/50))
    textx, texty = font.getsize(str(scalebar)+" "+unit)
    draw.line([(scalebarx, scalebary), (scalebarx+scalebarpix, scalebary)], fill = color, width = int(y/100))
    draw.text((int(0.9*x-textx/2), scalebary+int(y/100)), str(scalebar)+" "+unit, font=font, fill = color)

    return np.array(im)

def makeNewDir(name):
    newdir = getcwd()+'/{}/'.format(name)

    if path.exists(newdir):
        print('Directory exists.')
    else:
        mkdir(newdir)

    return newdir
