# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from EM import readwrite, registration, functions


class EM_image(object):
    """
    Electron microscope image. 2 or 3D.

    Input: Filename. Accepts '.bin', '.ser', '.dm3', '.dm4', '.tiff'. If 'ser' input, will look for emi of the same filename 
    in that folder for more metadata.
    """

    def __init__(self, filename):

        
        if isinstance(filename, str):
            self.f = filename.split('/')[-1].split('.')[-2]
            readwrite.read(self, filename)
        else:
            self.info = {}
            self.data = filename
            if self.data.ndim==3:
                self.is3D = True
                self.x, self.y, self.z = np.shape(self.data)
            else:
                self.is3D = False
                self.x, self.y = np.shape(self.data)

        if not 'pixelSize' in self.info:
            self.info['pixelSize'] = 1
            self.info['unit'] = 'pixels'

        self.info['pixelSizeOriginal'] = self.info['pixelSize']

        return

    def shape(self):
        return np.shape(self.data)

    def auto_frame_crop(self):
        self.data = fun.auto_frame_crop(self.data)
        self.x, self.y, self.z = np.shape(self.data)
        return

    def crop(self, xmin, xmax, ymin, ymax):
        self.data = self.data[xmin:xmax, ymin:ymax]
        self.x = np.shape(self.data)[0]
        self.y = np.shape(self.data)[1]
        return

    def enhance_contrast(self, percent1, percent2):
        self.data = functions.enhance_contrast(self.data, percent1, percent2)
        return


    def get_fft(self):
        if self.is3D:
            self.fft = np.zeros_like(self.data)
            for i in range(self.z):
                self.fft[:,:,i] = functions.fft(self.data[:,:,i])
        else:
            self.fft = functions.fft(self.data)
        return

    def get_pfft(self):
        if self.is3D:
            self.pfft = np.zeros_like(self.data)
            for i in range(self.z):
                self.pfft[:,:,i] = functions.pfft(self.data[:,:,i])
        else:
            self.pfft = functions.pfft(self.data)
        return

    def remove_brights(self, threshold=False):
        self.data = functions.remove_brights(self.data, threshold=threshold)
        return

    def rebin2d(self, factor):
        self.data = functions.rebin2d(self.data, factor)
        self.info['pixelSize'] = self.info['pixelSize']*factor
        return

    def scale(self):
        #scales intensities from 0 to 1
        return functions.scale(self.data)


    def max(self):
        return np.max(self.data)


    def min(self):
        return np.min(self.data)


    def low_pass_filter(self, b):
        self.data = functions.low_pass_filter(self.data, b)
        return

    def hamming_window(self):
        return functions.hamming_window(self.data)

    def flattop_window(self):
        return functions.flattop_window(self.data)

    def svd(self, low, high):
        if self.is3D:
            svd = np.zeros_like(self.data)
            for i in range(self.z):
                svd[:,:,i] = functions.svd(self.data[:,:,i], low, high)
            self.data = svd
        else:
            self.data = functions.svd(self.data, low, high)
        return


    def show(self, cmap='gray', **kwargs):
        if self.data.ndim == 3:
            if hasattr(self, 'aligned_sum'):
                imarr = self.aligned_sum
            else:
                print("Data is stack, showing first slice only.")
                imarr = self.data[:,:,0]
        else:
            imarr = self.data
        functions.show(imarr, cmap=cmap, **kwargs)
        return

    def show_fft(self, **kwargs):
        functions.show_fft(self.data, **kwargs)
        return

    def show_fft_profile(self):
        self.fft_profile = functions.show_fft_profile(self.data)
        return


    def align_slices(self, window=False, n=0, correlationType='cc', findMaxima='pixel', show=False):
        stack = registration.imstack(self.data)
        if window:
            stack.window('sin')
        else:
            stack.window(None)
        if n==0:
            stack.makeFourierMask(None)
        else:
            stack.makeFourierMask(mask='lowpass', n=n)
        stack.get_shifts(correlationType=correlationType, findMaxima=findMaxima, show=show)
        stack.get_average_shifts()
        stack.apply_shifts()

        self.aligned_sum = stack.alignedsum
        self.aligned_stack = stack.alignedstack
        self.xshifts = stack.xshifts
        self.yshifts = stack.yshifts

        return

    def saveTiff(self, filename=False, scalebar=False, bits=16):
        if filename==False:
            filename = self.f

        if hasattr(self, 'aligned_sum'):
            saveim = self.aligned_sum
        else:
            saveim = self.data
        
        if scalebar:
            if not 'pixelSize' in self.info.keys():
                print("I don't know the pixelsize, you'll have to enter it manually. Scaled image not saved.")
                return
            readwrite.saveTiff(readwrite.scaleBar(saveim, self.info['pixelSize'], self.info['unit']), filename, bits=8)
        else:
            readwrite.saveTiff(saveim, filename, bits=bits)

    def saveArr(self, filename=False, scalebar=False, bits=16):
        if filename==False:
            filename = self.f
        
        if hasattr(self, 'aligned_sum'):
            np.save(filename, self.aligned_sum)
        else:
            np.save(filename, self.data)


    def get_scalebar(self):
        self.scaled = readwrite.scaleBar(self.data, self.info['pixelSize'], self.info['unit'])
        return

    def zoom(self, amt, **kwargs):
        xorig = self.x
        if self.is3D:
            zx, zy = np.shape(functions.zoom(self.data[:,:,0], amt, **kwargs))
            zoom = np.zeros((zx, zy, self.z))
            for i in range(self.z):
                zoom[:,:,i] = functions.zoom(self.data[:,:,i], amt, **kwargs)
            self.data = zoom
            self.x = zx
            self.y = zy

        else:
            self.data = functions.zoom(self.data, amt, **kwargs)
            self.x, self.y = np.shape(self.data)

        self.info['pixelSize'] = self.info['pixelSize'] * xorig/self.x

        return
