# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:24:24 2016

@author: katie
"""
from __future__ import division
import numpy as np
import functions as fun
import matplotlib.pyplot as plt

class imstack(object):
    """
    Register and average stacks of noisy images
    """

    def __init__(self, image_stack):
        """
        Initializes imstack object from a 3D numpy array.

        Inputs:
            image_stack     ndarray of floats, shape (nx, ny, nz)
        """

        self.stack = image_stack
        self.x, self.y, self.z = np.shape(image_stack)

        return


    def make_fftstack(self, window=True, filter=False, b=1000, periodic=False):
        if window:
            self.fftstack = fun.fftstack(fun.hamming_window(self.stack), periodic=periodic)
        else:
            self.fftstack = fun.fftstack(self.stack, periodic=periodic)
        if filter:
            window = fun.low_pass_window(self.stack[:,:,0], b=b)
            for i in xrange(self.z):
                self.fftstack[:,:,i] = self.fftstack[:,:,i]*window
        return


    def make_conjstack(self):
        if not hasattr(self, 'fftstack'):
            self.make_fftstack()
        self.conjstack = np.conj(self.fftstack)

        return


    def get_shifts(self, window=False, filter=False, b=10, max_method='quadratic', peak_window=1, show=False, phase=False, periodic=False):

        self.xshifts = np.zeros((self.z, self.z))
        self.yshifts = np.zeros((self.z, self.z))

        self.make_fftstack(window = window, filter = filter, b = b, periodic=periodic)
        self.make_conjstack()

        def findpeak(array):
            if max_method=='gauss':
                #fit array maximum using 2D Gaussian
                return(fun.max_2d_gauss(array))
            elif max_method=='quadratic':
                #fit array maximum using 2D Quadratic
                return(fun.max_2d_quadratic(array))
            elif max_method=='com':
                #returns center of mass of array
                return(fun.COM(array))
            else:
                print "specify a different max_method"
                return

        for i in xrange(self.z-1):
            for j in xrange(i+1, self.z):
                if phase:
                    cc = fun.phase_correlate(self.fftstack[:,:,i], self.conjstack[:,:,j])
                else:
                    cc = fun.fft_correlate(self.fftstack[:,:,i], self.conjstack[:,:,j])
                xmaxcc, ymaxcc = np.unravel_index(np.argmax(cc), np.shape(cc))
                xmax = xmaxcc - self.x/2
                ymax = ymaxcc - self.y/2
                if phase:
                    xsp = 0
                    ysp = 0
                else:
                    xsp, ysp = findpeak(cc[xmaxcc-peak_window:xmaxcc+peak_window+1, ymaxcc-peak_window:ymaxcc+peak_window+1])
                self.xshifts[i,j], self.yshifts[i,j] = -(xmax+xsp), -(ymax+ysp)
                self.xshifts[j,i] = - self.xshifts[i,j]
                self.yshifts[j,i] = - self.yshifts[i,j]
                if i == 0:
                    if j<10:
                        if show:
                            print self.xshifts[i,j], self.yshifts[i,j]
                            a, b = np.shape(cc)
                            dim = a//10
                            xcrop = int((a-dim)//2)
                            ycrop = int((b-dim)//2)
                            plt.matshow(cc[xcrop:xcrop+dim, ycrop:ycrop+dim], cmap='jet')
                            plt.plot(ymaxcc - ycrop, xmaxcc - xcrop)
                            plt.arrow(dim/2, dim/2, -self.yshifts[i,j], -self.xshifts[i,j])
                            plt.xlim(0, dim)
                            plt.ylim(0, dim)
                            plt.show()

        if show:
            fun.show_map(self.xshifts)
            fun.show_map(self.yshifts)


        return

    def get_shifts_sequential(self, window=False, filter=False, b=1000, max_method='quadratic', peak_window=10, show=False):
        self.xshiftsavg = np.zeros(self.z)
        self.yshiftsavg = np.zeros(self.z)

        def findpeak(array):
            if max_method=='gauss':
                #fit array maximum using 2D Gaussian
                return(fun.max_2d_gauss(array))
            elif max_method=='quadratic':
                #fit array maximum using 2D Quadratic
                return(fun.max_2d_quadratic(array))
            elif max_method=='com':
                #returns center of mass of array
                return(fun.COM(array))
            else:
                print "specify a different max_method"
                return

        if filter:
            filter_window = fun.low_pass_window(self.stack[:,:,0], b)

        ref = self.stack[:,:,0]
        for i in range(1,self.z):

            slice = self.stack[:,:,i]

            if window:
                slice = fun.flattop_window(slice)
                sliceref = fun.flattop_window(ref)
            else:
                sliceref = ref

            if filter:
                cc = fun.fft_correlate(filter_window*fun.fft(self.stack[:,:,i]), filter_window*np.conj(fun.fft(ref)))
            else:
                cc = fun.fft_correlate(fun.fft(self.stack[:,:,i]), np.conj(fun.fft(ref)))
            xmaxcc, ymaxcc = np.unravel_index(np.argmax(cc), np.shape(cc))
            xmax = xmaxcc - self.x/2
            ymax = ymaxcc - self.y/2
            xsp, ysp = findpeak(cc[xmaxcc-peak_window:xmaxcc+peak_window+1, ymaxcc-peak_window:ymaxcc+peak_window+1])
            self.xshiftsavg[i] = -(xmax+xsp)
            self.yshiftsavg[i] = -(ymax+ysp)
            if show:
                print self.xshiftsavg[i], self.yshiftsavg[i]
                a, b = np.shape(cc)
                dim = a
                xcrop = int((a-dim)//2)
                ycrop = int((b-dim)//2)
                plt.matshow(cc[xcrop:xcrop+dim, ycrop:ycrop+dim], cmap='jet')
                plt.plot(ymaxcc - ycrop, xmaxcc - xcrop)
                plt.arrow(dim/2, dim/2, self.yshiftsavg[i], self.xshiftsavg[i])
                plt.xlim(0, dim)
                plt.ylim(0, dim)
                plt.show()
            ref = ref + fun.shift_subpixel(self.stack[:,:,i], self.xshiftsavg[i], self.yshiftsavg[i])

        return




    def get_average_shifts(self, mask=False, iqmult = 10):
        if not hasattr(self, 'xshifts'):
            self.get_shifts()

        if mask:
            q1 = np.percentile(self.xshifts, 25)
            q3 = np.percentile(self.xshifts, 75)
            iq = q3-q1
            min = q1-iqmult*iq
            max = q3+iqmult*iq
            xmask = ((self.xshifts > min) * (self.xshifts < max)).astype('bool')
            

            q1 = np.percentile(self.yshifts, 25)
            q3 = np.percentile(self.yshifts, 75)
            iq = q3-q1
            min = q1-iqmult*iq
            max = q3+iqmult*iq
            #print "interquartile {}, min = {}, max = {}".format(iq, min, max)
            ymask = ((self.yshifts > min) * (self.yshifts < max)).astype('bool')


            mask = (xmask)*(ymask)

            self.shiftmask = mask


            self.yshiftsavg = np.sum(self.yshifts*mask, axis=1)/np.sum(mask, axis=1)
            self.xshiftsavg = np.sum(self.xshifts*mask, axis=1)/np.sum(mask, axis=1)
            print "masked {} shifts".format(np.count_nonzero(1-mask.astype('int')))

        else:
            self.xshiftsavg = np.sum(self.xshifts, axis=1)/self.z
            self.yshiftsavg = np.sum(self.yshifts, axis=1)/self.z

        return


    def expand_image(self, factor):
        newdat = np.zeros((self.x*factor, self.y*factor, self.z))
        for i in xrange(self.x*factor):
            for j in xrange(self.y*factor):
                newdat[i,j,:] = self.stack[i//factor, j//factor, :]
        self.stack = newdat
        self.x, self.y, self.z = np.shape(self.stack)


    def apply_shifts(self, expand=1, crop=True):
        if not hasattr(self, 'xshifts') and not hasattr(self, 'xshiftsavg'):
            self.get_shifts()
        if not hasattr(self, 'xshiftsavg'):
            self.get_average_shifts()

        if expand > 1:
            self.expand_image(expand)
            self.xshiftsavg = self.xshiftsavg*expand
            self.yshiftsavg = self.yshiftsavg*expand

        self.alignedstack = np.zeros_like(self.stack)

        for slice in xrange(self.z):
            self.alignedstack[:,:,slice] = fun.shift_subpixel(self.stack[:,:,slice], self.xshiftsavg[slice], self.yshiftsavg[slice])

        if crop:
            print "uncropped shape {}".format(np.shape(self.alignedstack))
            xcropmin = int(np.ceil(max(self.xshiftsavg)))
            xcropmax = int(np.floor(min(self.xshiftsavg)))
            ycropmin = int(np.ceil(max(self.yshiftsavg)))
            ycropmax = int(np.floor(min(self.yshiftsavg)))
            self.alignedstack = self.alignedstack[xcropmin:xcropmax, ycropmin:ycropmax]
            self.alignedsum = np.sum(self.alignedstack, axis=2)
            print "cropped shape {}".format(np.shape(self.alignedstack))
        else:
            self.alignedsum = np.sum(self.alignedstack, axis=2)
            print "you turned off auto-cropping but the largest x shift is {} and y is {} pixels".format(np.max(abs(self.xshiftsavg)), np.max(abs(self.yshiftsavg)))

        return