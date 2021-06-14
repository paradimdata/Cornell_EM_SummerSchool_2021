# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:24:24 2016

@author: katie
"""
from __future__ import division, print_function
import numpy as np
from scipy.signal import flattop
from scipy.optimize import curve_fit
import multiprocessing as mp
import time
from scipy.interpolate import griddata
from itertools import combinations, chain, islice



import matplotlib.pyplot as plt
from EM import functions as fun



class imstack(object):
    """
    Register and average selfs of noisy images
    """

    def __init__(self, image_self):
        """
        Initializes imstack object from a 3D numpy array.

        Inputs:
            image_self     ndarray of floats, shape (nx, ny, nz)
        """

        self.stack = image_self
        self.nx, self.ny, self.nz = np.shape(image_self)

        if self.nx%2!=0:
            print("cropping x to even dimension")
            self.stack=self.stack[:-1, :, :]
            self.nx, self.ny, self.nz = np.shape(self.stack)

        if self.ny%2!=0:
            print("cropping y to even dimension")
            self.stack=self.stack[:, :-1, :]
            self.nx, self.ny, self.nz = np.shape(self.stack)

        # Define real and reciprocal space meshgrids
        rx,ry = np.meshgrid(np.arange(self.nx),np.arange(self.ny))
        self.rx,self.ry = rx.T,ry.T
        nx,ny = float(self.nx),float(self.ny)

        kx,ky = np.meshgrid(np.arange(-int(nx/2),int(nx/2),1),np.arange(-int(ny/2),int(ny/2),1))

        self.kx,self.ky = kx.T,ky.T
        self.kr = np.sqrt(self.kx**2+self.ky**2)

        self.Rij_mask = np.ones((self.nz, self.nz)).astype('bool')

        return

    def window(self, window=None):
        if window==None:
            mask_realspace = np.ones_like(self.stack[:,:,0])
        elif window=='sin':
            #sin squared window, as in BHS code
            mask_realspace = (np.sin(np.pi*self.rx/self.nx)*np.sin(np.pi*(self.ry/self.ny)))**2
        elif window=='hamming':
            mask_realspace = np.sqrt(np.outer(np.hamming(self.nx), np.hamming(self.ny)))
        elif window=='flattop':
            mask_realspace = np.outer(flattop(self.nx), flattop(self.ny))
        self.windowed = self.stack*np.stack(np.array([mask_realspace]*self.nz), axis=2)

    def make_fftstack(self):
        if not hasattr(self, 'windowed'):
            self.window()
        fftstack = np.zeros_like(self.stack, dtype='complex')
        for i in range(self.nz):
            fftstack[:,:,i] = fun.fft(self.windowed[:,:,i]-np.mean(self.windowed[:,:,i]))
        self.fftstack = fftstack
        return

    def makeFourierMask(self, mask=None, n=4):
        """
        Defines the Fourier space mask to be used when finding cross-correlation.

        Inputs:
            n   int or float        Defines upper frequency allowed.  For all masks,
                                    maximum frequency is k_max / n.  Thus features smaller
                                    than ~n pixels are smoothed.
                                    Heuristically n=4 works well, but will vary with data.
            mask    str             Either "bandpass", "lowpass", or "none".
        """
        nx,ny = float(self.nx),float(self.ny)
        if mask=="bandpass":
            self.mask_fourierspace = ((self.kr<ny/n/2)*(np.sin(2*n*np.pi*self.kr/ny)*np.sin(2*n*np.pi*self.kr/ny)))
        elif mask=="lowpass":
            self.mask_fourierspace = ((self.kr<ny/n/2)*(np.cos(n*np.pi*self.kr/ny)*np.cos(n*np.pi*self.kr/ny)))
        elif mask==None:
            self.mask_fourierspace = np.ones_like(self.kr)
        else:
            print("Mask type must be 'bandpass', 'lowpass', or 'none'.")
            print("Alternatively, define a custom mask by setting the self.mask_fourierspace attribute manually.  The self.kr coordinates may be useful.")
            return
        return



    def get_shifts(self, correlationType='cc', findMaxima='pixel', verbose=False, parallel=False, show=False):
        self.xshifts = np.zeros((self.nz, self.nz))
        self.yshifts = np.zeros((self.nz, self.nz))

        if not hasattr(self,'fftstack'):
            self.make_fftstack()
        if not hasattr(self, 'mask_fourierspace'):
            self.makeFourierMask()

        indexgrid = np.meshgrid(np.arange(self.nz), np.arange(self.nz))
        index_mask = indexgrid[1] < indexgrid[0]
        indices = np.argwhere(index_mask)

        if parallel:

            def collect_result(result):
                res.append(result)
                return

            pool = mp.Pool(4)
            t0 = time.time()
            res = []
            for index in indices:
                pool.apply_async(find_shift, args=(self, index, correlationType, findMaxima), callback = collect_result)

            pool.close()
            pool.join()
            res = np.array(res)
            #print res
            #print np.shape(res)
            xshifts = -res[:,0]
            yshifts = -res[:,1]

            self.xshifts[index_mask] = xshifts
            self.yshifts[index_mask] = yshifts

            for j in range(self.nz):
                for i in range(0, j):
                    self.xshifts[j,i] = -self.xshifts[i,j]
                    self.yshifts[j,i] = -self.yshifts[i,j]

            t1 = time.time()
            print("shifts found in {} seconds using multiprocessing".format(t1-t0))

        else:
            t0 = time.time()
            counter = 0

            for index in indices:
                #print index
                if (counter < 2 and show):
                    self.xshifts[index[0], index[1]], self.yshifts[index[0], index[1]] = find_shift(self, index, correlationType, findMaxima, showcc=True)
                else:
                    self.xshifts[index[0], index[1]], self.yshifts[index[0], index[1]] = find_shift(self, index, correlationType, findMaxima, showcc=False)
                counter=counter+1
                #print self.xshifts[index[0], index[1]], self.yshifts[index[0], index[1]]

            for j in range(self.nz):
                for i in range(0, j):
                    self.xshifts[j,i] = -self.xshifts[i,j]
                    self.yshifts[j,i] = -self.yshifts[i,j]

            t1 = time.time()
            print("shifts found in {} seconds".format(t1-t0))

            self.xshifts = self.xshifts.astype(float)
            self.yshifts = self.yshifts.astype(float)

        return



    ########### Methods for correlating image pairs #############

    def getSingleCrossCorrelation(self, fft1, fft2):
        """
        Cross correlates two images from previously calculated ffts.
        Applies self.mask_fourierspace.  If undefined, masks using bandpass filter with n=4.
        (See self.makeFourierMask for more info.)
        """
        try:
            cross_correlation = np.abs(np.fft.ifft2(self.mask_fourierspace * fft2 * np.conj(fft1)))
        except AttributeError:
            self.makeFourierMask()
            cross_correlation = np.abs(np.fft.ifft2(self.mask_fourierspace * fft2 * np.conj(fft1)))
        return cross_correlation

    def getSingleMutualCorrelation(self, fft1, fft2):
        """
        Calculates mutual correlation function for two images from previously calculated ffts.
        Applies self.mask_fourierspace.  If undefined, masks using bandpass filter with n=4.
        (See self.makeFourierMask for more info.)
        """
        try:
            mutual_correlation = np.abs(np.fft.ifft2(self.mask_fourierspace * fft2 * np.conj(fft1) / np.sqrt(np.abs(fft2 * np.conj(fft1)))))
        except AttributeError:
            self.makeFourierMask()
            mutual_correlation = np.abs(np.fft.ifft2(self.mask_fourierspace * fft2 * np.conj(fft1) / np.sqrt(np.abs(fft2 * np.conj(fft1)))))
        return mutual_correlation

    def getSinglePhaseCorrelation(self, fft1, fft2):
        """
        Calculates phase correlation function for two images from previously calculated ffts.
        Applies self.mask_fourierspace.  If undefined, masks using bandpass filter with n=4.
        (See self.makeFourierMask for more info.)
        """
        try:
            phase_correlation = np.abs(np.fft.ifft2(self.mask_fourierspace * fft2 * np.conj(fft1) / (np.abs(fft2)**2) ))
        except AttributeError:
            self.makeFourierMask()
            phase_correlation = np.abs(np.fft.ifft2(self.mask_fourierspace * fft2 * np.conj(fft1) / (np.abs(fft2)**2) ))
        return phase_correlation



    ########### Methods for getting shifts from correlation maxima ########## 

    def getSingleShift_pixel(self, cc):
        """
        Calculates the shift between two images from their cross correlation by finding the
        maximum pixel.
        """
        xshift, yshift = np.unravel_index(np.argmax(cc),(self.nx,self.ny))
        return xshift, yshift

    def setGaussianFitParams(self,num_peaks=5,sigma_guess=2,window_radius=6):
        self.num_peaks=num_peaks
        self.sigma_guess=sigma_guess
        self.window_radius=window_radius
        return

    def getSingleShift_gaussianFit(self,cc):
        """
        Calculates the shift between two images from their cross correlation by fitting a
        gaussian to the cc maximum.
        Fits gaussians to the self.num_peaks maximum pixels and uses the point with the
        maximum *fit* rather than the maximum *pixel* intensity, to handle sampling issues.
        Alter fitting params with self.setGaussianFitParams(), or by manually setting the
        selt.num_peaks, self.sigma_guess, or self.window_radius attributes.

        Notes:
        (1) Gaussian fits make sense for atomic resolution data, but may not be appropriate
        elsewere, depending on your data.
        (2) Fitting multiple gaussians can handle sampling artifacts which can lead to
        identifying the incorrect maximum point in the cc.
        (3) Absent sampling problems, subpixel fitting for images selfs with ~10 or more
        images may not differ significantly from pixel fitting, as final shifts are calculated
        by averaging all the shifts to a given image.
        """
        all_shifts = self.get_n_cross_correlation_maxima(cc,self.num_peaks)

        data = np.fft.fftshift(cc)
        est_positions = all_shifts
        est_sigmas = np.ones_like(all_shifts)*self.sigma_guess
        est_params=[est_positions,est_sigmas]

        amplitudes, positions, sigmas, thetas, offsets, success_mask = fit_peaks(data,est_params,self.window_radius,print_mod=1, verbose=False)

        shift_x, shift_y = positions[np.argmax(offsets+amplitudes),:]
        return shift_x-np.shape(cc)[0]/2.0, shift_y-np.shape(cc)[1]/2.0

    def setCoMParams(self,num_iter=2,min_window_frac=3):
        self.num_iter=num_iter
        self.min_window_frac=min_window_frac
        return

    def getSingleShift_com(self,cc):
        """
        TODO: Document this function
        """
        ccs=np.fft.fftshift(cc)
        norm=np.sum(ccs)
        x_com, y_com = np.sum(ccs*self.rx)/norm, np.sum(ccs*self.ry)/norm

        # Iterate
        n_vals=np.linspace(self.min_window_frac,0,self.num_iter,endpoint=False)[::-1]
        for n in n_vals:
            r_com = np.sqrt((x_com-self.rx)**2+(y_com-self.ry)**2)
            weights = (r_com<self.ny/n/2)*(np.cos(n*np.pi*r_com/self.ny))**2
            ccs_weighted = ccs*weights
            norm = np.sum(ccs_weighted)
            x_com,y_com = np.sum(self.rx*ccs_weighted)/norm, np.sum(self.ry*ccs_weighted)/norm

        return x_com-self.nx/2.0, y_com-self.ny/2.0

    def get_n_cross_correlation_maxima(self,cc,n):
        """
        Gets the maximum n pixels in a cross correlation.
        """
        cc_shift=np.fft.fftshift(cc)
        shifts=np.zeros((n,2))
        for i in range(n):
            shifts[i,0],shifts[i,1]=np.unravel_index(np.argmax(cc_shift),np.shape(cc_shift))

            cc[int(shifts[i,0]),int(shifts[i,1])]=0
        return shifts


########### Methods for masking Rij matrix #############

    def update_Rij_mask(self):
        if not hasattr(self,'bad_image_mask'):
            self.bad_image_mask = np.ones_like(self.xshifts).astype('bool')
        if not hasattr(self,'outlier_mask'):
            self.outlier_mask = np.ones_like(self.xshifts).astype('bool')
        """
        Rij_mask is comprised of:
            nz_mask: False outside the specified range of images between nz_min and nz_max
            bad_image_mask: False at specified bad images
            outlier_mask: False on datapoints determined to be outliers
        """
        self.Rij_mask = (self.bad_image_mask)*(self.outlier_mask)
        return

    def set_nz(self, nz_min, nz_max):
        """
        Sets range of images to include in averaging by setting self.nz_mask.

        Inputs:
            nz_min  int     first image in imstack to include
            nz_max  int     last image in imstack to include
        """
        self.nz_min, self.nz_max = nz_min, nz_max
        self.nz_mask = np.zeros((self.nz,self.nz),dtype=bool)
        self.nz_mask[nz_min:nz_max,nz_min:nz_max]=True
        self.update_Rij_mask()
        return

    def set_bad_images(self, bad_images):
        """
        Marks specified images as bad data, which won't be included in final average image.

        Inputs:
            bad_images      list of ints    indices of images to throw out
        """
        self.bad_images = list(bad_images)
        self.bad_image_mask = np.ones((self.nz,self.nz),dtype=bool)
        for image in bad_images:
            self.bad_image_mask[image,:]=False
            self.bad_image_mask[:,image]=False
        self.update_Rij_mask()
        return



    def get_outliers(self, method="ensemble", iqmult=10, threshold = 10, maxpaths = 5):
        """
        Find outliers in Rij matrix, which will not be used in calculating final average image.

        Inputs:
            method  str     Method to be used for outlier detection.
                            Currenly supported: 'NN'

        Currently supported outlier detection methods:
        NN - detects outliers by looking at the shifts of the nearest neighbor image pairs
        args:
            arg[0]: max_shift - outlier shift threshhold
        """
        if method=="ensemble":
            self.outlier_mask = self.get_outliers_ensemble(iqmult=iqmult)

        if method=="transitivity":
            self.outlier_mask = self.get_outliers_transitivity(threshold = threshold, maxpaths=maxpaths)

        self.update_Rij_mask()
        return


    def get_outliers_ensemble(self, iqmult=10):
        if not hasattr(self, 'xshifts'):
            self.get_shifts()

        xmask = fun.get_outliers(self.xshifts, iqmult=iqmult)
        ymask = fun.get_outliers(self.yshifts, iqmult=iqmult)

        return (xmask)*(ymask)

    def get_outliers_transitivity(self, threshold, maxpaths=5):
        transitivity_scores=np.zeros_like(self.xshifts)
        for i in range(len(self.xshifts)-1):
            for j in range(i+1,len(self.xshifts)):
                paths = getpaths(i,j,maxpaths,self.nz)
                for p in paths:
                    pdx = np.array([self.xshifts[ip] for ip in p])
                    pdy = np.array([self.yshifts[ip] for ip in p])
                    transitivity_scores[i,j] += np.sqrt((pdx.sum()-self.xshifts[j,i])**2+(pdy.sum()-self.yshifts[j,i])**2)
        transitivity_scores /= maxpaths
        for i in range(len(self.xshifts)-1):
            for j in range(i+1,len(self.yshifts)):
                transitivity_scores[j,i] = transitivity_scores[i,j]
        return transitivity_scores<threshold



    def make_corrected_Rij(self,maxpaths=5):
        good_images=np.nonzero(np.all(self.Rij_mask,axis=1)==False)[0] #images with no outliers in the shifts
        print(good_images)
        temp_mask = np.copy(self.Rij_mask)
        self.xshifts_c,self.yshifts_c = np.where(self.Rij_mask,self.xshifts,float('nan')),np.where(self.Rij_mask,self.yshifts,float('nan')) #creates corrected matrices, with outlier pixels set to Nan
        counter = 0
        print(temp_mask[good_images,:][:,good_images])
        while np.all(temp_mask[good_images,:][:,good_images])==False:
            counter += 1
            if counter%1000==0:
                print('pixel {} of {}'.format(counter, np.count_nonzero(self.Rij_mask.astype(int)-1)))
            for i in range(len(self.xshifts)-1):
                for j in range(i+1,len(self.yshifts)):
                    if not temp_mask[i,j]:
                        n = 0.
                        x,y = 0.,0.
                        paths = allpaths(i,j,maxpaths)
                        for p in paths:
                            if np.all([temp_mask[ip] for ip in p]):
                                x += np.array([self.xshifts_c[ip] for ip in p]).sum()
                                y += np.array([self.yshifts_c[ip] for ip in p]).sum()
                                n += 1
                        if n:
                            self.xshifts_c[i,j],self.xshifts_c[j,i] = -x/n,x/n
                            self.yshifts_c[i,j],self.yshifts_c[j,i] = -y/n,y/n
            temp_mask = (np.isnan(self.xshifts_c)==False)*(np.isnan(self.yshifts_c)==False)
        self.Rij_mask_c = temp_mask
        self.xshifts_c[np.isnan(self.xshifts_c)] = 0
        self.yshifts_c[np.isnan(self.yshifts_c)] = 0
        return



    def interpolate_shifts_pad(self, squaremask, subarrays=9, iqmult=1):
        # #Find periodic shift array direction for PAD data
        # xsum = abs(np.sum(self.xshifts[:, 0]))
        # ysum = abs(np.sum(self.yshifts[:, 0]))

        # if xsum > ysum:
        #     print('Correcting y matrix')
        #     periodic = self.yshifts
        #     ycorr=True
        # if ysum > xsum:
        #     peridic = self.xshifts
        #     print('Correcting x matrix')
        #     ycorr=False


        xindices = np.argwhere(squaremask)

        transform = np.lexsort((xindices[:,0], xindices[:,1]))
        inverse = np.array(transform)
        inverse[transform] = np.array(range(len(transform)))

        yordered = self.yshifts[transform]
        ol = fun.get_outliers(yordered, iqmult)

        rows = range(self.nz)
        ydiag = yordered[rows, rows]
        replace = fun.get_outliers(ydiag, 1, 0.75)
        if np.sum(replace.astype(int))==0:
            ydiagfix = ydiag
        elif not np.count_nonzero(1-replace.astype(int))==0:
            ydiagfix = np.array(ydiag)
            ydiagfix[~replace] = np.interp(np.argwhere(~replace)[:,0], np.argwhere(replace)[:,0], ydiag[replace])
        else:
            ydiagfix = ydiag

        x, y = np.indices(self.xshifts.shape)

        ymap = yordered-ydiagfix
        youts = fun.get_outliers(ymap, subarrays, iqmult)
        xouts = fun.get_outliers(self.xshifts, subarrays, iqmult)
        

        if not np.count_nonzero(1-youts.astype(int))==0:
            print("fixing {} outliers in y shift matrix".format(np.count_nonzero(1-youts.astype(int))))
        
            interp = np.array(ymap)
            interp[~youts] = griddata((x[youts], y[youts]), ymap[youts], (x[~youts], y[~youts]), method='cubic')

            interp[np.isnan(interp)] = griddata((x[~np.isnan(interp)], y[~np.isnan(interp)]), interp[~np.isnan(interp)], (x[np.isnan(interp)], y[np.isnan(interp)]), method='nearest')

            self.yshifts = (interp+ydiagfix)[inverse]
        

        if not np.count_nonzero(1-xouts.astype(int))==0:
            print("fixing {} outliers in x shift matrix".format(np.count_nonzero(1-xouts.astype(int))))
            interp = np.array(self.xshifts)
            interp[~xouts] = griddata((x[xouts], y[xouts]), self.xshifts[xouts], (x[~xouts], y[~xouts]), method='cubic')

            interp[np.isnan(interp)] = griddata((x[~np.isnan(interp)], y[~np.isnan(interp)]), interp[~np.isnan(interp)], (x[np.isnan(interp)], y[np.isnan(interp)]), method='nearest')
            self.xshifts = interp

        self.Rij_mask = np.ones((self.nz, self.nz))
        self.get_average_shifts()
        return










    def get_average_shifts(self):
        if not hasattr(self, 'xshifts'):
            self.get_shifts()

        if hasattr(self, 'Rij_mask'):
            mask = self.Rij_mask

            self.yshiftsavg = np.sum((self.yshifts).astype(float)*mask, axis=1)/np.sum(mask, axis=1)
            self.xshiftsavg = np.sum((self.xshifts).astype(float)*mask, axis=1)/np.sum(mask, axis=1)
            print("masked {} shifts".format(np.count_nonzero(1-mask.astype('int'))))

        else:
            self.xshiftsavg = np.sum((self.xshifts).astype(float), axis=1)/self.nz
            self.yshiftsavg = np.sum((self.yshifts).astype(float), axis=1)/self.nz

        return


    def expand_image(self, factor):
        newdat = np.zeros((self.nx*factor, self.ny*factor, self.nz))
        for i in range(self.nx*factor):
            for j in range(self.ny*factor):
                newdat[i,j,:] = self.stack[i//factor, j//factor, :]
        self.expanded = newdat
        self.nx, self.ny, self.nz = np.shape(self.stack)


    def apply_shifts(self, expand=1, crop=True):
        if not hasattr(self, 'xshifts') and not hasattr(self, 'xshiftsavg'):
            self.get_shifts()
        if not hasattr(self, 'xshiftsavg'):
            self.get_average_shifts()

        if expand > 1:
            self.expand_image(expand)
        else: 
            self.expanded = self.stack

        self.xshiftsavgexp = self.xshiftsavg*expand
        self.yshiftsavgexp = self.yshiftsavg*expand

        self.alignedstack = np.zeros_like(self.expanded)

        for slice in range(self.nz):
            self.alignedstack[:,:,slice] = fun.shift_subpixel(self.expanded[:,:,slice], self.xshiftsavgexp[slice], self.yshiftsavgexp[slice])

        if crop:
            try:
                print("uncropped shape {}".format(np.shape(self.alignedstack)))
                xcropmin = int(np.ceil(max(self.xshiftsavgexp)))
                xcropmax = int(np.floor(min(self.xshiftsavgexp)))
                if xcropmax==0:
                    xcropmax = self.alignedstack.shape[0]
                ycropmin = int(np.ceil(max(self.yshiftsavgexp)))
                ycropmax = int(np.floor(min(self.yshiftsavgexp)))
                if ycropmax==0:
                    ycropmax = self.alignedstack.shape[1]
                self.alignedstack = self.alignedstack[xcropmin:xcropmax, ycropmin:ycropmax]
                self.alignedsum = np.sum(self.alignedstack, axis=2)
                print("cropped shape {}".format(np.shape(self.alignedstack)))
            except:
                print("auto-cropping failed idk, returning uncropped shifted image.")
        else:
            self.alignedsum = np.sum(self.alignedstack, axis=2)
            print("you turned off auto-cropping but the largest x shift is {} and y is {} pixels".format(np.max(abs(self.xshiftsavgexp)), np.max(abs(self.yshiftsavgexp))))

        return



##################

def find_shift(self, index, correlationType='cc', findMaxima='pixel', showcc=False):
    if correlationType=="cc":
        getSingleCorrelation = self.getSingleCrossCorrelation
    elif correlationType=="mc":
        getSingleCorrelation = self.getSingleMutualCorrelation
    elif correlationType=="pc":
        getSingleCorrelation = self.getSinglePhaseCorrelation
    else:
        print("'correlationType' must be 'cc', 'mc', or 'pc'.")
        return

    # Define maximum finder function call
    if findMaxima=="pixel":
        findMaxima = self.getSingleShift_pixel
    elif findMaxima=="gf":
        findMaxima = self.getSingleShift_gaussianFit
        if not hasattr(self, 'num_peaks'):
            print("No Gaussian Fit parameters specified; using defaults")
            self.setGaussianFitParams()
    elif findMaxima=="com":
        findMaxima = self.getSingleShift_com
        if not hasattr(self, 'num_iter'):
            print("No CoM parameters specified; using defaults")
            self.setCoMParams()
    else:
        print("'findMaxima' must be 'pixel', 'gf', or 'com'.")
        return

    i = index[0]
    j = index[1]

    cc = getSingleCorrelation(self.fftstack[:,:,i], self.fftstack[:,:,j])
    if showcc:
        fun.show(np.fft.fftshift(cc), cmap='jet')
    xshift, yshift = findMaxima(cc)
    if xshift>=self.nx/2:
        xshift = xshift-self.nx
    if yshift>=self.ny/2:
        yshift = yshift-self.ny
    return xshift, yshift


######Ben's Utils
# Define 2D Gaussian function
def gauss2d(xy_meshgrid, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    # Returns result as a 1D array that can be passed to scipy.optimize.curve_fit
    (x,y) = xy_meshgrid
    x0, y0 = float(x0), float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + np.abs(amplitude)*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2) ))
    return g.ravel()


def fit_gaussian(x0, y0, sigma_x, sigma_y, theta, offset, data, plot=False, verbose=True):
    # Define mesh for input values and initial guess
    xy_meshgrid = np.meshgrid(range(np.shape(data)[0]),range(np.shape(data)[1]))
    initial_guess = (data[x0,y0], x0, y0, sigma_x, sigma_y, theta, offset)
    # Perform fit and pull out centers
    try:
        popt, pcov = curve_fit(gauss2d, xy_meshgrid, data.ravel(), p0=initial_guess)#, ftol=1.49012e-10, xtol=1.49012e-10)
    except RuntimeError:
        if verbose:
            print("Particle could not be fit to a 2D gaussian.  Returning guess parameters.")
        return np.array(initial_guess),None,False
    # Plotting for troubleshooting
    if plot:
        data_fitted = gauss2d(xs,ys, *popt)
        fig,ax=plt.subplots(1,1)
        ax.matshow(data,cmap='gray')
        ax.contour(xs,ys,data_fitted.reshape(np.shape(data)[0],np.shape(data)[1]),8, colors='w')
        plt.show()
    return popt, pcov, True


def on_edge(im,x0,y0,radius):
    x,y=np.shape(im)[0],np.shape(im)[1]
    if (x0-radius>=0 and y0-radius>=0 and x0+radius+1<x and y0+radius+1<y):
        return False
    else:
        return True

def get_cutout(im,x0,y0,radius):

    return im[int(x0-radius):int(x0+radius+1), int(y0-radius):int(y0+radius+1)]


def fit_peaks(data, est_params, window_radius, print_mod=100, verbose=True):
    """
    Inputs:
        data           ndarray, floats
        est_params     len 2 list of ndarrays, as follows:
        est_params[0]   -   (n,2) ndarray, positions (x,y)
        est_params[1]   -   (n,2) ndarray, sigmas (s_x,s_y)
        window_radius  int
    Outputs:
        fit_amplitudes    (n,) ndarray, floats
        fit_positions     (n,2) ndarray, floats, (x,y)
        fit_sigmas        (n,2) ndarray, floats, (s_x,s_y)
        fit_thetas        (n,) ndarray, floats
        fit_offsets       (n,) ndarray, floats
        fit_success_mask  (n,) ndarray, booleans
    """
    reference_fit_params = []
    reference_fit_success_mask = []
    for i in range(len(est_params[0])):
        #if i%print_mod==0:
        #    print "Fitting column {} of {}".format(i,len(est_params[0]))
        x0,y0,sigma_x,sigma_y = est_params[0][i,0],est_params[0][i,1],est_params[1][i,0],est_params[1][i,1]
        if not on_edge(data,x0,y0,window_radius):
            cutout = get_cutout(data,x0,y0,window_radius)
            popt, pcov, fit_success = fit_gaussian(window_radius,window_radius,sigma_x,sigma_y,0,0,cutout,plot=False,verbose=verbose)
            popt[1:3]=popt[1:3][::-1]+[x0,y0]-window_radius
            reference_fit_params.append(popt)
            reference_fit_success_mask.append(fit_success)
    reference_fit_params=np.array(reference_fit_params)
    reference_fit_success_mask=np.array(reference_fit_success_mask)

    fit_amplitudes=reference_fit_params[:,0]
    fit_positions=reference_fit_params[:,1:3]
    fit_sigmas=reference_fit_params[:,3:5]
    fit_thetas=reference_fit_params[:,5]
    fit_offsets=reference_fit_params[:,6]
    fit_success_mask=reference_fit_success_mask
    return fit_amplitudes, fit_positions, fit_sigmas, fit_thetas, fit_offsets, fit_success_mask

def makeslice(seq):
    """
    Produces a sequence of array elements from a sequence of integers
    i.e. [1,2,3] yields [(1,2),(2,3)]
    Inputs
        seq:    array_like of integers
    Returns
        slices: array_like of tuples
    """
    tups = []
    for i in range(len(seq)-1):
        tups += [(seq[i+1],seq[i])]
    return tups

def allpaths(i,j,maxpaths=200):
    """
    Finds all paths between integers i and j, returning as many as maxpaths.
    The number of paths grows as |j-i|!, so a cutoff is necessary for practicality.

    Inputs:
        i,j       ints      endpoints of path
        maxpaths  int       maximum number of paths to return
    Returns:
        index_paths         a list of length at most maxpaths, each element of which
                            is a sequence which connects matrix elements i and j
    """
    if i>j:
        tmp = j
        j = i
        i = tmp
    if j-i < 2:
        return [[(j,i)]]
    n = range(i+1,j)
    combs = chain(*(combinations(n,l) for l in range(1,len(n)+1)[::1]))
    seq = [[i]+list(c)+[j] for c in islice(combs,maxpaths)]
    return map(makeslice, seq)



def getpaths(i,j,maxpaths,nz):
    """
    Finds a set of paths between integers i and j of length maxpaths.
    Selects paths in a sensible order, perferencing, in order (a) paths with elements between (i,j) (i.e. forward in time),
    and (b) shorter paths.

    Inputs:
        i,j       ints    endpoints of path
        maxpaths  int     maximum number of paths to return
    Returns:
        index_paths       a list of length at most maxpaths, each element of which is a sequence which connects
                          matrix elements i and j
    """
    if i>j:
        tmp = j
        j = i
        i = tmp
    if j-i<2:
        seq = [[i,j]]
    else:
        n = range(i+1,j)
        combs = chain(*(combinations(n,l) for l in range(1,len(n)+1)))
        seq = [[i]+list(c)+[j] for c in islice(combs,maxpaths)]

    a,b,count=0,0,0
    while(len(seq)<maxpaths):
        if i-a<=0:
            b += 1
        elif j+b>=nz-1:
            a += 1
        elif a%2:
            b+=1
        else:
            a+=1
        i0,j0 = i-a,j+b

        n1,n2 = range(i-1,i0,-1),range(j0-1,i+1,-1)
        combs1 = chain(*(combinations(n1,l) for l in range(1,len(n1)+1)[::-1]))
        combs2 = chain(*(combinations(n2,l) for l in range(1,len(n2)+1)[::-1]))
        seq1 = [list(c) for c in islice(combs1,maxpaths)]
        seq2 = [list(c) for c in islice(combs2,maxpaths)]
        seq1.append([])
        seq2.append([])
        for seq11 in seq1:
            for seq22 in seq2:
                subseq=[]
                if i!=i0:
                    subseq.append(i0)
                if j!=j0:
                    subseq.append(j0)
                seq.append([i]+seq11+subseq+seq22+[j])

    return map(makeslice, seq[:maxpaths])
