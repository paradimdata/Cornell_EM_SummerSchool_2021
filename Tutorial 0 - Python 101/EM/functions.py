from __future__ import division, print_function
import numpy as np
from scipy.signal import flattop
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.ndimage as ndi

"""
This is for operations on arrays, which will be extended to a class of image data.
"""



#Functions for manipulating images


def enhance_contrast(array, p_low,p_high):
    #saturate p_low percent of dark pixels and p-high percent of the bright pixels in the image to spread the histogram over larger display range.
    n_low = int(np.floor(np.prod(np.shape(array))*p_low/200))
    n_high = int(np.floor(np.prod(np.shape(array))*p_high/200))
    a = np.unravel_index(np.argsort(array, axis = None), np.shape(array))
    return np.clip(array, array[a][n_low], array[a][-n_high-1])

def fft(im):
    return np.fft.fftshift(np.fft.fft2(im))

def pfft(im):
    [rows,cols] = np.shape(im)
    #Compute boundary conditions
    s = np.zeros( np.shape(im) )
    s[0,:] = im[0,:] - im[-1,:]
    s[-1,:] = -s[0,:]


    s[:,0] = s[:,0] + im[:,0] - im[:,-1]
    s[:,-1] = s[:,-1] - im[:,0] + im[:,-1]


    #Create grid for computing Poisson solution
    [cx, cy] = np.meshgrid(2*np.pi*np.arange(0,cols)/cols, 2*np.pi*np.arange(0,rows)/rows)

    #Generate smooth component from Poisson Eq with boundary condition
    D = (2*(2 - np.cos(cx) - np.cos(cy)))
    D[0,0] = np.inf    # Enforce 0 mean & handle div by zero
    S = np.fft.fft2(s)/D


    

    P = np.fft.fft2(im) - S # FFT of periodic component


    return np.fft.fftshift(P)


def gauss2D(data, sigma):
    data = ndi.gaussian_filter(data, sigma)
    return (data)


def zoom(array, amt, **kwargs):
    if array.ndim!=3:
        return ndi.interpolation.zoom(array, amt, order = 0, **kwargs)
    else:
        x, y, z= np.shape(array)
        new = np.zeros((int(x*amt), int(y*amt), z))
        for i in range(z):
            new[:,:,i] = ndi.interpolation.zoom(array[:, :, i], amt, order = 0, **kwargs)
        return new


def remove_brights(array, threshold=False):
    inthreshold=threshold
    if inthreshold==False:
        inthreshold=10
    threshold = np.mean(array) + inthreshold*np.std(array)
    brights = array>threshold
    print("Found {} bright pixels".format(np.count_nonzero(brights)))

    while np.count_nonzero(brights)>0.0001*np.prod(np.shape(array)):
        #print("num std devs above mean is {}".format(inthreshold))
        print('too many brights')
        inthreshold+=1
        threshold = np.mean(array) + inthreshold*np.std(array)
        brights = array>threshold
        print("Found {} bright pixels".format(np.count_nonzero(brights)))
    print("Final threshold {} standard deviations above mean".format(inthreshold))


    #wherever you have a bright pixel, set it to the average of its 4 nearest neighbors
    brights = brights.astype('bool')
    indices = np.argwhere(brights)
    m = np.max(array[~brights])

    mcount = 0
    if array.ndim==2:
        for index in indices:

            i = index[0]
            j = index[1]
            try:
                array[i,j] = np.mean((array[i-1, j], array[i+1, j], array[i, j-1], array[i, j+1]))
            except:
                mcount = mcount + 1
                array[i,j] = m

    elif array.ndim==3:
        for index in indices:

            i = index[0]
            j = index[1]
            z = index[2]
            try:
                array[i,j,z] = np.mean((array[i-1, j,z], array[i+1, j,z], array[i, j-1,z], array[i, j+1,z]))
            except:
                mcount = mcount + 1
                array[i,j,z] = m

    print("Number set to max instead of average = {}".format(mcount))


    # if array.ndim==2:
    #     a, b = np.shape(array)
    #     for i in range(a):
    #         for j in range(b):
    #             #print i, j
    #             #print i-1, j-1
    #             if brights[i,j]:
    #                 try:
    #                     array[i,j] = np.mean(array[i-1, j], array[i+1, j], array[i, j-1], array[i, j+1])
    #                 except:
    #                     array[i,j] = np.max(array[~brights])
    # elif array.ndim==3:
    #     for z in range(array.shape[2]):
    #         a, b = np.shape(array[:,:,z])
    #         for i in range(a):
    #             for j in range(b):
    #                 #print i, j
    #                 #print i-1, j-1
    #                 if brights[i,j,z]:
    #                     try:
    #                         array[i,j,z] = np.mean(array[i-1, j,z], array[i+1, j,z], array[i, j-1,z], array[i, j+1,z])
    #                     except:
    #                         array[i,j,z] = np.max(array[~brights])
    return array


def rebin2d(array, factor):
    shape = np.shape(array)
    if len(shape) ==3:
        newshape = (shape[0]//factor)//2*2, (shape[1]//factor)//2*2, shape[2]
        arraycrop = array[0:(newshape[0]*factor),0:(newshape[1]*factor), :]
        binned = np.zeros(newshape)
        for i in range(shape[2]):
            binned[:,:,i] = arraycrop[:,:,i].reshape([newshape[0], factor, newshape[1], factor]).sum(-1).sum(1)
    
    else:
        newshape = shape[0]//factor, shape[1]//factor
        arraycrop = array[0:(newshape[0]*factor),0:(newshape[1]*factor)]
        binned = arraycrop.reshape([newshape[0], factor, newshape[1], factor]).sum(-1).sum(1)
   
    return binned

def auto_frame_crop(array):
    if not array.ndim==3:
        print("no frames to crop, a 2D image")
        return
    else:
        means = np.mean(np.mean(array, axis=0), axis=0)
        mask = means > (np.mean(means) - 2*np.std(means))
        print("cropped {} dark frames".format(np.shape(array)[-1]-np.count_nonzero(mask)))
        return array[:,:,mask]


def radius_grid(array):
    """
    Function to make an array the shape of input array; each pixel is valued its radius from the center.
    """
    a, b = np.shape(array)
    xgrid, ygrid = np.meshgrid(range(a), range(b))
    xgrid = xgrid-a/2
    ygrid = ygrid-b/2
    rad = np.transpose(np.sqrt(xgrid**2+ygrid**2))
    return rad

def scale(array):
    return (array-array.min())/(array.max()-array.min())


def low_pass_window(array, b):
    """
    Defines window that low-pass filters an array when applied in Fourier space
    b is a real space size in pixels, b/FOV describes the 1/e fall-off in Fourier space.
    """
    r = max(np.shape(array))
    rad = radius_grid(array)
    weight = 10**(-rad**2/(r/b)**2)
    return weight

def low_pass_filter(array, b):
    """
    Actually filters the array, see low_pass_window for math
    """
    def lp(array, b):
        window = low_pass_window(array, b)
        return abs(np.fft.ifft2(window*fft(array)))

    if array.ndim == 2:
        return lp(array, b)
    elif array.ndim == 3:
        filtered = np.zeros_like(array)
        for i in range(np.shape(array)[-1]):
            filtered[:,:,i] = lp(array[:,:,i], b)
        return filtered

def flattop_window(array):
    """
    Applies flat top window to array, returns windowed array
    """
    def ft2d(array):
        for axis, axis_size in enumerate(array.shape):
            filter_shape = [1, ] * array.ndim
            filter_shape[axis] = axis_size
            window = flattop(axis_size).reshape(filter_shape)
            array = array*window
        return array

    if array.ndim == 2:
        return ft2d(array)

    elif array.ndim == 3:
        windowed = np.zeros_like(array)
        for i in range(np.shape(array)[-1]):
            windowed[:,:,i] = ft2d(array[:,:,i])
        return windowed

def hamming_window(array):
    """
    Applies hamming window to array, returns windowed array
    """
    def ft2d(array):
        for axis, axis_size in enumerate(array.shape):
            filter_shape = [1, ] * array.ndim
            filter_shape[axis] = axis_size
            window = np.hamming(axis_size).reshape(filter_shape)
            array = array*window
        return array

    if array.ndim == 2:
        return ft2d(array)

    elif array.ndim == 3:
        windowed = np.zeros_like(array)
        for i in range(np.shape(array)[-1]):
            windowed[:,:,i] = ft2d(array[:,:,i])
        return windowed


def boxcar(array, number):
    """
    Smoothes array by averaging "number" of pixels in x and y. Returns smoothed array.
    """
    shifted = np.zeros((np.shape(array)[0], np.shape(array)[1], number*number))
    index = 0
    for i in range(number):
        for j in range(number):
            shifted[:,:,index] = np.roll(np.roll(array, i), j)
    return np.mean(shifted, axis=2)[number:-number, number:-number]

def subtract_means(array):
    """
    Returns stack of array, with the mean subtracted from each slice.
    """
    normarr = np.zeros_like(array)
    for i in range(np.shape(array)[-1]):
        normarr[:,:,i] = array[:,:,i] - np.mean(array[:,:,i])
    return normarr



def svd(a, low, high):     
    U, s, V = scipy.linalg.svd(a, full_matrices=False)   # Do the SVD, output 3 matrices!

    s[high:]=0                                          # Keep only modes between high
    s[0:low]=0                                          # and low!
    
    new_data = np.dot(U, np.dot(np.diag(s), V))

    
    return new_data


#Functions for displaying images!
def show(data, cmap='gray', **kwargs):
    fig = plt.figure(**kwargs)
    plt.matshow(data, fignum = False, cmap = cmap)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.show()
    return

def show_lower(data, cmap='gray', **kwargs):
    fig = plt.figure(**kwargs)
    plt.matshow(data, fignum = False, cmap = cmap, origin = 'lower')
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.show()
    return

def show_fft(array, **kwargs):
    boop = np.log(abs(pfft(array)))
    show(np.clip(boop, np.mean(boop), np.percentile(boop, 99.9)), **kwargs)
    return

def show_fft_profile(array):
    x, y = np.shape(array)
    dim = min(x,y)
    rad = dim//2
    crop = array[x//2-rad:x//2+rad, y//2-rad:y//2+rad]
    crop = abs(pfft(crop))
    profile = np.zeros(rad-2)
    for i in range(len(profile)):
        mesh = np.arange(0,2*rad,1)
        xgrid, ygrid = np.meshgrid(mesh, mesh)
        mask = ((xgrid-dim/2)**2+(ygrid-dim/2)**2 <= (i+1)**2) * ((xgrid-dim/2)**2+(ygrid-dim/2)**2 > (i)**2)
        profile[i] = np.sum(crop[mask])/(np.count_nonzero(mask))
    plt.semilogy(range(len(profile)), profile, linestyle = '-', color = 'black')
    plt.xlabel('Radius (pixels)')

    plt.ylabel('intensity')
    plt.show()
    return profile

def show_map(array, cmap='bwr', v=False, vmin=False, vmax=False, **kwargs):
    fig = plt.figure(**kwargs)
    if vmin==False and vmax==False and v==False:
        v = np.max(abs(array), axis=None)
        vmin = -v
        vmax = v
    if v!= False:
        vmin = -v
        vmax = v
    plt.matshow(array, fignum = False, cmap = cmap, vmin = vmin, vmax = vmax)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.colorbar(shrink = 0.65)
    plt.show()
    return


def show_map_lower(array, cmap='bwr', v=False, vmin=False, vmax=False, **kwargs):
    fig = plt.figure(**kwargs)
    if vmin==False and vmax==False and v==False:
        v = np.max(abs(array), axis=None)
        vmin = -v
        vmax = v
    if v!= False:
        vmin = -v
        vmax = v
    plt.matshow(array, fignum = False, cmap = cmap, vmin = vmin, vmax = vmax, origin = 'lower')
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.colorbar(shrink = 0.65)
    plt.show()
    return


#Functions for cross-correlating:


def get_outliers(array, subarrays=1, iqmult=10):
    
    def outliers(array, iqmult=10):
        q1 = np.percentile(array, 25)
        q3 = np.percentile(array, 75)
        iq = q3-q1
        min = q1-iqmult*iq
        max = q3+iqmult*iq
        mask = ((array > min) * (array < max)).astype('bool')
        return mask
        
    if subarrays == 1:
        return outliers(array, iqmult=iqmult)
    else:
        split = int(np.sqrt(subarrays))
        splitx = array.shape[0]//split
        splity = array.shape[1]//split
        
        splitindicesx = [splitx*(i) for i in range(split)]
        splitindicesy = [splity*(i) for i in range(split)]
        
        splitindicesx.append(array.shape[0])
        splitindicesy.append(array.shape[1])
        
        
        print('using {} subarrays'.format(split*split))
        outliermask = np.zeros_like(array, dtype=bool)
        for i in range(split):
            for j in range(split):
                subarr = array[splitindicesx[i]:splitindicesx[i+1],splitindicesy[j]:splitindicesy[j+1]]
                outliermask[splitindicesx[i]:splitindicesx[i+1],splitindicesy[j]:splitindicesy[j+1]] = outliers(subarr, iqmult=iqmult)
        return outliermask


def fft_correlate(fft1, fft2):
    """
    Input in fourier space; returns the NORMALIZED cross-correlation. The center of the CC is 0,0 shifts.
    """
    return abs(np.fft.ifftshift(np.fft.ifft2((fft1*np.conj(fft2)))))
    #return abs(np.fft.ifftshift(np.fft.ifft2((fft1*conj2)/abs(fft1*conj2))))

def phase_correlate(fft1, conj2):
    """
    Input in fourier space; returns the NORMALIZED cross-correlation. The center of the CC is 0,0 shifts.
    """
    #return abs(np.conj(conj2)*np.conj(fft1)/abs(fft1*np.conj(fft1)))
    return np.abs(fft1*conj2/np.abs(fft1*conj2))


def shift_subpixel(array, xshift, yshift):
    a, b = np.shape(array)
    rx, ry = np.meshgrid(np.arange(a), np.arange(b))
    x, y = float(a), float(b)
    w = -np.exp(-(2j*np.pi)*(xshift*rx/x+yshift*ry/y))
    shifted_fft = fft(array)*w.T
    return np.abs(np.fft.ifft2(np.fft.ifftshift(shifted_fft)))

#Functions for finding peak of Cross-correlation

def max_2d_quadratic(array):
    """
    Finds extremum of 2D array with subpixel accuracy by fitting to 2D quadratic. 
    Returns offset from center of the input array.
    """
    a, b = np.shape(array)
    u = array.flatten()
    x, y = np.meshgrid(range(a), range(b))
    x = (x-a/2).ravel().astype(float)
    y = (y-b/2).ravel().astype(float)

    fit_arr = np.array([np.ones(a*b), x, y, x*y, x**2, y**2])

    A, resid, rank, s = np.linalg.lstsq(fit_arr.T, u)

    xmax = (-A[2]*A[3]+2*A[5]*A[1])/(A[3]**2-4*A[4]*A[5])
    ymax = -1/(A[3]**2-4*A[4]*A[5])*(A[3]*A[1]-2*A[4]*A[2])

    return xmax, ymax

def COM(array):
    """
    finds offset of array's COM from its center.
    """
    x, y = np.shape(array)
    xgrid, ygrid = np.meshgrid(range(x), range(y))
    xgrid = xgrid-x/2
    ygrid = ygrid-y/2

    tot = np.sum(array)

    comx = np.sum(array*xgrid.T)/tot
    comy = np.sum(array*ygrid.T)/tot

    return comx, comy


def max_2d_gauss(array):

    def gauss2d(x,y, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
        # Returns result as a 1D array that can be passed to scipy.optimize.curve_fit
        x0, y0 = float(x0), float(y0)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + np.abs(amplitude)*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2) ))
        return g.ravel()
    a, b = np.shape(array)
    xgrid, ygrid = np.meshgrid(range(a), range(b))
    xgrid = (xgrid-a/2).ravel().astype(float)
    ygrid = (ygrid-b/2).ravel().astype(float)
    popt, pcov = curve_fit(gauss2d, (xgrid, ygrid), array.ravel(), p0 = [1., 1., 1., a/4, b/4, 1., 1.])
    print(popt)
    return popt[1], popt[2]

    show(gauss2d((xgrid, ygrid), popt[0], popt[1], popt[2], popt[3], popt[4], poprt[5], popt[6]))

