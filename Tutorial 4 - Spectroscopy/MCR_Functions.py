"""

==================
Loading Functions:
==================

load_bsub(file):
    Load the background subtracted data. This function assumes that the npz file contains a background subtracted spectrum named 'bsub', an energy axis named 'energy' and edge parameters.
    The edge input notation is edge=[fit_start_ev,fit_end_ev,integration_start,integration_end,edge_name].

load_check(bsub, energy, edge):
    Print SI size and edge parameters.

shape_guess(guess, Eguess, SIdisp):
    Inputs:
    guess - imported guess spectra
    Eguess - corresponding energy axis of guess spectra
    SIdisp - energy channel size of SI

    Outputs:
    newguess - guess spectra reshaped to match SIdisp
    newenergy - corresponding reshaped energy axis

more_params(bsub, energy, edge):
    This function imports various useful parameters.
    Returns the dimensions of the SI, the dispersion and the edge parameters converted to channel number.

bin_ndarray(ndarray, bin_factors, operation='sum')
    Bins an ndarray in all axes based on the bin_size array, by summing or
        averaging. Bin_size is expected to be an array with length 3, one element for the x, y and z binning.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

==================
MCR Functions:
==================

guess(array, xy,guess_energy):
    Inputs:
    array - original background subtracted SI
    xy - xy coordinates of guess regions [xmin,xmax,ymin,max]
    guess_energy - energy window in channel number

    Outputs:
    guess_spec - summed spectra from guess regions with dimensions component,energy

ROI_Outlines(SI, xy,line='none'):
    Inputs:
    SI - original background subtracted SI
    xy - xy coordinates of guess regions [xmin,xmax,ymin,max]
    line - 'none' for 2D maps, 'horizontal'/'vertical' for direction of 1D line scan

    Outputs:
    outlines - array of coordinates of rectangle using ax.plot


MCR(SI, guesses, energy, E_win, **kwargs):
    Generates score maps and components of SI given guess spectra components.

full_scores(SI,scores,components,E_win):
    This function will generate score maps using a NNLS fit given the final components extracted from MCR on the SI.
    This is particularly useful when binned data was used to extract the components in MCR.
    Input the unbinned SI to generate unbinned score maps.

mldivide(A,B):
    Solve systems of linear equations Ax = B for x

mrdivide(B,A):
    Solve systems of linear equations xA = B for x

"""


import numpy as np
import matplotlib.pyplot as plt

import eels

from scipy.optimize import nnls
import scipy
import numpy.linalg as LA

import hyperspy.api as hs

# import pandas
# from skimage.draw import polygon
# from pymcr.mcr import McrAR
# from pymcr.regressors import OLS, NNLS
# from pymcr.constraints import ConstraintNonneg,ConstraintNorm, ConstraintZeroEndPoints


#loads a background subtracted spectrum from a npz file, expects bsub, energy, and edge parameters as shown in the EELS demo notebook. For other formats, manually load via np.loadz()
def load_bsub(file):
    """
    Load the background subtracted data. This function assumes that the npz file contains a background subtracted spectrum named 'bsub', an energy axis named 'energy' and edge parameters.
    The edge input notation is edge=[fit_start_ev,fit_end_ev,integration_start,integration_end,edge_name].
    """
    data=np.load(file)
    bsub=data['bsub'].astype(float)
    energy=data['energy'].astype(float)
    edge=data['edge']
    edge_params=np.delete(edge,-1)
    edge_params=edge_params.astype(float)
    real_edge=[edge_params[0],edge_params[1],edge_params[2],edge_params[3],edge[-1]]
    return(bsub, energy, real_edge)

#This function is just a shortcut for showing the intitial spectra.
def load_check(bsub, energy, edge):
    """
    Print SI size and edge parameters.
    """
    xdim, ydim, zdim = np.shape(bsub)
    print('Background subtracted SI size: '),
    print(np.shape(bsub))
    print('Edge info: '),
    print(edge)

    Sum_Spectrum=sum(sum(bsub))
    edgemap = np.zeros((xdim,ydim))
    edge_params=np.delete(edge,-1)
    edge_params=edge_params.astype(float)
    edge_ch = []
    for element in edge_params:
        edge_ch.append(eels.eVtoCh(element,energy))
    for i in range (xdim):
        for j in range(ydim):
            edgemap[i,j] = sum(bsub[i,j,edge_ch[2]:edge_ch[3]])

    fig,(ax1,ax2)=plt.subplots(1,2,figsize = (8,4))
    ax1.imshow(edgemap,cmap='gray')
    ax1.set_title(str(edge[4]+' edgemap'))

    ax2.plot(energy[edge_ch[0]:edge_ch[3]+25],Sum_Spectrum[edge_ch[0]:edge_ch[3]+25])
    ax2.set_title("Background subtracted spectrum")
    ax2.set_xlabel("Energy loss (eV)")
    ax2.set_ylabel("Counts (a.u.)")
    ax2.set_yticks([0])
    return

def shape_guess(guess, Eguess, SIdisp):
    """
    Inputs:
    guess - imported guess spectra
    Eguess - corresponding energy axis of guess spectra
    SIdisp - energy channel size of SI

    Outputs:
    newguess - guess spectra reshaped to match SIdisp
    newenergy - corresponding reshaped energy axis
    """
    offset = Eguess[0]
    Gdisp = np.round(Eguess[1]- offset,decimals=6)
    step = SIdisp/Gdisp
    size0 = len(guess)
    size1 = int(((size0-1)/step)+1)
    newguess = np.zeros(size1)
    for i in range(size1-1):
        j = i*step
        newguess[i] = guess[int(j)] + (guess[int(j+1)] - guess[int(j)])*(j-int(j))
    newguess[-1] = guess[-1]
    newenergy = np.round(np.arange(offset,offset+size1*SIdisp,SIdisp),4)
    return(newguess, newenergy)

#This is just a nice shortcut for getting parameters loaded in a more usable form
def more_params(bsub, energy, edge):
    """
    This function imports various useful parameters.
    Returns the dimensions of the SI, the dispersion and the edge parameters converted to channel number.
    """
    dimx,dimy,dimE = np.shape(bsub)
    edge_ch = []
    for i in range(len(edge)-1):
        edge_ch.append(eels.eVtoCh(edge[i],energy))
    disp=np.round(energy[1]-energy[0],decimals=6)
    return(dimx,dimy,dimE,disp,edge_ch)

#bin array.
def bin_ndarray(ndarray, bin_factors, operation='sum'):
    """
    Bins an ndarray in all axes based on the bin_size array, by summing or
        averaging. Bin_size is expected to be an array with length 3, one element for the x, y and z binning.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    try:
        dimx,dimy,dimE = np.shape(ndarray)
        new_shape=[int(dimx/bin_factors[0]),int(dimy/bin_factors[1]),int(dimE/bin_factors[2])]
    except:
        dimL,dimE=np.shape(ndarray)
        new_shape=[int(dimL/bin_factors[0]),int(dimE/bin_factors[1])]
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]

    flattened = [l for p in compression_pairs for l in p]
    #bin_ndarray=np.copy(ndarray)
    ndarray = np.reshape(ndarray,flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

#Define a guess from a rectangular selection. Input the processed SI, and an array defining the coordinates of the image.
def guess(array, xy,guess_energy):
    """
    Inputs:
    array - original background subtracted SI
    xy - xy coordinates of guess regions [xmin,xmax,ymin,max]
    guess_energy - energy window in channel number

    Outputs:
    guess_spec - summed spectra from guess regions with dimensions component,energy
    """
    if np.ndim(array)==3:
        guess_spec=[]
        for i in range(np.shape(xy)[0]):
            dummy_guess=np.sum(np.sum(array[xy[i,0]:xy[i,1],xy[i,2]:xy[i,3],guess_energy[0]:guess_energy[1]],axis=0),axis=0)
            area=(xy[i,1]-xy[i,0])*(xy[i,3]-xy[i,2])
            dummy_guess=dummy_guess/area
            guess_spec.append(dummy_guess)
    if np.ndim(array)==2:
        guess_spec=[]
        for i in range(np.shape(xy)[0]):
            dummy_guess=np.sum(array[xy[i,0]:xy[i,1],guess_energy[0]:guess_energy[1]],axis=0)
            guess_spec.append(dummy_guess)
    return(guess_spec)

def ROI_Outlines(SI, xy,line='none'):
    """
    Inputs:
    SI - original background subtracted SI
    xy - xy coordinates of guess regions [xmin,xmax,ymin,max]
    line - 'none' for 2D maps, 'horizontal'/'vertical' for direction of 1D line scan

    Outputs:
    outlines - array of coordinates of rectangle using ax.plot
    """
    xdim, ydim, zdim = np.shape(SI)
    if line=='none':
        outlines=[]
        for i in range(np.shape(xy)[0]):
            dummy_line=[[xy[i,0],xy[i,0],xy[i,1]-1,xy[i,1]-1,xy[i,0]],[xy[i,2],xy[i,3]-1,xy[i,3]-1,xy[i,2],xy[i,2]]]
            outlines.append(dummy_line)
    if line=='vertical':
        outlines=[]
        for i in range(np.shape(xy)[0]):
            dummy_line=[[xy[i,0],xy[i,0],xy[i,1]-1,xy[i,1]-1,xy[i,0]],[0,ydim-1,ydim-1,0,0]]
            outlines.append(dummy_line)

    if line=='horizontal':
        outlines=[]
        for i in range(np.shape(xy)[0]):
            dummy_line=[[0,0,xdim-1,xdim-1,0],[xy[i,0],xy[i,1]-1,xy[i,1]-1,xy[i,0],xy[i,0]]]
            outlines.append(dummy_line)
    outlines=np.asarray(outlines)
    return(outlines)

#A shortcut for calling on pyMCR's functions. I highly recommend reviewing MCR in general, as well as the documentation for this package, which is available at https://pages.nist.gov/pyMCR/
def MCR(SI, guesses, energy, E_win, **kwargs):
    """
    Inputs:
    SI - SI (optional binning)
    guesses - initial guess spectra
    energy - total energy axis
    E_win - energy window in channel number

    kwargs:
    c_regr - optional switch on concentrations constraints
        c_regr = "OLS", no constraints on c (default)
        c_regr = "NNLS", non-negative c [calls NNLS]
    s_regr - optional switch on spectra constraints
        s_regr = "OLS", no constraints on s (default)
        s_regr = "NNLS", non-negative s [calls NNLS]
    iterations - max number of iterations (default = 3)
    nnlstol - optional input for max iterations on NNLS.

    Outputs:
    scores - component maps, same dimension of SI input
    components - spectra of each component

    note: naming convention is kinda weird :/ ST=spectra (components), C=concentrations (scores)
    """
    if np.ndim(SI)==3:
        xdim, ydim, zdim = np.shape(SI)
        SIline = np.reshape(SI,(xdim*ydim,zdim))
        SIline2=np.copy(SIline[:,E_win[0]:E_win[1]])
        SIline2=SIline2.astype('float64')
    else:
        SIline2=np.copy(SI[:,E_win[0]:E_win[1]])
    guesses=np.asarray(guesses,dtype=np.float64)

    if 'show_guess' in kwargs.keys():
        show_guess = kwargs['show_guess']
    else:
        show_guess = True

    if 'c_regr' in kwargs.keys():
        c_regr = kwargs['c_regr']
    else:
        c_regr = "NNLS"
    if 's_regr' in kwargs.keys():
        s_regr = kwargs['s_regr']
    else:
        s_regr = "OLS"
    if 'iterations' in kwargs.keys():
        iterations = kwargs['iterations']
    else:
        iterations = 3
    if 'nnlstol' in kwargs.keys():
        nnlstol = kwargs['nnlstol']
    else:
        nnlstol = np.max(np.shape(SIline2))*3

    if s_regr == "OLS":
        scon = 0
    elif s_regr == "NNLS":
        scon = 1
    else:
        print('s_regr must be OLS or NNLS')

    if c_regr == "OLS":
        ccon = 0
    elif c_regr == "NNLS":
        ccon = 1
    else:
        print('c_regr must be OLS or NNLS')

    scores,components=mcr_func(SIline2, guesses, ccon=ccon,scon=scon,ittol=iterations,nnlstol=nnlstol)

    # colors=['cyan','magenta','yellow','black']
    if np.ndim(SI)==3:
        xdim, ydim, zdim = np.shape(SI)
        num_comp=np.shape(components)[0]
        scores_spatial=np.reshape(scores,(xdim,ydim,num_comp))
        for i in range(num_comp):
            fig,ax=plt.subplots(figsize = (5,3))
            plt.title("Component "+str(i+1)+" Score")
            plt.imshow(scores_spatial[:,:,i],cmap='gray')
            plt.show()
    else:
        xdim,zdim=np.shape(SI)
        num_comp=np.shape(components)[0]
        fig,ax=plt.subplots(figsize=(5,3))
        plt.title("Component scores")
        for i in range(num_comp):
            plt.plot(scores[:,i],label='Component '+str(i+1)+' Contribution',color='C'+str(i))
        plt.legend()
        plt.show()
    eels.easyplot("Components")
    for i in range(num_comp):
        plt.plot(energy[E_win[0]:E_win[1]],components[i,:],label="Component " +str(i+1)+" Result",color='C'+str(i))
        if show_guess:
            plt.plot(energy[E_win[0]:E_win[1]],guesses[i,:]/np.sum(guesses[i,:]),'--',label="Component " +str(i+1)+" Guess",color='C'+str(i),alpha=0.5)
    plt.legend()
    plt.show()

    return(scores,components)

#Do a non-negative least squares to extract the score map.
def full_scores(SI,scores,components,E_win):
    """
    Inputs:
    SI - unbinned background subtracted SI
    scores - component maps from MCR function
    components - spectra of each component from MCR function
    E_win - energy window in channel number

    Outputs:
    scores_spatial - final component maps, basically an extra half step of MCR process
    """
    num_comp=np.shape(scores)[1]
    xdim, ydim, zdim = np.shape(SI)
    full_scores_t=np.zeros((num_comp,xdim*ydim))
    SIline = np.reshape(SI,(xdim*ydim,zdim))
    SIline2=np.copy(SIline[:,E_win[0]:E_win[1]])
    SIline2=SIline2.astype('float64')
    for column in range(0,xdim*ydim):
        x=scipy.optimize.nnls(components.T,SIline2.T[:,column])[0]
        for component in range(num_comp):
            full_scores_t[component,column]=x[component]
    full_scores=full_scores_t.T
    scores_spatial=np.reshape(full_scores,(xdim,ydim,num_comp))
    return(scores_spatial)


def mldivide(A,B):
    """
    Solve systems of linear equations Ax = B for x
    """
    q, r = LA.qr(A)
    p = np.dot(q.T,B)
    return np.dot(LA.inv(r),p)

def mrdivide(B,A):
    """
    Solve systems of linear equations xA = B for x
    """
    return np.dot(B,np.linalg.pinv(A))


def mcr_func(x, c0, **kwargs):
    """
    Credit: Eigenvector Research, Inc. 1997-2000

    Inputs:
    x - matrix to be decomposed as X = CS
    c0 - initial guess for either C or S depending on size. For X (m by n) then C is (m by k) and S is (k by n)
    where k is the number of components determined from the size of the input c0.

    kwargs:
    ccon - optional switch on concentrations constraints
        ccon = 0, no constraints on c (default)
        ccon = 1, non-negative c [calls NNLS]
    scon - optional switch on spectra constraints
        scon = 0, no constraints on s (default)
        scon = 1, non-negative s [calls NNLS]
    ittol - optional convergance criteria
        ittol < 1, convergence tolerance
        ittol >= 1, max number of iterations (default = 100).
    nnlstol - optional input for setting the convergence criteria.

    Outputs:
    c - Conentration map
    s - Pure component spectra

    """
 ### Setting up kwargs
    cx,cy = np.shape(c0)
    xx,xy = np.shape(x)

    if cx == xx:
        ## initial guess are concetration maps
        C0 = np.copy(c0)
        ka = np.copy(cy)
    elif cy == xy:
        ## initial guess are spectral components
        S0 = np.copy(c0)
        ka = np.copy(cx)
    else:
        return print('inital guess must share a dimenision with x (m,n). Either be unwrapped concentration maps (m,k) or spectral components (k,n).')

    if 'ccon' in kwargs.keys():
        ccon = kwargs['ccon']
        if ccon > 2:
            return print('ccon must be 0,1,2. See mcr_func? to learn the corresponding constraints')
    else:
        ccon = 0

    if 'scon' in kwargs.keys():
        scon = kwargs['scon']
        if scon > 2:
            return print('scon must be 0,1,2. See mcr_func? to learn the corresponding constraints')
    else:
        scon = 0

    if 'ittol' in kwargs.keys():
        ittol = kwargs['ittol']
        if ittol < 1:
            itmin = ittol
            itmax = 1e6
        elif ittol > 1 or ittol ==1:
            itmin = 1e-8
            itmax = ittol
    else:
        itmin = 1e-8
        itmax = 1e6
        ittol = np.copy(itmax)

    if 'nnlstol' in kwargs.keys():
        nnlstol = kwargs['nnlstol']
    else:
        nnlstol = np.max(np.shape(x))*3

    # c case
    if ka == cy:
        c = np.copy(C0)
        if scon == 0:
            S0 = mldivide(c,x)
        elif scon ==1:
            S0 = np.zeros((ka,xy))
            for i in range(xy):
                S0[:,i] = nnls(c,x[:,i],maxiter=nnlstol)[0]
        s = np.copy(S0)
        for i in range(ka):
            s[i,:] = s[i,:]/sum(s[i,:])
    elif ka == cx:
        s = np.copy(S0)
        for i in range(ka):
            s[i,:] = s[i,:]/sum(s[i,:])
        if ccon == 0:
            C0 = mrdivide(x,s)
        elif ccon ==1:
            C0 = np.zeros((xx,ka))
            for i in range(xx):
                C0[i,:] = nnls(s.T,x[i,:].T,maxiter=nnlstol)[0].T
        c = np.copy(C0)

    it = 0
    while it < itmax:
        if ccon == 0:
            c = mrdivide(x,s)
        elif ccon ==1:
            for i in range(xx):
                c[i,:] = nnls(s.T,x[i,:].T,maxiter=nnlstol)[0].T

        if scon == 0:
            s = mldivide(c,x)
        elif scon ==1:
            for i in range(xy):
                s[:,i] = nnls(c,x[:,i],maxiter=nnlstol)[0]
        for i in range(ka):
            s[i,:] = s[i,:]/sum(s[i,:])

        garb = it
        it = it + 1

        if (ittol<1) & ((it/2 - it//2) == 0):
            resc, ress = 0,0
            for i in range(ka):
                resc = resc + np.sqrt(np.sum(C0[:,i]*C0[:,i]))
                ress = ress + np.sqrt(np.sum(S0[i,:]*S0[i,:]))

            ress = np.sqrt(sum(sum( (s.T - S0.T)**2))/(ka*ress))
            ress = np.sqrt(sum(sum( (c - C0)**2))/(ka*resc))

            if (ress<itmin) & (resc<itmin):
                it = itmax +1
            else:
                C0 = np.copy(c)
                S0 = np.copy(s)

    return c,s


################################################################################
############################### Obsolete functions##############################
################################################################################

def lineprofile(SI,axis,method='sum'):
    """
    Inputs:
    SI - SI
    axis - 0 for horizontal profile, 1 for vertical profile
    method - 'sum' or 'mean'

    Output:
    profile - summed line profile
    """
    if method == 'sum':
        profile=np.sum(SI,axis=axis)
    elif method=='mean':
        profile=np.mean(SI,axis=axis)
    return(profile)

def spsample(spec,energy,spshift):
    """
    Inputs:
    spec - eels spectra
    energy - corresponding energy axis
    spshift - subpixel shift (in eV) applied to spectra

    Outputs:
    specshifted - corrected spectra
    eshifted - corresponding energy axis shift
    """
    disp = energy[1]-energy[0]
    if abs(spshift/disp)>1:
        return(print("shift must be less than energy channel dispersion"))
    else:
        specshifted = np.zeros((len(spec)-1))
        for i in range(len(specshifted)):
            if spshift>0:
                specshifted[i] = spec[i] - (spshift/disp)*(spec[i] - spec[i+1])
            else:
                specshifted[i] = spec[i+1] - (spshift/disp)*(spec[i] - spec[i+1])
        if spshift>0:
            eshifted = (energy + spshift)[:-1]
        else:
            eshifted = (energy + spshift)[1:]
        return(specshifted,eshifted)

def real_components(SI,scores,components,E_win):
    """
    Inputs:
    SI - unbinned background subtracted SI
    scores - component maps from MCR function
    components - spectra of each component from MCR function
    E_win - energy window in channel number

    Outputs:
    real_comp - average component in for each scoremap in SI
    """
    real_comp = np.zeros_like(components)
    score_bool = np.argmax(scores,axis=2)

    for i in range(np.shape(scores)[2]):
        real_comp[i,:] = np.mean(SI[score_bool==i,:],axis=0)[E_win[0]:E_win[1]]

    return(real_comp)

def load_guess(file):
    rawSI=hs.load(file)
    params=rawSI.axes_manager
    ch1=np.round(rawSI.axes_manager[2j].get_axis_dictionary()['offset'],4)
    disp=np.round(rawSI.axes_manager[2j].get_axis_dictionary()['scale'],4)
    rawSI.z=int(rawSI.axes_manager[2j].get_axis_dictionary()['size'])
    energy= np.round(np.arange(ch1,ch1+rawSI.z*disp,disp),4)
    if len(energy)!= rawSI.z:
        energy = energy[:-1]
    return(energy, rawSI.data, disp, params)

# A shortcut for calling on pyMCR's functions. I highly recommend reviewing MCR in general, as well as the documentation for this package, which is available at https://pages.nist.gov/pyMCR/
# def MCRv1(SI, guesses, energy, E_win, iterations, st_regr="OLS",c_regr="NNLS", c_constraints=[ConstraintNonneg()],
#               st_constraints=[],tol_increase=10.0,tol_n_above_min=30,tol_err_change=10e-3,show_guess='True'):
#     """
#     A shortcut for calling on pyMCR's functions. Full documentation available at https://pages.nist.gov/pyMCR/
#
#     Inputs:
#     SI - SI (optional binning)
#     guesses - initial guess spectra
#     energy - total energy axis
#     E_win - energy window in channel number
#     iterations - number of iterations of MCR
#     default kwargs - st_regr="OLS",c_regr="NNLS", c_constraints=[ConstraintNonneg()], st_constraints=[],tol_increase=10.0,tol_n_above_min=30,tol_err_change=10e-3,show_guess='True'
#
#     Outputs:
#     scores - component maps, same dimension of SI input
#     components - spectra of each component
#
#     note: naming convention is kinda weird :/ ST=spectra (components), C=concentrations (scores)
#     """
#     if np.ndim(SI)==3:
#         xdim, ydim, zdim = np.shape(SI)
#         SIline = np.reshape(SI,(xdim*ydim,zdim))
#         SIline2=np.copy(SIline[:,E_win[0]:E_win[1]])
#         SIline2=SIline2.astype('float64')
#     else:
#         SIline2=np.copy(SI[:,E_win[0]:E_win[1]])
#     guesses=np.asarray(guesses,dtype=np.float64)
#     mcrar = McrAR(max_iter=iterations, st_regr="OLS",c_regr="NNLS", c_constraints=[ConstraintNonneg()],
#               st_constraints=[],tol_increase=10.0,tol_n_above_min=30,tol_err_change=10e-3)
#     mcrar.fit(SIline2,ST=guesses,verbose=True)
#     scores=mcrar.C_opt_
#     components=mcrar.ST_opt_
#     #return(scores, components)
#
# #Display the results of MCR
# #def show_MCR(SI, scores, components,energy_axis,E_win):
#     colors=['cyan','magenta','yellow','black']
#     if np.ndim(SI)==3:
#         xdim, ydim, zdim = np.shape(SI)
#         num_comp=np.shape(components)[0]
#         scores_spatial=np.reshape(scores,(xdim,ydim,num_comp))
#         for i in range(num_comp):
#             fig,ax=plt.subplots(figsize = (5,3))
#             plt.title("Component "+str(i+1)+" Score")
#             plt.imshow(scores_spatial[:,:,i],cmap='gray')
#             plt.show()
#     else:
#         xdim,zdim=np.shape(SI)
#         num_comp=np.shape(components)[0]
#         fig,ax=plt.subplots(figsize=(5,3))
#         plt.title("Component scores")
#         for i in range(num_comp):
#             plt.plot(scores[:,i],label='Component '+str(i+1)+' Contribution',color=colors[i])
#         plt.legend()
#         plt.show()
#     eels.easyplot("Components")
#     for i in range(num_comp):
#         plt.plot(energy[E_win[0]:E_win[1]],components[i,:]/np.sum(components[i,:]),label="Component " +str(i+1)+" Result",color=colors[i])
#         if show_guess:
#             plt.plot(energy[E_win[0]:E_win[1]],guesses[i,:]/np.sum(guesses[i,:]),label="Component " +str(i+1)+" Guess",color=colors[i],alpha=0.4)
#     plt.legend()
#     plt.show()
#
#     return(scores,components)
