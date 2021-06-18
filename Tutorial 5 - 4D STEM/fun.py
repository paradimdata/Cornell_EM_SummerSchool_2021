# -*- coding: utf-8 -*-
"""
Helper functions for the 4D-STEM script
"""
import numpy as np
import pickle
from tqdm import tqdm
from scipy.special import erf
import matplotlib.patches as patches
from scipy import optimize
from scipy import ndimage,linalg 
import time
import matplotlib.pyplot as plt
import cv2
from IPython.display import display
import warnings
import ipywidgets as widgets


def disp2img(img1,img2):
    '''
    Display real & diffraction space images (static, non-interactive)
    Parameters:
    ----------
    img1 : 2D ndarray
        Image to be shown on the left of the figure.
    img2 : 2D ndarray
        Image to be shown on the right of the figure.
    rcoord : TYPE
        DESCRIPTION.
    dcoord : TYPE
        DESCRIPTION.

    Returns ax1 (the leftmost subplot), ax2 (the rightmost subplot).

    '''
    fig=plt.figure(figsize=(10,5));ax1=fig.add_subplot(1,2,1);
    ax1.imshow(img1,cmap='gray');ax1.axis('off')
    ax2=fig.add_subplot(1,2,2);
    ax2.imshow(img2,cmap='gray');ax2.axis('off')
    return ax1,ax2

def statDisp(img1,rcoord,dcoord):
    ax1,ax2=disp2img(img1[:,:,rcoord[1],rcoord[0]],img1[dcoord[1],dcoord[0],:,:]**0.1);
    ax1.set_title('Real Space (%d,%d)'%(dcoord[0],dcoord[1]))
    ax1.scatter(dcoord[0],dcoord[1],s=5,c='red',marker='o');
    ax2.scatter(rcoord[0],rcoord[1],s=5,c='red',marker='o');
    ax2.set_title('Diffraction Space (%d,%d)'%(rcoord[0],rcoord[1]))
    return ax1,ax2

def ewpc2D(data,useWindow=True,minlog=0.1,bg_sigma=20):
    '''
    converts dp slice into ewpc
    %ewpc calculates the EWPC transform fft(log(data)) for 2d or 4d data
    %   input:
    %       data -- 2d or 4d scanning electron diffraction data, ordered (kx,
    %               ky, x, y)
    %       useWindow -- logical indicating whether or not to apply hann window
    %               before FFT.  Default is 1.  The window is useful to prevent
    %               FFT artifacts from non-periodicity.  This is especially
    %               important it the diffraction patterns have significant
    %               intensity at their edges.
    '''
    N_kx,N_ky=data.shape
    minval=np.min(data)
    if useWindow:#window to prevent artifacts caused by non-periodic boundaries
        win=np.outer(np.hanning(N_kx),np.hanning(N_ky))
    else:
        win=np.ones((N_kx,N_ky))
    logdp=np.log(data-minval+minlog)
    cep=np.abs(np.fft.fftshift(np.fft.fft2(logdp*win)))
    return cep


def ConstrainedFun(x,func,win1,win2):
    '''
    %adds constraint to objective function fun, which is assumed to be always
    %negative, by adding a positive "cone of shame" outside the specified
    %window
    '''
    if x[0]<win1[0] or x[0]>win1[1] or x[1]<win2[0] or x[1]>win2[1]:
        cent=[np.mean(win1),np.mean(win2)]
        y=np.sqrt((x[0]-cent[0])**2+(x[1]-cent[1])**2)
    else:
        y = func(x)
    return y

# def window2(N,M,w_func):
# # %Makes a 2D window function for FFT
#     wc=window(w_func,N)
#     wr=window(w_func,M)
#     [maskr,maskc]=np.meshgrid(wr,wc)
#     return maskr*maskc


def calculateSpotMapVectors(spotMaps,center=[62,62]):
    '''
    Calculates vector components, length, and angle assuming the detector 
    has 124 pixels, i.e. zero=(63,63). These are added to the spotMaps struct.

    Parameters
    ----------
    spotMaps : TYPE
        DESCRIPTION.
    center : TYPE, optional
        DESCRIPTION. The default is [62,62].

    Returns
    -------
    None.

    '''

    numSpots = len(spotMaps['Q1map'])
    spotMaps_updated=spotMaps.copy()
    spotMaps_updated['VectorX1']=np.zeros_like(spotMaps['Q1map'])
    spotMaps_updated['VectorX2']=np.zeros_like(spotMaps['Q1map'])
    spotMaps_updated['VectorLength']=np.zeros_like(spotMaps['Q1map'])
    spotMaps_updated['VectorAngle']=np.zeros_like(spotMaps['Q1map'])    
    for i in range(numSpots):
        x1map=spotMaps['Q1map'][i]
        x1map=x1map-center[0]
        x2map=spotMaps['Q2map'][i]
        x2map=x2map-center[1]
        spotMaps_updated['VectorX1'][i]=x1map
        spotMaps_updated['VectorX2'][i]=x2map
        spotMaps_updated['VectorLength'][i]=np.sqrt(x1map**2+x2map**2)
        spotMaps_updated['VectorAngle'][i]=np.arctan2(x1map,x2map)
    return spotMaps_updated  

def cft2(f,q1,q2,zeroCentered=0):
    '''
    Calculates the continuous fourier transform

    %cft2 2D continuous Fourier tranform of f evaluated at point q1, q2
    %   inputs:
    %       f -- the 2D array the fourier transform is calculated from
    %       q1,q2 -- indices where the transform is to be evaluated, following
    %              the same convention as fft2.  q1,q2 can be non-integers.
    %       zeroCentered -- boolean indicating the q index corresponding to
    %                       zero: 0 - default, zero is at index 1,1 (same as
    %                                 fft2(f))
    %                             1 - zero is at the image center,
    %                                 corresponding to fftshift(fft2(f))
    %   outputs:
    %       F -- value of the fourier transform of f at q1,q2.  This is a complex
    %            number, rather than an array.
    %
    %This function is part of the PC-STEM Package by Elliot Padgett in the 
    %Muller Group at Cornell University.  Last updated by Megan Holtz for speed July 27, 2020.
    '''
    (m,n)=f.shape
    jgr = np.arange(m)
    kgr = np.arange(n)

    if zeroCentered:
        q1=q1+m/2
        q2=q2+n/2

    F = np.sum(f*np.outer(np.exp(-2*np.pi*1j*jgr*q1/m),np.exp(-2*np.pi*1j*kgr*q2/n)))

    return F

def central_beam_mask(dp,bright_disk_radius=5,erf_sharpness=5):

##### generates mask that blanks the center region (used for ewpc patterns)
    xcols=dp.shape[-2]
    yrows=dp.shape[-1]
    kx = np.arange(-xcols, xcols, 2)/2
    ky = np.arange(-yrows, yrows, 2)/2
    ky,kx = np.meshgrid(ky, kx)
    dist = np.hypot(kx, ky)
    bdisk_filter = erf((dist-bright_disk_radius)*erf_sharpness)/2 - erf((dist+bright_disk_radius)*erf_sharpness)/2 + 1
    return bdisk_filter

  
def load_raw_to_dp(fname , yrows , xcols ):
    with open(fname, 'rb') as file:
        dp = np.fromfile(file, np.float32)
                
    sqpix = dp.size/yrows/xcols
    pix = int(sqpix**(0.5))
    
    dp = np.reshape(dp, (pix, pix, yrows, xcols), order = 'C')
    if yrows>xcols:
        dp = dp[:,:,1:xcols+1, :]
        dp[:,:,0,:]=0
        dp[:,:,-1,:]=0
        dp = np.nan_to_num(dp)
        dp[dp <= 20] = 1e-10
    
    dp = dp[:,:,1:-3,2:-2]
    dp = np.swapaxes(dp, 2,3)
    dp = np.flip(dp, 0)
    dp = np.flip(dp, 2)

    return dp #edited; before: return dp

def saturate_array(masked_array,mask,saturation_lims):
    [min_val,max_val]=np.percentile(masked_array[np.logical_not(mask)],saturation_lims)
    binary_mask=np.logical_and(masked_array>max_val,np.logical_not(mask))
    masked_array[binary_mask]=max_val
    binary_mask=np.logical_and(masked_array<min_val,np.logical_not(mask))
    masked_array[binary_mask]=min_val
    return masked_array

def convert_dp_to_ewpc(dp,flatten_center=False,bright_disk_radius=5,erf_sharpness=5):
  pix, pix, yrows, xcols = dp.shape
  cep=np.zeros_like(dp)
  kx = np.arange(-xcols, xcols, 2)/2
  kx,ky = np.meshgrid(kx, kx)
  dist = np.hypot(kx, ky)
  if flatten_center:
      bdisk_filter = erf((dist-bright_disk_radius)*erf_sharpness)/2 - erf((dist+bright_disk_radius)*erf_sharpness)/2 + 1
  else:
      bdisk_filter= np.ones((yrows,xcols))

  for i in tqdm(range(pix)):
      for j in range(pix):
          current_slice=dp[i,j,:,:].copy()
          current_slice[current_slice<=0]=current_slice[current_slice>0].min()
          curr_cep = np.log10(current_slice)
          curr_cep=curr_cep*np.outer(np.hanning(yrows),np.hanning(xcols))
          curr_cep = np.fft.fft2(curr_cep)
          curr_cep = np.fft.fftshift(curr_cep, (-1, -2))
          cep[i,j] = np.abs(curr_cep)*bdisk_filter
  return cep

def create_haadf_mask(array_shape,radii):
    [r0,r1]=radii
    center=[array_shape[-2]/2,array_shape[-1]/2]
    kx = np.arange(array_shape[-1])-int(center[-1])
    ky = np.arange(array_shape[-2])-int(center[-2])
    kx,ky = np.meshgrid(kx,ky)
    kdist = (kx**2.0 + ky**2.0)**(1.0/2)
    haadf_mask = np.array(kdist <= r1, np.int)*np.array(kdist >= r0, np.int)
    return haadf_mask  

def disp_haadf(data4d,radii,*args):
    dim =data4d.shape #N_x1,N_x2,N_k1,N_k2
    try:
        i=args[0];j=args[1];
    except:
        print("real space position not specified correctly. Using default values.")
        i=int(dim[0]/2); j=int(dim[0]/2);
    haadf_mask=create_haadf_mask((dim[2],dim[3]),radii)
    haadf=np.mean(data4d*haadf_mask,axis=(-2,-1))
    ratio_array=np.log(data4d[i,j,:,:])
    haadf_bndry=np.logical_xor(ndimage.binary_dilation(haadf_mask),haadf_mask)

    normalized_ratio_array=(ratio_array-ratio_array.min())/(ratio_array.max()-ratio_array.min())
    img=plt.cm.gray(normalized_ratio_array)#use grayscale colormap

    img[haadf_bndry]=[1,0,0,1] #make the boundary red
    
    ax1,ax2=disp2img(haadf,img);
    ax1.set_title('Image'); ax2.set_title('Mean Pattern')

def show_roi(ewpc,roi,wins):
    win_mask=np.zeros((ewpc.shape[2],ewpc.shape[3])).astype('bool')
    for i in range(len(wins)):
        win_mask[wins[i,2]:wins[i,3],wins[i,0]:wins[i,1]]=True
    cep_df=np.sum(ewpc*win_mask,axis=(-2,-1))
    ax1,ax2=disp2img(cep_df,cep_df[roi[2]:roi[3]+1,roi[0]:roi[1]+1])
    ax1.add_patch(patches.Rectangle((roi[0],roi[2]),roi[1]+1-roi[0],roi[3]+1-roi[2],linewidth=1,edgecolor='r',facecolor='none'))
    
def show_wins(data4d,wins,roi):
    data4d_roi=data4d[roi[2]:roi[3]+1,roi[0]:roi[1]+1,:,:].copy()
    valid = np.ones([data4d_roi.shape[0],data4d_roi.shape[1]])
    valid = valid.astype(bool)
    
    ### store the windows positions in spotlist dictionary
    spotList=create_spotList(wins)
    
    (rx,ry,sx,sy)=data4d_roi.shape
    dp_mean=np.mean(data4d_roi.reshape((rx*ry,sx*sy)).T.reshape((sx,sy,rx,ry)), axis=(-2,-1))
   
    ewpc_img=ewpc2D(dp_mean)*central_beam_mask(dp_mean)
    ewpc_img=(ewpc_img-ewpc_img.min())/(ewpc_img.max()-ewpc_img.min())
    ax1,ax2=disp2img(np.log( dp_mean * 1e5 + 1.0 ),ewpc_img)
    
    for j in range(len(wins)):
        win=patches.Rectangle((wins[j,0],wins[j,2]),wins[j,1]-wins[j,0],wins[j,3]-wins[j,2],linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(win)
    
    ax1.set_title('log(Average DP)');ax2.set_title('selected EWPC peaks')
    
    return data4d_roi,ewpc_img

def create_spotList(wins):
        ### store the windows positions in spotlist dictionary
    spotList={}
    spotList['spotRangeQ1']=[]
    spotList['spotRangeQ2']=[]
    for i in range(len(wins)):
        spotList['spotRangeQ1'].append([wins[i][2],wins[i][3]]) #check
        spotList['spotRangeQ2'].append([wins[i][0],wins[i][1]])
    return spotList

def get_spotMaps(data4d_roi,wins,valid=None,tol=1e-4,method='Nelder-Mead'):
    (N_x1,N_x2,N_k1,N_k2)=data4d_roi.shape
    q1range=np.arange(N_k1)
    q2range=np.arange(N_k2)
    [Q2,Q1]=np.meshgrid(q2range,q1range)
    ###hann window 
    win=np.outer(np.hanning(N_k1),np.hanning(N_k2))
    spotList=create_spotList(wins)
    ### create the spotMaps dictionary, where the peak locations will be saved
    spotMaps={}; spotMaps['Q1map']=[]; spotMaps['Q2map']=[]
    for s in range(len(wins)):#edit to allow multiple windows
        spotMaps['Q1map'].append(np.zeros(data4d_roi.shape[0:2]))
        spotMaps['Q1map'][s][:,:]= np.nan; #### first set all values to nan 
        spotMaps['Q2map'].append(np.zeros(data4d_roi.shape[0:2]))
        spotMaps['Q2map'][s][:,:]= np.nan   
    if np.sum(valid==None):
        valid = np.ones([data4d_roi.shape[0],data4d_roi.shape[1]]) 
    valid = valid.astype(bool)
    mask_pos=np.where(valid)
    t1=time.time()
    ### now go through the points within the valid mask and calculate the peak locations
    for num_pos in tqdm(range(valid.sum())):
        j=mask_pos[0][num_pos]
        k=mask_pos[1][num_pos]
        CBED = data4d_roi[j,k,:,:]
        minval=CBED.min()
        EWPC = ewpc2D(CBED)
        #define continuous Fourier transform
        PeakFun = lambda x: -np.abs(cft2(win*np.log(CBED-minval+0.1),x[0],x[1],1)) 
        #iterate through spots of interest
        for s in range(len(spotList['spotRangeQ1'])):
            #Get spot locations from input struct
            spot_ROI_q1 = spotList['spotRangeQ1'][s]
            spot_ROI_q2 = spotList['spotRangeQ2'][s]
            spotNeighborhood = EWPC[spot_ROI_q1[0]:spot_ROI_q1[1]+1,spot_ROI_q2[0]:spot_ROI_q2[1]+1]
            #Find rough location of maximum peak 
            maxidx= np.unravel_index(np.argmax(spotNeighborhood),spotNeighborhood.shape)
            Q1_roi = Q1[spot_ROI_q1[0]:spot_ROI_q1[1]+1,spot_ROI_q2[0]:spot_ROI_q2[1]+1]
            Q2_roi = Q2[spot_ROI_q1[0]:spot_ROI_q1[1]+1,spot_ROI_q2[0]:spot_ROI_q2[1]+1]
            Q1max = Q1_roi[maxidx]
            Q2max = Q2_roi[maxidx]
            #Search for spot peak in continuous Fourier transform
            constrainedPeakFun = lambda x: ConstrainedFun(x,PeakFun,[spot_ROI_q1[0],spot_ROI_q1[-1]],[spot_ROI_q2[0],spot_ROI_q2[-1]])
            if method=='Nelder-Mead':
                peakQ = optimize.fmin(constrainedPeakFun,x0=np.array([Q1max,Q2max]),ftol=tol,xtol=tol,disp=False)
            elif method in ['L-BFGS-B','Powell','TNC']:
                bnds=((spot_ROI_q1[0],spot_ROI_q1[1]+1),(spot_ROI_q2[0],spot_ROI_q2[1]+1))
                peakQ = optimize.minimize(PeakFun,x0=np.array([Q1max,Q2max]),method=method,bounds=bnds,tol=tol).x
            # Assign in maps
            spotMaps['Q1map'][s][j,k] = peakQ[0]
            spotMaps['Q2map'][s][j,k] = peakQ[1]    
    t2=time.time()
    spotMaps_upd=calculateSpotMapVectors(spotMaps,center=[int(N_k1/2),int(N_k2/2)])
    print('Time spent: '+ "{:.0f}".format(t2-t1) + 's')
    return spotMaps_upd

def plotSpotMaps(wins,ewpc_img,spotMaps_upd,figureSize=(10,5),sat_lims=[0,100],pix_size=None,unit_label='pixels',cmap='viridis',plot_ids=None):
    fig = plt.figure(figsize=figureSize,constrained_layout=True)
    j=len(spotMaps_upd['VectorLength'])
    if pix_size==None:
        pix_size=1
    if plot_ids==None:
        plot_ids=np.arange(j)
    for i in range(j):
        ax1=fig.add_subplot(j,3,3*i+1);
        ax2=fig.add_subplot(j,3,3*i+2);ax3=fig.add_subplot(j,3,3*i+3)
        
        im1 = ax1.imshow(ewpc_img)
        win=patches.Rectangle((wins[i,0],wins[i,2]),wins[i,1]-wins[i,0],wins[i,3]-wins[i,2],linewidth=1,edgecolor='r',facecolor='none')
        ax1.add_patch(win); ax1.set_title('EWPC peak #'+str(plot_ids[i]));ax1.axis('off')

        array=spotMaps_upd['VectorLength'][i].copy()
        mask=np.isnan(array)
        mask_pos=np.where(np.logical_not(mask))
        a1=mask_pos[0].min()
        a2=mask_pos[0].max()
        b1=mask_pos[1].min()
        b2=mask_pos[1].max()
        
        if sat_lims!=[0,100]:
            array=saturate_array(array,mask,sat_lims)
        array=array[a1:a2,b1:b2]*pix_size
        vmin = np.nanmean(array) - np.nanstd(array)
        vmax =  np.nanmean(array) + np.nanstd(array)
        im2=ax2.imshow(array, vmin=vmin, vmax=vmax,cmap=cmap)
        cb2 = fig.colorbar(im2, ax=ax2, label = unit_label)
        ax2.set_title('Vector Length');ax2.axis('off')
        array=spotMaps_upd['VectorAngle'][i].copy()
        if sat_lims!=[0,100]:
            array=saturate_array(array,mask,sat_lims)
        array=180*array[a1:a2,b1:b2]/np.pi
        vmin = np.nanmean(array) - np.nanstd(array)
        vmax =  np.nanmean(array) + np.nanstd(array)
        im3=ax3.imshow(array, vmin=vmin, vmax=vmax,cmap=cmap) 
        fig.colorbar(im3, ax=ax3, label = 'deg') 
        ax3.set_title('Vector Angle');ax3.axis('off')
    fig.set_constrained_layout_pads(hspace=0.2, wspace=0.2)


def makeRelativeSpotReference( spotMaps_upd, ref_roi ):

    spotRef = {'id':[], 'point': []}

    num = len(spotMaps_upd['Q1map'])
    
    for i in range(num):
        spotRef["id"].append(i)
        ref1 = np.nanmean(spotMaps_upd['VectorX1'][i][ref_roi[0]:ref_roi[1], ref_roi[2]:ref_roi[3]])
        ref2 = np.nanmean(spotMaps_upd['VectorX2'][i][ref_roi[0]:ref_roi[1], ref_roi[2]:ref_roi[3]])
        spotRef["point"].append( np.array([ref1, ref2]) )

    return spotRef

def calculateStrainMap(spotMaps_upd, spotRef, latticeCoords=0):
    [N_x1,N_x2] = spotMaps_upd["Q1map"][0].shape
    
    StrainComponents = {'Eps11':np.zeros((N_x1, N_x2)), 'Eps22':np.zeros((N_x1, N_x2)), 'Eps12':np.zeros((N_x1, N_x2)), 'Theta':np.zeros((N_x1, N_x2)), 'majAx':np.zeros((N_x1, N_x2)), 'minAx':np.zeros((N_x1, N_x2)), 'strainAngle':np.zeros((N_x1, N_x2))}
    
    E = np.zeros((N_x1, N_x2, 2, 2))
    R = np.zeros((N_x1, N_x2, 2, 2))
    
    #prepare reference point list 
    num = len(spotRef['id'])
    for i in range(num):
        refPoints = np.array([ [0,0], spotRef['point'][0], spotRef['point'][1] ])
        refPoints = np.float32(refPoints)
    #dataPoints = [[0,0]]
    
    for j in range(N_x1):
        for k in range(N_x2):
            dataPoints = [[0,0]]
            
            for s in range(num):
                
                q1c = spotMaps_upd["VectorX1"][s][j,k]
                q2c = spotMaps_upd["VectorX2"][s][j,k]
                
                #include in list for tranformation calculation
                dataPoints.append([q1c, q2c])
                
                dataPoints_array = np.float32(np.array(dataPoints))
                
            if( np.sum(np.isnan(dataPoints_array)) ):
                StrainComponents["Eps11"][j,k]=np.nan
                StrainComponents["Eps22"][j,k] = np.nan
                StrainComponents["Eps12"][j,k] = np.nan
                StrainComponents["Theta"][j,k]=np.nan
                StrainComponents["majAx"][j,k] = np.nan
                StrainComponents["minAx"][j,k] = np.nan
                StrainComponents["strainAngle"][j,k] = np.nan                
                
            else:       
                M = cv2.getAffineTransform(refPoints, dataPoints_array)
                M =  M[:,0:2]
                # may need to take transpose to be consistent with Matlab
                
                
                r, u = linalg.polar(M, 'right') # M = ru
                r, v = linalg.polar(M, 'left') # M = vr
                
                if latticeCoords==1:
                    strain_mat = u - np.eye(2)
                else:
                    strain_mat = v - np.eye(2)
                    
                E[j,k,:,:] = strain_mat
                R[j,k,:,:] = r
                
                StrainComponents["Eps11"][j,k] = strain_mat[0,0]
                StrainComponents["Eps22"][j,k] = strain_mat[1,1]
                StrainComponents["Eps12"][j,k] = strain_mat[0,1]
                StrainComponents["Theta"][j,k] = 180*np.arctan2( r[1,0], r[0,0] )/np.pi
                

                #strain ellipse parameters
                
                eigval, eigvec = np.linalg.eig(strain_mat)
                
                if( eigval[0] > eigval[1]):
                    StrainComponents["majAx"][j,k] = eigval[0]
                    StrainComponents["minAx"][j,k] = eigval[1]
                    StrainComponents["strainAngle"][j,k] = np.arctan2( eigvec[0,0], eigvec[1,0] )                    
                else:
                    StrainComponents["majAx"][j,k] = eigval[1]
                    StrainComponents["minAx"][j,k] = eigval[0]
                    StrainComponents["strainAngle"][j,k] = np.arctan2( eigvec[0,0], eigvec[1,0] )                    
    
    return StrainComponents


def plotStrainEllipse(StrainComponents,figureSize=(8,3)):
    
    color = 'viridis'
    
    plt.figure(figsize=figureSize)
    plt.subplot(1,3,1)
    img1=plt.imshow(StrainComponents["majAx"], cmap=color)
    plt.colorbar(img1,shrink = 0.75)
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.title("Major axis", fontsize=12)
    
    plt.subplot(1,3,2)
    plt.imshow(StrainComponents["minAx"], cmap=color)
    plt.colorbar(shrink = 0.75)
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.title("Minor axis", fontsize=12)
    
    plt.subplot(1,3,3)
    plt.imshow(StrainComponents["strainAngle"], cmap=color)
    plt.colorbar(shrink = 0.75)
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.title("Axis angle", fontsize=12)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    
    
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotStrainTensor(StrainComponents,figureSize=(8,8),sat_lims=[0,100],cmap='viridis'):    
    titles=["$\epsilon_{11}$","$\epsilon_{22}$","$\epsilon_{12}$","$\Theta $"]
    keys=["Eps11","Eps22","Eps12","Theta"]
    mask=np.isnan(StrainComponents["Eps11"])
    mask_pos=np.where(np.logical_not(mask))
    a1=mask_pos[0].min()
    a2=mask_pos[0].max()
    b1=mask_pos[1].min()
    b2=mask_pos[1].max()

    fig,axes=plt.subplots(2,2,figsize=figureSize)
    for i in range(4):
        array=StrainComponents[keys[i]].copy()
        ax_loc=np.unravel_index(i,(2,2))
        if sat_lims!=[0,100]:
            array=saturate_array(array,mask,sat_lims)
        array=array[a1:a2,b1:b2]
        im=axes[ax_loc].imshow(array,cmap=cmap)
        axes[ax_loc].set_xticks([])
        axes[ax_loc].set_yticks([])        
        divider = make_axes_locatable(axes[ax_loc])
        cax = divider.append_axes("right", size="10%", pad=0.05)
        axes[ax_loc].set_title(titles[i])
        if keys[i]=='Theta':
            plt.colorbar(im,cax=cax,label='deg')
        else:
            plt.colorbar(im,cax=cax)
    plt.subplots_adjust(wspace=0.1, hspace=0.15)        

    
def pick_window_ewpc(data4d,adf_img,wins=[],roi=[],cmap='gray'):
    plot_output = widgets.Output()
    
    mean_reciprocal=np.sum(data4d,axis=(0,1))
    
    dx,dy = np.shape(adf_img)
    
    fig=plt.figure(figsize=(9, 6))
    ax1=fig.add_axes([0.1,0.6,0.2,0.3])
    ax2=fig.add_axes([0.35,0.6,0.2,0.3])
    ax3=fig.add_axes([0.1,0.2,0.2,0.3])
    ax4=fig.add_axes([0.35,0.2,0.2,0.3])
    ax6=fig.add_axes([0.6,0.2,0.2,0.3])
    ax5=fig.add_axes([0.6,0.6,0.2,0.3])

    ax1.matshow(adf_img,cmap='gray')
    ax2.matshow(adf_img,cmap='gray')
    ax3.matshow(mean_reciprocal,cmap='gray')
    ax3.matshow(ewpc2D(mean_reciprocal),cmap='gray')
    ax1.set_title('ADF image')
    ax3.set_title('CBED')
    ax4.set_title('LOG(EWPC)')
    ax5.set_title('EWPC WIN')
    ax6.set_title('DF EWPC ROI')
    for ax in [ax1,ax3,ax5]:
        ax.set_xticks([]);ax.set_yticks([]);

    # widgets
    dx_range_slider_1 = widgets.IntRangeSlider(value=(0, dx-1), min=0, max=dx-1, step=1, description='ROI x range')
    dy_range_slider_1 = widgets.IntRangeSlider(value=(0, dy-1), min=0, max=dy-1, step=1, description='ROI y range')

    dx_range_slider_2 = widgets.IntSlider(value=0, min=0, max=dx-1, step=1, description='CBED at x ')
    dy_range_slider_2 = widgets.IntSlider(value=0, min=0, max=dy-1, step=1, description='CBED at y')
    
    dkx,dky=np.shape(data4d)[-2:]
    dx_range_slider_3 = widgets.IntSlider(value=int(dkx-1)/2, min=0, max=dkx-1, step=1, description='EWPC win x')
    dy_range_slider_3 = widgets.IntSlider(value=int(dkx-1)/2, min=0, max=dky-1, step=1, description='EWPC win y')
    
    win_size_slider = widgets.IntSlider(value=3, min=1, max=6, step=1, description='EWPC win size')
    
    check_region_2 = widgets.Checkbox(value=False,description='CBED in LOG')

    ewpc_df_button = widgets.Button(description="Update EWPC DF")
    
    add_win_list_button= widgets.Button(description="Add Window")
    
    
    # update display with updated widgets
    
    def update_edgemap(xr_1,yr_1,xr_2,yr_2,check_2,xr_3,yr_3,win_size):
        ax2.cla()
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        plot_output.clear_output()

        xmin_1, xmax_1 = xr_1[0],xr_1[1]
        ymin_1, ymax_1 = yr_1[0],yr_1[1]
        box_1 = [[xmin_1-0.5,xmin_1-0.5,xmax_1+0.5,xmax_1+0.5,xmin_1-0.5],[ymin_1-0.5,ymax_1+0.5,ymax_1+0.5,ymin_1-0.5,ymin_1-0.5]]
        xmin_3, xmax_3 = [xr_3-win_size,xr_3+win_size]
        ymin_3, ymax_3 = [yr_3-win_size,yr_3+win_size]
        box_3 = [[xmin_3-0.5,xmin_3-0.5,xmax_3+0.5,xmax_3+0.5,xmin_3-0.5],[ymin_3-0.5,ymax_3+0.5,ymax_3+0.5,ymin_3-0.5,ymin_3-0.5]]

        if check_2:
            cbed=np.log(data4d[xr_2,yr_2,:,:])
        else:
            cbed=data4d[xr_2,yr_2,:,:]
            
        with plot_output:
            [l.remove() for l in ax1.lines]
            [l.remove() for l in ax4.lines]
            ax1.plot(box_1[1],box_1[0],color='C0')
            ax2.matshow(adf_img[xmin_1:xmax_1,ymin_1:ymax_1],extent=[ymin_1,ymax_1,xmax_1,xmin_1],cmap=cmap)
            ax2.plot(yr_2,xr_2,'ro',markersize=3)
            ax3.imshow(cbed,cmap=cmap)
            ewpc=ewpc2D(data4d[xr_2,yr_2,:,:])
            ax4.imshow(np.log(ewpc),cmap=cmap)
            ax4.plot(box_3[1],box_3[0],color='C1')
            ax5.matshow(ewpc[xmin_3:xmax_3,ymin_3:ymax_3],cmap=cmap)
            for ax in [ax1,ax3,ax5]:
                ax.set_xticks([]);ax.set_yticks([]);
            plt.show()

    # update widgets
    def update_xrange2(*args):
        dx_range_slider_2.min=dx_range_slider_1.value[0]
        dx_range_slider_2.max=dx_range_slider_1.value[1]
    def update_yrange2(*args):
        dy_range_slider_2.min=dy_range_slider_1.value[0]
        dy_range_slider_2.max=dy_range_slider_1.value[1]
    def dx_range_eventhandler_1(change):
        yr_1=dy_range_slider_1.value
        xr_1=change.new

        update_edgemap(change.new,dy_range_slider_1.value,
                       dx_range_slider_2.value,dy_range_slider_2.value,check_region_2.value,
                       dx_range_slider_3.value,dy_range_slider_3.value,win_size_slider.value)                       
        update_xrange2(change.new)
        roi=[yr_1[0],yr_1[1],xr_1[0],xr_1[1]]
        
    def dy_range_eventhandler_1(change):
        xr_1=dx_range_slider_1.value
        yr_1=change.new
        update_edgemap(dx_range_slider_1.value,change.new,
                       dx_range_slider_2.value,dy_range_slider_2.value,check_region_2.value,
                       dx_range_slider_3.value,dy_range_slider_3.value,win_size_slider.value)
        update_yrange2(change.new)
        roi=[yr_1[0],yr_1[1],xr_1[0],xr_1[1]]
    def dx_range_eventhandler_2(change):
        update_edgemap(dx_range_slider_1.value,dy_range_slider_1.value,
                       change.new,dy_range_slider_2.value,check_region_2.value,
                       dx_range_slider_3.value,dy_range_slider_3.value,win_size_slider.value)                       
    def dy_range_eventhandler_2(change):
        update_edgemap(dx_range_slider_1.value,dy_range_slider_1.value,
                       dx_range_slider_2.value,change.new,check_region_2.value,
                       dx_range_slider_3.value,dy_range_slider_3.value,win_size_slider.value)
    def dx_range_eventhandler_3(change):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        yr_3=dy_range_slider_3.value
        xr_3=change.new
        win_size=win_size_slider.value
        xmin_3, xmax_3 = [xr_3-win_size,xr_3+win_size]
        ymin_3, ymax_3 = [yr_3-win_size,yr_3+win_size]
        box_3 = [[xmin_3-0.5,xmin_3-0.5,xmax_3+0.5,xmax_3+0.5,xmin_3-0.5],[ymin_3-0.5,ymax_3+0.5,ymax_3+0.5,ymin_3-0.5,ymin_3-0.5]]
        [l.remove() for l in ax4.lines]
        ax4.plot(box_3[1],box_3[0],color='C1')
        xr_2=dx_range_slider_2.value
        yr_2=dy_range_slider_2.value
        ewpc=np.log(ewpc2D(data4d[xr_2,yr_2,:,:]))
        ax5.matshow(ewpc[xmin_3:xmax_3,ymin_3:ymax_3],cmap=cmap)
        ax5.set_xticks([]);ax5.set_yticks([])

    def dy_range_eventhandler_3(change):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        xr_3=dx_range_slider_3.value
        yr_3=change.new
        win_size=win_size_slider.value        
        xmin_3, xmax_3 = [xr_3-win_size,xr_3+win_size]
        ymin_3, ymax_3 = [yr_3-win_size,yr_3+win_size]
        box_3 = [[xmin_3-0.5,xmin_3-0.5,xmax_3+0.5,xmax_3+0.5,xmin_3-0.5],[ymin_3-0.5,ymax_3+0.5,ymax_3+0.5,ymin_3-0.5,ymin_3-0.5]]
        [l.remove() for l in ax4.lines]
        ax4.plot(box_3[1],box_3[0],color='C1')
        xr_2=dx_range_slider_2.value
        yr_2=dy_range_slider_2.value
        ewpc=np.log(ewpc2D(data4d[xr_2,yr_2,:,:]))
        ax5.matshow(ewpc[xmin_3:xmax_3,ymin_3:ymax_3],cmap=cmap)
        ax5.set_xticks([]);ax5.set_yticks([])
    def checkbox_eventhandler_2(change):
        update_edgemap(dx_range_slider_1.value,dy_range_slider_1.value,
                       dx_range_slider_2.value,dy_range_slider_2.value,change.new,
                       dx_range_slider_3.value,dy_range_slider_3.value,win_size_slider.value)
    def win_size_eventhandler(change):
        update_edgemap(dx_range_slider_1.value,dy_range_slider_1.value,
                       dx_range_slider_2.value,dy_range_slider_2.value,check_region_2.value,
                       dx_range_slider_3.value,dy_range_slider_3.value,change.new)
        
    def ewpc_df_button_update(change):
        xr_3=dx_range_slider_3.value
        yr_3=dy_range_slider_3.value
        xr_1=dx_range_slider_1.value
        yr_1=dy_range_slider_1.value
        win_size=win_size_slider.value
        xmin_1, xmax_1 = xr_1[0],xr_1[1]
        ymin_1, ymax_1 = yr_1[0],yr_1[1]
        xmin_3, xmax_3 = [xr_3-win_size,xr_3+win_size]
        ymin_3, ymax_3 = [yr_3-win_size,yr_3+win_size]
        ewpc_roi=np.zeros((xmax_1-xmin_1,ymax_1-ymin_1,xmax_3-xmin_3,ymax_3-ymin_3))
        for x in range(xmin_1,xmax_1):
            for y in range(ymin_1,ymax_1):
                ewpc_roi[x-xmin_1,y-ymin_1,:,:]=ewpc2D(data4d[x,y,:,:])[xmin_3:xmax_3,ymin_3:ymax_3]
        ax6.imshow(np.sum(ewpc_roi,axis=(-2,-1)),cmap=cmap)
    def add_wins_button_update(change):
        xr_3=dx_range_slider_3.value
        yr_3=dy_range_slider_3.value
        win_size=win_size_slider.value
        xmin_3, xmax_3 = [xr_3-win_size,xr_3+win_size]
        ymin_3, ymax_3 = [yr_3-win_size,yr_3+win_size]        
        wins.append([ymin_3,ymax_3,xmin_3,xmax_3])
        xr_1=dx_range_slider_1.value
        yr_1=dy_range_slider_1.value
        roi.append([yr_1[0],yr_1[1],xr_1[0],xr_1[1]])

    # observe updated widgets
    dx_range_slider_1.observe(dx_range_eventhandler_1,names='value')
    dy_range_slider_1.observe(dy_range_eventhandler_1,names='value')
    dx_range_slider_2.observe(dx_range_eventhandler_2,names='value')
    dy_range_slider_2.observe(dy_range_eventhandler_2,names='value')
    check_region_2.observe(checkbox_eventhandler_2,names='value')
    dx_range_slider_3.observe(dx_range_eventhandler_3,names='value')
    dy_range_slider_3.observe(dy_range_eventhandler_3,names='value')
    win_size_slider.observe(win_size_eventhandler,names='value')
    ewpc_df_button.on_click(ewpc_df_button_update)
    add_win_list_button.on_click(add_wins_button_update)
    # display
    input_widgets_1 = widgets.HBox([dx_range_slider_1, dy_range_slider_1])
    input_widgets_2 = widgets.HBox([check_region_2,dx_range_slider_2, dy_range_slider_2])
    input_widgets_3 = widgets.HBox([dx_range_slider_3, dy_range_slider_3,win_size_slider])
    input_widgets_4 = widgets.HBox([ewpc_df_button,add_win_list_button])
    input_widgets = widgets.VBox([input_widgets_1,input_widgets_2,input_widgets_3,input_widgets_4])
    display(input_widgets)

from matplotlib.widgets import RectangleSelector,Button

def browser_with_peak_selection(data4d,cmap='gray', wins=[],rois=[],half_width=8):
    rx,ry,kx,ky=np.shape(data4d)
    bf_img=data4d[:,:,int(kx/2),int(ky/2)]    
    fig=plt.figure(figsize=(10, 6))
    ax1=fig.add_axes([0.10,0.1,0.25,0.8])
    ax2=fig.add_axes([0.40,0.1,0.25,0.8])
    ax3=fig.add_axes([0.05,0.05,0.15,0.07])
    ax4=fig.add_axes([0.70,0.1,0.25,0.8]);ax4.axis('off')
    ax5=fig.add_axes([0.3,0.05,0.15,0.07]);ax5.axis('off')
    wins=[];rois=[]
    ax1.imshow(bf_img,cmap=cmap,origin='upper');ax1.axis('off')
    ax2.imshow(data4d[int(rx/2),int(ry/2),:,:],cmap=cmap,origin='upper')
    ax2.set_title('Cepstral/Diffraction space');ax2.axis('off')
    ax1.set_title('Real space (Dark Field Image)')
    ax4.set_title('Select peak for analysis')
    ax5.text(0.1,0.5,"Number of rois saved:"+str(len(rois)),horizontalalignment='center',verticalalignment='center')
    def select_zoom(eclick,erelease):
        zoom_roi=np.array(add_selector.extents).astype('int')
        updated_r_img=np.mean(data4d[:,:,int(zoom_roi[2]):int(zoom_roi[3]),int(zoom_roi[0]):int(zoom_roi[1])],axis=(-2,-1))
        ax1.imshow(updated_r_img,cmap=cmap);ax1.axis('off')
        
    def onselect_function_real_space(eclick, erelease):
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))        
        ax2.imshow(np.log(updated_k_img),cmap=cmap);ax2.axis('off')
    
    def onselect_function_reciprocal_space(eclick, erelease):
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        updated_r_img=np.mean(data4d[:,:,int(reciprocal_roi[2]):int(reciprocal_roi[3]),int(reciprocal_roi[0]):int(reciprocal_roi[1])],axis=(-2,-1))
        ax1.imshow(updated_r_img,cmap=cmap);ax1.axis('off')
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))        
        ewpc_win=[int(0.5*(reciprocal_roi[0]+reciprocal_roi[1]))-half_width,int(0.5*(reciprocal_roi[0]+reciprocal_roi[1]))+half_width,int(0.5*(reciprocal_roi[2]+reciprocal_roi[3]))-half_width,int(0.5*(reciprocal_roi[2]+reciprocal_roi[3]))+half_width]
        ax4.imshow(np.log(updated_k_img)[int(ewpc_win[2]):int(ewpc_win[3]),int(ewpc_win[0]):int(ewpc_win[1])],extent=[ewpc_win[0],ewpc_win[1],ewpc_win[3],ewpc_win[2]],cmap=cmap)
        ax4.axis('off')
        
    def save_results(event):
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        zoom_roi=np.array(add_selector.extents).astype('int')
        real_roi = np.array(rect_selector.extents).astype('int')
        wins.append(zoom_roi)
        rois.append(real_roi);ax5.clear()
        ax5.text(0.1,0.5,"Number of rois saved:"+str(len(rois)),horizontalalignment='center',verticalalignment='center')
        ax5.axis('off')
    
    
    add_selector= RectangleSelector(ax4, select_zoom, drawtype='box', button=[1],
                                      useblit=True,minspanx=20, minspany=20,spancoords='pixels',interactive=True)    
    rect_selector = RectangleSelector(ax1, onselect_function_real_space, drawtype='box', button=[1],
                                      useblit=True ,minspanx=20, minspany=20,spancoords='pixels',interactive=True)
    reciprocal_rect_selector = RectangleSelector(ax2, onselect_function_reciprocal_space, drawtype='box', button=[1],
                                      useblit=True,minspanx=20, minspany=20,spancoords='pixels',interactive=True)    
    save_results_button=Button(ax3, 'Save Results')
    save_results_button.on_clicked(save_results)
    return (rect_selector,reciprocal_rect_selector,add_selector,save_results_button),wins,rois

def browser(data4d,cmap='gray'):
    rx,ry,kx,ky=np.shape(data4d)
    bf_img=data4d[:,:,int(kx/2),int(ky/2)]    
    fig=plt.figure(figsize=(8, 5))
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    
    ax1.imshow(bf_img,cmap=cmap,origin='upper');ax1.axis('off')
    ax2.imshow(data4d[int(rx/2),int(ry/2),:,:],cmap=cmap,origin='upper')
    ax2.set_title('Cepstral/Diffraction space');ax2.axis('off')
    ax1.set_title('Real space (Dark Field Image)')
    
    def onselect_function_real_space(eclick, erelease):
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))        
        ax2.imshow(np.log(updated_k_img),cmap=cmap)
    
    def onselect_function_reciprocal_space(eclick, erelease):
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        updated_r_img=np.mean(data4d[:,:,int(reciprocal_roi[2]):int(reciprocal_roi[3]),int(reciprocal_roi[0]):int(reciprocal_roi[1])],axis=(-2,-1))
        ax1.imshow(updated_r_img,cmap=cmap)


    rect_selector = RectangleSelector(ax1, onselect_function_real_space, drawtype='box', button=[1],
                                      useblit=True ,minspanx=1, minspany=1,spancoords='pixels',interactive=True)
    reciprocal_rect_selector = RectangleSelector(ax2, onselect_function_reciprocal_space, drawtype='box', button=[1],
                                      useblit=True,minspanx=1, minspany=1,spancoords='pixels',interactive=True)    
    
    
    
    return (rect_selector,reciprocal_rect_selector)


###PCA Decomposition

from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib.cm import ScalarMappable as scm
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle as Rect

def pca_decomposition(ewpc,n_components,circular_mask,include_center=False,normalization=True):
    if include_center:
        array=ewpc.reshape((ewpc.shape[0]*ewpc.shape[1],ewpc.shape[2]*ewpc.shape[3]))
    else:
        array=flatten_with_circular_mask(ewpc, circular_mask)
    if normalization:
        norm_means=np.mean(array, axis=1)
        norm_stds=np.std(array,axis=1)
        print('normalization of the ewpc pattern')
        for i in tqdm(range(array.shape[1])):
            array[:,i]-=norm_means
            array[:,i]/=norm_stds

    pca = PCA(n_components)
    scores = pca.fit_transform(array)
    return pca,scores

def generate_false_color_image(images, first_index = 1, last_index = None):
    imgs = np.copy(images[first_index:last_index])
    num_images = imgs.shape[0]
    hues = (np.arange(0, num_images)/(110**0.5)) % 1.0

    colors = np.array([hsv_to_rgb((hue, 1.0, 255)) for hue in hues])
    imgs = imgs.T
    imgs -= np.mean(imgs, axis = (0,1))
    imgs /= np.max(imgs, axis = (0,1))

    imgs = imgs.T
    
    colored_imgs = np.zeros(imgs.shape+(3,))
    for i in range(num_images):
        for j in range(3):
            colored_imgs[i,:,:,j] = imgs[i]*colors[i,j]
            
    false_color_img = np.mean(colored_imgs, axis = 0)
    false_color_img -= np.min(false_color_img)
    false_color_img /= np.max(false_color_img)
    
    #Color correcting
    false_color_hsv = rgb_to_hsv(false_color_img)
    
    v_values = false_color_hsv[:,:,2]
    vmean = np.mean(v_values)
    vstd = np.std(v_values)
    v_values = np.clip(v_values, vmean-2*vstd, vmean+2*vstd)
    v_values -= np.min(v_values)
    v_values /= np.max(v_values)
    v_floor = 0.1
    v_values *= 0.95-v_floor
    v_values += v_floor
    false_color_hsv[:,:,2] = v_values
    
    s_values = false_color_hsv[:,:,1]
    smean = np.mean(s_values)
    sstd = np.std(s_values)
    s_values = np.clip(s_values, smean - 3*sstd, smean + 3*sstd)
    s_values -= np.min(s_values)
    s_values /= np.max(s_values)
    s_floor = 0.50
    s_values *= 1-s_floor
    s_values += s_floor
    false_color_hsv[:,:,1] = s_values
    
    false_color_hsv[:,:,0] -= np.min(false_color_hsv[:,:,0])
    false_color_hsv[:,:,0] /= np.max(false_color_hsv[:,:,0])
    
    cc_false_color_img = hsv_to_rgb(false_color_hsv)
    
    return cc_false_color_img, false_color_img

def flatten_with_circular_mask(ewpc,circular_mask):
    flat_cep=np.zeros((ewpc.shape[0]*ewpc.shape[1],np.sum(circular_mask)))
    print('flattenning the cepstral signal')
    for i in tqdm(range(flat_cep.shape[0])):
        ii,jj=np.unravel_index(i,(ewpc.shape[0],ewpc.shape[1]))
        cur_slice=ewpc[ii,jj,:,:]
        flat_cep[i,:]=cur_slice[circular_mask.astype('bool')]
    return flat_cep

def plot_false_color_img(pca,scoresT,circular_mask,cmap='jet'):
    n_components=scoresT.shape[1]
    xy_shape=int(np.sqrt(scoresT.shape[0]))
    scores=np.reshape(scoresT.T,(n_components,xy_shape,xy_shape))
    output = widgets.Output()
    cutoff_slider = widgets.IntSlider(value=1, min=1, max=n_components, step=1, description='cut off')
    check_region = widgets.Checkbox(value=False,description='Log Y axis')
    
    fig=plt.figure(figsize=(8,8))
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)
    
    ax1.plot(np.arange(1,1+n_components),pca.explained_variance_ratio_)
    ax1.set_xlabel('# components')
    ax1.set_ylabel('Explained Variance Ratio')
    vline=ax1.axvline(color='k')
    cut_off=cutoff_slider.value
    fc = generate_false_color_image(scores[:cut_off,:,:], 0)[0]
    ax2.set_title('False colored image')
    ax2.imshow(fc)
    ax3.imshow(scores[cut_off,:,:],cmap=cmap)
    ax4.imshow(unflatten_circular_mask(pca.components_[cut_off,:],circular_mask),cmap=cmap)
    ax4.set_title('Component #'+str(cut_off))     
    ax3.set_title('Score #'+str(cut_off))
    for ax in [ax2,ax3,ax4]:
        ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout()
    def update_fc(cut_off):
        vline.set_xdata(cut_off)
        fc = generate_false_color_image(scores[:cut_off,:,:], 0)[0]
        ax2.imshow(fc)
        ax3.imshow(scores[cut_off-1,:,:],cmap=cmap)
        ax3.set_title('Score #'+str(cut_off))
        ax4.imshow(unflatten_circular_mask(pca.components_[cut_off-1,:],circular_mask),cmap=cmap)
        ax4.set_title('Component #'+str(cut_off))     

    def yaxis_scale(change):
        if change.new:
            ax1.set_yscale('log')
        else:
            ax1.set_yscale('linear')
    def slider_eventhandler(change):
        update_fc(change.new)
    cutoff_slider.observe(slider_eventhandler,names='value')
    check_region.observe(yaxis_scale,names='value')
    input_widgets_1 = widgets.HBox([check_region,cutoff_slider])
    display(input_widgets_1)

    
def unflatten_circular_mask(flat_array,haadf_mask):
    mask_pos=np.where(haadf_mask)
    cur_slice=np.zeros(haadf_mask.shape,flat_array.dtype)
    for i in range(np.sum(haadf_mask.astype('bool'))):
        cur_slice[mask_pos[0][i],mask_pos[1][i]]=flat_array[i]
    return cur_slice        

def plot_scores_components(pca,scores,n1,n2,circular_mask,cmap='jet',figsize=(9,9)):
    n_components=scores.shape[1]
    fig,axes=plt.subplots(n1,n2,figsize=figsize)
    k=0
    for i in range(n1):
        for j in range(n2):
            axes[i,j].imshow(scores[:,k].reshape((rx,ry)),cmap=cmap)
            axes[i,j].set_title('Score #'+str(k+1))

            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            k=k+1
            if k==n_components:
                break            
#     fig.savefig(saveDir+'/scores_all.png')
#     fig.savefig(saveDir+'/scores_all.pdf')
    fig,axes=plt.subplots(n1,n2,figsize=figsize)
    k=0
    for i in range(n1):
        for j in range(n2):
            axes[i,j].cla()
            axes[i,j].imshow(unflatten_circular_mask(pca.components_[k,:],circular_mask),cmap=cmap)
            axes[i,j].set_title('Component #'+str(k+1))     
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            k=k+1
            if k==n_components:
                break                        
#     fig.savefig(saveDir+'/components_all.png')
#     fig.savefig(saveDir+'/components_all.pdf')


def plot_kmeans_dict(kmeans_dict):

    clusters=list(kmeans_dict.keys())
    clusters.sort()
    output = widgets.Output()
    slider = widgets.IntSlider(value=clusters[0], min=clusters[0], max=clusters[-1], step=1, description='# Clusters')
    check_region = widgets.Checkbox(value=False,description='Log Y axis')
    
    fig=plt.figure(figsize=(9,4))
    ax1=fig.add_axes([0.1,0.1,0.3,0.675])
    ax2=fig.add_axes([0.5,0.1,0.3,0.675])
    ax_cm_t=fig.add_axes([0.9,0.3,0.05,0.4])
    
    wss=[]
    for i in clusters:
        wss.append(kmeans_dict[i]['wss'])
        
    ax1.plot(clusters,wss)
    ax1.set_xlabel('# Clusters')
    ax1.set_ylabel('Within cluster sum of squares')
    n_clusters=int(slider.value)
    
    vline=ax1.axvline(color='k')
    vline.set_xdata(n_clusters)
    ax1.set_xlim([clusters[0]-0.5,clusters[-1]+0.5])
    current_cmap = plt.get_cmap('RdBu', int(n_clusters)) 
    fc = kmeans_dict[n_clusters]['label']
    ax2.set_title('Cluster labels')
    ax2.imshow(fc,cmap=current_cmap)
    ax_cm_t.cla()
    plt.colorbar(scm(norm=mpl.colors.Normalize(vmin=-0.5,vmax=n_clusters-0.5),cmap=current_cmap),cax=ax_cm_t, ticks=np.arange(0,n_clusters))        

    for ax in [ax2]:
        ax.set_xticks([]);ax.set_yticks([])
    def update_fc(n_clusters):
        current_cmap = plt.get_cmap('RdBu', int(n_clusters)) 
        vline.set_xdata(n_clusters)
        fc = kmeans_dict[n_clusters]['label']
        ax2.imshow(fc,cmap=current_cmap)
        ax2.set_xticks([]);ax2.set_yticks([])
        ax_cm_t.cla()
        plt.colorbar(scm(norm=mpl.colors.Normalize(vmin=-0.5,vmax=n_clusters-0.5),cmap=current_cmap),cax=ax_cm_t, ticks=np.arange(0,n_clusters))        
        
    def yaxis_scale(change):
        if change.new:
            ax1.set_yscale('log')
        else:
            ax1.set_yscale('linear')
    def slider_eventhandler(change):
        update_fc(change.new)
    slider.observe(slider_eventhandler,names='value')
    check_region.observe(yaxis_scale,names='value')
    input_widgets_1 = widgets.HBox([check_region,slider])
    display(input_widgets_1)


def perform_kmeans(scores,cut_off,clusters_range,mask=None):
    rx=int(np.sqrt(scores.shape[0]))
    ry=rx
    kmeans_dict={}
    if mask==None:
        mask=np.ones((rx,ry)).astype('bool')
    mask_scores=np.zeros((cut_off,np.sum(mask)))
    mask_pos=np.where(mask)
    for i in range(cut_off):
        curr_slice=scores[:,i].copy().reshape((rx,ry))
        curr_slice-=curr_slice[mask].min()
        curr_slice/=curr_slice[mask].max()
        mask_scores[i,:]=curr_slice[mask]
    print('Performing clustering')
    for n_clusters in tqdm(range(clusters_range[0],clusters_range[1])):
        if n_clusters in kmeans_dict.keys():
            continue
        kmeans_dict[n_clusters]={}
        kmeans=KMeans(n_clusters=n_clusters).fit(mask_scores.T)
        labels=kmeans.labels_
        labels_unf=np.ones((rx,ry))*n_clusters
        for i in range(np.sum(mask)):
            labels_unf[mask_pos[0][i],mask_pos[1][i]]=labels[i]
        kmeans_dict[n_clusters]['wss']=kmeans.inertia_
        kmeans_dict[n_clusters]['label']=labels_unf
    
    return kmeans_dict


    
def pick_window_ewpc(data4d,adf_img,all_wins=[],all_roi=[],cmap='gray'):
    global roi,wins
    global add_window_button
    dx,dy = np.shape(adf_img)
    rx,ry,kx,ky=np.shape(data4d)
    if dx != rx or ry!=dy:
        print('Incompatible dimensions!')
        return
    roi=[0,dy,0,dx]
    cbed_pos=[int(dy/2),int(dx/2)]
    cbed=data4d[cbed_pos[1],cbed_pos[0],:,:]
    ewpc=ewpc2D(cbed)
    ewpc_win=[0,ky,0,kx]
    fine_ewpc_win=[0,ky,0,kx]
    wins=[]
    
    fig=plt.figure(figsize=(9, 6))
    ax1=fig.add_axes([0.1,0.6,0.2,0.3])
    ax2=fig.add_axes([0.35,0.6,0.2,0.3])
    ax4=fig.add_axes([0.1,0.2,0.2,0.3])
    ax5=fig.add_axes([0.35,0.2,0.2,0.3])
    ax6=fig.add_axes([0.6,0.2,0.2,0.3])
    ax3=fig.add_axes([0.6,0.6,0.2,0.3])

    add_window_ax = fig.add_axes([0.05, 0.05,0.3, 0.1])
    add_window_button = Button(add_window_ax,'Add window')
    
    
    df_ewpc_update_ax = fig.add_axes([0.50, 0.05,0.3, 0.1])
    df_ewpc_update_button = Button(df_ewpc_update_ax,'Update EWPC DF')    

    
    ax1.imshow(adf_img,cmap=cmap,origin='upper')
    ax2.imshow(adf_img,cmap=cmap,origin='upper')
    pix = ax2.add_patch(Rect((0,0), 1, 1))

    ax3.imshow(np.log(cbed),cmap=cmap)
    ax4.imshow(np.log(ewpc),cmap=cmap)
    ax1.set_title('Choose ROI')
    ax2.set_title('Pick probe position')    
    ax3.set_title('LOG(CBED)')
    ax4.set_title('LOG(EWPC)')
    ax5.set_title('EWPC WIN')
    ax6.set_title('DF EWPC ROI')
    for ax in [ax1,ax3]:
        ax.set_xticks([]);ax.set_yticks([]);

#     # widgets

    def click_cbed(event):
        global roi,cbed_pos,cbed,ewpc,ewpc_win,fine_ewpc_win
        if event.inaxes==add_window_ax:
            add_window()
            return
        elif event.inaxes==df_ewpc_update_ax:
            update_df_ewpc()
            return
        elif not event.inaxes == ax2:
            return
        [l.remove() for l in ax2.lines]
        cbed_pos = [int(event.xdata), int(event.ydata)]
        cbed=data4d[cbed_pos[1],cbed_pos[0],:,:]
        ewpc=ewpc2D(cbed)
        ax2.plot(cbed_pos[0],cbed_pos[1],'ro')
        ax3.imshow(np.log(cbed),cmap=cmap)
        ax3.set_xticks([]);ax3.set_yticks([]);
        ax4.imshow(np.log(ewpc),cmap=cmap)
        ax4.set_xticks([]);ax4.set_yticks([]);        
        ax5.imshow(ewpc[int(ewpc_win[2]):int(ewpc_win[3]),int(ewpc_win[0]):int(ewpc_win[1])],extent=[ewpc_win[0],ewpc_win[1],ewpc_win[3],ewpc_win[2]],cmap=cmap)
        colors=plt.cm.jet(np.linspace(0,1,len(wins)))
        for i in range(len(wins)):
            window=wins[i]
            xmin_1, xmax_1 = window[2],window[3]
            ymin_1, ymax_1 = window[0],window[1]
            box_1 = [[xmin_1-0.5,xmin_1-0.5,xmax_1+0.5,xmax_1+0.5,xmin_1-0.5],[ymin_1-0.5,ymax_1+0.5,ymax_1+0.5,ymin_1-0.5,ymin_1-0.5]]    
            ax4.plot(box_1[1],box_1[0],color=colors[i])
        
    def onselect_function(eclick, erelease):
        global roi,cbed,ewpc,ewpc_win,fine_ewpc_win
        roi = np.array(rect_selector.extents).astype('int')
        ax2.cla()
        ax2.imshow(adf_img[int(roi[2]):int(roi[3]),int(roi[0]):int(roi[1])],extent=[roi[0],roi[1],roi[3],roi[2]],cmap=cmap)
        ax2.set_title('Pick probe position')
    

        
    def win_select_function(eclick, erelease):
        global roi,cbed,ewpc,ewpc_win,fine_ewpc_win
        ewpc_win = np.array(ewpc_wind_selector.extents).astype('int')
        ax5.imshow(ewpc[int(ewpc_win[2]):int(ewpc_win[3]),int(ewpc_win[0]):int(ewpc_win[1])],extent=[ewpc_win[0],ewpc_win[1],ewpc_win[3],ewpc_win[2]],cmap=cmap)

    def fine_win_select_function(eclick, erelease):
        pass

        
        
    def update_df_ewpc():
        global roi,cbed,ewpc,ewpc_win,fine_ewpc_win        
        fine_ewpc_win = np.array(fine_ewpc_wind_selector.extents).astype('int')
        ewpc_roi=np.zeros((roi[3]-roi[2],roi[1]-roi[0],
                           fine_ewpc_win[3]-fine_ewpc_win[2],                           
                           fine_ewpc_win[1]-fine_ewpc_win[0]))
        for x in range(roi[2],roi[3]):
            for y in range(roi[0],roi[1]):
                ewpc_roi[x-roi[2],y-roi[0],:,:]=ewpc2D(data4d[x,y,:,:])[fine_ewpc_win[2]:fine_ewpc_win[3],fine_ewpc_win[0]:fine_ewpc_win[1]]
        ax6.imshow(np.sum(ewpc_roi,axis=(-2,-1)),cmap=cmap)
    
    def add_window():
        global wins,fine_ewpc_win
        wins.append(fine_ewpc_win)
        [l.remove() for l in ax4.lines]
        colors=plt.cm.jet(np.linspace(0,1,len(wins)))
        for i in range(len(wins)):
            window=wins[i]
            xmin_1, xmax_1 = window[2],window[3]
            ymin_1, ymax_1 = window[0],window[1]
            box_1 = [[xmin_1-0.5,xmin_1-0.5,xmax_1+0.5,xmax_1+0.5,xmin_1-0.5],[ymin_1-0.5,ymax_1+0.5,ymax_1+0.5,ymin_1-0.5,ymin_1-0.5]]    
            ax4.plot(box_1[1],box_1[0],color=colors[i])

    rect_selector = RectangleSelector(ax1, onselect_function, drawtype='box', button=[1],
                                      useblit=True,minspanx=5, minspany=5,spancoords='pixels',interactive=True)    
    ewpc_wind_selector = RectangleSelector(ax4, win_select_function, drawtype='box', button=[1],
                                           rectprops=dict(facecolor="red", edgecolor="red", alpha=1.0, fill=False),
                                           useblit=True,minspanx=1, minspany=1,spancoords='pixels',interactive=True)
    fine_ewpc_wind_selector = RectangleSelector(ax5, fine_win_select_function, drawtype='box', button=[1],
                                      useblit=True,minspanx=1, minspany=1,spancoords='pixels',interactive=True)

    cid = fig.canvas.mpl_connect('button_press_event', click_cbed)



def trim_spotMaps(spotMaps,strain_id): 
    new_spotMaps={}
    for key in spotMaps.keys():
        if not key in ['wins','roi']:
            new_spotMaps[key]=[]
            for i in strain_id:
                new_spotMaps[key].append(spotMaps[key][i-1])
            new_spotMaps[key]=np.array(new_spotMaps[key])
    return new_spotMaps


def calculate_DF(ewpc,wins):
    wins=np.array(wins)
    cep_mask=np.zeros((ewpc.shape[2],ewpc.shape[3])).astype('bool')
    for i in range(len(wins)):
        cep_mask[wins[i,2]:wins[i,3],wins[i,0]:wins[i,1]]=True
    img=np.sum(ewpc*cep_mask,axis=(-2,-1))
    return img



def segment_manually(img,thresh=None,figureSize=(8,4),bins=100):

    fig=plt.figure(figsize=figureSize)
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)
    
    if thresh==None:
        thresh=img.mean()

    bImg=img>thresh

    ax1.set_title('DF C-STEM')
    ax2.set_title('Segmentation results')
    ax1.imshow(img)
    ax2.imshow(bImg)
    a,b=np.histogram(img.flatten(),bins=bins)
    ax3.bar(b[:-1],a,b[1]-b[0],align='edge',edgecolor=None,facecolor='k')

    ylim=ax3.get_ylim()
    ax3.plot(np.ones(2)*thresh,ylim,'r--')
    ax3.set_ylim(ylim)
    for ax in [ax1,ax2]:
        ax.set_xticks([]);ax.set_yticks([])

    fig.tight_layout()
