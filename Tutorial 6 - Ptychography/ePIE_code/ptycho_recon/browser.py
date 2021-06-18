import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector,Button

def browser(data4d,cmap='gray'):
    rx,ry,kx,ky=np.shape(data4d)

    dim =data4d.shape #N_x1,N_x2,N_k1,N_k2
    i=int(dim[0]/2); j=int(dim[0]/2);


    haadf_mask=create_haadf_mask((dim[2],dim[3]),[40,62])
    haadf=np.mean(data4d*haadf_mask,axis=(-2,-1))

    #bf_img=data4d[:,:,int(kx/2),int(ky/2)]  

    fig=plt.figure(figsize=(9, 5))
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)

    ax1.imshow(haadf,cmap=cmap,origin='upper');ax1.axis('off')
    ax2.imshow(data4d[int(rx/2),int(ry/2),:,:],cmap=cmap,origin='upper')
    ax2.set_title('Diffraction space (linear)');ax2.axis('off')
    ax1.set_title('Real space (ADF Image)')

    ax3.imshow(data4d[int(rx/2),int(ry/2),:,:],cmap=cmap,origin='upper')
    ax3.set_title('Diffraction space (log)');ax3.axis('off')

    
    def onselect_function_real_space(eclick, erelease):
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))        
        ax3.imshow(np.log(updated_k_img),cmap=cmap)
        ax2.imshow(updated_k_img,cmap=cmap)
    
    def onselect_function_reciprocal_space(eclick, erelease):
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        updated_r_img=np.mean(data4d[:,:,int(reciprocal_roi[2]):int(reciprocal_roi[3]),int(reciprocal_roi[0]):int(reciprocal_roi[1])],axis=(-2,-1))
        ax1.imshow(updated_r_img,cmap=cmap)


    rect_selector = RectangleSelector(ax1, onselect_function_real_space, drawtype='box', button=[1],
    useblit=True ,minspanx=1, minspany=1,spancoords='pixels',interactive=True)
    #reciprocal_rect_selector = RectangleSelector(ax2, onselect_function_reciprocal_space, drawtype='box', button=[1],
                                      #useblit=True,minspanx=1, minspany=1,spancoords='pixels',interactive=True)    
    
    
    
    return rect_selector


def create_haadf_mask(array_shape,radii):
    [r0,r1]=radii
    center=[array_shape[-2]/2,array_shape[-1]/2]
    kx = np.arange(array_shape[-1])-int(center[-1])
    ky = np.arange(array_shape[-2])-int(center[-2])
    kx,ky = np.meshgrid(kx,ky)
    kdist = (kx**2.0 + ky**2.0)**(1.0/2)
    haadf_mask = np.array(kdist <= r1, np.int)*np.array(kdist >= r0, np.int)
    return haadf_mask  