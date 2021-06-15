import numpy as np

def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1,a=1):
    return a * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def incostem(xs,ys,ints,imdim=None,probe=None,sigma=2):
    '''
    Generates image from gaussians convolved with point potentials.
    ~ from incostem.cpp by Kirkland
    coordinate convention: (rows, columns) as specified by imdim, xs are horizontal (columns), ys are vertical (rows) 

    xs: xcoordinates of atom positions, length n
    ys: ycoordinates of atom positions , length n
    ints: intensity for atom, length n
    imdim : tuple with output image shape, should have length 2, default is 2+floored max coordinate on each axis
    probe: 2d image with same shape as imdim convolved with scaled interpolated atom positions, default is a symmetric gaussian
    sigma: gaussian standard deviation used only if default probe is used
    '''

    if(imdim is None):
        imdim = (int(np.max(ys))+2, int(np.max(xs))+2)

    if(probe is None):
        x = np.arange(imdim[1])
        y = np.arange(imdim[0])
        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
        probe = gaus2d(x,y,imdim[1]/2,imdim[0]/2,sigma,sigma,1)

    
    
    imgen = np.zeros(imdim)
    
    #for each atom
    for it in range(len(xs)):

        xa = xs[it]
        ya = ys[it]
        
        # get (under) x,y
        ix = int(xa)
        iy = int(ya)
        
        # weights along x
        bxx = (ix+1) - xa
        cxx = (xa+1.0) - (ix+1)
        #weights along y
        byy = (iy+1)-ya
        cyy = (ya+1.0)-(iy+1)
        # weights
        w11 = bxx*byy #byy?
        w21 = cxx*byy #byy?
        w12 = bxx*cyy
        w22 = cxx*cyy
        
        wall = w11+w21+w12+w22
        
        if(abs(wall-1.0)>.01):
            w11 /=wall
            w21 /=wall
            w12 /=wall
            w22 /=wall
        

        ix2 = ix + 1
        iy2 = iy+1
        
        imgen[iy,ix] += w11*ints[it]
        imgen[iy2,ix] += w21*ints[it]
        imgen[iy,ix2] += w12*ints[it]
        imgen[iy2,ix2] +=w22*ints[it]
    imgen = np.flip(np.real(np.fft.fft2(np.fft.fft2(imgen)*np.fft.fft2(np.fft.fftshift(probe)))))
    
    return imgen


def make_aberrated_probe(imdim, ang_per_px, obj_ap_radius, keV, ab_fn):
    # geometry
    wavelen = 12.3986/np.sqrt((2*511+keV)*keV) * 10**(-10) # electron wavelength in meters
    simdim = wavelen/(ang_per_px*10**-10)#rad -- full width : alpha = wavelen*k

    XX, YY = np.meshgrid(np.linspace(-simdim/2,simdim/2,imdim),np.linspace(-simdim/2,simdim/2,imdim))

    RR = np.sqrt(XX**2+YY**2)
    TT = np.arctan2(YY,XX) #-pi .... pi

    obj_ap = RR < obj_ap_radius

    # aberration function

    chi = np.zeros((imdim,imdim))

    for it,at in enumerate(ab_fn):
        chi = chi + at['mag'] * np.cos(at['m']*(TT-at['angle'])) * (RR**(at['n']+1))/(at['n']+1)



    chi0 = 2*np.pi/wavelen * chi

    # probe
    expchi0 = np.exp(-1j * chi0) * obj_ap

    psi_p = np.fft.fft2(expchi0)

    probe = np.abs(np.fft.fftshift(psi_p))**2
    return probe