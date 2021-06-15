import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from time import time
from source.ctemh import ctemh, wavelen
from source.stemh import stemcalc, stemhr, stemhrCc
from source.incostem import incostem, make_aberrated_probe

def ctem_gui():

    k = np.linspace(0, 0.7, 1000)

    # The parametrized function to be plotted
    def f(k, energy, C3, C5, defocus, semiangle, ddf):
        return ctemh(k, [energy, C3, C5, defocus, ddf, semiangle],0)

    # Define initial parameters
    init_energy = 200
    init_C3 = 1.3
    init_defocus = 700
    init_C5 = 50
    init_semiangle = 0.5
    init_ddf = 100

    # Create the figure and the line that we will manipulate
    plt.ion()
    fig, ax = plt.subplots(figsize=[10,7])
    ydata = ctemh(k, [init_energy, init_C3, 50, init_defocus, 100, 0.5],0)
    line, = plt.plot(k, ydata, lw=2, label = 'CTF')
    ax.set_xlabel('Spatial Frequency (1/$\AA$)',fontsize = 16)
    ax.set_ylabel('MTF', fontsize = 16)
    ax.set_ylim([-1,1])
    axcolor = 'lightgoldenrodyellow'
    ax.set_title('CTEM Contrast Transfer Function Plot')
    ax.margins(x = 0)
    ax.axhline(y = 0, color ="C1", linestyle ="--")
    vline_5pct = plt.axvline(x = k[np.where((ydata > 0.05) | (ydata < -0.05))[0][-1]], linewidth=2, 
                             color= 'C2', label = '5% Info Limit')
    vline_crossover = plt.axvline(x = k[np.where(ydata * ydata[1] < 0)[0][0]-1], linewidth = 2, 
                                  color = 'C3', label = 'First Crossover')
    plt.legend(fontsize = 12, loc='upper right')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.15, right = 0.92, bottom=0.45)


    # Make a vertically oriented slider to control the amplitude
    ax_C3 = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
    C3_slider = Slider(
        ax=ax_C3,
        label="C3 (mm)",
        valmin=0,
        valmax=100,
        valinit=init_C3,
    )

    ax_defocus = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    Defocus_slider = Slider(
        ax=ax_defocus,
        label="Defocus ($\AA$)",
        valmin=0,
        valmax=2000,
        valinit=init_defocus,
    )

    ax_C5 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    C5_slider = Slider(
        ax=ax_C5,
        label="C5 (mm)",
        valmin=0,
        valmax=100,
        valinit=init_C5,
    )

    ax_semiangle = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    semiangle_slider = Slider(
        ax=ax_semiangle,
        label="Semi-angle (mrad)",
        valmin=0,
        valmax=20,
        valinit=init_semiangle,
    )

    ax_ddf = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ddf_slider = Slider(
        ax=ax_ddf,
        label="Defocus spread ($\AA$)",
        valmin=0,
        valmax=500,
        valinit=init_ddf,
    )

    ax_energy = plt.axes([0.25, 0.025, 0.45, 0.03], facecolor=axcolor)
    energy_slider = Slider(
        ax=ax_energy,
        label='Beam Energy [keV]',
        valmin=0.1,
        valmax=500,
        valinit=init_energy,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        ydata = f(k, energy_slider.val, C3_slider.val, C5_slider.val, Defocus_slider.val, 
                         semiangle_slider.val, ddf_slider.val)
        line.set_ydata(ydata)
        vline_5pct.set_xdata(k[np.where((ydata > 0.05) | (ydata < -0.05))[0][-1]])
        vline_crossover.set_xdata(k[np.where(ydata * ydata[1] < 0)[0][0]-1])
        fig.canvas.draw_idle()


    # register the update function with each slider
    C3_slider.on_changed(update)
    Defocus_slider.on_changed(update)
    energy_slider.on_changed(update)
    C5_slider.on_changed(update)
    semiangle_slider.on_changed(update)
    ddf_slider.on_changed(update)

    # Create a button to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        C3_slider.reset()
        Defocus_slider.reset()
        energy_slider.reset()
        C5_slider.reset()
        semiangle_slider.reset()
        ddf_slider.reset()
    button.on_clicked(reset)

    plt.show()
    return button
    
def stem_gui():
    # use smaller k array for STEM as it takes longer to calculate STEM 
    k = np.linspace(0, 2.2, 100)
    r = np.linspace(0, 5.0, 100)

    # The parametrized function to be plotted
    def f(k, energy, C3, C5, defocus, semiangle, ddf):
        return stemcalc(k, [energy, C3, C5, defocus, semiangle, 0])


    # Define initial parameters
    init_energy = 300
    init_C3 = 2e-6
    init_defocus = 0
    init_C5 = 1.3e-6
    init_semiangle = 15

    # Create the figure and the line that we will manipulate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10,7])
    ctf, psf = f(k, init_energy, init_C3, init_C5, init_defocus, init_semiangle, 0)

    line1, = ax1.plot(k, ctf, lw=2)
    ax1.set_xlabel('Spatial Frequency (1/$\AA$)',fontsize = 16)
    ax1.set_ylabel('CTF/PSF', fontsize = 16)
    ax1.set_ylim([-1,1])
    ax1.set_title('STEM Contrast Transfer Function Plot')
    axcolor = 'lightgoldenrodyellow'
    ax1.margins(x = 0)
    ax1.axhline(y = 0, color ="C1", linestyle ="--")
    vline_5pct = ax1.axvline(x = k[np.where((ctf > 0.05) | (ctf < -0.05))[0][-1]], linewidth=2, 
                                 color= 'C2', label = '5% Info Limit')
    ax1.legend(fontsize = 12, loc = 'upper right')

    line2, = ax2.plot(r, psf, lw=2,color='#1f77b4')
    line22, = ax2.plot(-r, psf, lw=2,color='#1f77b4')
    ax2.set_xlabel('Radius ($\AA$)',fontsize = 16)
    # ax2.set_ylabel('PSF', fontsize = 16)
    ax2.set_ylim([-1,1])
    axcolor = 'lightgoldenrodyellow'
    ax2.margins(x = 0)
    ax2.axhline(y = 0, color ="C1", linestyle ="--")
    ax2.set_title('STEM Point Spread Function Plot')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.15, right = 0.92, bottom=0.45)

    # Make a vertically oriented slider to control the amplitude
    ax_C3 = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
    C3_slider = Slider(
        ax=ax_C3,
        label="C3 (mm)",
        valmin=0,
        valmax=5,
        valinit=init_C3,
    )

    ax_defocus = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    Defocus_slider = Slider(
        ax=ax_defocus,
        label="Defocus ($\AA$)",
        valmin=-500,
        valmax=500,
        valinit=init_defocus,
    )

    ax_C5 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    C5_slider = Slider(
        ax=ax_C5,
        label="C5 (mm)",
        valmin=0,
        valmax=100,
        valinit=init_C5,
    )

    ax_semiangle = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    semiangle_slider = Slider(
        ax=ax_semiangle,
        label="Semi-angle (mrad)",
        valmin=0,
        valmax=28,
        valinit=init_semiangle,
    )


    ax_energy = plt.axes([0.25, 0.10, 0.45, 0.03], facecolor=axcolor)
    energy_slider = Slider(
        ax=ax_energy,
        label='Beam Energy [keV]',
        valmin=0.1,
        valmax=500,
        valinit=init_energy,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        ctf, psf = f(k, energy_slider.val, C3_slider.val, C5_slider.val, Defocus_slider.val, 
                         semiangle_slider.val, 0)
        wav = wavelen(energy_slider.val)
        rmax = 2.0*np.sqrt( np.sqrt( C3_slider.val*wav*wav*wav) )
        line1.set_ydata(ctf)
        line2.set_ydata(psf)
        line22.set_ydata(psf)
        vline_5pct.set_xdata(k[np.where((ctf > 0.05) | (ctf < -0.05))[0][-1]])
        fig.canvas.draw_idle()


    # register the update function with each slider
    C3_slider.on_changed(update)
    Defocus_slider.on_changed(update)
    energy_slider.on_changed(update)
    C5_slider.on_changed(update)
    semiangle_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.10, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        C3_slider.reset()
        Defocus_slider.reset()
        energy_slider.reset()
        C5_slider.reset()
        semiangle_slider.reset()
    button.on_clicked(reset)

    plt.show()
    return button
    
    
    
def perov_gui():
    #building crystal: square lattice with 3.9 angstrom bond length (corresp. to A-A distance in ~STO)
    a = 3.9 #Ang
    padding =1 #Ang
    a1 = np.array([a,0])
    a2 = np.array([0,a])
    n = 3
    xs = np.zeros((n,n))
    ys = np.zeros((n,n))
    for it1 in range(n):
        for it2 in range(n):
            xs[it1,it2] = it1*a1[0]+it2*a2[0]
            ys[it1,it2] = it1*a1[1]+it2*a2[1]
    xs = xs.ravel()
    ys = ys.ravel()
    xs = xs-xs.min()+padding
    ys = ys-ys.min()+padding
    Zs = np.ones(xs.shape)
    sample_dim = max((xs.max(),ys.max()))+padding

    # sim params
    imdim = 2*100
    imrad = int(np.floor(imdim/2))
    ang_per_px = .05

    # The parametrized function to be plotted
    def f(energy, C3, C12, defocus, semiangle, ddf):
        ab_fn = [{'n':1,'m':0,'angle':0,'mag':defocus*1e-10},{'n':3,'m':0,'angle':0,'mag':C3*1e-3},{'n':1,'m':2,'angle':0,'mag':C12*1e-10}]
        probe = make_aberrated_probe(imdim, ang_per_px, semiangle*1e-3, energy, ab_fn)
        probe_prof = probe[imrad,imrad:]
        probe_prof = probe_prof/probe_prof.max()
        im = incostem(xs/ang_per_px,ys/ang_per_px,Zs,imdim=(imdim,imdim),probe=probe)
        return (probe_prof,probe_prof,im)
        #return stemcalc(k, [energy, C3, C5, defocus, semiangle, ddf])


    # Define initial parameters
    init_energy = 60
    init_C3 = 0.0
    init_defocus = 42.3
    init_C12 = 32.6
    init_semiangle = 21.98
    init_ddf = 0

    # Calculate the initial r coordinates using the default setup
    # wav = wavelen(init_energy)
    # rmax = 2.0*np.sqrt( np.sqrt( init_C3*wav*wav*wav) )
    # r = np.linspace(0, rmax, 500)

    # Create the figure and the line that we will manipulate
    fig, ax1 = plt.subplots(1, 1, figsize=[10,7])
    iim = ax1.matshow(f(init_energy, init_C3, init_C12, init_defocus, init_semiangle, init_ddf)[2],cmap='gray')
    ax1.axis('off')
    axcolor = 'lightgoldenrodyellow'
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.15, right = 0.92, bottom=0.45)

    # Make a vertically oriented slider to control the amplitude
    '''
    ax_C3 = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
    C3_slider = Slider(
        ax=ax_C3,
        label="C3 (mm)",
        valmin=0,
        valmax=2,
        valinit=init_C3,
        #valstep = 20
    )
    '''

    ax_defocus = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    Defocus_slider = Slider(
        ax=ax_defocus,
        label="Defocus ($\AA$)",
        valmin=-50,
        valmax=50,
        valinit=init_defocus,
    )

    ax_C12 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    C12_slider = Slider(
        ax=ax_C12,
        label="C12 ($\AA$)",
        valmin=0,
        valmax=50,
        valinit=init_C12,
    )

    ax_semiangle = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    semiangle_slider = Slider(
        ax=ax_semiangle,
        label="Semi-angle (mrad)",
        valmin=0,
        valmax=40,
        valinit=init_semiangle,
    )
    '''
    ax_ddf = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ddf_slider = Slider(
        ax=ax_ddf,
        label="Defocus spread ($\AA$)",
        valmin=0,
        valmax=500,
        valinit=init_ddf,
    )
    '''
    ax_energy = plt.axes([0.25, 0.10, 0.45, 0.03], facecolor=axcolor)
    energy_slider = Slider(
        ax=ax_energy,
        label='Beam Energy [keV]',
        valmin=0.1,
        valmax=500,
        valinit=init_energy,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        ctf, psf,im = f(energy_slider.val, 0, C12_slider.val, Defocus_slider.val, 
                         semiangle_slider.val, 0)
        wav = wavelen(energy_slider.val)
        #rmax = 2.0*np.sqrt( np.sqrt( C3_slider.val*wav*wav*wav) )
        #line1.set_ydata(ctf)
        #     line2.set_xdata(linspace(0, rmax, 500))
        #line2.set_ydata(psf)
        iim.set_data(im)
        #iim.set_clim(0, 0.5)
        iim.autoscale()
        fig.canvas.draw_idle()

    
    # register the update function with each slider
    #C3_slider.on_changed(update)
    Defocus_slider.on_changed(update)
    energy_slider.on_changed(update)
    C12_slider.on_changed(update)
    semiangle_slider.on_changed(update)
    #ddf_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.10, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        #C3_slider.reset()
        Defocus_slider.reset()
        energy_slider.reset()
        C12_slider.reset()
        semiangle_slider.reset()
        #ddf_slider.reset()
    button.on_clicked(reset)

    plt.show()
    return button
    
def c23_gui():
    a = 3.9 #Ang
    padding =1 #Ang
    a1 = np.array([a,0])
    a2 = np.array([a*0.5,a*0.87])
    n = 3
    xs = np.zeros((n,n))
    ys = np.zeros((n,n))
    for it1 in range(n):
        for it2 in range(n):
            xs[it1,it2] = it1*a1[0]+it2*a2[0]
            ys[it1,it2] = it1*a1[1]+it2*a2[1]
    xs = xs.ravel()
    ys = ys.ravel()
    xs = xs-xs.min()+padding
    ys = ys-ys.min()+padding
    Zs = np.ones(xs.shape)
    sample_dim = max((xs.max(),ys.max()))+padding

    # sim params
    imdim = 255
    imrad = int(np.floor(imdim/2))
    ang_per_px = .05

    # The parametrized function to be plotted
    def f(energy, C23_angle, C23, defocus, semiangle, ddf):
        ab_fn = [{'n':1,'m':0,'angle':0,'mag':defocus*1e-10},{'n':2,'m':3,'angle':C23_angle,'mag':C23*1e-9}]
        probe = make_aberrated_probe(imdim, ang_per_px, semiangle*1e-3, energy, ab_fn)
        probe_prof = probe[imrad,imrad:]
        probe_prof = probe_prof/probe_prof.max()
        im = incostem(xs/ang_per_px,ys/ang_per_px,Zs,imdim=(imdim,imdim),probe=probe)
        return (probe_prof,probe_prof,im[:175,:])


    # Define initial parameters
    init_energy = 120
    init_C23_angle = 11
    init_defocus = 75
    init_C23 = 190
    init_semiangle = 30
    init_ddf = 0

    # Create the figure and the line that we will manipulate
    fig, ax1 = plt.subplots(1, 1, figsize=[10,7])
    iim = ax1.matshow(f(init_energy, init_C23_angle, init_C23, init_defocus, init_semiangle, init_ddf)[2],
                      cmap='gray')
    ax1.axis('off')
    axcolor = 'lightgoldenrodyellow'

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.15, right = 0.92, bottom=0.45)

    # Make a vertically oriented slider to control the amplitude

    ax_defocus = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
    Defocus_slider = Slider(
        ax=ax_defocus,
        label="Defocus ($\AA$)",
        valmin=-150,
        valmax=150,
        valinit=init_defocus,
    )

    ax_C23_angle = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    C23_angle_slider = Slider(
        ax=ax_C23_angle,
        label="C23 Angle (degrees)",
        valmin=0,
        valmax=30,
        valinit=init_C23_angle,
    )

    ax_C23 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    C23_slider = Slider(
        ax=ax_C23,
        label="C23 magnitude (nm)",
        valmin=0,
        valmax=250,
        valinit=init_C23,
    )

    ax_semiangle = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    semiangle_slider = Slider(
        ax=ax_semiangle,
        label="Semi-angle (mrad)",
        valmin=0,
        valmax=40,
        valinit=init_semiangle,
    )


    ax_energy = plt.axes([0.25, 0.10, 0.45, 0.03], facecolor=axcolor)
    energy_slider = Slider(
        ax=ax_energy,
        label='Beam Energy [keV]',
        valmin=0.1,
        valmax=500,
        valinit=init_energy,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        ctf, psf,im = f(energy_slider.val, C23_angle_slider.val, C23_slider.val, Defocus_slider.val, 
                         semiangle_slider.val, 0)
        wav = wavelen(energy_slider.val)
        iim.set_data(im)
        iim.autoscale()
        fig.canvas.draw_idle()


    # register the update function with each slider
    C23_angle_slider.on_changed(update)
    Defocus_slider.on_changed(update)
    energy_slider.on_changed(update)
    C23_slider.on_changed(update)
    semiangle_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.10, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        C23_angle_slider.reset()
        Defocus_slider.reset()
        energy_slider.reset()
        C23_slider.reset()
        semiangle_slider.reset()

    button.on_clicked(reset)

    plt.show()
    return button
    
def cc_gui():
    # use smaller k array for STEM as it takes longer to calculate STEM 
    k = np.linspace(0, 2.2, 500)
    r = np.linspace(0, 5.0, 100)

    # The parametrized function to be plotted
    def f(k, energy, C3, C5, defocus, semiangle, ddf):
        return stemcalc(k, [energy, C3, C5, defocus, semiangle, ddf])


    # Define initial parameters
    init_energy = 60
    init_C3 = 0#1.3
    init_defocus = 0#500
    init_C5 = 0#50
    init_semiangle = 22
    init_cc = 1.1
    init_de = 0.3

    # Create the figure and the line that we will manipulate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10,7])
    ctf, psf = f(k, init_energy, init_C3, init_C5, init_defocus, init_semiangle, init_cc * init_de / init_energy * 1e4)

    line1, = ax1.plot(k, ctf, lw=2)
    ax1.set_xlabel('Spatial Frequency (1/$\AA$)',fontsize = 16)
    ax1.set_ylabel('CTF/PSF', fontsize = 16)
    ax1.set_ylim([-1,1])
    ax1.set_title('STEM Contrast Transfer Function Plot')
    axcolor = 'lightgoldenrodyellow'
    ax1.margins(x = 0)
    ax1.axhline(y = 0, color ="C1", linestyle ="--")

    line2, = ax2.plot(r, psf, lw=2,color='#1f77b4')
    line22, = ax2.plot(-r, psf, lw=2,color='#1f77b4')

    ax2.set_xlabel('Radius ($\AA$)',fontsize = 16)
    # ax2.set_ylabel('PSF', fontsize = 16)
    ax2.set_ylim([-1,1])
    axcolor = 'lightgoldenrodyellow'
    ax2.margins(x = 0)
    ax2.axhline(y = 0, color ="C1", linestyle ="--")
    ax2.set_title('STEM Point Spread Function Plot')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.15, right = 0.92, bottom=0.45)

    # Make a vertically oriented slider to control the amplitude
    ax_C3 = plt.axes([0.25, 0.32, 0.65, 0.03], facecolor=axcolor)
    C3_slider = Slider(
        ax=ax_C3,
        label="C3 (mm)",
        valmin=0,
        valmax=5,
        valinit=init_C3,
    )

    ax_defocus = plt.axes([0.25, 0.27, 0.65, 0.03], facecolor=axcolor)
    Defocus_slider = Slider(
        ax=ax_defocus,
        label="Defocus ($\AA$)",
        valmin=-1000,
        valmax=1000,
        valinit=init_defocus,
    )

    ax_C5 = plt.axes([0.25, 0.22, 0.65, 0.03], facecolor=axcolor)
    C5_slider = Slider(
        ax=ax_C5,
        label="C5 (mm)",
        valmin=0,
        valmax=100,
        valinit=init_C5,
    )

    ax_semiangle = plt.axes([0.25, 0.17, 0.65, 0.03], facecolor=axcolor)
    semiangle_slider = Slider(
        ax=ax_semiangle,
        label="Semi-angle (mrad)",
        valmin=0,
        valmax=28,
        valinit=init_semiangle,
    )

    ax_cc = plt.axes([0.25, 0.12, 0.65, 0.03], facecolor=axcolor)
    cc_slider = Slider(
        ax=ax_cc,
        label="Cc (mm)",
        valmin=0,
        valmax=5,
        valinit=init_cc,
    )

    ax_de = plt.axes([0.25, 0.07, 0.65, 0.03], facecolor=axcolor)
    de_slider = Slider(
        ax=ax_de,
        label="$\Delta E$ (eV)",
        valmin=0,
        valmax=1,
        valinit=init_de,
    )

    ax_energy = plt.axes([0.25, 0.025, 0.45, 0.03], facecolor=axcolor)
    energy_slider = Slider(
        ax=ax_energy,
        label='Beam Energy [keV]',
        valmin=0.1,
        valmax=500,
        valinit=init_energy,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        ddf = cc_slider.val * de_slider.val / energy_slider.val * 1e4
        ctf, psf = f(k, energy_slider.val, C3_slider.val, C5_slider.val, Defocus_slider.val, 
                         semiangle_slider.val, ddf)
        wav = wavelen(energy_slider.val)
        rmax = 2.0*np.sqrt( np.sqrt( C3_slider.val*wav*wav*wav) )
        line1.set_ydata(ctf)
        line2.set_ydata(psf)
        line22.set_ydata(psf)
        fig.canvas.draw_idle()
        print(ddf)


    # register the update function with each slider
    C3_slider.on_changed(update)
    Defocus_slider.on_changed(update)
    energy_slider.on_changed(update)
    C5_slider.on_changed(update)
    semiangle_slider.on_changed(update)
    cc_slider.on_changed(update)
    de_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.005, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        C3_slider.reset()
        Defocus_slider.reset()
        energy_slider.reset()
        C5_slider.reset()
        semiangle_slider.reset()
        cc_slider.reset()
        de_slider.reset()
        
    button.on_clicked(reset)

    plt.show()
    return button

