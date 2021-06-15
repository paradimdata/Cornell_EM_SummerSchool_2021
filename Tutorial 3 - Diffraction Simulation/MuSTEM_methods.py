#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import subprocess
import shlex
import glob

#Make a driver file for cbed simulations 
def make_driver_cbed(driver_name, params, device_type):
    driver_params = ["Output filename \n    " + params[0], #0
        "Input crystal file name \n    " + params[1], #1
        "Probe accelerating voltage (kV) \n    " + params[2], #2
        "Slice unit cell <1> yes <2> no \n    " + "1", #3
        "Number of distinct potentials \n    " + params[3], #4
        "<1> manual or <2> auto slicing \n    " + "2", #5
        "Thickness \n    " + params[4], #6
        "Scattering factor accuracy \n    " + "1", #7
        "Tile supercell x \n    " + params[5][0], #8
        "Tile supercell y \n    " + params[5][1], #9
        "Number of pixels in x \n    " + params[6][0], #10
        "Number of pixels in y \n    " + params[6][1], #11
        "<1> Continue <2> Change \n    " + "1", #revisit #12
        "Calculation type \n    " + "2", #13 <-- Hybrid (not as accurate)
        "aperture cutoff \n    " + params[7], #14
        "Defocus \n    0", #15
        "Change probe forming lens parameters \n    " + "0", #16
        "<1> QEP <2> ABS \n    2", #17
        "<1> Absorption <2> No absorption \n    1", #18
        "Elastic and inelastic scattering output choice \n    1", #19
        "<0> Continue <1> Beam tilt <2> Specimen tilt \n    0", #20
        "<0> continue <1> save <2> load \n    " + "0", #21
        "<1> CBED <2> STEM/PACBED \n    " + "1", #22
        "Initial probe position \n    " + "0.5 0.5"] #23

    #Tilt: 
    new_list = ["<0> Continue <1> Beam tilt <2> Specimen tilt"]
    if params[8][0] == ["1","0","0"] and params[8][1] == ["2","0","0"]:
        new_list.append("    0")
    else:
        for n1 in range(len(params[8])):
            new_list.append("    " + params[8][n1][0])
            if params[8][n1][0] == "1":
                new_list.append("Beam tilt in mrad")
                new_list.append("    " + params[8][n1][1])
                new_list.append("Beam tilt azimuth in mrad")
                new_list.append("    " + params[8][n1][2])
            if params[8][n1][0] == "2":
                new_list.append("Specimen tilt in mrad")
                new_list.append("    " + params[8][n1][1])
                new_list.append("Specimen tilt azimuth in mrad")
                new_list.append("    " + params[8][n1][2])
            new_list.append("<0> Continue <1> Beam tilt <2> Specimen tilt")
        new_list.append("    0")
    driver_params[20] = new_list

    #Write driver file
    outf = open(driver_name,'w')
    if device_type == 1:
        outf.write("Device used for calculation \n")
        outf.write("0 \n")
        
    for n1 in range(len(driver_params)):
        if type(driver_params[n1]) is list:  #check for nested lists to iterate over
            for n2 in range(len(driver_params[n1])):
                outf.write(driver_params[n1][n2]+"\n")
        else:
            outf.write(driver_params[n1] + "\n")
            
    if device_type ==1:
        outf.write("<0> Precalculated potentials <1> On-the-fly calculation \n")
        outf.write("    0")
    outf.close()
    
    #Write user_input.txt
    outf = open("user_input.txt",'w')
    outf.write("play \n")  
    outf.write(driver_name)
    outf.close()

#Make STEM driver file    
def make_driver_stem(driver_name, params, device_type):
    driver_params = ["Output filename \n    " + params[0], #0
        "Input crystal file name \n    " + params[1], #1
        "Probe accelerating voltage (kV) \n    " + params[2], #2
        "Slice unit cell <1> yes <2> no \n    " + "1", #3
        "Number of distinct potentials \n    " + params[3], #4
        "<1> manual or <2> auto slicing \n    " + "2", #5
        "Thickness \n    " + params[4], #6
        "Scattering factor accuracy \n    " + "1", #7
        "Tile supercell x \n    " + params[5][0], #8
        "Tile supercell y \n    " + params[5][1], #9
        "Number of pixels in x \n    " + params[6][0], #10
        "Number of pixels in y \n    " + params[6][1], #11
        "<1> Continue <2> Change \n    " + "1", #revisit #12
        "Calculation type \n    " + "2", #13 <-- Hybrid (not as accurate)
        "aperture cutoff \n    " + params[7], #14
        "Defocus \n    " + params[8], #15
        "Change probe forming lens parameters \n    " + "0", #16
        "<1> QEP <2> ABS \n    2", #17
        "<1> Absorption <2> No absorption \n    1", #18 *
        "Elastic and inelastic scattering output choice \n    1", #19
        "<0> Continue <1> Beam tilt <2> Specimen tilt \n    0", #20
        "<0> continue <1> save <2> load \n    " + "0", #21
        "<1> CBED <2> STEM/PACBED \n    " + "2", #22
        "STEM modes \n 1", #23
        "STEM modes \n 0", #24
        "Probe scan menu choice \n    1", #25
        "output interpolation max pixels \n    " + params[10], #26
        "output interpolation tilex \n    " + params[11][0], #27
        "output interpolation tiley \n    " + params[11][1], #28
        "<0> Proceed <1> Output probe intensity \n    0", #29
        "Number of detectors \n    " + str(len(params[12])), #30
        "manual detector <1> auto <2> \n    1", #31
        "<1> mrad <2> inv A \n    1"] #32 * 

    #Tilt: 
    new_list = ["<0> Continue <1> Beam tilt <2> Specimen tilt"]
    if params[9][0] == ["1","0","0"] and params[9][1] == ["2","0","0"]:
        new_list.append("    0")
    else:
        for n1 in range(len(params[9])):
            new_list.append("    " + params[9][n1][0])
            if params[9][n1][0] == "1":
                new_list.append("Beam tilt in mrad")
                new_list.append("    " + params[9][n1][1])
                new_list.append("Beam tilt azimuth in mrad")
                new_list.append("    " + params[9][n1][2])
            if params[9][n1][0] == "2":
                new_list.append("Specimen tilt in mrad")
                new_list.append("    " + params[9][n1][1])
                new_list.append("Specimen tilt azimuth in mrad")
                new_list.append("    " + params[9][n1][2])
            new_list.append("<0> Continue <1> Beam tilt <2> Specimen tilt")
        new_list.append("    0")
    driver_params[20] = new_list
        

    #Annular detector radii:
    new_list = []
    new_list.append(driver_params[32])
    for n1 in range(len(params[12])):
        new_list.append("inner \n    " + params[12][n1][0] + \
                             "\nouter \n    " + params[12][n1][1])
    driver_params[32] = new_list

    #Write driver file
    outf = open(driver_name,'w')
    if device_type == 1:
        outf.write("Device used for calculation \n")
        outf.write("0 \n")
        
    for n1 in range(len(driver_params)):
        if type(driver_params[n1]) is list:  #check for nested lists to iterate over
            for n2 in range(len(driver_params[n1])):
                outf.write(driver_params[n1][n2]+"\n")
        else:
            outf.write(driver_params[n1] + "\n")
    
    if device_type == 1:
        outf.write("<0> Precalculated potentials <1> On-the-fly calculation \n")
        outf.write("    0")
    
    outf.close()
    
    #Write user_input.txt
    outf = open("user_input.txt",'w')
    outf.write("play \n")  
    outf.write(driver_name)
    outf.close()

#Run MuSTEM calculation from python, print Terminal output
def run_mustem(command, verbose):  
    #adapted from https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
    #text encoding idea from: https://stackoverflow.com/questions/42019117/unicodedecodeerror-\
    #charmap-codec-cant-decode-byte-0x8f-in-position-xxx-char
    #Note: check if text encoding is different on windows
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, encoding='cp850')
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            if output.strip().startswith("Calculating absorptive scattering"):
                print(output.strip(), end = "\r")
            elif output.strip().startswith("Cell"):
                print(output.strip(), end = "\r")
            elif output.strip().startswith("QEP pass:"):
                print(output.strip(), end = "\r")
            elif output.strip().startswith("Calculating transmission functions"):
                print(output.strip())
            elif output.strip().startswith("df:"):
                print(output.strip(), end = "\r")
            elif output.strip().startswith("Calculation is"):
                print("\n"+output.strip()+"\n")
            #elif output.strip().startswith("Time elapsed"):
            #    print("\n"+output.strip())
            else:
                if verbose == True: #if True, print everything
                    print(output.strip())
                elif output.strip().startswith("Kinematic mean free"):
                    #Just to add an empty line between the two "\r"s above
                    print("\n")
    rc = process.poll()  #idk what this does, a problem for later... 

def load_files(fnm_prefix):
    #print("Reading all files at " + fnm_prefix)
    listTot = sorted(glob.glob(fnm_prefix+"*")) #append all the filenames into a list sorted by name
    #print("Found " + str(len(listTot)) + " files")

    #cbed = [] #np.zeros((len(listTot),int(dims[0]),int(dims[1])))
    #dimensions = []
    #for n1 in range(len(listTot)):
    with open(listTot[0], 'rb') as file:
            #cbed[n1][:][:] = np.fromfile(file, dtype='>f4').reshape((dims[0],dims[1]))
        dimX = int((listTot[0].split("_"))[-1].split("x")[0])
        dimY = int((listTot[0].split(str(dimX)+"x"))[1].split(".bin")[0])
        #cbed = np.fromfile(file, dtype='>f4').reshape((dimX, dimY))
        cbed = np.fromfile(file, dtype='>f4').reshape((dimY, dimX))
        #cbed.append(tmp)
        #dimensions.append([dimX, dimY])

    #print(listTot)
    return cbed #, dimensions  #removed dimensions 5/13/21

def load_files_series(fnm_prefix):
    print("Reading all files at " + fnm_prefix)
    listTot = sorted(glob.glob(fnm_prefix+"*")) #append all the filenames into a list sorted by name
    print("Found " + str(len(listTot)) + " files")

    cbed = [] #np.zeros((len(listTot),int(dims[0]),int(dims[1])))
    dimensions = []
    for n1 in range(len(listTot)):
        with open(listTot[n1], 'rb') as file:
            ##cbed[n1][:][:] = np.fromfile(file, dtype='>f4').reshape((dims[0],dims[1]))
            dimX = int((listTot[0].split("_"))[-1].split("x")[0])
            dimY = int((listTot[0].split(str(dimX)+"x"))[1].split(".bin")[0])
            ##cbed = np.fromfile(file, dtype='>f4').reshape((dimX, dimY))
            tmp = np.fromfile(file, dtype='>f4').reshape((dimY, dimX))
            cbed.append(tmp)
            dimensions.append([dimX, dimY])

    #print(listTot)
    return cbed #, dimensions  #removed dimensions 5/13/21


#Take the log of the cbed pattern
def log_cbed(cbed):
    cbed_log = cbed.copy()
    #cbed_log = copy.deepcopy(cbed)
    #print(np.amin(cbed))
    #print(np.amax(cbed))

    size1 = np.shape(cbed_log)[0]
    size2 = np.shape(cbed_log)[1]
    #print(np.shape(cbed_log))
    
    for n1 in range(size1):
        for n2 in range(size2):
            cbed_log[n1][n2] = math.log(abs(cbed_log[n1][n2])+1E-9) #deleted +1, 5/12/21

    return cbed_log

#Crop the cbed pattern to a specified size
def crop_cbed(cbed, crop_size):
    crop = math.ceil(crop_size/2)
    center = [math.ceil(x/2) for x in cbed[:][:].shape]

    cbed_cropped = cbed[center[0]-crop:center[0]+crop, center[1]-crop:center[1]+crop]
    
    return cbed_cropped 

def make_xtl(params, fnm_out):
    
    elements = params[0]
    atom_nums = params[1]
    uc = params[2]
    dw = params[3]
    coords = params[4]

    num_el = len(atom_nums)
    
    coords_frac = []
    num_atoms = []

    for n1 in range(num_el):
        for n2 in range(len(coords[n1])):
            num_atoms.append(len(coords[n1]))
            
            x = coords[n1][n2][0]
            y = coords[n1][n2][1]
            z = coords[n1][n2][2]
            
            coords_frac.append([str(x/uc[0]), str(y/uc[1]), str(z/uc[2])])
    
    outf = open(fnm_out,'w')
    
    outf.write(fnm_out + "\n")
    outf.write(str(uc[0])+" "+str(uc[1])+" "+str(uc[2]) +" ")
    outf.write("90.0 90.0 90.0\n")
    outf.write(str(num_el)+"\n\n")
    
    cnt = 0
    for n1 in range(num_el):
        outf.write(" "+elements[n1]+"\n")
        outf.write(str(num_atoms[n1])+"    "+ atom_nums[n1]+"    1.0    "\
                       +dw[n1] + "\n")
        for n2 in range(num_atoms[n1]):
            outf.write(coords_frac[n2+cnt][0] +"\t" + coords_frac[n2+cnt][1] + "\t"\
                           +coords_frac[n2+cnt][2] +"\n")
        cnt = cnt + 1
        outf.write("\n")
    outf.close()

def make_vasp(params, fnm_vasp):
    
    elements = params[0]
    uc = params[2]
    coords = params[4]
    
    outf = open(fnm_vasp, 'w')
    outf.write("CELL\n")
    outf.write("1.000\n")
    outf.write("\t" + str(uc[0]) + "\t 0.000 \t 0.000 \n")
    outf.write("\t 0.000 \t " + str(uc[1]) + "0.000 \n")
    outf.write("\t 0.000 \t 0.000 \t " + str(uc[2]) + " \n")
    outf.write("\t ")

    num_atoms = []

    for n1 in range(len(elements)):
        num_atoms.append(len(coords[n1]))
        outf.write(elements[n1]+" \t ")

    outf.write("\n \t  ")
    
    for n1 in range(len(elements)):
        outf.write(str(num_atoms[n1])+" \t ")
    
    outf.write("\nDirect\n")
    
    for n1 in range(len(elements)):
        for n2 in range(num_atoms[n1]):
            x = coords[n1][n2][0]/uc[0]
            y = coords[n1][n2][1]/uc[1]
            z = coords[n1][n2][2]/uc[2]
            
            outf.write(str(x)+" \t " + str(y) + " \t " + str(z) + "\n")

    outf.close()

def make_driver_stem_old(driver_name, params, device_type):
    driver_params = ["Output filename \n    " + params[0], #0
        "Input crystal file name \n    " + params[1], #1
        "Probe accelerating voltage (kV) \n    " + params[2], #2
        "Slice unit cell <1> yes <2> no \n    " + "1", #3
        "Number of distinct potentials \n    " + params[3], #4
        "<1> manual or <2> auto slicing \n    " + "2", #5
        "Thickness \n    " + params[4], #6
        "Scattering factor accuracy \n    " + "1", #7
        "Tile supercell x \n    " + params[5][0], #8
        "Tile supercell y \n    " + params[5][1], #9
        "Number of pixels in x \n    " + params[6][0], #10
        "Number of pixels in y \n    " + params[6][1], #11
        "<1> Continue <2> Change \n    " + "1", #revisit #12
        "Calculation type \n    " + "2", #13 <-- Hybrid (not as accurate)
        "aperture cutoff \n    " + params[7], #14
        "Defocus \n    " + params[8], #15
        "Change probe forming lens parameters \n    " + "0", #16
        "<1> QEP <2> ABS \n    2", #17
        "<1> Absorption <2> No absorption \n    1", #18 *
        "Elastic and inelastic scattering output choice \n    1", #19
        "<0> Continue <1> Beam tilt <2> Specimen tilt \n    " + params[9][0], #20
        "<0> continue <1> save <2> load \n    " + "0", #21
        "<1> CBED <2> STEM/PACBED \n    " + "2", #22
        "STEM modes \n 1", #23
        "STEM modes \n 0", #24
        "Probe scan menu choice \n    1", #25
        "output interpolation max pixels \n    " + params[10], #26
        "output interpolation tilex \n    " + params[11][0], #27
        "output interpolation tiley \n    " + params[11][1], #28
        "<0> Proceed <1> Output probe intensity \n    0", #29
        "Number of detectors \n    " + str(len(params[12])), #30
        "manual detector <1> auto <2> \n    1", #31
        "<1> mrad <2> inv A \n    1"] #32 * 

    #If beam/specimen tilt:
    if params[9][0] == "2": #specimen tilt
        new_list = [
            driver_params[18], 
            "Specimen tilt in mrad \n    " + params[9][1],
            "Specimen tilt azimuth in mrad \n    " + params[9][2],
            "<0> Continue <1> Beam tilt <2> Specimen tilt \n    " + "0"
        ]
        driver_params[18] = new_list
    elif params[9][0] == "1": #beam tilt
        new_list = [
            driver_params[18],
            "Beam tilt in mrad \n    " + params[9][1],
            "Beam tilt azimuth in mrad \n    " + params[9][2],
            "<0> Continue <1> Beam tilt <2> Specimen tilt \n    " + "0"
        ]
        driver_params[18] = new_list
        

    #Annular detector radii:
    new_list = []
    new_list.append(driver_params[32])
    for n1 in range(len(params[12])):
        new_list.append("inner \n    " + params[12][n1][0] + \
                             "\nouter \n    " + params[12][n1][1])
    driver_params[32] = new_list

    #Write driver file
    outf = open(driver_name,'w')
    if device_type == 1:
        outf.write("Device used for calculation \n")
        outf.write("0 \n")
        
    for n1 in range(len(driver_params)):
        if type(driver_params[n1]) is list:  #check for nested lists to iterate over
            for n2 in range(len(driver_params[n1])):
                outf.write(driver_params[n1][n2]+"\n")
        else:
            outf.write(driver_params[n1] + "\n")
    
    if device_type == 1:
        outf.write("<0> Precalculated potentials <1> On-the-fly calculation \n")
        outf.write("    0")
    
    outf.close()
    
    #Write user_input.txt
    outf = open("user_input.txt",'w')
    outf.write("play \n")  
    outf.write(driver_name)
    outf.close()
