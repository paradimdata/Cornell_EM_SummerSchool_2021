# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 23:29:40 2021

@author: Xiyue Zhang
Converted from Yi Jiang's 'postProcess.m''
"""
import numpy as np
from skimage.transform import rotate, rescale

def sind(x):
    return np.sin(x * np.pi / 180)

def cosd(x):
    return np.cos(x * np.pi / 180)

def removePhaseRamp(input, dx):
    Ny = input.shape[0]
    Nx = input.shape[1]
    y = np.linspace(-np.floor(Ny/2),np.ceil(Ny/2)-1,Ny)
    x = np.linspace(-np.floor(Nx/2),np.ceil(Nx/2)-1,Nx)
    [X,Y] = np.meshgrid(x,y)
    X = X*dx
    Y = Y*dx
    phase_image = np.angle(input)

    #fit ramp
    Xf = X.flatten()
    Yf = Y.flatten()
    A = np.array([Xf*0+1, Xf, Yf]).T
    B = phase_image.flatten()
    coeff, r, rank, s = np.linalg.lstsq(A, B)
    background = X*coeff[1]+Y*coeff[2]
    output = phase_image - background
    return output


def postProcess(obj, rot_angle, px, py, dx):
    #Post process reconstructed object
    # Convert from Matlab postProcess.m  Yi Jiang (yj245@cornell.edu)

    #rotate object to 0 degree
    py_rot = px*-sind(-rot_angle) + py*cosd(-rot_angle)
    px_rot = px*cosd(-rot_angle) + py*sind(-rot_angle)

    obj_rot_r = rescale(obj.real,2) 
    obj_rot_i = rescale(obj.imag,2)#upsample to reduce rotation artifacts
    obj_rot_r = rotate(obj_rot_r, -rot_angle)
    obj_rot_i = rotate(obj_rot_i, -rot_angle)
    obj_rot = obj_rot_r + 1j*obj_rot_i

    cen_rot = np.floor(np.size(obj_rot,1)/2)+1

    dx = dx/2
    y_lb = np.ceil(min(py_rot[0])/dx+cen_rot)
    y_ub = np.floor(max(py_rot[0])/dx+cen_rot)    
    x_lb = np.ceil(min(px_rot[0])/dx+cen_rot)   
    x_ub = np.floor(max(px_rot[0])/dx+cen_rot)
    
    obj_crop = obj_rot[int(y_lb):int(y_ub),int(x_lb):int(x_ub)]
    
    #remove phase ramp
    obj_crop_phase = removePhaseRamp(obj_crop, dx)
    obj_crop = abs(obj_crop) * np.exp(1j*obj_crop_phase)
    obj_crop_r = rescale(obj_crop.real,1/2) 
    obj_crop_i = rescale(obj_crop.imag,1/2) #scale back to original size
    obj_crop = obj_crop_r + 1j*obj_crop_i

    return obj_crop