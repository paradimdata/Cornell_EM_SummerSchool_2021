#
#  ctemh.py
#
#------------------------------------------------------------------------
#Copyright 2011-2014 Earl J. Kirkland
#
#This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#---------------------- NO WARRANTY ------------------
#THIS PROGRAM IS PROVIDED AS-IS WITH ABSOLUTELY NO WARRANTY
#OR GUARANTEE OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#MERCHANABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#IN NO EVENT SHALL THE AUTHOR BE LIABLE
#FOR DAMAGES RESULTING FROM THE USE OR INABILITY TO USE THIS
#PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA
#BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR
#THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH
#ANY OTHER PROGRAM). 
#-----------------------------------------------------------------------------
#
#  python function to calculate CTEM bright 
#  field phase contrast transfer function with partial
#  coherence for weak phase objects
#    input array k has the spatial freq. values (in 1/A)
#    input array params has the optical parameters
#    input type = 0 for phase contrast
#           and 1 for amplitude contrast
#    output array contains the transfer function vs k
#
#  p = [ kev, Cs3, Cs5, df, ddf, beta]
#
#  Cs3,5  = spherical aberration (in mm)
#  df     = defocus (in Angstroms)
#  kev    = electron energy (in keV)
#  ddf    = chrom. aberr. def. spread (in Angst.)
#  beta   = spread in illum. angles (in mrad)
#
#  started 25-jun-2011 ejk
#  small comsmetic changes 30-jun-2014 ejk
#  last modified 30-jun-2014 ejk
#
#from numpy import *
from pylab import *   #  gets sqrt()

# electron wavelength
def wavelen(kev: float):
    return( 12.3986/sqrt((2*511.0+kev)*kev) )

#  BF CTEM transfer function
def ctemh(k,params,type):
    kev = params[0]
    Cs3 = params[1]*1.0e7
    Cs5 = params[2]*1.0e7
    df = params[3]
    ddf = params[4]
    beta = params[5]*0.001
    wav = wavelen(kev)
    wavsq = wav*wav
    w1 = pi*Cs3*wavsq*wav
    w2 = pi*wav*df
    w3 = pi*Cs5*wavsq*wavsq*wav
    e0 = (pi*beta*ddf)**2
    k2 = k * k
    wr = ((w3*k2+w1)*k2-w2)*k*beta/wav
    wi = pi*wav*ddf*k2
    wi = wr*wr + 0.25*wi*wi
    wi = exp(-wi/(1+e0*k2))
    wr = w3*(1-2.0*e0*k2)/3.0
    wr = wr*k2 + 0.5*w1*(1-e0*k2)
    wr = (wr*k2 - w2)*k2/(1+e0*k2)
    if type == 0: y = sin(wr)* wi
    else:  y = cos(wr)* wi
    return y
