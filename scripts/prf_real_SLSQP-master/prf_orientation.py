#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:39:04 2022

@author: kurtb
"""


import sys
import numpy as np
import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats

#%%

expF='/home/kurtb/code/prf_real_SLSQP'
os.chdir(expF)
sys.path.append(expF)
import prf_analysis_tools as pt

#%%
vHrf = np.array([         0,
    0.0866,
    0.3749,
    0.3849,
    0.2161,
    0.0769,
    0.0016,
   -0.0306,
   -0.0373,
   -0.0308,
   -0.0205,
   -0.0116,
   -0.0058,
   -0.0026,
   -0.0011,
   -0.0004,
   -0.0001])

stimP = 'stim_256x256.nii'



#%%
def convert_xy_to_colrow(x0,y0,framerowpix=mStim.shape[0]):
    
    x = np.round((x0+1)/2 * framerowpix).astype(int)
    y = np.round((y0+1)/2 * framerowpix).astype(int)
    return x,y
    
def visStim(mStim,x0,y0,voxHrf,isNeural=False):
    '''
    l_r,br_tl, t_b, bl_tr, r_l, tl_br, b_t, tr_bl
    
    left to right, bottom right to top left, top to bottom,
bottom left to top right, right to left, top left to bottom right,
bottom to top, and top right to bottom left.'''
    plt.figure(figsize=(40,20))
    
    for tr in range(0,mStim.shape[-1]):
        plt.subplot(1,2,1)
        plt.imshow(mStim[:,:,0,tr],origin='lower')
        col,row = convert_xy_to_colrow(x0, y0)
        plt.scatter(col,row,s=100,c='r',marker='x')
        
        plt.subplot(1,2,2)
        if not isNeural:
            plt.plot(np.arange(mStim.shape[-1]),voxHrf)
        else:
            wh = np.where(voxHrf>.5)
            [plt.axvline(wh) for wh in voxHrf]
        plt.axvline(tr,linestyle='dashed',c='k')
        plt.pause(.5)
        # k = input('ok')
        plt.clf()
        
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#%%

imStim = nib.load(stimP)
mStim = imStim.get_fdata()[::-1,:]

mStim2d = np.squeeze(mStim).reshape(np.prod(mStim.shape[:2]), mStim.shape[-1])
#%%
x0,y0=.5,-.5
ecc, polangle = cart2pol(x0,y0)
sg=.05
R=1
mRf = pt.gen_ellipse(x0,y0,sg,R,0,mStim.shape[0])
# plt.imshow(mRf,origin='lower')

#%%
mBold2d = pt.convolve_matrix_neural_bold(mStim2d, 
                                       vHrf, 
                                       2, 
                                       mStim.shape[-1],
                                       stimFrameShape=mStim.shape[:2])
voxHrf = stats.zscore(pt.normVoxHrfResp(np.dot(mRf.reshape(mStim.shape[0]**2),mBold2d )))
voxNeur = np.dot(mRf.reshape(mStim.shape[0]**2),mStim2d )
plt.close('all')
# visStim(mStim,x0,y0,voxNeur,isNeural=True)
visStim(mStim,x0,y0,voxHrf,isNeural=False)
plt.close('all')



