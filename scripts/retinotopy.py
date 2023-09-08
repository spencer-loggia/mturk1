#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:50:38 2022

@author: duffieldsj

Retinopy processing for wooster
"""
#%% Import packages

import sys
import glob
import os
import nilearn as nl
from nilearn import maskers as mask
from nilearn import signal as sig
from nilearn import image as image 
from nilearn import masking as masking
import nibabel as nib
import popeye as pop
from natsort import natsorted
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import popeye.og_hrf as og
import popeye.utilities as utils
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus
import math
import ctypes
import sharedmem
import datetime
import multiprocessing
from multiprocessing import Pool
import pickle

#%% Set up functions 

def load_binary_file_as_array(path, nb_cols, bits=8):
    import struct
    f = open(path, 'rb')
    xraw = f.read()
    size = f.tell()
    nb_vals = int(size/8) #8 bits values
    print("%d samples saved in file %s"%(nb_vals/nb_cols, path))
    out = np.array(struct.unpack('d'*nb_vals, xraw)).reshape(-1, nb_cols)
    f.close()
    return out

def create_circular_mask(h, w, center=None, radius=None): ## from alkasm on stackoverflow.net
# https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def mion_hrf(onset, tr, oversampling=1, time_length=40.):
    """ Implementation of the monocrystalline iron oxide nanoparticle 
    (MION) hrf model (Leite, et al. NeuroImage 16:283-294 (2002);
    
    http://dx.doi.org/10.1006/nimg.2002.1110
    https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html
    
    Parameters
    ----------
    tr : float
        scan repeat time, in seconds

    oversampling : int, optional
        temporal oversampling factor

    time_length : float, optional
        hrf kernel length, in seconds

    onset : float, optional
        hrf onset time, in seconds

    Returns
    -------
    hrf: array of shape(length / tr * oversampling, dtype=float)
         hrf sampling on the oversampled time grid
    """
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length, np.rint(float(time_length) / dt).astype(np.int))
    time_stamps -= onset

    hrf = np.zeros_like(time_stamps)

    for i,t in enumerate(time_stamps):
        hrf[i]  = - 16.4486 * ( - 0.184/ 1.5 * np.exp(-t / 1.5)
                                + 0.330/ 4.5 * np.exp(-t / 4.5)
                                + 0.670/13.5 * np.exp(-t / 13.5) )
    hrf /= np.abs(hrf.sum())
    return hrf


#%% Set up paths
subject = 'wooster'
sessions = ['20220826','20220829']
base_projP = '/home/ssbeast/Projects/SS/monkey_fmri/MTurk1'
behaviorP = base_projP+'/other_data/Retinotopy'
projP = base_projP  #+'/MTurk1/MTurk1' # On server slightly different pathing, may change when we resync the fmri computer to the server
subjectsP = projP+'/subjects'
subjP = subjectsP+'/%s'%subject
sessionsP = subjP+'/sessions'
sessPs = [sessionsP+'/%s'%session for session in sessions]
analF = subjP+'/analysis'
mriF = subjP+'/mri'

#%% We should clean these images and then average
TR = 3 # Seconds
bar_sweep_length = 14 # TRs


cleaned_avg_path = analF+'/pRFavg.nii.gz'

roilabel = 334

load = 1

if not load:
    print('Average run data')
    #%% Let's load the functional images

    imgFs = []

    for sess in sessPs:
        imgFs.extend(glob.glob(sess+'/*/reg_moco.nii.gz'))
        
    imgFs = natsorted(imgFs) # So these paths are properly sorted

    #%% Now we load the behavior

    behaviorPs = glob.glob(behaviorP+'/*.bin')
    behaviorPs = natsorted(behaviorPs)


    behaviorFs = []
    for behaviorfile in behaviorPs:
        behaviorFs.append(load_binary_file_as_array(behaviorfile, 11))
                          
    # Looks like we just want anything with over 371000 samples
    cutoff = 371000
    behaviorFiles = []
    for behaviorF in behaviorFs:
        if np.shape(behaviorF)[0] > 371000:
            behaviorFiles.append(behaviorF)

    #%% col_x, col_y, col_t = 8, 9, 4
    # We're now going to calculate which runs had good fixation
    prop_thresh = 0.85
    proportion = []

    for behavior in behaviorFiles:
        proportion.append(np.mean(np.hypot(behavior[:,8],behavior[:,9]) < 1))

    good_run_indx = np.array(proportion) > prop_thresh

    #%% Load images

    imgs = []
    shapes = []
    for i,imgF in enumerate(imgFs):
        if good_run_indx[i] == True:
            temp_img = nib.load(imgF)
            imgs.append(temp_img)
            
            shapes.append(temp_img.shape)
            print('Loaded %s'%imgF)

    affine = temp_img.affine




    #%%
    high_pass = 1/(TR*bar_sweep_length)
    low_pass = 1/TR

    cleaned_imgs = []
    for i,img in enumerate(imgs):
        cleaned_imgs.append(image.clean_img(img,low_pass=low_pass,high_pass=high_pass,t_r=TR))
        print('Done with image number %s'%str(i))
        #cleaned_imgs.append(mask.NiftiMasker(img,standardize='zscore',low_pass=low_pass,high_pass=high_pass,t_r=TR).mask_img)

    #%% 

    cleaned_imgs_arrays = []
    for img in cleaned_imgs:
        cleaned_imgs_arrays.append(img.get_fdata())
        



    cleaned_avg = np.mean(cleaned_imgs_arrays,axis=0)

    print(np.shape(cleaned_avg))

    num_TRs= np.shape(cleaned_avg)[3]

    #%% Save out cleaned average
    maskF = mriF+'/inverse_trans_ds_brainmask.nii'
    maskImg = nib.load(maskF)
    maskAffine = maskImg.affine
    maskImg_data = maskImg.get_fdata()
    maskImg_data = np.round(maskImg_data)
    maskImg = nib.Nifti1Image(maskImg_data,maskAffine)

    cleaned_avg_img = nib.Nifti1Image(cleaned_avg,affine)
    cleaned_masked_data = masking.apply_mask(cleaned_avg_img, maskImg)
    cleaned_masked_img = masking.unmask(cleaned_masked_data,maskImg)
    #cleaned_masked_img = nib.Nifti1Image(cleaned_masked_data, affine)
    nib.save(cleaned_masked_img,cleaned_avg_path)

if load:
#%% Reload?
    print('Loading cleaned data')
    cleaned_masked_img = nib.load(cleaned_avg_path)
    affine = cleaned_masked_img.affine
    maskF = mriF+'/inverse_trans_reg_atlas.nii'
    maskImg = nib.load(maskF)
    maskAffine = maskImg.affine
    maskImg_data = maskImg.get_fdata()
    maskImg_data = (maskImg_data == roilabel).astype('int') 
    maskImg = nib.Nifti1Image(maskImg_data,maskAffine)

    cleaned_masked_data = masking.apply_mask(cleaned_masked_img, maskImg)

    num_TRs= np.shape(cleaned_masked_img)[3]
#%%

# crds = [80,31,25]

# plt.plot(cleaned_masked_data[:,1])



#%%
screen_height_pixels = 1080
screen_width_pixels = 1080
screen_distance = 57
screen_height_cm = 21.5
screen_width_cm = 38.2
num_trs_sweep = 14
num_trs_blank = 3

# Now we create the sweeping bar stimulus. Below code is from Marianne Duyck

side_px = screen_height_pixels # target goal of the aperture diameter
side_dva = math.atan(screen_height_cm/screen_distance)*180/math.pi
nb_TRs = num_TRs
PRF_array = np.zeros((side_px, side_px, nb_TRs))
ratio_visible = 1 #proportion of the full height of the screen
vols_per_sweep = num_trs_sweep
bar_width_p = 0.15/ratio_visible
directions_code = [270, 45, 180, 315, 90, 225, 0, 135] # <==> [L-R, BR-TL, T-B, BL-TR, R-L, TL-BR, B-T, TR-BL]
directions_img = [0, 135, -90, 45, 180, -45, 90, -135]

# use sunit circle for ease
unit = 1
bar_width = bar_width_p * 2 * unit
coords = np.linspace(-unit, unit, side_px)


xv, yv = np.meshgrid(coords, coords, sparse=False, indexing='xy')
bar_centers = np.linspace(-unit/ratio_visible, +unit/ratio_visible, vols_per_sweep) #how it is currently implemented

full_disk = (xv**2 + yv**2) <= unit**2

zero_sweep = np.zeros((side_px, side_px, vols_per_sweep))
for i in range(vols_per_sweep):
    zero_sweep[:, :, i] = np.logical_and(full_disk, np.logical_and(xv > bar_centers[i] - bar_width/2, xv < bar_centers[i] + bar_width/2))

directions = directions_img
ix_start_sweep = np.arange(num_trs_blank, nb_TRs-num_trs_blank, vols_per_sweep)
for d in range(len(directions)):
    PRF_array[:, :, ix_start_sweep[d]:ix_start_sweep[d]+zero_sweep.shape[-1]] = ndimage.rotate(zero_sweep, directions[d], reshape=False, order=0)


#%% Plot PRF Array
# for i in range(np.shape(PRF_array)[-1]):
#     plt.figure()
#     plt.imshow(PRF_array[:,:,i])
#     plt.title('Frame %3.2f'%(i))
    
#%% Popeye

stimulus = VisualStimulus(PRF_array,screen_distance,screen_width_cm,0.5,TR,ctypes.c_int16)

### MODEL
## initialize the gaussian model
model = og.GaussianModel(stimulus, mion_hrf)

#%%
data = cleaned_masked_data.T



#%% Simulation
# voxel = 1
# data_check = data[voxel]
# ## generate a random pRF estimate
# x = 0
# y = 0
# sigma = 1
# hrf_delay = 0
# beta = -0.006
# baseline = 0

# ## create the time-series for the invented pRF estimate
# data_sim = model.generate_prediction(x, y, sigma, hrf_delay, beta, baseline)

# plt.plot(data_sim,c='r',lw=3,label='Simulated model',zorder=1)
# plt.plot(data_check,label='data',c='b',zorder=2)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel('Time',fontsize=18)
# plt.ylabel('Amplitude',fontsize=18)
# plt.xlim(0,len(data_sim.data))
# #plt.legend(loc=0)

#%%
### FIT
## define search grids
# these define min and max of the edge of the initial brute-force search. 
x_grid = (-22,22)
y_grid = (-22,22)
s_grid = (1/stimulus.ppd + 0.25,12)
h_grid = (-3.0,3.0)

## define search bounds
# these define the boundaries of the final gradient-descent search.
x_bound = (-22.0,22.0)
y_bound = (-22.0,22.0)
s_bound = (1/stimulus.ppd, 12.0) # smallest sigma is a pixel
h_bound = (-3.0,3.0)
b_bound = (-10,-0.0001)
u_bound = (None,None)

## package the grids and bounds
grids = (x_grid, y_grid, s_grid, h_grid)
bounds = (x_bound, y_bound, s_bound, h_bound, b_bound, u_bound,)


#%%
## fit the response
# auto_fit = True fits the model on assignment
# verbose = 0 is silent
# verbose = 1 is a single print
# verbose = 2 is very verbose
# fit = og.GaussianFit(model, data[voxel], grids, bounds, Ns=3,
#                      voxel_index=voxel, auto_fit=True,verbose=2)



# #%% Actual
# ## plot the results
# import matplotlib.pyplot as plt
# plt.plot(fit.prediction,c='r',lw=3,label='model',zorder=1)
# plt.plot(fit.data,label='data',c='b',zorder=2)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel('Time',fontsize=18)
# plt.ylabel('Amplitude',fontsize=18)
# plt.xlim(0,len(fit.data))
# plt.legend(loc=0)



#%%
## multiprocess all

indices = np.arange(0,np.shape(data)[0])
# FYI you may need to edit line 402 of popeye base.py so it doesn't index the voxel index
cores = 50

bundle = utils.multiprocess_bundle(og.GaussianFit, model, data[indices], 
                                   grids, bounds, indices, 
                                   auto_fit=True, verbose=1, Ns=5)




## run
print("popeye will analyze %d voxels across %d cores" %(len(bundle), cores))




with Pool(cores) as pool:
    t1 = datetime.datetime.now()
    output = pool.map(utils.parallel_fit, bundle)
    t2 = datetime.datetime.now()
    delta = t2-t1
    print("popeye multiprocessing finished in %s.%s seconds" %(delta.seconds,delta.microseconds))

print('packing out data')
with open(analF+'/prfFit.pkl','wb') as f:
    pickle.dump(output,f)

data_array = np.array([o.data for o in output]).T
hrf_delays = np.array([o.hrf_delay for o in output]).T
receptive_fields = np.array([o.receptive_field for o in output])
rsquareds = np.array([o.rsquared for o in output]).T
sigmas = np.array([o.sigma for o in output]).T
thetas = np.array([o.theta for o in output]).T
xes = np.array([o.x for o in output]).T
yes = np.array([o.y for o in output]).T
rhos = np.array([o.rho for o in output]).T

                      

data_img = masking.unmask(data_array,maskImg)
hrf_img = masking.unmask(hrf_delays,maskImg)
rsquared_img = masking.unmask(rsquareds,maskImg)
sigma_img = masking.unmask(sigmas,maskImg)
theta_img = masking.unmask(thetas,maskImg)
x_img = masking.unmask(xes,maskImg)
y_img =masking.unmask(yes,maskImg)
rho_img = masking.unmask(rhos,maskImg)

nib.save(data_img,analF+'/predictedPRFtimeseries.nii.gz')
nib.save(hrf_img,analF+'/PRFhrf.nii.gz')
nib.save(rsquared_img,analF+'/PRFrsquared.nii.gz')
nib.save(sigma_img,analF+'/PRFsigma.nii.gz')
nib.save(theta_img,analF+'/PRFtheta.nii.gz')
nib.save(x_img,analF+'/PRFx.nii.gz')
nib.save(y_img,analF+'/PRFy.nii.gz')
nib.save(rho_img,analF+'/PRFrho.nii.gz')




#%% Plot different voxels 
# index = 3

# plt.plot(output[index].data,c='r',lw=3,label='Model Fit',zorder=1)
# plt.plot(data[indices[index]],label='data',c='b',zorder=2)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel('Time',fontsize=18)
# plt.ylabel('Amplitude',fontsize=18)
# plt.xlim(0,len(data[0]))




