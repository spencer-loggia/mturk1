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
from sklearn.linear_model import LinearRegression
from scipy.signal import convolve2d


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