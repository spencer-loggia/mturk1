#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:35:03 2022

@author: kurtb
"""
from nilearn.glm.first_level import hemodynamic_models as hrf_mods
from skimage.transform import resize
import matplotlib.pyplot as plt
# import prf_analysis_tools as pat
# import mPreproc as mp
import sys
import glob
import clarte as cl
from sklearn.linear_model import LinearRegression
import itertools as it
import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import nilearn.plotting as nip
from joblib import Parallel, delayed
import nilearn.image as nli
from scipy.optimize import minimize
from scipy import stats
np.seterr(divide='ignore')
from scipy.signal import convolve2d


if 'arwin' in sys.platform:
    expF = '/Volumes/Untitled/macman_align'
    fsl_b02b0 = '/Applications/fsl/etc/flirtsch/b02b0.cnf'
else:
    expF='/media/veracrypt1/macman_align'
    fsl_b02b0 = '/usr/local/fsl/etc/flirtsch/b02b0.cnf'
    

humScriptF = os.path.join(expF,'bin','preproc_hum')
sys.path.append(humScriptF); import hPreproc as hp
monkScriptF = os.path.join(expF,'bin','preproc_monk_og')
sys.path.append(monkScriptF); import mPreproc as mp
prfF = os.path.join(expF,'bin','first_lev','prf')
sys.path.append(prfF);

#%%

fwhm=2
    
subjs = ['sub-001','sub-003','sub-004','sub-005']
subjs = ['sub-004']
species=['hum','monk'][0]


derivF= os.path.join(expF,'data',species,'derivatives')
preprocF = os.path.join(derivF,'preproc')
hrfToUse='vista'

# anName = 'slsqp_%s_precompute'%hrfToUse
prfResF = derivF+'/prf'
# anF = os.path.join(prfResF,anName)
# csvResF = anF+'/csvRes'
# niiResF = anF+'/niiRes'
# surfResF = anF+'/surfRes'
# fsToCreate = [csvResF,niiResF,surfResF
#               ]
# for f in fsToCreate:
#     if not os.path.exists(f):
#         os.makedirs(f)
    
fsF = os.path.join(derivF,'fs')
    
imgCol = 'clean_in_dsFsT1P'#'clean_in_dsFsT1_dilatedRibbonP'

avgF = prfResF+'/avg_clean_in_dsFsT1'
if not os.path.exists(avgF):
    os.makedirs(avgF)
            
#%%  avg all runs
def correctPs(dfP):
    cs = [ imgCol,
            ]
    if 'arwin' in sys.platform:
        rpF='/media/veracrypt1/macman_align'
    else:
        rpF = '/Volumes/Untitled/macman_align'
    for c in cs:
        for i in dfP.index:
            newp = dfP.loc[i,c].replace(rpF,expF)
            assert(os.path.exists(newp))
            dfP.loc[i,c]  = newp
    return dfP


plt.close('all')
plt.figure()

    
for subj in subjs:
    
    csvF = os.path.join(preprocF,subj,'paths') 
    bgImg = os.path.join(fsF,subj,'mri','brainmask.mgz')
    
    # find sessions
    ps = glob.glob(os.path.join(preprocF,
                                subj,
                                '*/*/*_ret_*.nii.gz'))
    sessions = np.unique([os.path.basename(os.path.split(os.path.split(p)[0])[0]) for p in ps])
    assert(len(sessions)==1); sessN=sessions[0]
        
    csvP = glob.glob(os.path.join(csvF,'%s_%s.csv'%(subj,sessN)))[0]
    dfP = pd.read_csv(csvP,index_col=0)       
    idx = ['ret' in v for v in dfP['task']]
    dfP = dfP.loc[idx,:].reset_index(drop=True)
    dfP = correctPs(dfP)
    assert(np.all([os.path.exists(p) for p in dfP[imgCol]]))
    
    ribbonMaskP = fsF+'/%s/mri/bi.ribbon.nii.gz'%subj
    if not os.path.exists(ribbonMaskP):
        ribbonMaskP = hp.mgz_to_niigz(ribbonMaskP.replace('.nii.gz','.mgz'))
    # check mask
    if 0:
        dat = cl.fmri_data([dfP.loc[0,imgCol]],ribbonMaskP)
        mn = dat.dat.mean(axis=0)
        imMn = dat.unmasker(mn)
        nip.plot_epi(imMn,bg_img=bgImg)


    avgP = avgF+'/%s_fwhm%.2d_fullRibbon.nii.gz'%(subj,fwhm)
    
    if os.path.exists(avgP):
        print(os.path.basename(avgP),'exists, skipping')
    else:
        # niiPs = np.sort(glob.glob(inImF+'/c*masked.nii.gz'))
        #s print([os.path.basename(p) for p in niiPs])
        for i in tqdm(dfP.index,desc='%s: averaging %d runs'%(subj,
                                                              len(dfP))):
            # im = nim.apply_mask(imMask,fwhm=2)
            # smP = '%s/fwhm%.2d_%s'%(os.path.split(p)[0],fwhm,os.path.split(p)[1])
            # cl.smooth_within_mask(atlasGreyP,
            #                       p,
            #                       fwhm,
            #                       outP=smP)
            p = dfP.loc[i,imgCol]
            dat=cl.fmri_data([p],ribbonMaskP,fwhm=fwhm)
            plt.clf()
            plt.hist(dat.dat.flatten(),bins=30)
            plt.title(os.path.basename(p))
            plt.pause(.01)
            if i==0:
                mAv = dat.dat
            else:
                mAv += dat.dat
        mAv /= np.float32(len(dfP))
        
        imBold = dat.unmasker(mAv)
        imBold.to_filename(avgP)
        import scipy  
        matP = '%s/runAvg.mat'%os.path.split(p)[0]
        scipy.io.savemat(matP,{'mAv':mAv})
        
        del mAv, imBold, dat
        
