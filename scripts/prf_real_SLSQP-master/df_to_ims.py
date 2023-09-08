#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:27:13 2022

@author: braunlichkr
"""

import glob
import clarte as cl
import os
import nibabel as nib
import nilearn.image as nli
import numpy as np
from subprocess import call
import pandas as pd

#%%

atlasN='juelich'
anatF='/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/nii/sub-001/anat_clinical'
anatP = glob.glob(anatF+'/4_t1_mprage_sag_p2_0.75mm.nii')[0]

atlasGrayP = glob.glob('%s/%sVisualRegions_masked_by_ribbon.nii.gz'%(anatF,
                                                            atlasN))[0]

sessDerivF='/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214'
anN='prf_grid_slsqp'
f='%s/%s'%(sessDerivF,anN)

polP = glob.glob(f+'/pol.nii.gz')[0]
dat = cl.fmri_data([polP],atlasGrayP)

dfp = glob.glob(f+'/allRuns_prfs.csv')[0]
df=pd.read_csv(dfp,index_col=0)

nRowsIncomplete = dat.dat.shape[0]-len(df)
vComplete = np.arange(len(df))
m = df.values
mEmpt = np.zeros((nRowsIncomplete,len(df.columns)))-10
assert(mEmpt.shape[0] + m.shape[0] == dat.dat.shape[0])
mConcat = np.vstack([m,mEmpt])
df = pd.DataFrame(mConcat,columns=df.columns)


# nComplete = np.max(df.index)
# dfEmpt = pd.DataFrame(,
#                       index=range(nComplete,nComplete+nRowsIncomplete))
# dfFull = pd.concat([df,dfEmpt],axis=0)

#%%
def matrixCoords2polar(x,y,returnPosTh=False):
    '''matrix coords have sign y flipped relative to cartesian.
    
     returns ecc, angle'''
    y *= -1 # corect for matrix coords 
    z = x + 1j * y
    ecc, angle = ( np.abs(z), np.angle(z) )
    if returnPosTh and angle<0:
        angle += np.pi*2
        
    return ecc, angle
#%%
print(df.columns)
csToIm= ['sigma', 'err', 'polarNeg', 'eccentricity', 'pearsonr',
       'r2', 'intercept', 'beta']
df['polarNeg']= np.nan
# change positive only thetas to range -3.14 to 3.14 (to compare to afni results)
for i in vComplete:
    x,y = df.loc[i,['X','Y']]
    df.loc[i,'polarNeg'] = matrixCoords2polar(x,y)[1]

#%%
lp=[]
for c in csToIm:
    im = dat.unmasker(df[c].values)
    p=f+'/%s_2.nii.gz'%c
    im.to_filename(p)
    lp.append(p)


cl.fsey([anatP]+lp)

#%%
volP = '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/e_clean/ccmst_04_ret_cmrr_1-2mm_MB2_iPAT2_TR2_loc_APmasked.nii.gz'
outP = '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/e_clean/smFwhm2_ccmst_04_ret_cmrr_1-2mm_MB2_iPAT2_TR2_loc_APmasked.nii.gz'
imp = nib.load(volP)
rAtlP = '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/e_clean/rJuelich_by_ribbon.nii.gz'
rAtlIm = nli.resample_img(atlasGrayP,imp.affine,interpolation='nearest')
#%%
rAtlIm.to_filename(rAtlP)

cl.smooth_within_mask(rAtlP,p,2,outP)
cl.fsey([outP])

#%%
layniF='/Users/braunlichkr/Documents/software/laynii'
cmd = './LN_GRADSMOOTH -input  %s -gradfile %s -FWHM 2 -mask %s -within -selectivity .05 -output %s'%(volP,rAtlP,rAtlP,outP)
print(cmd)
os.chdir(layniF)
call(cmd,shell=True)

