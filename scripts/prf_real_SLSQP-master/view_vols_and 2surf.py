#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:34:51 2022

@author: braunlichkr
"""

import sys
import glob
import clarte as cl
import os

scriptF = '/Users/braunlichkr/Documents/experiments/macman_align/bin/preproc_hum'
sys.path.append(scriptF); import hPreproc as hp
#%%
anatP = '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/nii/sub-001/anat_clinical/brain.nii.gz'
anatF = os.path.split(anatP)[0]
prfResF='/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/prf_afni/res'
# ps=['/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/prf_afni/res/ecc.nii.gz',
#     '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/prf_afni/res/pol.nii.gz',
#     '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/prf_afni/res/r2.nii.gz']
ps = [glob.glob(prfResF+'/%s*.nii.gz'%v)[0] for v in ['ecc','pol','sigma']]
funcF = os.path.split(ps[0])[0]
surfF = '%s/surf'%os.path.split(ps[0])[0]
if not os.path.exists(surfF):
    os.makedirs(surfF)
#%% view volume
cmd = [p+' -cm hot -un ' for p in ps]
cl.fsey([anatP]+cmd)
#%%
hems = ['lh','rh']
fsF = '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/fs'
subj = 'sub-001'
for volP in ps:
    for hem in hems:
        outMghP = '%s/%s_%s.mgz'%(surfF,
                                  os.path.basename(volP).replace('.nii.gz',''),
                                  hem)
        refP = glob.glob(fsF+'/%s/mri/orig.mgz'%subj)[0]
        # hp.vol2surf(fsF, subj,hem,volP,outMghP)
        # cl.fsey([refP,volP])
        #%
        hp. mri_vol2surf(volP,
                         'orig.mgz',
                         outMghP,
                         fsF,
                         subj,
                         hem,
                         reg='',
                         interp='trilinear',
                         surf='white',#%hem,
                         projfracavg=(0,1,.1),
                         surf_fwhm=3)
        
#%% view inflated surface
variable = ['ecc','pol','r2'][0]
subFsF = fsF+'/%s'%subj
# for hem in hems:
hem='rh'
inflSurfP = glob.glob(subFsF+'/surf/%s.inflated'%hem)[0]# for hem in hems]
statSurfP = glob.glob(funcF+'/surf/%s_%s.mgz'%(variable,
                                                  hem))[0]# for hem in hems] 
hp.freeview_surface_and_overlay(inflSurfP,
                                statSurfP,thrMinMax=[0.1,1])