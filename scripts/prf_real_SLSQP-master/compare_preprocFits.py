#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 13:03:28 2022

@author: braunlichkr
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
import nilearn.image as nli
from scipy.optimize import minimize
from scipy import stats
np.seterr(divide='ignore')
from scipy.signal import convolve2d
sys.path.append(
    '/Users/braunlichkr/Documents/experiments/macman_align/bin/first_lev/prf')
sys.path.append(
    '/Users/braunlichkr/Documents/experiments/macman_align/bin/preproc_monk')
# sys.path.append('/Users/braunlichkr/Documents/experiments/PRFs/gari_sims')

# import prf_tools as pt
#%%
TR=2
prfF = '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/prf_afni'
anatF='/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/nii/sub-001/anat_clinical'
atlasN='juelich'
sessF = '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214'
os.chdir(prfF)
cleanF = sessF+'/e_clean'

atlasGrayP = glob.glob('%s/%sVisualRegions_masked_by_ribbon.nii.gz'%(anatF,
                                                            atlasN))[0]
imAtlasGray = nib.load(atlasGrayP)
vHrf = hrf_mods.spm_hrf(TR, oversampling=1)

# afniTmpF = 'res/_tmp_afni'#%expF
# anTag = 'all_afni'
# voxBoldP = '%s/%s.1D'%(afniTmpF,anTag)

#%% create og mBold2d
recreateOgBold = False
ogBoldP = cleanF+'/avgAllRuns.nii.gz'
if (not recreateOgBold) and os.path.exists(ogBoldP):
    print('loading og bold')
    ratlasGrayIm = nli.resample_to_img(atlasGrayP,
                                   ogBoldP,
                                   interpolation='nearest')
    dat = cl.fmri_data([ogBoldP],ratlasGrayIm)
    imBold_2d = dat.dat.T
    mBold2d = dat.dat.T
else:
    niiPs = np.sort(glob.glob(cleanF+'/*masked.nii.gz'))
    for i,p in tqdm(enumerate(niiPs),desc='averaging %d runs'%len(niiPs)):
        dat=cl.fmri_data([p],imAtlasGray)
        if i==0:
            mAv = dat.dat
        else:
            mAv += dat.dat
    mAv /= np.float32(len(niiPs))
    
    imBold = dat.unmasker(mAv)
    
    import scipy  
    matP = '%s/runAvg.mat'%os.path.split(p)[0]
    scipy.io.savemat(matP,{'mAv':mAv})
    
    #% reshape bold
    
    dat = cl.fmri_data([imBold],imAtlasGray)
    mBold2d = dat.dat.T
    dat.unmasker(dat.dat).to_filename(ogBoldP)

# mBold2d = pd.read
#%% create new mBold2d
# og is: detrend, standardize, confounds, high-pass=0.17852, mask_img=atlasGrey
preprocN = ['stndrz_cnfnds_highPs_mask', 
             ][0]
# print(preprocN)
createNew_mBold_new = False
newCleanP= cleanF+'/mnAll_%s.nii.gz'%preprocN

if (not createNew_mBold_new) and os.path.exists(newCleanP):
    print('loading ', preprocN)
    ratlasGrayIm = nli.resample_to_img(atlasGrayP,
                                   newCleanP,
                                   interpolation='nearest')
    dat = cl.fmri_data([newCleanP],ratlasGrayIm)
    mBold2d_new = dat.dat.T
else:
    print('creating new',preprocN)
    cregF='/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/d_alignToT1_rigid'
    niiPs = np.sort(glob.glob(cregF+'/cmst*.nii.gz'))
    for i,p in tqdm(enumerate(niiPs),desc='cleaning & averaging %d runs'%len(niiPs)):
        n = os.path.basename(p).replace('.nii.gz','')
        noiseF = '/Users/braunlichkr/Documents/experiments/macman_align/fmri/hum/derivatives/sub-001/ses-20211214/noise_regressors'
        noiseP = glob.glob(noiseF+'/%s.csv'%n)[0]
        dfAllNoise = pd.read_csv(noiseP,index_col=0)
        ratlasGrayIm = nli.resample_to_img(atlasGrayP,
                                           p,
                                           interpolation='nearest')
        im = nli.clean_img(
            p,
            # detrend=True,
            standardize=True,
            confounds = dfAllNoise.values,
            t_r=np.float32(TR),
            # low_pass=.25, 
            high_pass=.017852, # 56=s 2sweeps (1sweep=14TRs)
            mask_img =ratlasGrayIm,
            )
        dat=cl.fmri_data([im],ratlasGrayIm)
        if i==0:
            mAv = dat.dat
        else:
            mAv += dat.dat
        mAv /= np.float32(len(niiPs))
    
    imBold = dat.unmasker(mAv)
    
    
    dat = cl.fmri_data([imBold],ratlasGrayIm)
    dat.unmasker(dat.dat).to_filename(newCleanP)
    mBold2d_new = dat.dat.T
    
    del imBold
    del im
    del mAv
    del dat


#%% visualize random old and new bold vectors

plt.close('all')
plt.figure(figsize=(10,10))
n = 2
nVox = mBold2d.shape[0]
ridx = np.random.permutation(range(nVox))
idx = [ridx[:n],ridx[n:n*2]]
for iplt in range(2):
    plt.subplot(2,2,iplt+1)
    for i in idx[iplt]:
        plt.plot(mBold2d[i,:])
    plt.legend([iplt-2])
    plt.title('og hPreproc')
    
for iplt in range(2,4):
    plt.subplot(2,2,iplt+1)
    for i in idx[iplt-2]:
        plt.plot(mBold2d_new[i,:])
    plt.legend([iplt-2])
    plt.title('new')


# %% downsample stim
'''afni's nii format (from Ed) has dims swapped. To match my code. I swap before writing'''
print('resampling stim')
view = False
write=False
og_afniStimP = '/Users/braunlichkr/Desktop/prf_sims/stim/stimMasks144Padded.nii'
og_afniStimIm = nib.load(og_afniStimP)
mAfni = np.swapaxes( og_afniStimIm.get_fdata(),0,1)
stimP = '/Users/braunlichkr/Documents/experiments/macman_align/bin/first_lev/prf/PRF_mask_per_TR.npy'

mStim = np.load(stimP)#,0,1)

# previously gnerated 256*256 

newAfniP = '/Users/braunlichkr/Documents/experiments/macman_align/bin/first_lev/prf/stim_256x256.nii'
showLast=True
try: 
    new256 = np.swapaxes( nib.load(newAfniP).get_fdata(),0,1)
except:
    showLast=False

d = [100,256][0]
nTr = mStim.shape[-1]
rmStim = np.zeros((d, d, nTr))
if view:
    plt.figure()
for i in range(nTr):
    im = mStim[:, :, i]
    rim = np.round(resize(im, (d, d), order=0))
    if view:
        plt.subplot(1, 3, 1)
        plt.imshow(im)
        plt.title('200x200')
        plt.subplot(1, 3, 2)
        plt.imshow(rim)
        plt.title(rim.shape)
        if showLast:
            plt.subplot(1, 3, 3)
            plt.imshow(new256[:,:,0,i])
            plt.title('afni space (256*256)\n loaded from file')
    assert(len(np.unique(rim)) <= 2)
    rmStim[:, :, i] = rim
    if view:
        plt.suptitle(i)
        plt.pause(.5)
        plt.clf()
plt.close('all')
if write:
    srmStim = np.swapaxes(rmStim,0,1)
    dsStimIm = nib.Nifti1Image( np.expand_dims(srmStim,2),og_afniStimIm.affine)
    # ndsStimIm = np.expand_dims(rmStim,2)
    dsStimIm.to_filename(newAfniP)


#%%

def gen_gauss(x0, y0, sigma, frameRowPix):
    _x = np.linspace(-1, 1, frameRowPix+1)
    step = _x[1]-_x[0]
    xs, ys = np.mgrid[-1:1:step, -1:1:step]
    pos = np.dstack((xs, ys))
    mX = pos[:, :, 1]
    mY = pos[:, :, 0]
    return np.exp(- ((mX-x0)**2 + (mY-y0)**2) / (2 * sigma**2))


def convolve_neural_boldPred(vHrf, mStim2d, nTr, frameRowPix):
    if len(vHrf.shape) == 1:
        vHrf = vHrf.reshape(1, len(vHrf))
        mBold = convolve2d(mStim2d, vHrf)[:, :nTr].reshape(frameRowPix,
                                                           frameRowPix,
                                                           nTr)

        return mBold.squeeze().reshape(frameRowPix**2, nTr)


def make_param_grid(xs, ys, sgs):
    lParam = list(it.product(xs, ys, sgs))
    cols = ['sX', 'sY', 'sSigma']
    df = pd.DataFrame(index=range(len(lParam)),
                      data=lParam,
                      columns=cols)
    return df


reg = LinearRegression()


def get_hrf(x, y, sigma, mPredBold2d, frameRowPix):
    imRf = gen_gauss(x, y, sigma, frameRowPix)
    voxHrfResp = stats.zscore(
        np.dot(imRf.reshape(frameRowPix**2), mPredBold2d))
    return voxHrfResp


def fitPred(vPred, targHrf):
    reg.fit(
        targHrf.reshape(len(vPred), -1),
        vPred.reshape(len(vPred), -1))
    return vPred + reg.intercept_[0]


def fitSL(mPredBold2d, voxBold, frameRowPix=100, glmFit=False):
    def loss(params, targHrf=voxBold, glmFit=glmFit):
        x, y, sigma = params
        hrf_ = get_hrf(x, y, sigma, mPredBold2d, frameRowPix)
        if glmFit:
            try:
                hrf_ = fitPred(hrf_, targHrf)
            except:
                pass
        return np.sum((targHrf-hrf_)**2)*.001

    bounds = ((-1., 1.),
              (-1., 1.),
              (.001, 2.),
              )

    def con(params):
        x,y,_ = params
        euc = np.linalg.norm([x,y])
        return float( 1.0-euc)
    cons = {'type':'ineq','fun':con}
    
    # setup grid search
    topN = 10
    xs, ys = np.linspace(-.95, .95, 8), np.linspace(-.95, .95, 8)
    # sgs = np.linspace(.005, 2., 20)
    sgs = np.arange(0.01,.1,.01)
    sgs2 = np.arange(.1,1,.025)
    sgs =np.concatenate([sgs,sgs2])
    dfGrid = make_param_grid(xs, ys, sgs)
    # remove start locs > periph (1)
    lEuc = np.array([np.linalg.norm(dfGrid.loc[i,['sX','sY']] ) for i in dfGrid.index])
    dfGrid = dfGrid.loc[lEuc<=1.0,:]

    dfGrid['err'] = np.nan
    for i in (dfGrid.index):#,desc='initial grid search'):
        dfGrid.loc[i, 'err'] = loss(dfGrid.loc[i, dfGrid.columns[:-1]],
                                    glmFit=glmFit)

    dfGrid = dfGrid.sort_values('err',
                                ascending=True,
                                kind='mergesort',
                                ignore_index=True)
    dfGrid = dfGrid.loc[:topN, :]


    resCols = ['rX', 'rY', 'rSg']  # ,'rR','rTh']
    for c in resCols+['fitErr']:
        dfGrid[c] = np.nan

    for i in (dfGrid.index):#,desc='fitting with top %d start params'%topN):
        # print(loss(dfGrid.loc[i, ['sX', 'sY', 'sSigma']]))
        res = minimize(loss,
                       # , 'sSigrat', 'sTheta']],
                       dfGrid.loc[i, ['sX', 'sY', 'sSigma']],
                       bounds=bounds,
                       method='SLSQP',
                       options={'disp': False},
                       constraints=cons,
                       )
        dfGrid.loc[i, 'fitErr'] = res.fun
        for ii in range(len(resCols)):
            dfGrid.loc[i, resCols[ii]] = res.x[ii]

    # vErr = [[err,100][int(np.isnan(err))] for err in dfGrid['fitErr']]
    vNan = np.array([np.isnan(v) for v in dfGrid['fitErr']])
    idxBest = dfGrid['fitErr'] == np.min(dfGrid.loc[vNan==False,'fitErr'])
    idxBest = np.where(idxBest)[0][0]
    return dfGrid.loc[idxBest, resCols+['fitErr']]


# def xy_to_polar(x, y):
#     return np.arctan2(y, x)


# def xy_to_ecc(x, y):
#     return np.linalg.norm([x, y])
def polar2matrixCoords(r,theta):
    '''matrix coords have sign y flipped relative to cartesian.
    
     returns x,y'''
    x = r * np.cos(theta)
    y = r * np.sin(theta) * -1 # corect for matrix coords 
    return x,y

def matrixCoords2polar(x,y,returnPosTh=True):
    '''matrix coords have sign y flipped relative to cartesian.
    
     returns ecc, angle'''
    y *= -1 # corect for matrix coords 
    z = x + 1j * y
    ecc, angle = ( np.abs(z), np.angle(z) )
    if returnPosTh and angle<0:
        angle += np.pi*2
        
    return ecc, angle



# %% setup neural and bold timeseries based on stim and hrf:


mStim2d = rmStim.reshape(rmStim.shape[0] * rmStim.shape[1], rmStim.shape[2])
mPredBold2d = convolve_neural_boldPred(vHrf, mStim2d, nTr, frameRowPix=100)


#%% fit all runs
view=True
glmFit=True
mRes = np.zeros((mBold2d.shape[0],4))
mRes_new = np.zeros_like(mRes)
cs = ['X', 'Y', 'sigma', 'err', 'polarAngle', 'eccentricity','pearsonr','r2']
newcs = ['%s_new'%c for c in cs]
df = pd.DataFrame(columns=cs+newcs)

plt.close('all')
if view: 
    plt.figure(figsize=(10,10))
#%%
for idx,i in enumerate(np.random.permutation(range(0,mBold2d.shape[0]))):
    voxBold = stats.zscore(mBold2d[i, :])
    voxBold_new = stats.zscore(mBold2d_new[i, :])
    
    try:
        df.loc[i, ['X', 'Y', 'sigma', 'err']] = fitSL(mPredBold2d,
                                                        voxBold,
                                                        frameRowPix=100,
                                                        glmFit=glmFit).values
        
        df.loc[i, ['X_new', 'Y_new',
                   'sigma_new', 'err_new']] = fitSL(mPredBold2d,
                                                        voxBold_new,
                                                        frameRowPix=100,
                                                        glmFit=glmFit).values
    except:
        df.loc[i, ['X', 'Y', 'sigma', 'err']] = 0,0,0,100
        df.loc[i, ['X_new', 'Y_new', 'sigma_new', 'err_new']] = 0,0,0,100
    x, y, sigma, err = df.loc[i, ['X', 'Y', 'sigma', 'err']]
    x_new, y_new, sigma_new, err_new = df.loc[i, ['X_new', 'Y_new', 'sigma_new', 'err_new']]
    
    df.loc[i, ['eccentricity','polarAngle' ]] = matrixCoords2polar(x,y)
    df.loc[i, ['eccentricity_new','polarAngle_new' ]] = matrixCoords2polar(x_new,
                                                                           y_new)
    
    mRes[i, :2] = df.loc[i, ['polarAngle_new', 'eccentricity_new']]
    mRes_new[i, :2] = df.loc[i, ['polarAngle_new', 'eccentricity_new']]
    
    fitBoldPred = get_hrf(x_new, y_new, sigma_new, mPredBold2d, frameRowPix=100)
    fitBoldPred_new = get_hrf(x_new, y_new, sigma_new, mPredBold2d, frameRowPix=100)
    try:
        if glmFit:
            fitBoldPred = fitPred(fitBoldPred, voxBold)
            fitBoldPred_new = fitPred(fitBoldPred_new, voxBold_new)
    
        # slope, intercept, r_value, p_value, std_err = stats.linregress(fitBoldPred, voxBold)
        
        df.loc[i,'pearsonr'] = stats.pearsonr(fitBoldPred, voxBold)[0]
        df.loc[i,'pearsonr_new'] = stats.pearsonr(fitBoldPred_new, voxBold_new)[0]
        
        df.loc[i,'r2'] = stats.pearsonr(fitBoldPred, voxBold)[0]**2
        df.loc[i,'r2_new'] = stats.pearsonr(fitBoldPred_new, voxBold_new)[0]**2
        
        
        mRes[i,2] = df.loc[i,'r2']
        mRes[i,3] = df.loc[i,'pearsonr']
        
        
        mRes_new[i,2] = df.loc[i,'r2_new']
        mRes_new[i,3] = df.loc[i,'pearsonr_new']
            
        mnR2Og = np.mean(df.loc[:i,'r2'])
        mnR2New = np.mean(df.loc[:i,'r2_new'])
        
        zrhos = np.array([cl.arctanh(n) for n in mRes[:i,'pearsonr']])
        zrhos_new = np.array([cl.arctanh(n) for n in mRes[:i,'pearsonr']])
        
        t,p = stats.ttest_ind(zrhos,zrhos_new)
        print('%4d: mnOg=%.2f, mnNew = %.2f (t=%.2f, p=%.3f)'%(idx,mnR2Og,mnR2New,
                                                               t,p))
        df.to_csv(prfF+'/%s_prfs.csv'%'compare_%s'%preprocN)
        # imPol = dat.unmasker(mRes[:,0])
        # imEcc = dat.unmasker(mRes[:,1])
        # imR2 =  dat.unmasker(mRes[:,2])
        # imRho = dat.unmasker(mRes[:,3])
        
        # for im,n in zip([imPol,imEcc,imR2,imRho],
        #                 ['pol','ecc','r2','pearsonr']):
        #     outp = prfF+'/%s.nii.gz'%n
        #     im.to_filename(outp)
            
        if view:
            
            plt.clf()
            plt.subplot(2,2,1)
            plt.plot(voxBold)
            plt.plot(fitBoldPred)
            plt.legend(['data %s'%'all', 'fit %s'%'all'])
            plt.title('r2=%.2f'%df.loc[i,'r2'])
            plt.subplot(2,2,2)
            plt.plot(voxBold_new)
            plt.plot(fitBoldPred_new)
            plt.legend(['data %s'%'all', 'fit %s'%'all'])
            plt.title('r2=%.2f'%df.loc[i,'r2_new'])
        #     plt.hist(df.loc[:i,'r2'],bins=30)
        #     plt.title("r2's so far")
        #     ax=plt.subplot(2,2,3)
        #     nip.plot_stat_map(imPol,
        #                       bg_img=anatP,
        #                       axes=ax,
        #                       draw_cross=False,
        #                       title='polar coordinates',
        #                       cmap='hsv',
        #                       )
        #     ax=plt.subplot(2,2,4)
        #     nip.plot_stat_map(imEcc,
        #                       bg_img=anatP,
        #                       axes=ax,
        #                       draw_cross=False,
        #                       title='eccentricity')
        #     plt.suptitle('voxel %d/%d (%d percent)'%(i+1,
        #                                             dat.dat.shape[1],
        #                                              (1+i)/dat.dat.shape[1]))
        #     plt.show();plt.pause(.01)
            
    except: pass
        
    