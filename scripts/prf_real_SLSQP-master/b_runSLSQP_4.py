#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:36:42 2021

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
from joblib import Parallel, delayed
import nilearn.image as nli
from scipy.optimize import minimize
from scipy import stats
np.seterr(divide='ignore')
from scipy.signal import convolve2d


# if 'arwin' in sys.platform:
#     expF = '/Volumes/Untitled/macman_align'
#     fsl_b02b0 = '/Applications/fsl/etc/flirtsch/b02b0.cnf'
# else:
#     expF='/media/veracrypt1/macman_align'
#     fsl_b02b0 = '/usr/local/fsl/etc/flirtsch/b02b0.cnf'
expF='/misc/data8/braunlichkr/macman_align'
fsl_b02b0='/opt/fsl/etc/flirtsch/b02b0.cnf'


humScriptF = os.path.join(expF,'bin','preproc_hum')
sys.path.append(humScriptF); import hPreproc as hp
monkScriptF = os.path.join(expF,'bin','preproc_monk_og')
sys.path.append(monkScriptF); import mPreproc as mp
prfBinF = os.path.join(expF,'bin','first_lev','prf')
sys.path.append(prfBinF);

#%% setup
n_jobs=50
subj = ['sub-001','sub-003','sub-004','sub-005'][0]
# iSubjToRun = 0
species=['hum','monk'][0]
hrfToUse=['vista'][0]

derivF= os.path.join(expF,'data',species,'derivatives')
preprocF = os.path.join(derivF,'preproc')


maskType = ['ribbonSubCort','desikan'][1]
lDesikan = ['cuneus','latOccip']#,'latOccip','lingual','periCalc','fusiform']

if 'ribb' in maskType:
    anName = 'slsqp_%s_precompute'%hrfToUse
else:
    roisN = ''.join(lDesikan)
    anName = 'slsqp_%s_%s_precompute'%(hrfToUse,roisN)

subAnatF = os.path.join(preprocF,subj,'anat')
prfF = derivF+'/prf'
dataF = prfF+'/avg_clean_in_dsFsT1'
subAvgDataP = glob.glob(dataF+'/*%s*.nii.gz'%subj)[0]
roiAnF = os.path.join(prfF,'roi',anName)

csvResF = roiAnF+'/csvRes'
niiResF = roiAnF+'/niiRes'
surfResF = roiAnF+'/surfRes'
fsToCreate = [csvResF,niiResF,surfResF
              ]
for f in fsToCreate:
    if not os.path.exists(f):
        os.makedirs(f)
    
fsF = os.path.join(derivF,'fs')
    
imgCol = 'clean_in_dsFsT1'#'_dilatedRibbonP'

            
#%% plot hrfs 
plt.close('all')
realSpmHrf_TR2 = np.array([         0, # pulled from spm,not nilearn
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
vistaHrf_Tr2P = os.path.join(prfBinF,'hrf_vista_twogammas_TR2_absarea.csv')
os.path.exists(vistaHrf_Tr2P)
vistaHrf = np.loadtxt(open(vistaHrf_Tr2P, "rb"), delimiter=",", skiprows=0)
plt.plot(stats.zscore(realSpmHrf_TR2))
plt.plot(stats.zscore(vistaHrf))
plt.legend(['spm','vista'])
hrf = [realSpmHrf_TR2,vistaHrf]['vista' in hrfToUse]




# %% resample stim


viewStim = False
write=False
d = [100,256][0]
# og_afniStimP = '/Users/braunlichkr/Desktop/prf_sims/stim/stimMasks144Padded.nii'
# og_afniStimIm = nib.load(og_afniStimP)
# mAfni = np.swapaxes( og_afniStimIm.get_fdata(),0,1)
stimP = os.path.join(expF,'bin','first_lev','prf','PRF_mask_per_TR.npy')

iStart=10 # we cut out blank trs in beginning from stim and mri data to minimize processing
mStim = np.load(stimP)[::-1,:,iStart:]
# mStim = imStim.get_fdata()[::-1,:]

# previously gnerated 256*256 

# newAfniP = os.path.join(expF,'bin','first_lev','prf','stim_256x256.nii')
# showLast=True
# try: 
#     new256 = np.swapaxes( nib.load(newAfniP).get_fdata(),0,1)
# except:
#     showLast=False

# iStart = 10
nTr = mStim.shape[-1]  # skip trs before onset
rmStim = np.zeros((d, d, nTr))
# if viewStim:
#     plt.figure()
for i in range(nTr):
    im = mStim[:, :, i]
    rim = np.round(resize(im, (d, d), order=0))
    if 0:
        plt.subplot(1, 3, 1)
        plt.imshow(im,origin='lower')
        plt.title('200x200')
        plt.subplot(1, 3, 2)
        plt.imshow(rim,origin='lower')
        plt.title(rim.shape)
        plt.pause(.3)
        plt.clf()
        # if showLast:
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(new256[:,:,0,i],origin='lower')
        #     plt.title('afni space (256*256)\n loaded from file')
    # assert(len(np.unique(rim)) <= 2)
    rmStim[:, :, i] = rim
    # if viewStim:
    #     plt.suptitle(i)
    #     plt.pause(.5)
    #     plt.clf()
plt.close('all')

# rmStim = rmStim[:,:,iStart:]

# if swapStimAxes:
#     rmStim = np.swapaxes(rmStim,0,1)
# nTr-=iStart
plt.close('all')
if viewStim:
    '''    l_r,br_tl, t_b, bl_tr, r_l, tl_br, b_t, tr_bl
    '''
    plt.figure(figsize=(4,8))
    for i in range(rmStim.shape[-1]):
        plt.subplot(1,2,1)
        plt.imshow(rmStim[:,:,i],origin='lower')
        plt.title(i)
        
        plt.pause(.01)
        plt.clf()


# %%

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
    # mParam = np.array(list(it.product(xs, ys, sgs)))
    mXy = np.vstack([xs.T,ys]).T
    mXySg = np.zeros((mXy.shape[0],3))
    mXySg[:,:2] = mXy
    mTmpOg = mXySg.copy()
    mXySg = np.array([], dtype=np.int64).reshape(0,3)
    for sg in sgs:
        mTmp = mTmpOg.copy()
        mTmp[:,2] = sg
        mXySg = np.vstack([mXySg,mTmp])
    cols = ['sX', 'sY', 'sSigma']
    df = pd.DataFrame(index=range(len(mXySg)),
                      data=mXySg,
                      columns=cols)
    return df


reg = LinearRegression(positive=True)


def get_hrf(x, y, sigma, mPredBold2d, frameRowPix):
    imRf = gen_gauss(x, y, sigma, frameRowPix)
    voxHrfResp = stats.zscore(
        np.dot(imRf.reshape(frameRowPix**2), mPredBold2d))
    return voxHrfResp


def fitPred(vPred, targHrf,intercept_only=True):
    reg.fit(
        targHrf.reshape(len(vPred), -1),
        vPred.reshape(len(vPred), -1))
    if intercept_only:
        return vPred + reg.intercept_[0],reg
    else:
        return vPred * reg.coef_[0] + reg.intercept_[0], reg


def makePred(vPred,intercept,beta=None):
    if beta is None:
        return vPred+intercept
    else:
        return vPred*beta+intercept
    
# from scipy.signal import butter,filtfilt
# def butter_lowpass_filter(data, cutoff, fs, order):
#     nyq = 
#     normal_cutoff = cutoff / nyq
#     # Get the filter coefficients 
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

def makebatch(lst, n):
    return [lst[i::n] for i in range(n)]
def makebatch(a, n):
    k, m = divmod(len(a), n)
    return list((a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))


def createGridPredictions(dfGrid,mPredBold2d):
    
    nTr = rmStim.shape[-1]
    trCols = ['tr%.3d'%i for i in range(nTr)]
    dfGridPreds = pd.DataFrame(index=dfGrid.index,
                               columns=['x','y','sg']+trCols)
    frameRowPix = rmStim.shape[0]
    
    batches = makebatch(dfGrid.index, n_jobs)
    
    def runbatch(batch):
        dfTmp = pd.DataFrame(index=batch,columns=dfGridPreds.columns)
        for i in tqdm(batch,desc='precomputing grid predictions'):
            x,y,sg = dfGrid.loc[i,['sX', 'sY', 'sSigma']]
            hrf_ = get_hrf(x, y, sg, mPredBold2d, frameRowPix)
            dfTmp.loc[i,['x', 'y', 'sg',]] = dfGrid.loc[i,['sX', 'sY', 'sSigma',]].values
            dfTmp.loc[i,trCols] = hrf_
        return dfTmp
    ldf = Parallel(n_jobs=n_jobs)(delayed(runbatch)(batch) for batch in batches)
    
    dfGridPreds = pd.concat(ldf,verify_integrity=True)#.sort_index(drop=True)
    # for i in tqdm(dfGrid.index,desc='precomputing grid predictions'):
    #     x,y,sg = dfGrid.loc[i,['sX', 'sY', 'sSigma']]
    #     hrf_ = get_hrf(x, y, sg, mPredBold2d, frameRowPix)
    #     dfGridPreds.loc[i,trCols] = hrf_
        
    #     dfGridPreds.loc[i,['x','y','sg']] =  x,y,sg
    return dfGridPreds


def calcSseWithGrid(mGridPreds,voxTs):
    '''mGridPreds is dfGridPreds[trCols].values'''
    return np.sum((mGridPreds - voxTs)**2, axis=1)


def getGridFinalStartParams(dfGridPreds,voxTs,nKeep,nFit=None):
    trCols = [c for c in dfGridPreds.columns if 'tr' in c]
    dfGridPreds['sse'] = calcSseWithGrid(dfGridPreds[trCols].values,
                                        voxTs)
    idx = np.argsort( dfGridPreds['sse'])[:nKeep]
    
    if nFit !=None:
        dfFit = dfGridPreds.loc[idx,['x','y','sg','sse']].copy()
        # scale params
        cs = [c for c in dfGridPreds if not 'tr' in c]
        dfFit['sg']/= np.max(dfGridPreds['sg'])
        dfFit['eucs'] =    [np.linalg.norm(dfFit.loc[i,cs]) for i in dfFit.index]
        dfFit = dfFit.sort_values(by=['eucs'])
        idxFit = dfFit.index[-nFit:]
        if not idx[0] in idxFit:
            idxFit = np.hstack([idx[0],idxFit])
        return dfFit.loc[idxFit,['x','y','sg','sse']]
        
    else:
        return dfGridPreds.loc[idx,['x','y','sg','sse']]
    


def fit(mPredBold2d,voxTs,startParams,frameRowPix=100):
    def loss(params, targHrf=voxTs):
        x, y, sigma = params
        hrf_ = get_hrf(x, y, sigma, mPredBold2d, frameRowPix)
        return np.sum((targHrf-hrf_)**2)*.001
    
    bounds = ((-1., 1.),
              (-1., 1.),
              (.001, 2.),
              )
    def con(params):
        x,y,_ = params
        euc = np.linalg.norm([x,y])
        return float( 1.0-euc) # inequality constraint means that it is to be non-negative
    
    cons = {'type':'ineq','fun':con} 
    
    res = minimize(loss,
                   startParams[['x','y','sg']],
                   bounds=bounds,
                   method='SLSQP',
                   options={'disp': False},
                   constraints=cons,
                   )
    return res


def fitVoxelWithGridStarts(mPredBold2d,voxTs,dfGridStartParams,frameRowPix=100,plot=False):
    dfRes = pd.DataFrame(columns = ['sX','sY','sSg',
                                    'x','y','sg',
                                    'sse','R2'])
    dfRes[['sX','sY','sSg']] = dfGridStartParams[['x','y','sg']].copy()
    for i in dfGridStartParams.index:
        res = fit(mPredBold2d,
                  voxTs,
                  dfGridStartParams.loc[i,['x','y','sg']],
                  frameRowPix=frameRowPix)
        dfRes.loc[i,'sse'] = res.fun
        dfRes.loc[i,['x','y','sg']] = res.x
    idxBest = dfRes['sse']==np.min(dfRes['sse'])
    dfBest = dfRes.loc[idxBest,:].reset_index(drop=True)
    x,y,sg = dfBest.loc[0,['x','y','sg']]
    # rf = gen_gauss(x,y,sg)
    fitHrf = get_hrf(x,y,sg,mPredBold2d,frameRowPix)
    dfBest.loc[0,'R2'] = stats.pearsonr(voxTs,fitHrf)[0]**2
    if plot:
        plt.clf()
        plt.plot(voxTs)
        plt.plot(fitHrf)
        plt.legend(['real','fit'])
        plt.title('R2=%.2f'%dfBest.loc[0,'R2'] )
        plt.show(); plt.pause(1)
    return dfBest.loc[0,:]

# def xy_to_polar(x, y):
#     return np.arctan2(y, x)


# def xy_to_ecc(x, y):
#     return np.linalg.norm([x, y])
def polar2cartesian(r,theta):
    '''cartesian coords have sign y flipped relative to cartesian.
    
      returns x,y'''
    x = r * np.cos(theta)
    y = r * np.sin(theta) #* -1 # corect for cartesian coords 
    return x,y

def cartesian2polar(x,y,returnPosTh=False):
    '''cartesian coords have sign y flipped relative to cartesian.
    
      returns ecc, angle'''
    # y *= -1 # corect for cartesian coords 
    z = x + 1j * y
    ecc, angle = ( np.abs(z), np.angle(z) )
    if returnPosTh and angle<0:
        angle += np.pi*2
        
    return ecc, angle
def cart2pol(x, y):
    ecc = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(ecc, theta)

def pol2cart(ecc, theta):
    x = ecc * np.cos(theta)
    y = ecc * np.sin(theta)
    return(x, y)

def convert_xy_to_colrow(x0,y0,framerowpix=rmStim.shape[0]):
    
    x = np.round((x0+1)/2 * framerowpix).astype(int)
    y = np.round((y0+1)/2 * framerowpix).astype(int)
    return x,y

# def visStim(mStim,x0,y0,voxHrf):
#     '''
#     l_r,br_tl, t_b, bl_tr, r_l, tl_br, b_t, tr_bl
    
#     left to right, bottom right to top left, top to bottom,
#     bottom left to top right, right to left, top left to bottom right,
#     bottom to top, and top right to bottom left.'''
#     plt.figure(figsize=(12,4))
    
#     for tr in range(0,mStim.shape[-1]):
#         plt.subplot(1,2,1)
#         plt.imshow(mStim[:,:,tr],origin='lower')
#         col,row = convert_xy_to_colrow(x0, y0)
#         plt.scatter(col,row,s=100,c='r',marker='x')
        
#         plt.scatter(col,row,s=100,c='r',marker='x')
        
#         plt.subplot(1,2,2)
#         plt.plot(np.arange(mStim.shape[-1]),voxHrf)
#         plt.axvline(tr,linestyle='dashed',c='k')
#         plt.tight_layout()
#         plt.pause(.3)
#         plt.clf()
        

def visStim(mStim,mRf,voxHrf):
    '''
    l_r,br_tl, t_b, bl_tr, r_l, tl_br, b_t, tr_bl
    
    left to right, bottom right to top left, top to bottom,
    bottom left to top right, right to left, top left to bottom right,
    bottom to top, and top right to bottom left.'''
    plt.figure(figsize=(12,4))
    
    for tr in range(0,mStim.shape[-1]):
        plt.subplot(1,2,1)
        mrf_bar = mRf.copy()
        mrf_bar[mStim[:,:,tr]>.5]=np.max(mRf)
        plt.imshow(mrf_bar,origin='lower')
                
        plt.subplot(1,2,2)
        plt.plot(np.arange(mStim.shape[-1]),voxHrf)
        plt.axvline(tr,linestyle='dashed',c='k')
        plt.tight_layout()
        plt.pause(.3)
        plt.clf() 
        

   
# %% setup neural and bold timeseries based on stim and hrf:

# mStim2d = rmStim.reshape(rmStim.shape[0] * rmStim.shape[1], rmStim.shape[2])
mStim2d = np.squeeze(rmStim).reshape(np.prod(rmStim.shape[:2]), 
                                     rmStim.shape[-1])

dBoldPred2d = {}
mPredBold2d = convolve_neural_boldPred(hrf, 
                                        mStim2d, 
                                        nTr, 
                                        frameRowPix=rmStim.shape[0])

#%% test rf, stim correspondence:

x0, y0 = -.5, -.5
sg = .05
ecc, theta = cart2pol(x0, y0)
print('ecc = %.2f, pol = %.2f'%(ecc,theta))
mRf = gen_gauss(x0,y0,sg,rmStim.shape[0])

voxHrf = get_hrf(x0,
                 y0,
                 sg,
                 mPredBold2d, 
                 rmStim.shape[0])

plt.close('all')
visStim(rmStim, mRf, voxHrf)
'''    l_r,br_tl, t_b, bl_tr, r_l, tl_br, b_t, tr_bl
'''
plt.close('all')


#%%  setup grid search
gridPredsP = prfF+'/stimulusGridPredictions_skipFirstTrs_evenLocSpace2.csv'
if os.path.exists(gridPredsP):
    dfGridPreds = pd.read_csv(gridPredsP,index_col=0)
    dfGridPreds = dfGridPreds.reset_index(drop=True)
else:
    
    # eccs,pols = list(np.linspace(.0,.95,10)), np.linspace(-3.14,3.14,33)[:-1]#[:-1]
    
    # setup so that space is evenly samples at each ecc
    def arc_to_theta(arcLength,radius):
        return arcLength/radius
    
    # def circ_to_arcLength(circumference,arcLengthAt1):
    #     return circumference/arcLengthAt1
        
    def getThetas(radius,arcLengthAt1):
        # circumference = 2 * np.pi * radius
        # arcLength =arcLengthAt1# circ_to_arcLength(circumference,arcLengthAt1)
        thetaStepSize = arc_to_theta(arcLengthAt1,radius)
        return np.arange(-3.14,3.14,thetaStepSize)
        
    eccs = list(np.linspace(.0,.95,10))
    dPol = {}
    nPolAtEcc1 = 4 # a tthe inner most ring, we wminunum
    circAtOne = 2*np.pi*eccs[1]
    arcLengthAt1 = circAtOne/nPolAtEcc1
    
    for ecc in eccs[1:]:
        dPol[ecc] = getThetas(ecc,arcLengthAt1)
    
    locs = [[0,0]]
    for ecc in dPol.keys():
        for pol in dPol[ecc]:
            locs.append([ecc ,pol])

    
    # locs = list(it.product(eccs,pols))
    
    mXy = np.array([pol2cart(*loc) for loc in locs])
    xs,ys = mXy[:,0],mXy[:,1]
    dfXy = pd.DataFrame(mXy,columns=['x','y'])

    sgs =  np.arange(.001,1.1,.0025)
    
    dfGrid = make_param_grid(xs, ys, sgs)
    idxKeep = dfGrid.duplicated(keep='first')==False
    dfGrid = dfGrid.loc[idxKeep,:].reset_index(drop=True)
    
    print('-----------------\nN grid params:',
          dfGridPreds.shape[0],
          '\n--------------------------')
    idxKeep = np.ones(dfGridPreds.shape[0]).astype(bool)#dfGridPreds[['sX','sY']].duplicated(keep='first')
    plt.figure()
    plt.scatter(dfGridPreds.loc[idxKeep,'x'],
                dfGridPreds.loc[idxKeep,'y'],
                s=1)
    
    dfGridPreds = createGridPredictions(dfGrid,mPredBold2d)
    dfGridPreds.to_csv(gridPredsP)
    
#%%    


#%% test rf, stim correspondence:
i = np.random.permutation(dfGridPreds.index)[0]
x0, y0,sg = dfGridPreds.loc[i,['x','y','sg']]
# sg = .05
ecc, theta = cart2pol(x0, y0)
print('ecc = %.2f, pol = %.2f'%(ecc,theta))
mRf = gen_gauss(x0,y0,sg,rmStim.shape[0])

voxHrf = get_hrf(x0,
                 y0,
                 sg,
                 mPredBold2d, 
                 rmStim.shape[0])

plt.close('all')
visStim(rmStim, mRf, voxHrf)
'''    l_r,br_tl, t_b, bl_tr, r_l, tl_br, b_t, tr_bl
'''
plt.close('all')
#%% test (and time) fit for 1 synthetic voxel:
nKeep=5
# nFit = 5
i = np.random.permutation(dfGridPreds.index)[0]
trCols = [c for c in dfGridPreds if 'tr' in c]
voxTs = dfGridPreds.loc[i,trCols].values
ground = dfGridPreds.loc[i,['x', 'y', 'sg']]

# the key function we use below: 
def fitVox(voxTs,mPredBold2d=mPredBold2d, dfGridPreds=dfGridPreds,
           nKeep=nKeep,  frameRowPix=100):
    
    dfGridStartParams = getGridFinalStartParams(dfGridPreds,voxTs,nKeep,nFit=None) # if nFit !=None, strives to increase parameter variability
    res= fitVoxelWithGridStarts(mPredBold2d,
                                voxTs,
                                dfGridStartParams,
                                frameRowPix=100,
                                plot=False)
    return res
%timeit fitVox(voxTs)


res=fitVox(voxTs)
x,y,sg = res[['x','y','sg']]
ecc,pol = cart2pol(x,y)
print('ground:\n',ground)
print('res:',np.round([x,y,sg],2))
# rf = gen_gauss(x,y,sg,100).T
# plt.imshow(rf,origin='lower')
            

#%% load averaged subject data

# subj = subjs[iSubjToRun]
dSubP={}
dSubP['fsF'] = os.path.join(fsF,subj)
fwhm=2
view=True

# avgP = avgF+'/%s_fwhm%.2d_fullRibbon.nii.gz'%(subj,fwhm)
avgP = subAvgDataP
bg_img = glob.glob(fsF+'/%s/mri/dsT1.nii.gz'%subj)[0]

ribbonMaskP = fsF+'/%s/mri/bi.ribbon.nii.gz'%subj# ribbon_and_subcort.mgz

ribbonMaskP = fsF+'/%s/mri/ribbon_and_subcort.nii.gz'%subj
if not os.path.exists(ribbonMaskP):
    ribbonMaskP = hp.mgz_to_niigz(ribbonMaskP.replace('.nii.gz','.mgz'))
    
dDesikan = {'cuneus':  {'lh': 1005,'rh':2005},
            'fusiform' :{'lh': 1007 ,'rh': 2007},
            'inferiorParietal': {'lh': 1008,'rh': 2008 },
            'superiorParietal': {'lh': 1029,'rh': 2029 },
            'insula':   {'lh': 1035,'rh': 2035},
            'latOccip': {'lh':1011,'rh':2011},
            'lingual': {'lh':1013,'rh':2013},
            'periCalc': {'lh':1021,'rh':2021},
            'precuneus':{'lh':1025,'rh':2025},
            'banksts':{'lh':1001,'rh':2001},
            'supramarginal':{'lh':1031,'rh':2031},
            'inferiortemporal':{'lh':1009,'rh':2009},
            'middletemporal':{'lh':1015,'rh':2015},
            'superiortemporal':{'lh':1030,'rh':2030},
            
            }

# hem = 'lh'
lDesikan = dDesikan.keys()

atlasVals = [dDesikan[r][h] for r,h in it.product(lDesikan,['lh','rh'])]


fsSubF = os.path.join(fsF,subj)
desikanMgzP = fsSubF+'/mri/aparc+aseg.mgz'#)[0]
if not os.path.exists(desikanMgzP):
    # use ants-warped dk atlas
    desikanMgzP = glob.glob(subAnatF+'/desikan_killiany_in_native.nii.gz')[0]
imDes = nib.load(desikanMgzP)
mDes = imDes.get_fdata()
# mMask = (mDes==dDesikan[roi]['lh']).astype(np.float32)
mMask = np.zeros_like(mDes)
for atlasVal in atlasVals:
    mMask +=  (mDes==atlasVal).astype(np.float32)
mMask[mMask>1] = 1.
# mMask += (mDes==dDesikan[roi]['rh']).astype(np.float32)
imMask = nib.Nifti1Image(mMask,imDes.affine)
maskP = roiAnF+'/%s_desikan_%s_mask.nii.gz'%(subj, ''.join(lDesikan))
imMask.to_filename(maskP)

desP = fsSubF+'/mri/aparc+aseg.mgz:colormap=lut:opacity=0.4'
# hp.freeview([bg_img,desP,maskP])


from subprocess import call
if 0:
    # view desikan atlas in subject space:
    # fsSubF = os.path.join(dP['hum']['fsF'],subj)
    cmd ='''freeview -v fsSubF/mri/orig.mgz \
    fsSubF/mri/aparc+aseg.mgz:colormap=lut:opacity=0.4 \
    -f fsSubF/surf/lh.white:annot=aparc.annot'''

    cmd = cmd.replace('fsSubF',fsSubF)
    print(cmd)
    call(cmd,shell=True)

#%%
print('loading %s'%subj)
dat = cl.fmri_data([avgP],maskP)
mBold2d = dat.dat.T
nVox = mBold2d.shape[0]
print('nVox=%d'%nVox)
mBold2d = mBold2d[:,iStart:]

mnBold =mBold2d.mean(axis=1)
imMn = dat.unmasker(mnBold)

# nip.plot_stat_map(imMn,bg_img=bg_img)
# cl.freeview([bg_img,ribbonMaskP])

#%%
plt.close('all')

cs = ['x', 'y', 'sg', 'sse','R2','pearsonr',
              'polarAngle', 'eccentricity',]
# dfRes = pd.DataFrame(np.zeros((nVox,len(cs))),
#                      columns=cs)
# dfRes.values = 
plt.close('all')
if view: 
    plt.figure(figsize=(7,5))


#%%
nKeep=5
plt.close('all')
plt.figure(figsize=(15,10))
# def fitBatch(rows):
#     # _dfRes = dfRes.loc[rows,:]
#     _dfRes = pd.DataFrame(index=rows,columns=cs)
#     for i in tqdm(rows):
#         voxTs = stats.zscore(mBold2d[i, :])
#         res = fitVox(voxTs,mPredBold2d=mPredBold2d, dfGridPreds=dfGridPreds,
#                    nKeep=nKeep,  frameRowPix=100)
#         _dfRes.loc[i,['x','y','sg','sse','R2']] = res[['x','y','sg','sse','R2']]
    
#         _dfRes.loc[i, ['eccentricity','polarAngle' ]] = cartesian2polar(res['x'],
#                                                                        res['y'])
#     return _dfRes


    
import colorcet as cc
cc_cmap = cc.cm.cyclic_ymcgy_60_90_c67_s25
def preview(imPol,imEcc,imR2,imSg,bg_img=bg_img):
                
    # for polar, an alternative to wilight is 
    plt.clf()
    ax=plt.subplot(4,1,1)
    nip.plot_stat_map(imR2,
                      bg_img=bg_img,
                      axes=ax,
                      display_mode='z',
                      cut_coords=np.linspace(-20,20,10),
                      draw_cross=False,
                      title='R2')
    ax=plt.subplot(4,1,2)
    nip.plot_stat_map(imPol,
                      bg_img=bg_img,
                      axes=ax,
                      display_mode='z',
                      cut_coords=np.linspace(-20,20,10),
                      draw_cross=False,
                      title='polar coordinates',
                      cmap=cc_cmap,#'twilight',
                      )
    ax=plt.subplot(4,1,3)
    nip.plot_stat_map(imEcc,
                      bg_img=bg_img,
                      axes=ax,
                      display_mode='z',
                      cut_coords=np.linspace(-20,20,10),
                      draw_cross=False,
                      title='eccentricity')
    ax=plt.subplot(4,1,4)
    nip.plot_stat_map(imSg,
                      bg_img=bg_img,
                      axes=ax,
                      display_mode='z',
                      cut_coords=np.linspace(-20,20,10),
                      cmap='viridis',
                      vmax=.5,
                      draw_cross=False,
                      title='sigma')

rows = np.arange(nVox)
csvResP = csvResF+'/%s.csv'%subj
m = np.zeros((len(rows),len(cs)))
dfRes =  pd.DataFrame(m,index=rows,columns=cs)
#%%
def runbatch(rs):
    dfTmp = pd.DataFrame(index=rs,columns=dfRes.columns)
    for rw in tqdm(rs,desc='fitting %s'%subj):
        voxTs = stats.zscore(mBold2d[rw, :])
        try:
            res = fitVox(voxTs,mPredBold2d=mPredBold2d, dfGridPreds=dfGridPreds,
                       nKeep=nKeep,  frameRowPix=100)
            dfTmp.loc[rw,['x','y','sg','sse','R2']] = res[['x','y','sg','sse','R2']]
            
            dfTmp.loc[rw, ['eccentricity','polarAngle' ]] = cartesian2polar(res['x'],
                                                                       res['y'])
        except:
            pass
    return dfTmp


batches = makebatch(rows, n_jobs)
ldf = Parallel(n_jobs=n_jobs)(delayed(runbatch)(batch) for batch in batches)
dfRes = pd.concat(ldf,verify_integrity=True)

if 0: # run serially
    for row in tqdm(rows,desc='fitting %s'%subj):
        voxTs = stats.zscore(mBold2d[row, :])
        try:
            res = fitVox(voxTs,mPredBold2d=mPredBold2d, dfGridPreds=dfGridPreds,
                       nKeep=nKeep,  frameRowPix=100)
            dfRes.loc[row,['x','y','sg','sse','R2']] = res[['x','y','sg','sse','R2']]
            
            dfRes.loc[row, ['eccentricity','polarAngle' ]] = cartesian2polar(res['x'],
                                                                       res['y'])
        except:
            pass
        if (row>0) and (np.mod(row,500)==0):
            print('plottig')
            preview(dat.unmasker(dfRes['polarAngle'].values.astype(np.float32)),
                 dat.unmasker(dfRes['eccentricity'].values.astype(np.float32)),
                 dat.unmasker(dfRes['R2'].values.astype(np.float32)), #imR2,
                 dat.unmasker(dfRes['sg'].values.astype(np.float32)),
                 bg_img=bg_img)
            plt.suptitle('%s\n%d vox'%(subj,row))
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show();plt.pause(.01)
    
            dfRes.to_csv(csvResP)
            
            l
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show();plt.pause(.01)
# df values to nii
cs = ['sse', 'R2', 'sg', 'polarAngle', 'eccentricity']
for c in cs:
    im = dat.unmasker(dfRes[c].values.astype(np.float32))
    fname = niiResF+'/%s_%s.nii.gz'%(subj,c)
    im.to_filename(fname)
# n_jobs=6
# rowss = cl.divideBatch(np.arange(nVox),n_jobs)
# Parallel(n_jobs=2)(delayed(fitBatch)(rows) for rows in rowss)

#%% to surface

cs = ['sse', 'R2', 'sg', 'polarAngle', 'eccentricity']
for c in cs:
    if c=='maskP':
        niiP = maskP
    else:
        niiP = glob.glob(niiResF+'/%s_%s.nii.gz'%(subj,c))[0]
        plt.close('all')
    
    for hem in ['lh','rh']:
        surfP = surfResF+'/%s_%s_%s.mgz'%(subj,c,hem)
        
        os.environ['SUBJECTS_DIR'] = fsF
        scripF = humScriptF #= os.path.join(expF,'bin','preproc_hum')
        cmd = '''/opt/freesurfer-730/bin/mri_vol2surf \
                --mov movP  \
                --projdist-avg 0 1 0.1 \
                --regheader SUBJN \
                 --surf-fwhm FWIDTHHM \
                 --surf white \
                --hemi HEMIS \
                --out outMgzP
                '''
        cmd = cmd.replace('movP',niiP).replace('outMgzP',surfP)
        cmd = cmd.replace("HEMIS",hem).replace('SUBJN',subj)
        cmd = cmd.replace('FWIDTHHM','0')
        print(cmd)
        
        call(cmd,shell=True)

#%%
plt.close('all')
fsInflPs = {'%sh'%h:glob.glob(os.path.join(fsF,subj,'surf','%sh.inflated'%h))[0] for h in ['l','r']}
fsSulcPs ={'%sh'%h:glob.glob(os.path.join(fsF,subj,'surf','%sh.sulc'%h))[0] for h in ['l','r']}

cmap = ['twilight', 'hsv',cc.cm.colorwheel][-1]
srchStr = 'pol'
r2Thresh = .2
dThr = {'ecc':[0,1],
        'pol':[-3.14,3.14],
        'R2':[0,1],
        'sg':[0,1]}

dR2Mask = {}

for hem in ['lh','rh']:
    stat_map = glob.glob(surfResF+'/*%s*%s*.mgz'%('R2',hem))[0]
    imR2 =  nib.MGHImage.from_filename(stat_map)
    mMask = (imR2.get_fdata().squeeze()>r2Thresh).astype(np.float32)
    dR2Mask[hem] = mMask #nib.MGHImage(mMask, imR2.affine)

for hem in ['lh','rh']:
    
    stat_map = glob.glob(surfResF+'/*%s*%s*.mgz'%(srchStr,hem))[0]
    statIm = nib.MGHImage.from_filename(stat_map)
    mStat = statIm.get_fdata().squeeze()
    mStat[dR2Mask[hem]==0]=0
    thrStatIm = nib.MGHImage(mStat.astype(np.float32), statIm.affine)
    
    threshP = surfResF+'/%s_%s_r2Thr%.2f.mgz'%(srchStr,hem,r2Thresh)
    threshP = threshP.replace('r0.','r0-')
    thrStatIm.to_filename(threshP)
    
    
    fig=plt.figure(figsize=(7,7))
    nip.plot_surf_stat_map(fsInflPs[hem],
                           threshP,
                            hemi=['left','right'][hem=='rh'],
                            view='posterior',
                            threshold=[.0001,.05]['R2' in srchStr],
                            bg_map=fsSulcPs[hem],
                            figure=fig,
                            # vmin =[None,.2]['R2' in srchStr],
                            # vmax = dThr[srchStr][1],
                            cmap=['cold_hot',cmap ]['pol' in srchStr],
                            title=srchStr,
                            )
#%% mask
maskLabelP = surfResF+'/%s'%os.path.basename(maskP)
maskLabelP= maskLabelP.replace('.nii.gz','.label')
cmd = ['mri_vol2label --c ',
       '%s --id 1'%maskP,
       '--l %s'%maskLabelP]
cmd = ' '.join(cmd)
print(cmd)
call(cmd, shell=True)
#%% freesurfer
''' I need to fix these display props to A) show thresholded, B) circular for
    polar'''

if 0:
    showAllContrasts = int(True)
    hems = [['lh'],['rh'],['lh','rh']][2]
    addWhite = False
    addPial = False
    cs= [c for c in cs if (not 'pear' in c) and (not 'mask' in c)]
    # surfResF = firstLevResF+'/surf_zmaps'
    dThresh = {'eccentricity':[0,1],
               'R2':[.2,1],
               'sse':[0,100],
               'polarAngle':[-np.pi,np.pi],
               'sg':[0,1.2],
        }
    
    for hem in hems:
        lCmd = ['freeview',
              '-f',
              ]
        if addWhite:
            lCmd+=['%s/surf/%s.white:visible=0'%(dSubP['fsF'],hem)]
            
        if addPial:
            lCmd+=['%s/surf/%s.pial:visible=0'%(dSubP['fsF'],hem)]
    
        for c in cs:
            threshs = dThresh[c]
            surfP = glob.glob(surfResF+'/%s_%s_%s.mgz'%(subj,c,hem))[0]
            lCmd+= ['%s/surf/%s.inflated:curvature=%s.avg_curv:overlay=%s:overlay_threshold=%s,%s::name=%s:visible=1'%(dSubP['fsF'],
                                                                                       hem,
                                                                                       hem,
                                                                                       surfP,
                                                                                       threshs[0],
                                                                                       threshs[1],
                                                                                       c+'_%s'%hem)]
        lCmd+=['--view lateral']
            
        lCmd+=['-colorscale',' --viewport 3d']
        # lCmd+=['--screenshot SUBJ_%name.jpeg 2 '.replace("SUBJ",subj)]
        cmd = ' '.join(lCmd)
        print(cmd)
    
        # os.chdir(summaryF)
        call(cmd,shell=True)