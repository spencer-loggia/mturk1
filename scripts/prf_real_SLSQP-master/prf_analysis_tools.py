#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:56:54 2021

@author: braunlichkr
"""
import numpy as _np
import itertools as it
import pandas as _pd
from subprocess import call
from scipy.optimize import minimize
from scipy import stats as _stats
import os as _os
from scipy.signal import find_peaks,convolve2d, peak_widths

from nilearn.glm.first_level import hemodynamic_models as hrf_mods
#%%
def convolve_matrix_neural_bold(mStim2d,vHrf,TR,nTr,stimFrameShape=(100,100),
                                hrf='spm',flatten=True):
    '''convolve binary mStim ("neural signal") with spm hrf.

    return 2d mtrx: vox*time
    '''
    # mBold_ = _np.zeros((mStim.shape[0],mStim.shape[1],1,nDim4))
    # for ix in (range(mStim.shape[1])):
    #     for iy in range(mStim.shape[0]):
    #         vNeural = mStim[int(iy),int(ix),0:].flatten()
    #         vBold = convolve(vNeural,TR,nDim4,plot=False)
    #         mBold_[int(iy),int(ix),0,:] = vBold

    # stimSz = mStim.shape[0]
    # mStim_2d = mStim.reshape(stimSz*stimSz,mStim.shape[3])
    # maxSpm = hrf_mods.spm_hrf(TR,oversampling=1).max()
    # if 'spm' in hrf:
    #     vHrf_ = hrf_mods.spm_hrf(TR,oversampling=1)
    # elif type(hrf)==str:
    #     vHrf_ = hrf_mods.gamma.pdf(range(30),7.5)[::TR]
    #     vHrf_ = vHrf_* (maxSpm/_np.max(vHrf_))
    # else:
    #     vHrf_ = hrf* (maxSpm/_np.max(hrf))
    # if type(hrf)==str:
    #     vHrf_ = vHrf_.reshape(1,len(vHrf_))
    if len(vHrf.shape)==1:
        vHrf=vHrf.reshape(1,len(vHrf))
    mBold = convolve2d(mStim2d,vHrf)[:,:nTr].reshape(stimFrameShape[0],
                                                     stimFrameShape[1],
                                                     nTr)
    if not flatten:
        return mBold
    else:
        return mBold.squeeze().reshape(_np.prod(stimFrameShape),nTr)

def make_param_grid(xs,ys,sgs,rs,ths):
    lParam = list(it.product(xs,ys,sgs,rs,ths))
    cols = ['sX','sY','sSigma','sSigrat','sTheta']
    df =  _pd.DataFrame(index=range(len(lParam)),
                           data=lParam,
                           columns=cols)

    idxCut = _np.logical_and(df['sSigrat']==1,df['sTheta']!=ths[0])
    return df.loc[idxCut==False,:].reset_index(drop=True)


def normVoxHrfResp(voxHrfResp):
    voxHrfResp_dm = voxHrfResp - _np.mean(voxHrfResp)
    voxHrfResp_sd = voxHrfResp_dm /_np.max(voxHrfResp_dm)
    return voxHrfResp_sd


def gen_ellipse(x0,y0,sigma,sigrat,theta,frameRowPix):
    ''' From AFNI's 3dNLfim doc:

    %                 e^-(A(x-x0)^2 + 2*B(x-x0)(y-y0) + C(y-y0)^2), where
    %
    %                      cos^2(theta)     sin^2(theta)
    %                  A = ------------  +  ------------
    %                       2sigma_x^2       2sigma_y^2
    %
    %                        sin(2theta)     sin(2theta)
    %                  B =   -----------  -  -----------     (signs mean CCW rot)
    %                        4sigma_x^2      4sigma_y^2
    %
    %                      sin^2(theta)     cox^2(theta)
    %                  C = ------------  +  ------------
    %                       2sigma_x^2       2sigma_y^2
    %
    %             Substituting sigma_x = R*sigma_y, sigma_y = sigma yields,
    %
    %                      cos^2(theta) + R^2*sin^2(theta)
    %                  A = -------------------------------
    %                                2*R^2sigma^2
    %
    %                                sin(2theta)
    %                  B = (1-R^2) * -----------
    %                                4*R^2sigma^2
    %
    %                      sin^2(theta) + R^2*cos^2(theta)
    %                  C = -------------------------------
    %                                2*R^2sigma^2

    '''
    A = (_np.cos(theta)**2 + sigrat**2 * _np.sin(theta)**2) / (2 * sigrat**2 * sigma**2)
    B = (1 - sigrat**2) * ((_np.sin(2*theta)) / (4 * sigrat**2 * sigma**2)  )
    C = (_np.sin(theta)**2 + sigrat**2 * _np.cos(theta)**2 ) / (2 * sigrat**2 * sigma**2)

    _x = _np.linspace(-1,1,frameRowPix+1)
    step = _x[1]-_x[0]
    import numpy as ognp
    xs,ys = ognp.mgrid[-1:1:step,-1:1:step]

    pos = _np.dstack((xs,ys))
    mX = pos[:,:,1]
    mY = pos[:,:,0]

    return _np.exp(-(A*(mX-x0)**2 + 2 * B * (mX-x0) * (mY-y0) + C * (mY-y0)**2))



def get_hrf(x,y,sigma,sigrat,theta,mBold_2d,frameRowPix):
    imRf = gen_ellipse(x,y,sigma,1,0,frameRowPix)
    voxHrfResp = _stats.zscore(normVoxHrfResp(_np.dot(imRf.reshape(frameRowPix**2),mBold_2d )))
    return voxHrfResp


def fitRf_noRf(voxBold,mBold_2d,stimP,outRf,anMethod,vHrf,frameRowPix=100,
          anTag=None,TR=2):
    '''anMethod= "AFNI" or "SLQP"'''
    # if 'gamma' in hrf_type:
    #     print('USING Glover instead of vista_twogammas')
    #     vHrf = hrf_mods.glover_hrf(TR,oversampling=1)
    # elif 'spm' in hrf_type:
    #     vHrf = hrf_mods.spm_hrf(TR,oversampling=1)
    # print(voxBold.shape,mBold_2d.shape,imRf.shape,stimP,inRf,outRf,anMethod)
    # cl.keyboard()
    if 'AF' in anMethod:
        sRes = run_afni(voxBold, stimP,outRf,TR,anTag,vHrf)
        if 'ell' in outRf:
            X,Y,sigma,sigrat,theta = sRes[['X','Y','sigma','sigrat','theta']]
            # if theta>=0: # afni has theta=0 as vertical (a rotation of np.pi/2)
            #     theta += _np.pi/2
            # else:
            #     theta -= _np.pi/2
        elif 'gau' in outRf:
            X,Y,sigma = sRes[['Y','X','sigma']]
            sigrat,theta = 1,0

    elif 'SL' in anMethod:
        def loss(params,targHrf=voxBold):
           x,y,sigma,sigrat,theta = params
           hrf_ = get_hrf(x,y,sigma,sigrat,theta,mBold_2d,frameRowPix)
           return _np.sum((targHrf-hrf_)**2)

        bounds = ((-1.,1.),
                  (-1.,1.),
                  (.0,5.),
                  (1.,1.),
                  (0, 0),
                    )

        # setup grid search
        topN = 10
        xs,ys = _np.linspace(-1,1,10), _np.linspace(-1,1,10)
        sgs = _np.linspace(.05,.4,15)
        rs = [1]#_np.linspace(1,5,15)
        ths = [0]#_np.linspace(-_np.pi/2, _np.pi/2,16)[:-1]
        dfGrid = make_param_grid(xs,ys,sgs,rs,ths)

        dfGrid['err'] = _np.nan
        for i in (dfGrid.index):#,desc='initial grid search'):
            dfGrid.loc[i,'err'] = loss(dfGrid.loc[i,dfGrid.columns[:-1]])

        dfGrid= dfGrid.sort_values('err',
                           ascending=True,
                           kind='mergesort',
                           ignore_index=True)
        dfGrid = dfGrid.loc[:topN,:]


        if 'gau' in outRf:
            bounds = list(bounds)
            bounds[3] = (1,1) # sigrat
            bounds[4] = (0,0) # theta
            bounds = tuple(bounds)

        resCols = ['rX','rY','rSg','rR','rTh']
        for c in resCols+['fitErr']:
            dfGrid[c] = _np.nan

        for i in (dfGrid.index):#,desc='fitting with top %d start params'%topN):
            res = minimize(loss,
                      dfGrid.loc[i,['sX', 'sY', 'sSigma', 'sSigrat', 'sTheta']],
                      bounds=bounds,
                      method='SLSQP',
                      options={'disp':True})
            dfGrid.loc[i,'fitErr'] = res.fun
            for ii in range(len(resCols)):
                dfGrid.loc[i,resCols[ii]] = res.x[ii]


        idxBest = dfGrid['fitErr'] == _np.min(dfGrid['fitErr'])
        idxBest = _np.where(idxBest)[0][0]
        X,Y,sigma,sigrat,theta = dfGrid.loc[idxBest,resCols]
    else:
        raise Exception

    return X,Y,sigma, sigrat,theta


def run_afni(voxBold,stimP,inRf,outRf,TR,anTag,vHrf):
    '''vHrf is e.g, hrf_mods.spm_hrf(TR,oversampling=1)'''
    # hrfP = 'spm_hrf_TR2.1D'
    hrfP = 'hrf.1D'
    # vHrf = hrf_mods.spm_hrf(TR,oversampling=1)
    # if anTag is None:
    #     anTag = _random_string()
    # anTag
    # afniTmpF = '%s/res/_tmp_afni'%expF
    afniTmpF = 'res/_tmp_afni'#%expF
    # makeFs([afniTmpF])
    voxBoldP = '%s/%s.1D'%(afniTmpF,anTag)
    buckP = '%s/Buck_%s.PRF'%(afniTmpF, anTag)
    _np.savetxt(hrfP,
               vHrf)#.reshape(1,len(vHrf)),
               # delimiter='  ')
    _np.savetxt(voxBoldP,
               voxBold.reshape(1,len(voxBold)),
               delimiter='  ')

    setupCmd = 'setenv AFNI_CONVMODEL_REF hrfP;setenv AFNI_MODEL_PRF_STIM_DSET stimP;setenv AFNI_MODEL_PRF_ON_GRID NO;setenv AFNI_MODEL_DEBUG 0;setenv AFNI_NIFTI_TYPE_WARN NO'
    setupCmd = setupCmd.replace('hrfP',hrfP).replace('stimP',stimP)

    if 'ell' in outRf:
        cmd = '3dNLfim -input inP -noise Zero -signal Conv_PRF_6 -sconstr 0 -10 10  -sconstr 1 -1 1  -sconstr 2 -1 1 -sconstr 3 0 1  -BOTH -nrand 1000 -nbest 1 -bucket 0 buckP -snfit snfitP -jobs 1';
    else:
        cmd = '3dNLfim -input inP -noise Zero -signal Conv_PRF -sconstr 0 -10 10  -sconstr 1 -1 1  -sconstr 2 -1 1 -sconstr 3 0 1  -BOTH -nrand 1000 -nbest 1 -bucket 0 buckP -snfit snfitP -jobs 1'

    cmd = cmd.replace('inP', voxBoldP)
    cmd = cmd.replace('buckP', buckP)
    snfitP = '%s/snft_%s.PRF'%(afniTmpF, anTag)
    cmd = cmd.replace('snfitP',snfitP)

    cmdFinal = setupCmd + ' ;' +cmd
    print('')
    print('')
    print(cmdFinal)
    print('')

    # again, mac is a pita:
    import sys
    if not 'darwin' in sys.platform:
        cmdFinal = '/usr/bin/tcsh -i -f -c "'+ setupCmd + ' ;' +cmd+'"'
        call(cmdFinal,shell=True)
    else:
        cmdFinal = setupCmd + ' ;' +cmd
        _os.system ("tcsh -c '%s'"%cmdFinal)
        
    # _os.remove(voxBoldP)
    return read_afni(buckP,inRf,outRf,anTag)#cmdFinal


def read_afni(buckP,inRf,outRf,anTag):
    import glob
    buckP+='.1D'
    df = _pd.read_csv(buckP,skiprows=9,sep=' ',header=None,index_col=0)
    df.index=range(len(df))
    cols=df.columns
    if 'ell' in outRf:
        # Conv_PRF_6               : 6-param Population Receptive Field Model
        #     %                              (A, X, Y, sigma, sigrat, theta)
        #     %                              see model_conv_PRF_6.c
        df.columns = ['A','X','Y','sigma','sigrat','theta']+list(cols[6:])
    elif 'gau' in outRf:
        # %   Conv_PRF                 : 4-param Population Receptive Field Model
        # %                              (A, X, Y, sigma)
        # %                              see model_conv_PRF.c
        df.columns = ['A','Y','X','sigma']+list(cols[4:])
    f,n = _os.path.split(buckP)
    ps = glob.glob(f+'/*%s*'%anTag)
    # for p in ps:
    #     _os.remove(p)
    return df.loc[0,:]
