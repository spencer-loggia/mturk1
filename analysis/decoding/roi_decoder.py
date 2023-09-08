import os
import matplotlib
import pandas as pd

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from dataloader import TrialDataLoader
import nibabel as nib
import pickle
import numpy as np

from sklearn.svm import LinearSVC as Classifier
from sklearn.decomposition import PCA as Embedder
from sklearn.metrics import ConfusionMatrixDisplay


def get_full_set(set_name, resample=True): # Helen look here
    # hack data loader to produce a single batch of desired amount of data 
    examples = len(eval("MTurk1." + set_name))
    stim_data = []
    targets = []
    MTurk1.batch_size = int(examples / 48)
    for stim, target in MTurk1.batch_iterator(set_name, num_train_batches=180, resample=resample, n_workers=4, cube=False): #360
        stim_data.append(stim)
        targets.append(target)
    train_stim_data = np.concatenate(stim_data, axis=0)
    train_targets = np.concatenate(targets, axis=0)
    return train_stim_data, train_targets


if __name__=='__main__':

    use_binary_target = False
    COTENT_ROOT = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1"
    USE_CLASSES = [2, 3, 6, 7, 10, 11]
    SUBJECT = "jeeves"
    FUNC_WM_PATH = "/home/ssbeast/Projectsz/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/mri/func_wm.nii"
    BRAIN_MASK = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/mri/no_cereb_decode_mask.nii.gz"
    DATA_KEY_PATH = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/analysis/shape_color_attention_decode_stimulus_response_data_key.csv"
    # TEST DECODER WITH LINEAR WOOSTER: this path leads to the datakey for the linear backup - it has all sessions up to 0813 with linear
    # 3d rep to functional target registration. Need to also change root for wherever the betas themselves are loaded, but otherwise should be able to just use this
    #DATA_KEY_PATH = "/media/ssbeast/DATA/Projects/MTurk1_LinearBackup/subjects/wooster/analysis/shape_color_attention_decode_stimulus_response_data_key.csv"
    CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]
    CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]

    #all_classes = set(range(1, 15)) # This should be correct but it was the source of Helen's and Spencer's plots not matching up for jeeves
    all_classes = set(range(14)) # This is what it used to be. When you use this, it fails to take hat data out of the decoder
    ignore_classes = all_classes - set(USE_CLASSES)

    if SUBJECT == "jeeves":
        crop = CROP_JEEVES
    elif SUBJECT == "wooster":
        crop = CROP_WOOSTER
    else:
        raise ValueError()

    MTurk1 = TrialDataLoader(
        DATA_KEY_PATH,
        250,
        #content_root="/media/ssbeast/DATA/Projects/MTurk1_LinearBackup", ignore_class=ignore_classes, crop=crop, use_behavior=True) # Wooster linear backup path
        content_root="/home/ssbeast/Projects/SS/monkey_fmri/MTurk1", ignore_class=ignore_classes, crop=crop, use_behavior=True)
    print(MTurk1.color_train)
    print(MTurk1.color_all)
    print(MTurk1.color_test)
    print(MTurk1.color_cross_dev)
    print(MTurk1.color_cross_test)
    # dir_path = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/wooster/mri/ROIs/functional/DYLOC_REGIONS" # Wooster old regions
    dir_path = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/jeeves/rois/dyloc_regions/he_func_space_rois"

    roi_mask_paths = [f for f in os.listdir(dir_path) if ".nii.gz" in f and "~" not in f]
       
    # list of pca embedding dimensions to try.
    comp = [6]

    # number of bootstrap (sorta fake) iterations to preform.  
    boot_folds = 20
    
    # set of data to train model on
    train_set = "uncolored_shape_all"
    
    # set of data that cross decoding is preformed on
    test_set = "color_all"

    roi_train_data = [list() for _ in roi_mask_paths]
    roi_cross_data = [list() for _ in roi_mask_paths]
    roi_names = []

    # Helen added to save values out into csv
    #df = pd.DataFrame()
    #df['roi'] = []
    #df['train_var'] = []
    #df['cross_var'] = []
    #df['train_mean'] = []
    #df['cross_mean'] = []
    for i in range(boot_folds):
        og_train_stim_data, train_targets = get_full_set(train_set, resample=True)
        og_test_stim_data, test_targets = get_full_set(test_set, resample=False)

        for j, mask_path in enumerate(roi_mask_paths):
            train_accs = []
            cross_accs = []
            roi_name = mask_path.split(".")[0]
            if i == 0:
                roi_names.append(roi_name)
            print("PROC. ROI", roi_name)
            roi_mask = nib.load(os.path.join(dir_path, mask_path)).get_fdata()
            roi_mask = roi_mask[MTurk1.crop[0][0]:MTurk1.crop[0][1], MTurk1.crop[1][0]:MTurk1.crop[1][1],
                       MTurk1.crop[2][0]:MTurk1.crop[2][1]]
            roi_mask = (roi_mask > .7).squeeze()

            all_mask = roi_mask

            if use_binary_target:
                targets = sorted(list(set(range(1, 15)) - set(ignore_classes)))
            else:
                targets = [None]
            for target_class in targets:
                train_accs.append([])
                cross_accs.append([])
                MTurk1.one_v_rest_target = target_class
                # get train set data
                train_stim_data = og_train_stim_data[:, :, all_mask]
                train_stim_data = train_stim_data.reshape((train_stim_data.shape[0], -1))
                train_stim_data = (train_stim_data - train_stim_data.mean()) / train_stim_data.std()
                # = np.random.normal(0, 1, size=(train_stim_data.shape[0], 200))

                # get test set data
                test_stim_data = og_test_stim_data[:, :, all_mask] # <trials, channels, voxels_in_roi>
                test_stim_data = test_stim_data.reshape((test_stim_data.shape[0], -1)) # <trials, channels * voxels_in_roi> 
                test_stim_data = (test_stim_data - test_stim_data.mean()) / test_stim_data.std() # standardization
                # test_stim_data = np.random.normal(0, 1, size=(test_stim_data.shape[0], 200))
                all_data = np.concatenate([train_stim_data, test_stim_data], axis=0)
                mean_data = all_data.mean(axis=0)
                
                # discard small signal voxels if roi is very large (mb not needed)
                select_voxels = np.argsort(mean_data)[:min(len(mean_data), 150)] 
                sel_all_data = all_data[:, select_voxels]
                sel_train_stim_data = train_stim_data[:, select_voxels]
                sel_test_stim_data = test_stim_data[:, select_voxels]
                
                print("Initial Dim:", all_data.shape[1])
                
                # fit and transform train and test data with pca. 
                for c in comp:
                    pca = Embedder(n_components=c)
                    pca.fit(sel_all_data) # for prod should be fit on just train data or seperatly for train and test data.
                    #pca.fit(sel_train_stim_data)
                    proj_train_stim_data = pca.transform(sel_train_stim_data)
                    proj_test_stim_data = pca.transform(sel_test_stim_data)


                    print("TRAIN: target:", target_class, " examples:", proj_train_stim_data.shape[0], " roi_features:", proj_train_stim_data.shape[1])
                       
                    # try with fitting intercept and not.
                    svc_model = Classifier(max_iter=5000) #, fit_intercept=False
                    svc_model.fit(proj_train_stim_data, train_targets)

                    # save out model
                    out_dir = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/wooster/analysis/single_roi/wooster"
                    out_name = os.path.join(out_dir, roi_name + "_svc_model.pkl")
                    with open(out_name, "wb") as f:
                        pickle.dump(svc_model, f)

                    chance = 1 / len(USE_CLASSES)

                    # train set eval
                    in_y_hat = svc_model.predict(proj_train_stim_data)
                    train_acc = np.count_nonzero(in_y_hat == train_targets) / len(train_targets)
                    train_accs[-1].append(train_acc)
                    print(train_acc)
                    # ConfusionMatrixDisplay.from_estimator(svc_model, proj_train_stim_data, train_targets)
                    #plt.title("In-Modality CM")
                    # plt.show()

                    print("TEST: n_examples:", proj_test_stim_data.shape[0], " roi_features:", proj_test_stim_data.shape[1])

                    # test set eval
                    cross_y_hat = svc_model.predict(proj_test_stim_data)
                    test_acc = np.count_nonzero(cross_y_hat == test_targets) / len(test_targets)
                    print(test_acc)
                    cross_accs[-1].append(test_acc)
                    print("\n")
                    # ConfusionMatrixDisplay.from_estimator(svc_model, proj_test_stim_data, test_targets)
                    # plt.title("Cross-Modality CM")

            train_accs = np.array(train_accs)
            cross_accs = np.array(cross_accs)
            roi_train_data[j].append(np.mean(train_accs))
            roi_cross_data[j].append(np.mean(cross_accs))

            print(roi_name, " All Cross Accs:", cross_accs)
            print(roi_name, "All Train Accs:", train_accs)
        print("Done with bootfold", i)

    fig, ax = plt.subplots(1)
    roi_train_data = np.array(roi_train_data)
    roi_cross_data = np.array(roi_cross_data)

    train_var = roi_train_data.std(axis=1)
    #cross_var = roi_train_data.std(axis=1)
    cross_var = roi_cross_data.std(axis=1) # Helen and Karthik changed to cross bc we think it was a typo and said 'roi_train_data' again; this works, keep this
    roi_train_data = roi_train_data.mean(axis=1)
    roi_cross_data = roi_cross_data.mean(axis=1)

    print(roi_train_data.shape)

    x_ax = np.arange(len(roi_train_data))
    ax.bar(x_ax, roi_train_data)
    ax.bar(x_ax, roi_cross_data)
    ax.errorbar(x_ax, roi_train_data, yerr=train_var, ls='none', c="black")
    ax.errorbar(x_ax, roi_cross_data, yerr=cross_var, ls='none', c="black")

    ax.set_xticks(np.arange(len(roi_train_data)))
    ax.set_xticklabels(roi_names, visible=True)
    plt.show()
    print(train_var)
    print(cross_var)

    # Helen added to save values out into csv
    #df['roi'] = roi_names
    #df['train_var'] = train_var
    #df['cross_var'] = cross_var
    #df['train_mean'] = roi_train_data
    #df['cross_mean'] = roi_cross_data
    #df.to_csv('/media/ssbeast/DATA/Users/Helen/20230906_jeeves.csv')