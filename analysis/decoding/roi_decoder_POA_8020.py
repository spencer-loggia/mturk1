import os
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from dataloader_POA_8020 import TrialDataLoader
import nibabel as nib
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC as Classifier
from sklearn.decomposition import PCA as Embedder
from sklearn.metrics import ConfusionMatrixDisplay


def get_full_set(set_name, resample=True):
    # hack data loader to produce a single batch of desired amount of data 

    examples = len(eval("MTurk1." + set_name))

    stim_data = []
    targets = []
    MTurk1.batch_size = int(examples / 42) # originally was 48. must be 33 or less in order to get an iteration with 8% of the data as a training set bc resampling tries to grab batches > 0.08*the amount of data examples there are
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
    # This csv is the jeeves datakey with only the 6 shapes/colors we care about in it
    #DATA_KEY_PATH = '/media/ssbeast/DATA/Users/Helen/trial.csv'
    CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]
    CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]

    all_classes = set(range(1, 15))
    ignore_classes = all_classes - set(USE_CLASSES)

    if SUBJECT == "jeeves":
        crop = CROP_JEEVES
    elif SUBJECT == "wooster":
        crop = CROP_WOOSTER
    else:
        raise ValueError()
finaldata = pd.DataFrame()
roi_train_data_df = pd.DataFrame()
roi_train_data_df['iteration'] = []
roi_train_data_df['frac'] = []
roi_train_data_df['roi'] = []
roi_train_data_df['roi_train'] = []
roi_train_data_df['roi_cross'] = []
roi_train_data_df['train_var'] = []
roi_train_data_df['cross_var'] = []
for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for times in range(8):
        MTurk1 = TrialDataLoader(
            DATA_KEY_PATH,
            250, frc=fraction,
            content_root="/home/ssbeast/Projects/SS/monkey_fmri/MTurk1", ignore_class=ignore_classes, crop=crop, use_behavior=True)
        dir_path = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/jeeves/rois/dyloc_regions/he_func_space_rois"

        roi_mask_paths = [f for f in os.listdir(dir_path) if ".nii.gz" in f and "~" not in f]

        # list of pca embedding dimensions to try.
        comp = [4]

        # number of bootstrap (sorta fake) iterations to preform.
        boot_folds = 20

        # set of data to train model on
        train_set = "uncolored_subset"

        # set of data that cross decoding is preformed on
        test_set = "color_subset"

        # for independent training and testing
        # set of data to train model on
        train_shape = "uncolored_shape_train"
        # set of data to perform within-class (identity) decoding on
        test_shape = "uncolored_shape_test"

        roi_train_data = [list() for _ in roi_mask_paths]
        roi_cross_data = [list() for _ in roi_mask_paths]
        roi_names = []
        for i in range(boot_folds):
            og_train_stim_data, train_targets = get_full_set(train_set, resample=True)
            og_test_stim_data, test_targets = get_full_set(test_set, resample=False)
            #og_shape_train_stim_data, shape_train_targets = get_full_set(train_shape, resample=True)
            #og_shape_test_stim_data, shape_test_targets = get_full_set(test_shape, resample=False)

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
                    targets = sorted(list(set(range(1, 15)) - set(ignore_classes))) # didn't we switch this back ???
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

                    # get shape train set data
                    #shape_train_stim_data = og_shape_train_stim_data[:, :, all_mask]
                    #shape_train_stim_data = shape_train_stim_data.reshape((shape_train_stim_data.shape[0], -1))
                    #shape_train_stim_data = (shape_train_stim_data - shape_train_stim_data.mean()) / shape_train_stim_data.std()

                    # get shape test set data
                    #shape_test_stim_data = og_shape_test_stim_data[:, :, all_mask]
                    #shape_test_stim_data = shape_test_stim_data.reshape((shape_test_stim_data.shape[0], -1))
                    #shape_test_stim_data - (shape_test_stim_data - shape_test_stim_data.mean()) / shape_test_stim_data.std()

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
                        proj_train_stim_data = pca.transform(sel_train_stim_data)
                        proj_test_stim_data = pca.transform(sel_test_stim_data)


                        print("TRAIN: target:", target_class, " examples:", proj_train_stim_data.shape[0], " roi_features:", proj_train_stim_data.shape[1])

                        # try with fitting intercept and not.
                        svc_model = Classifier(max_iter=5000) #, fit_intercept=False
                        svc_model.fit(proj_train_stim_data, train_targets)

                        # helen's new model


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
                        #ConfusionMatrixDisplay.from_estimator(svc_model, proj_train_stim_data, train_targets)
                        #plt.title("In-Modality CM")
                        #plt.show()

                        print("TEST: n_examples:", proj_test_stim_data.shape[0], " roi_features:", proj_test_stim_data.shape[1])

                        # test set eval
                        cross_y_hat = svc_model.predict(proj_test_stim_data)
                        test_acc = np.count_nonzero(cross_y_hat == test_targets) / len(test_targets)
                        print(test_acc)
                        cross_accs[-1].append(test_acc)
                        print("\n")
                        #ConfusionMatrixDisplay.from_estimator(svc_model, proj_test_stim_data, test_targets)
                        #plt.title("Cross-Modality CM")
                        #plt.show()
                #train accuracy for each bootfold for each roi
                train_accs = np.array(train_accs)
                cross_accs = np.array(cross_accs)
                #append mean accuracy across bootfold i for roi j
                roi_train_data[j].append(np.mean(train_accs))
                roi_cross_data[j].append(np.mean(cross_accs))

                print(roi_name, " All Cross Accs:", cross_accs)
                print(roi_name, "All Train Accs:", train_accs)

        fig, ax = plt.subplots(1)
        roi_train_data = np.array(roi_train_data)
        roi_cross_data = np.array(roi_cross_data)

        train_var = roi_train_data.std(axis=1)
        cross_var = roi_cross_data.std(axis=1)
        # takes mean accuracy for a given roi across bootfolds
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
        #plt.show() messes with savefig
        #plt.savefig('/media/ssbeast/DATA/Users/KarthikKasi/plot8020' + str(fraction) + '.png')
        #plt.clf()
        print("roi train")
        print(roi_train_data)
        print("roi cross")
        print(roi_cross_data)
        print("train var")
        print(train_var)
        print("cross var")
        print(cross_var)
        roi_train_data_df = pd.DataFrame()
        roi_train_data_df['iteration'] = [times, times, times, times, times, times, times, times, times, times]
        roi_train_data_df['frac'] = [fraction, fraction, fraction, fraction, fraction, fraction, fraction, fraction, fraction, fraction]
        roi_train_data_df['roi'] = roi_names
        roi_train_data_df['roi_train'] = roi_train_data
        roi_train_data_df['roi_cross'] = roi_cross_data
        roi_train_data_df['train_var'] = train_var
        roi_train_data_df['cross_var'] = cross_var
        finaldata = [finaldata, roi_train_data_df]
        finaldata = pd.concat(finaldata)
    #roi_train_data_df = pd.concat([fraction, fraction], axis = 0)
#    roi_train_data_df.to_csv('/media/ssbeast/DATA/Users/KarthikKasi/trythis.csv')
finaldata.to_csv('/media/ssbeast/DATA/Users/Helen/woostertry20230901.csv')
