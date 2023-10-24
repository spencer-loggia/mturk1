import os
import sys

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from dataloader_POA import TrialDataLoader
import nibabel as nib
import pickle
import numpy as np
import pandas as pd

# from sklearn.ensemble import RandomForestClassifier as Classifier
from sklearn.svm import SVC as Classifier
# fro, sklearn.neighbors import K
from sklearn.decomposition import PCA as Embedder
from sklearn.metrics import ConfusionMatrixDisplay


def get_full_set(set_name, resample=True):
    examples = len(eval("MTurk1." + set_name))
    stim_data = []
    targets = []
    MTurk1.batch_size = int(examples / 48)
    for stim, target in MTurk1.batch_iterator(set_name, num_train_batches=120, resample=False, n_workers=1, cube=False, standardize=False): #360
        stim_data.append(stim)
        targets.append(target)
    train_stim_data = np.concatenate(stim_data, axis=0)
    train_targets = np.concatenate(targets, axis=0)
    return train_stim_data, train_targets


if __name__ == '__main__':
    ROOT = sys.argv[1]
    SUBJECT = sys.argv[2]
    TRAIN_SET = sys.argv[3]
    ROI_SET_NAME = sys.argv[4]
    PCA_COMP = int(sys.argv[5])
    boot_folds = int(sys.argv[6])
    power_analysis = sys.argv[7] # True or false
    fractions = int(sys.argv[8]) # Which fractions of the dataset to take
    fraction_iterations = int(sys.argv[9]) # How many times to repeat at each fraction of the dataset

    if SUBJECT not in ["wooster", "jeeves"]:
        raise ValueError("Subject must be either wooster or jeeves")

    # if power_analysis == "False":
    #     if TRAIN_SET == "uncolored":
    #         train_set = "uncolored_shape_all"
    #         in_train_set = "uncolored_shape_train"
    #         in_test_set = "uncolored_shape_test"
    #         test_set = "color_all"
    #     elif TRAIN_SET == "color":
    #         train_set = "color_all"
    #         in_train_set = "color_train"
    #         in_test_set = "color_test"
    #         test_set = "uncolored_shape_all"
    #     else:
    #         raise ValueError("Train set must be one of colored, uncolored")
    # else:
    #     if TRAIN_SET == "uncolored":
    #         train_set = "uncolored_shape_subset"
    #         test_set = "color_subset"
    #     elif TRAIN_SET == "color":
    #         train_set = "color_subset"
    #         test_set = "uncolored_shape_subset"
    #     else:
    #         raise ValueError("Train set must be one of colored, uncolored")

    use_binary_target = False
    CONTENT_ROOT = os.path.abspath(ROOT)
    USE_CLASSES = [2, 3, 6, 7, 10, 11]
    FUNC_WM_PATH = os.path.join(CONTENT_ROOT, "subjects", SUBJECT, "mri", "func_wn.nii")
    BRAIN_MASK = os.path.join(CONTENT_ROOT, "subjects", SUBJECT, "mri", "no_cereb_decode_mask.nii.gz")
    DATA_KEY_PATH = os.path.join(CONTENT_ROOT, "subjects", SUBJECT, "analysis", "shape_color_attention_decode_stimulus_response_data_key.csv")

    CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]
    CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]

    if SUBJECT == "wooster":
        crop = CROP_WOOSTER
    else:
        crop = CROP_JEEVES

    all_classes = set(range(1, 15))
    ignore_classes = all_classes - set(USE_CLASSES)

# This is where I want to start defining it as a function
    def within_roi_decoder():
        # MTurk1 = TrialDataLoader(
        #     DATA_KEY_PATH,
        #     250,
        #     frc = frac,
        #     content_root=CONTENT_ROOT, ignore_class=ignore_classes, crop=crop, use_behavior=True, verbose=False)


        dir_path = os.path.join(CONTENT_ROOT, "subjects", SUBJECT, "rois", ROI_SET_NAME)

        roi_mask_paths = [f for f in os.listdir(dir_path) if ".nii.gz" in f and "~" not in f]

        comp = [PCA_COMP]

        roi_train_data = [list() for _ in roi_mask_paths]
        roi_cross_data = [list() for _ in roi_mask_paths]
        roi_names = []
        print("This is the train set", train_set)
        for i in range(boot_folds):
            og_train_stim_data, train_targets = get_full_set(train_set, resample=True)
            og_in_train_stim_data = og_train_stim_data[:int(85 * len(train_targets) / 100)]
            in_train_targets = train_targets[:int(85 * len(train_targets) / 100)]
            og_in_test_stim_data = og_train_stim_data[int(85 * len(train_targets) / 100):]
            in_test_targets = train_targets[int(85 * len(train_targets) / 100):]
            og_test_stim_data, test_targets = get_full_set(test_set, resample=False)

            for j, mask_path in enumerate(roi_mask_paths):
                train_accs = []
                cross_accs = []
                roi_name = mask_path.split(".")[0]
                if i == 0:
                    roi_names.append(roi_name)
                print("\nPROC. ROI", roi_name)
                roi_mask = nib.load(os.path.join(dir_path, mask_path)).get_fdata()
                roi_mask = roi_mask[MTurk1.crop[0][0]:MTurk1.crop[0][1], MTurk1.crop[1][0]:MTurk1.crop[1][1],
                           MTurk1.crop[2][0]:MTurk1.crop[2][1]]
                roi_mask = (roi_mask > .7).squeeze()

                all_mask = roi_mask

                if use_binary_target:
                    targets = sorted(list(set(range(14)) - set(ignore_classes)))
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

                    # get in train set data
                    in_train_stim_data = og_in_train_stim_data[:, :, all_mask]
                    in_train_stim_data = in_train_stim_data.reshape((in_train_stim_data.shape[0], -1))
                    in_train_stim_data = (in_train_stim_data - in_train_stim_data.mean()) / in_train_stim_data.std()

                    # get in  test set data
                    in_test_stim_data = og_in_test_stim_data[:, :, all_mask]
                    in_test_stim_data = in_test_stim_data.reshape((in_test_stim_data.shape[0], -1))
                    in_test_stim_data = (in_test_stim_data - in_test_stim_data.mean()) / in_test_stim_data.std()

                    # get test set data
                    test_stim_data = og_test_stim_data[:, :, all_mask]
                    test_stim_data = test_stim_data.reshape((test_stim_data.shape[0], -1))
                    test_stim_data = (test_stim_data - test_stim_data.mean()) / test_stim_data.std()


                    # all_data = np.concatenate([train_stim_data, test_stim_data], axis=0)
                    # mean_data = all_data.mean(axis=0)
                    # select_voxels = np.argsort(mean_data)[:min(len(mean_data), 150)] #
                    # sel_all_data = all_data[:, select_voxels]
                    # sel_train_stim_data = train_stim_data[:, select_voxels]
                    # sel_test_stim_data = test_stim_data[:, select_voxels]
                    # print("Initial Dim:", all_data.shape[1])


                    def fit_transform_svm(x, y, tx, ty):
                        svc_model = Classifier(max_iter=10000)  # max_iter=5000, fit_intercept=False
                        svc_model.fit(x, y)
                        chance = 1 / len(USE_CLASSES)

                        # train set eval
                        in_y_hat = svc_model.predict(x)
                        train_acc = np.count_nonzero(in_y_hat == y) / len(y)
                        # train_accs[-1].append(train_acc)
                        print("TRAIN_TRAIN_ACC", train_acc)

                        # test set eval
                        cross_y_hat = svc_model.predict(tx)
                        test_acc = np.count_nonzero(cross_y_hat == ty) / len(ty)
                        print("TRAIN_TEST_ACC", test_acc)
                        # cross_accs[-1].append(test_acc)
                        return train_acc, test_acc


                    for m_c in comp:
                        c = m_c

                        pca = Embedder(n_components=c, whiten=False)
                        pca.fit(in_train_stim_data)
                        proj_train_stim_data = pca.transform(in_train_stim_data)
                        proj_test_stim_data = pca.transform(in_test_stim_data)
                        # proj_train_stim_data = (proj_train_stim_data - proj_train_stim_data.mean()) / proj_train_stim_data.std()
                        # proj_test_stim_data = (proj_test_stim_data - proj_test_stim_data.mean()) / proj_test_stim_data.std()

                        print("IN 2 IN TRAIN: target:", target_class, " examples:", proj_train_stim_data.shape[0], " roi_features:", proj_train_stim_data.shape[1])

                        _, test_acc = fit_transform_svm(proj_train_stim_data, in_train_targets, proj_test_stim_data, in_test_targets)
                        train_accs[-1].append(test_acc)

                        pca = Embedder(n_components=c, whiten=False)
                        pca.fit(train_stim_data)
                        proj_train_stim_data = pca.transform(train_stim_data)
                        proj_test_stim_data = pca.transform(test_stim_data)

                        print("IN 2 X TRAIN: n_examples:", proj_test_stim_data.shape[0], " roi_features:", proj_test_stim_data.shape[1])
                        _, test_acc = fit_transform_svm(proj_train_stim_data, train_targets, proj_test_stim_data, test_targets)
                        cross_accs[-1].append(test_acc)

                train_accs = np.array(train_accs)
                cross_accs = np.array(cross_accs)
                roi_train_data[j].append(np.mean(train_accs))
                roi_cross_data[j].append(np.mean(cross_accs))

                print(roi_name, " All Cross Accs:", cross_accs)
                print(roi_name, "All Train Accs:", train_accs)

        fig, ax = plt.subplots(1)
        roi_train_data = np.array(roi_train_data)
        roi_cross_data = np.array(roi_cross_data)

        train_var = roi_train_data.std(axis=1)
        cross_var = roi_cross_data.std(axis=1)
        roi_train_data = roi_train_data.mean(axis=1)
        roi_cross_data = roi_cross_data.mean(axis=1)

        print(roi_train_data.shape)

        # x_ax = np.arange(len(roi_train_data))
        # ax.bar(x_ax, roi_train_data)
        # ax.bar(x_ax, roi_cross_data)
        # ax.errorbar(x_ax, roi_train_data, yerr=train_var, ls='none', c="black")
        # ax.errorbar(x_ax, roi_cross_data, yerr=cross_var, ls='none', c="black")
        #
        # ax.set_xticks(np.arange(len(roi_train_data)))
        # ax.set_xticklabels(roi_names, visible=True)
        # plt.xticks(rotation=50)
        # plt.show()

    if power_analysis == "False":
        if TRAIN_SET == "uncolored":
            train_set = "uncolored_shape_all"
            in_train_set = "uncolored_shape_train"
            in_test_set = "uncolored_shape_test"
            test_set = "color_all"
        elif TRAIN_SET == "color":
            train_set = "color_all"
            in_train_set = "color_train"
            in_test_set = "color_test"
            test_set = "uncolored_shape_all"
        else:
            raise ValueError("Train set must be one of colored, uncolored")
        within_roi_decoder(frac=1)
    else:
        if TRAIN_SET == "uncolored":
            train_set = "uncolored_shape_subset"
            test_set = "color_subset"
        elif TRAIN_SET == "color":
            train_set = "color_all"
            test_set = "uncolored_shape_subset"
        else:
            raise ValueError("Train set must be one of colored, uncolored")
        power = pd.DataFrame()
        power_data = pd.DataFrame()
        power_data['iteration'] = []
        power_data['frac'] = []
        power_data['roi'] = []
        power_data['roi_train'] = []
        power_data['roi_cross'] = []
        power_data['train_var'] = []
        power_data['cross_var'] = []
        for fraction in range(fractions):
            for iter in range(fraction_iterations):
                MTurk1 = TrialDataLoader(
                    DATA_KEY_PATH,
                    250,
                    frc = fraction,
                    content_root=CONTENT_ROOT, ignore_class=ignore_classes, crop=crop, use_behavior=True, verbose=False)
                within_roi_decoder() # where frc is set back in the dataloader as the link between this and the subsets...
                power_data = pd.DataFrame()
                power_data['iteration'] = np.repeat(fraction_iterations, len(roi_names))
                power_data['frac'] = np.repeat(fraction, len(roi_names))
                power_data['roi'] = roi_names
                power_data['roi_train'] = roi_train_data
                power_data['roi_cross'] = roi_cross_data
                power_data['train_var'] = train_var
                power_data['cross_var'] = cross_var
                power = [power, roi_train_data_df]
                power = pd.concat(power)
        power.to_csv('/media/ssbeast/DATA/Users/Helen/20230917_poa_try1.csv')


######################################################
#
#
# def get_full_set(set_name, resample=True):
#     # hack data loader to produce a single batch of desired amount of data
#
#     examples = len(eval("MTurk1." + set_name))
#
#     stim_data = []
#     targets = []
#     MTurk1.batch_size = int(examples / 42) #changed this to 42 because on 10% was too small
#     for stim, target in MTurk1.batch_iterator(set_name, num_train_batches=180, resample=resample, n_workers=4, cube=False): #360
#         stim_data.append(stim)
#         targets.append(target)
#     train_stim_data = np.concatenate(stim_data, axis=0)
#     train_targets = np.concatenate(targets, axis=0)
#     return train_stim_data, train_targets
#
#
# if __name__=='__main__':
#
#     use_binary_target = False
#     COTENT_ROOT = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1"
#     USE_CLASSES = [2, 3, 6, 7, 10, 11]
#     SUBJECT = "jeeves"
#     FUNC_WM_PATH = "/home/ssbeast/Projectsz/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/mri/func_wm.nii"
#     BRAIN_MASK = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/mri/no_cereb_decode_mask.nii.gz"
#     DATA_KEY_PATH = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/analysis/bkp_shape_color_attention_decode_stimulus_response_data_key.csv"
#     # This csv is the jeeves datakey with only the 6 shapes/colors we care about in it
#     #DATA_KEY_PATH = '/media/ssbeast/DATA/Users/Helen/trial.csv'
#     CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]
#     CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]
#
#     all_classes = set(range(1, 15))
#     ignore_classes = all_classes - set(USE_CLASSES)
#
#     if SUBJECT == "jeeves":
#         crop = CROP_JEEVES
#     elif SUBJECT == "wooster":
#         crop = CROP_WOOSTER
#     else:
#         raise ValueError()
# finaldata = pd.DataFrame()
# roi_train_data_df = pd.DataFrame()
# roi_train_data_df['iteration'] = []
# roi_train_data_df['frac'] = []
# roi_train_data_df['roi'] = []
# roi_train_data_df['roi_train'] = []
# roi_train_data_df['roi_cross'] = []
# roi_train_data_df['train_var'] = []
# roi_train_data_df['cross_var'] = []
# for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     for times in range(10):
#         MTurk1 = TrialDataLoader(
#             DATA_KEY_PATH,
#             250, frc=fraction,
#             content_root="/home/ssbeast/Projects/SS/monkey_fmri/MTurk1", ignore_class=ignore_classes, crop=crop, use_behavior=True)
#         #print(MTurk1.color_train)
#         #print("Printing Mturk1.color_all in roi_decoder")
#         #print(type(MTurk1.color_all))
#         #print(MTurk1.color_test)
#         #print(MTurk1.color_cross_dev)
#         #print(MTurk1.color_cross_test)
#         dir_path = "/media/ssbeast/DATA/Users/Helen/jeeves_roi_subset"
#
#         roi_mask_paths = [f for f in os.listdir(dir_path) if ".nii.gz" in f and "~" not in f]
#
#         # list of pca embedding dimensions to try.
#         comp = [6]
#
#         # number of bootstrap (sorta fake) iterations to preform.
#         boot_folds = 20
#
#         # set of data to train model on
#         train_set = "uncolored_subset"
#
#         # set of data that cross decoding is preformed on
#         test_set = "color_subset"
#
#         roi_train_data = [list() for _ in roi_mask_paths]
#         roi_cross_data = [list() for _ in roi_mask_paths]
#         roi_names = []
#         for i in range(boot_folds):
#             og_train_stim_data, train_targets = get_full_set(train_set, resample=True)
#             og_test_stim_data, test_targets = get_full_set(test_set, resample=False)
#             # train80_data, train80_targets = get_full_set(80train_set, resample=True)
#             # test_20data, test20_targets = get_full_set(20test_set, resample=True)
#
#
#             for j, mask_path in enumerate(roi_mask_paths):
#                 train_accs = []
#                 cross_accs = []
#                 roi_name = mask_path.split(".")[0]
#                 if i == 0:
#                     roi_names.append(roi_name)
#                 print("PROC. ROI", roi_name)
#                 roi_mask = nib.load(os.path.join(dir_path, mask_path)).get_fdata()
#                 roi_mask = roi_mask[MTurk1.crop[0][0]:MTurk1.crop[0][1], MTurk1.crop[1][0]:MTurk1.crop[1][1],
#                            MTurk1.crop[2][0]:MTurk1.crop[2][1]]
#                 roi_mask = (roi_mask > .7).squeeze()
#
#                 all_mask = roi_mask
#
#                 if use_binary_target:
#                     targets = sorted(list(set(range(14)) - set(ignore_classes)))
#                 else:
#                     targets = [None]
#                 for target_class in targets:
#                     train_accs.append([])
#                     cross_accs.append([])
#                     MTurk1.one_v_rest_target = target_class
#                     # get train set data
#                     train_stim_data = og_train_stim_data[:, :, all_mask]
#                     train_stim_data = train_stim_data.reshape((train_stim_data.shape[0], -1))
#                     train_stim_data = (train_stim_data - train_stim_data.mean()) / train_stim_data.std()
#                     # = np.random.normal(0, 1, size=(train_stim_data.shape[0], 200))
#
#                     # get test set data
#                     test_stim_data = og_test_stim_data[:, :, all_mask] # <trials, channels, voxels_in_roi>
#                     test_stim_data = test_stim_data.reshape((test_stim_data.shape[0], -1)) # <trials, channels * voxels_in_roi>
#                     test_stim_data = (test_stim_data - test_stim_data.mean()) / test_stim_data.std() # standardization
#                     # test_stim_data = np.random.normal(0, 1, size=(test_stim_data.shape[0], 200))
#                     all_data = np.concatenate([train_stim_data, test_stim_data], axis=0)
#                     mean_data = all_data.mean(axis=0)
#
#                     # discard small signal voxels if roi is very large (mb not needed)
#                     select_voxels = np.argsort(mean_data)[:min(len(mean_data), 150)]
#                     sel_all_data = all_data[:, select_voxels]
#                     sel_train_stim_data = train_stim_data[:, select_voxels]
#                     sel_test_stim_data = test_stim_data[:, select_voxels]
#
#                     print("Initial Dim:", all_data.shape[1])
#
#                     # fit and transform train and test data with pca.
#                     for c in comp:
#                         pca = Embedder(n_components=c)
#                         pca.fit(sel_train_stim_data)
#                         #pca.fit(sel_all_data) # for prod should be fit on just train data or seperatly for train and test data.
#                         proj_train_stim_data = pca.transform(sel_train_stim_data)
#                         proj_test_stim_data = pca.transform(sel_test_stim_data)
#
#
#                         print("TRAIN: target:", target_class, " examples:", proj_train_stim_data.shape[0], " roi_features:", proj_train_stim_data.shape[1])
#
#                         # try with fitting intercept and not.
#                         svc_model = Classifier(max_iter=5000) #, fit_intercept=False
#                         svc_model.fit(proj_train_stim_data, train_targets)
#
#                         # save out model
#                         out_dir = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/wooster/analysis/single_roi/wooster"
#                         out_name = os.path.join(out_dir, roi_name + "_svc_model.pkl")
#                         with open(out_name, "wb") as f:
#                             pickle.dump(svc_model, f)
#
#                         chance = 1 / len(USE_CLASSES)
#
#                         # train set eval
#                         in_y_hat = svc_model.predict(proj_train_stim_data)
#                         train_acc = np.count_nonzero(in_y_hat == train_targets) / len(train_targets)
#                         train_accs[-1].append(train_acc)
#                         print(train_acc)
#                         #ConfusionMatrixDisplay.from_estimator(svc_model, proj_train_stim_data, train_targets)
#                         #plt.title("In-Modality CM")
#                         #plt.show()
#
#                         print("TEST: n_examples:", proj_test_stim_data.shape[0], " roi_features:", proj_test_stim_data.shape[1])
#
#                         # test set eval
#                         cross_y_hat = svc_model.predict(proj_test_stim_data)
#                         test_acc = np.count_nonzero(cross_y_hat == test_targets) / len(test_targets)
#                         print(test_acc)
#                         cross_accs[-1].append(test_acc)
#                         print("\n")
#                         #ConfusionMatrixDisplay.from_estimator(svc_model, proj_test_stim_data, test_targets)
#                         #plt.title("Cross-Modality CM")
#                         #plt.show()
#                 #train accuracy for each bootfold for each roi
#                 train_accs = np.array(train_accs)
#                 cross_accs = np.array(cross_accs)
#                 #append mean accuracy across bootfold i for roi j
#                 roi_train_data[j].append(np.mean(train_accs))
#                 roi_cross_data[j].append(np.mean(cross_accs))
#
#                 print(roi_name, " All Cross Accs:", cross_accs)
#                 print(roi_name, "All Train Accs:", train_accs)
#             print("Done with bootfold", i)
#         print("Done with iteration", times)
#
#         fig, ax = plt.subplots(1)
#         roi_train_data = np.array(roi_train_data)
#         roi_cross_data = np.array(roi_cross_data)
#
#         train_var = roi_train_data.std(axis=1)
#         cross_var = roi_cross_data.std(axis=1)
#         # takes mean accuracy for a given roi across bootfolds
#         roi_train_data = roi_train_data.mean(axis=1)
#         roi_cross_data = roi_cross_data.mean(axis=1)
#
#         print(roi_train_data.shape)
#
#         x_ax = np.arange(len(roi_train_data))
#         ax.bar(x_ax, roi_train_data)
#         ax.bar(x_ax, roi_cross_data)
#         ax.errorbar(x_ax, roi_train_data, yerr=train_var, ls='none', c="black")
#         ax.errorbar(x_ax, roi_cross_data, yerr=cross_var, ls='none', c="black")
#
#         ax.set_xticks(np.arange(len(roi_train_data)))
#         ax.set_xticklabels(roi_names, visible=True)
#         #plt.show() messes with savefig
#         #plt.savefig('/media/ssbeast/DATA/Users/KarthikKasi/plot' + str(fraction) + '.png')
#         #plt.clf()
#         #print("roi train")
#         #print(roi_train_data)
#         #print("roi cross")
#         #print(roi_cross_data)
#         #print("train var")
#         #print(train_var)
#         #print("cross var")
#         #print(cross_var)
#         roi_train_data_df = pd.DataFrame()
#         roi_train_data_df['iteration'] = [times, times, times, times]
#         roi_train_data_df['frac'] = [fraction, fraction, fraction, fraction]
#         roi_train_data_df['roi'] = roi_names
#         roi_train_data_df['roi_train'] = roi_train_data
#         roi_train_data_df['roi_cross'] = roi_cross_data
#         roi_train_data_df['train_var'] = train_var
#         roi_train_data_df['cross_var'] = cross_var
#         finaldata = [finaldata, roi_train_data_df]
#         finaldata = pd.concat(finaldata)
#     print("Done with fraction", fraction)
#     #roi_train_data_df = pd.concat([fraction, fraction], axis = 0)
# #    roi_train_data_df.to_csv('/media/ssbeast/DATA/Users/KarthikKasi/trythis.csv')
# finaldata.to_csv('/media/ssbeast/DATA/Users/Helen/20230905_poa_ant_subset_10_20.csv')
