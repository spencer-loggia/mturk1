import os
import sys

import matplotlib
import nibabel

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

import nibabel as nib
import numpy as np

from dataloader import TrialDataLoader


def get_full_set(set_name, resample=False):
    examples = len(eval("MTurk1." + set_name))
    stim_data = []
    targets = []
    MTurk1.batch_size = 50
    for stim, target in MTurk1.batch_iterator(
        set_name,
        num_train_batches=40,
        resample=False,
        n_workers=1,
        cube=False,
        standardize=True,
    ):  # 360
        stim_data.append(stim)
        targets.append(target)
    train_stim_data = np.concatenate(stim_data, axis=0)
    train_targets = np.concatenate(targets, axis=0)
    return train_stim_data, train_targets


def get_subject_accs(acc_map, roi_dir, roi_keys, train_set, test_set, thresh=0.22):
    acc_map = nib.load(acc_map).get_fdata()
    rois = [f for f in os.listdir(roi_dir) if ".nii.gz" in f and "~" not in f]
    roi_masks = [None]*len(roi_keys)
    train_accs = []
    cross_accs = []
    counts = [0] * len(roi_keys)

    og_train_stim_data, train_targets = get_full_set(train_set, resample=True)
    og_in_train_stim_data = og_train_stim_data[
                            : int(85 * len(train_targets) / 100)]
    in_train_targets = train_targets[: int(85 * len(train_targets) / 100)]
    og_in_test_stim_data = og_train_stim_data[
                           int(85 * len(train_targets) / 100):]
    in_test_targets = train_targets[int(85 * len(train_targets) / 100):]
    og_test_stim_data, test_targets = get_full_set(test_set, resample=False)

    for i, key in enumerate(roi_keys):
        for f in rois:
            if key in f:
                path = os.path.join(roi_dir, f)
                roi = nibabel.load(path).get_fdata()
                ind = np.logical_and(roi > .5, acc_map > thresh)
                acc_roi = MTurk1.crop_volume(ind, cube=False)
                if roi_masks[i] is None:
                    roi_masks[i] = acc_roi
                else:
                    roi_masks[i] = np.logical_or(acc_roi, roi_masks[i])
                counts[i] += 1
        # get train set data
        train_stim_data = og_train_stim_data[:, :, roi_masks[i]]
        train_stim_data = train_stim_data.reshape(
            (train_stim_data.shape[0], -1)
        )
        train_stim_data = (
                                  train_stim_data - train_stim_data.mean()
                          ) / train_stim_data.std()

        # get in train set data
        in_train_stim_data = og_in_train_stim_data[:, :, roi_masks[i]]
        in_train_stim_data = in_train_stim_data.reshape(
            (in_train_stim_data.shape[0], -1)
        )
        in_train_stim_data = (
                                     in_train_stim_data - in_train_stim_data.mean()
                             ) / in_train_stim_data.std()

        # get in  test set data
        in_test_stim_data = og_in_test_stim_data[:, :, roi_masks[i]]
        in_test_stim_data = in_test_stim_data.reshape(
            (in_test_stim_data.shape[0], -1)
        )
        in_test_stim_data = (
                                    in_test_stim_data - in_test_stim_data.mean()
                            ) / in_test_stim_data.std()

        # get test set data
        test_stim_data = og_test_stim_data[:, :, roi_masks[i]]
        test_stim_data = test_stim_data.reshape(
            (test_stim_data.shape[0], -1))
        test_stim_data = (
                                 test_stim_data - test_stim_data.mean()
                         ) / test_stim_data.std()

        def fit_transform_svm(x, y, tx, ty, roi):
            svc_model = LinearSVC(
                max_iter=20000, fit_intercept=False
            )  # max_iter=5000, fit_intercept=False
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
            return train_acc, test_acc

        print(
            "IN 2 IN TRAIN: target:",
            None,
            " examples:",
            in_train_stim_data.shape[0],
            " roi_features:",
            in_train_stim_data.shape[1],
        )
        _, test_acc = fit_transform_svm(
            in_train_stim_data,
            in_train_targets,
            in_test_stim_data,
            in_test_targets,
            None,
        )
        train_accs.append(test_acc)

        # pca = Embedder(n_components=c, whiten=False)
        # pca.fit(train_stim_data)
        proj_train_stim_data = (
            train_stim_data  # pca.transform(train_stim_data)
        )
        proj_test_stim_data = (
            test_stim_data  # pca.transform(test_stim_data)
        )
        # proj_train_stim_data = train_stim_data
        # proj_test_stim_data = test_stim_data

        print(
            "IN 2 X TRAIN: n_examples:",
            proj_test_stim_data.shape[0],
            " roi_features:",
            proj_test_stim_data.shape[1],
        )
        _, test_acc = fit_transform_svm(
            proj_train_stim_data,
            train_targets,
            proj_test_stim_data,
            test_targets,
            None,
        )
        cross_accs.append(test_acc)
    return np.array(train_accs), np.array(cross_accs)


if __name__ == '__main__':
    SUBJECT = "wooster"

    M1_ROI_DIR = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/rois/IT-subdivisions/auto_func_space_rois"

    M1_S2C_ACC_MAP = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/analysis/global_decode/shape2color_all_searchlight_ACC.nii.gz"

    M1_C2S_ACC_MAP = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/analysis/global_decode/color2shape_all_searchlight_ACC.nii.gz"

    ROI_KEYS = ["color", "shape", "v1v2v3", "V4", "post_IT", "mid_IT", "ant_IT", "far_IT", "TP", "entorhinal", "parahippo"]

    CONTENT_ROOT = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1"

    CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]
    CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]

    if SUBJECT == "wooster":
        USE_CLASSES = [1, 4, 5, 8, 9, 12]
        ignore_sessions = ['scd_20230804',
                           'scd_20230806']  # wooster NIF receive coil sessions
        crop = CROP_WOOSTER
    else:
        USE_CLASSES = [2, 3, 6, 7, 10, 11]
        ignore_sessions = []  # jeeves NIF receive coil sessions
        crop = CROP_JEEVES

    all_classes = set(range(1, 15))
    ignore_classes = all_classes - set(USE_CLASSES)

    DATA_KEY_PATH = os.path.join(
        CONTENT_ROOT,
        "subjects",
        SUBJECT,
        "analysis",
        "shape_color_attention_decode_stimulus_response_data_key.csv",
    )

    MTurk1 = TrialDataLoader(
        DATA_KEY_PATH,
        250,
        content_root=CONTENT_ROOT,
        ignore_class=ignore_classes,
        ignore_sessions=ignore_sessions,
        crop=crop,
        use_behavior=True,
        verbose=False,
    )

    shape_train_set = "uncolored_shape_all"

    color_train_set = "color_all"

    m1_s2s_in_accs, m1_s2c_x_accs = get_subject_accs(M1_S2C_ACC_MAP, M1_ROI_DIR, ROI_KEYS, train_set=shape_train_set, test_set=color_train_set)

    m1_c2c_in_accs, m1_c2s_x_accs = get_subject_accs(M1_C2S_ACC_MAP, M1_ROI_DIR, ROI_KEYS, train_set=color_train_set, test_set=shape_train_set)


    fig, ax = plt.subplots(3)
    fig.set_size_inches(24, 8)
    fig.tight_layout()
    x_ax = np.arange(len(ROI_KEYS))
    ax[0].bar(x_ax, (m1_s2c_x_accs + m1_c2s_x_accs) / 2)
    ax[1].bar(x_ax, (m1_s2s_in_accs + m1_c2c_in_accs) / (m1_s2c_x_accs + m1_c2s_x_accs))
    ax[2].bar(x_ax, (m1_c2s_x_accs / m1_s2c_x_accs))

    # ax.bar(x_ax + .2, c2s_accs)
    # ax.errorbar(x_ax, roi_train_data, yerr=train_var, ls='none', c="black")
    # ax.errorbar(x_ax, roi_cross_data, yerr=cross_var, ls='none', c="black")

    ax[0].set_xticklabels([""] + ROI_KEYS, visible=True)
    ax[1].set_xticklabels([""] + ROI_KEYS, visible=True)
    ax[2].set_xticklabels([""] + ROI_KEYS, visible=True)
    plt.xticks(rotation=50)
    plt.show()
