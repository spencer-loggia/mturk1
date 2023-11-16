import os
import sys

import matplotlib

# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from dataloader import TrialDataLoader
import nibabel as nib
import pickle
import numpy as np
import pandas as pd

# from sklearn.ensemble import RandomForestClassifier as Classifier
from sklearn.svm import LinearSVC as Classifier

# fro, sklearn.neighbors import K
from sklearn.decomposition import PCA as Embedder
from sklearn.metrics import ConfusionMatrixDisplay


def get_full_set(set_name, resample=False):
    examples = len(eval("MTurk1." + set_name))
    stim_data = []
    targets = []
    MTurk1.batch_size = int(examples / 48)
    for stim, target in MTurk1.batch_iterator(
        set_name,
        num_train_batches=120,
        resample=resample,
        n_workers=1,
        cube=False,
        standardize=True,
    ):  # 360
        stim_data.append(stim)
        targets.append(target)
    train_stim_data = np.concatenate(stim_data, axis=0)
    train_targets = np.concatenate(targets, axis=0)
    return train_stim_data, train_targets


if __name__ == "__main__":
    ROOT = sys.argv[1]
    SUBJECT = sys.argv[2]
    TRAIN_SET = sys.argv[3]
    ROI_SET_NAME = sys.argv[4]
    PCA_COMP = int(sys.argv[5])
    boot_folds = int(sys.argv[6])

    if SUBJECT not in ["wooster", "jeeves"]:
        raise ValueError("Subject must be either wooster or jeeves")

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

    use_binary_target = False
    CONTENT_ROOT = os.path.abspath(ROOT)
    if SUBJECT == "wooster":
        USE_CLASSES = [1, 4, 5, 8, 9, 12]
        ignore_sessions = ['scd_20230804',
                           'scd_20230806']  # wooster NIF receive coil sessions
    else:
        USE_CLASSES = [2, 3, 6, 7, 10, 11]
        ignore_sessions = ['scd_20230712', 'scd_20230713', 'scd_20230803',
                           'scd_20230805', 'scd_20230806',
                           'scd_20230807']  # jeeves NIF receive coil sessions

    FUNC_WM_PATH = os.path.join(CONTENT_ROOT, "subjects", SUBJECT, "mri", "func_wn.nii")
    BRAIN_MASK = os.path.join(
        CONTENT_ROOT, "subjects", SUBJECT, "mri", "no_cereb_decode_mask.nii.gz"
    )
    DATA_KEY_PATH = os.path.join(
        CONTENT_ROOT,
        "subjects",
        SUBJECT,
        "analysis",
        "shape_color_attention_decode_stimulus_response_data_key.csv",
    )
    # DATA_KEY_PATH = "/media/data/Users/Helen/dk_test.csv"

    CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]
    CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]

    if SUBJECT == "wooster":
        crop = CROP_WOOSTER
    else:
        crop = CROP_JEEVES

    all_classes = set(range(1, 15))
    ignore_classes = all_classes - set(USE_CLASSES)

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

    dir_path = os.path.join(CONTENT_ROOT, "subjects", SUBJECT, "rois", ROI_SET_NAME)

    roi_mask_paths = [
        f for f in os.listdir(dir_path) if ".nii.gz" in f and "~" not in f
    ]

    comp = [PCA_COMP]

    roi_train_data = [list() for _ in roi_mask_paths]
    roi_cross_data = [list() for _ in roi_mask_paths]
    roi_names = []

    # big_cl = [] # 0928 works
    # big_class_accs = [] # 0928 works
    # big_rois = [] # 0928 works

    for i in range(boot_folds):
        og_train_stim_data, train_targets = get_full_set(train_set, resample=True)
        og_in_train_stim_data = og_train_stim_data[: int(85 * len(train_targets) / 100)]
        in_train_targets = train_targets[: int(85 * len(train_targets) / 100)]
        og_in_test_stim_data = og_train_stim_data[int(85 * len(train_targets) / 100) :]
        in_test_targets = train_targets[int(85 * len(train_targets) / 100) :]
        og_test_stim_data, test_targets = get_full_set(test_set, resample=False)

        # fold_cl = [] # 0928 works
        # fold_class_acc = [] # 0928 works
        # fold_rois = [] # 0928 works

        for j, mask_path in enumerate(roi_mask_paths):
            train_accs = []
            cross_accs = []
            # all_cls = [] # 0928 works
            # all_class_accs = [] # 0928 works
            # roi_by_class = [] works
            roi_name = mask_path.split(".")[0]
            if i == 0:
                roi_names.append(roi_name)
            print("\nPROC. ROI", roi_name)
            roi_mask = nib.load(os.path.join(dir_path, mask_path)).get_fdata()
            roi_mask = roi_mask[
                MTurk1.crop[0][0] : MTurk1.crop[0][1],
                MTurk1.crop[1][0] : MTurk1.crop[1][1],
                MTurk1.crop[2][0] : MTurk1.crop[2][1],
            ]
            roi_mask = (roi_mask > 0.7).squeeze()

            all_mask = roi_mask

            if use_binary_target:
                targets = sorted(list(set(range(14)) - set(ignore_classes)))
            else:
                targets = [None]
            for target_class in targets:
                train_accs.append([])
                cross_accs.append([])
                # all_cls.append([]) # 0928
                # all_class_accs.append([]) # 0928
                # roi_by_class.append([]) # 0928

                MTurk1.one_v_rest_target = target_class

                # get train set data
                train_stim_data = og_train_stim_data[:, :, all_mask]
                train_stim_data = train_stim_data.reshape(
                    (train_stim_data.shape[0], -1)
                )
                train_stim_data = (
                    train_stim_data - train_stim_data.mean()
                ) / train_stim_data.std()

                # get in train set data
                in_train_stim_data = og_in_train_stim_data[:, :, all_mask]
                in_train_stim_data = in_train_stim_data.reshape(
                    (in_train_stim_data.shape[0], -1)
                )
                in_train_stim_data = (
                    in_train_stim_data - in_train_stim_data.mean()
                ) / in_train_stim_data.std()

                # get in  test set data
                in_test_stim_data = og_in_test_stim_data[:, :, all_mask]
                in_test_stim_data = in_test_stim_data.reshape(
                    (in_test_stim_data.shape[0], -1)
                )
                in_test_stim_data = (
                    in_test_stim_data - in_test_stim_data.mean()
                ) / in_test_stim_data.std()

                # get test set data
                test_stim_data = og_test_stim_data[:, :, all_mask]
                test_stim_data = test_stim_data.reshape((test_stim_data.shape[0], -1))
                test_stim_data = (
                    test_stim_data - test_stim_data.mean()
                ) / test_stim_data.std()

                # all_data = np.concatenate([train_stim_data, test_stim_data], axis=0)
                # mean_data = all_data.mean(axis=0)
                # select_voxels = np.argsort(mean_data)[:min(len(mean_data), 150)] #
                # sel_all_data = all_data[:, select_voxels]
                # sel_train_stim_data = train_stim_data[:, select_voxels]
                # sel_test_stim_data = test_stim_data[:, select_voxels]
                # print("Initial Dim:", all_data.shape[1])

                def fit_transform_svm(x, y, tx, ty, roi):
                    svc_model = Classifier(
                        max_iter=5000, fit_intercept=False
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
                    # cross_accs[-1].append(test_acc)

                    # 0928 works
                    # cl = []
                    # class_accs = []
                    # rois = []
                    # for i in set(ty):
                    #    ty2 = np.array(ty)
                    #    print(ty2)
                    #    idx = np.where(ty2 == i)
                    #    print(idx)
                    #    ty2 = ty2[idx]
                    #    print(ty2)
                    #    cross_y_hat2 = cross_y_hat[idx]
                    #    print("preds:", cross_y_hat2)
                    #    print("break")
                    #    class_acc = np.count_nonzero(cross_y_hat2 == ty2) / len(ty2)
                    #    cl.append(i)
                    #    class_accs.append(class_acc)
                    #    rois.append(roi)
                    # all_cls.append(cl) # no
                    # all_class_accs.append(class_accs) # no
                    # roi_by_class.append(roi_name) # no
                    return train_acc, test_acc

                for m_c in comp:
                    # pca = Embedder(n_components=c, whiten=False)
                    #pca.fit(in_train_stim_data)
                    proj_train_stim_data = (
                        in_train_stim_data  # pca.transform(in_train_stim_data) i
                    )
                    proj_test_stim_data = (
                        in_test_stim_data  # pca.transform(in_test_stim_data)
                    )
                    # proj_train_stim_data = in_train_stim_data
                    # proj_test_stim_data = in_test_stim_data
                    # proj_train_stim_data = (proj_train_stim_data - proj_train_stim_data.mean()) / proj_train_stim_data.std()
                    # proj_test_stim_data = (proj_test_stim_data - proj_test_stim_data.mean()) / proj_test_stim_data.std()

                    print(
                        "IN 2 IN TRAIN: target:",
                        target_class,
                        " examples:",
                        proj_train_stim_data.shape[0],
                        " roi_features:",
                        proj_train_stim_data.shape[1],
                    )

                    _, test_acc = fit_transform_svm(
                        proj_train_stim_data,
                        in_train_targets,
                        proj_test_stim_data,
                        in_test_targets,
                        roi_name,
                    )
                    train_accs[-1].append(test_acc)

                    # pca = Embedder(n_components=c, whiten=False)
                    #pca.fit(train_stim_data)
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
                        roi_name,
                    )
                    cross_accs[-1].append(test_acc)
                    # all_cls[-1].append(cl)  # 0928
                    # all_class_accs[-1].append(class_accs)  # 0928
                    # roi_by_class[-1].append(roi_name)
                    # all_cls.append(cl) # 0928 works
                    # all_class_accs.append(class_accs) # 0928 works
                    # roi_by_class.append(roi_name) # 0928 works

            train_accs = np.array(train_accs)
            cross_accs = np.array(cross_accs)
            roi_train_data[j].append(np.mean(train_accs))
            roi_cross_data[j].append(np.mean(cross_accs))

            # fold_cl.append(all_cls) # 0928 works
            # fold_class_acc.append(all_class_accs) # 0928 works
            # fold_rois.append(roi_by_class) # 0928 works

            # all_class_accs = np.array(all_class_accs)
            # print("look here")
            # print(all_cls)
            # print(all_class_accs)

            print(roi_name, " All Cross Accs:", cross_accs)
            print(roi_name, "All Train Accs:", train_accs)
        print("Done with bootfold:", i)

        # big_cl.append(fold_cl) # these all work
        # big_class_accs.append(fold_class_acc)
        # big_rois.append(fold_rois)
        # by_class = pd.DataFrame({'class': big_cl, 'accuracy': big_class_accs, 'roi': big_rois})
        # by_class = by_class.groupby(['class', 'roi'], as_index=False)['accuracy'].mean()
        # by_class.to_csv('/media/data/Users/Helen/class_accuracies_color_oppositeshapes_preaugust.csv')

    fig, ax = plt.subplots(1)
    fig.tight_layout()
    roi_train_data = np.array(roi_train_data)
    roi_cross_data = np.array(roi_cross_data)

    train_var = roi_train_data.std(axis=1)
    cross_var = roi_cross_data.std(axis=1)
    roi_train_data = roi_train_data.mean(axis=1)
    roi_cross_data = roi_cross_data.mean(axis=1)

    print(roi_train_data.shape)

    # to_plot = pd.DataFrame({'within': roi_train_data, 'across': roi_cross_data, 'roiz': roi_names})
    # to_plot.to_csv('/media/data/Users/Helen/to_plot_color_oppositeshapes_preaugust.csv')

    x_ax = np.arange(len(roi_train_data))
    ax.bar(x_ax, roi_train_data)
    ax.bar(x_ax, roi_cross_data)
    ax.errorbar(x_ax, roi_train_data, yerr=train_var, ls="none", c="black")
    ax.errorbar(x_ax, roi_cross_data, yerr=cross_var, ls="none", c="black")

    ax.set_xticks(np.arange(len(roi_train_data)))
    ax.set_xticklabels(roi_names, visible=True)
    plt.xticks(rotation=50)
    plt.show()
    fig.savefig(TRAIN_SET + "_roi_decoder_out.svg")
    results = pd.DataFrame({'roi': roi_names, 'identity_mean': roi_train_data, 'identity_std': train_var,
                            'cross_mean': roi_cross_data, 'cross_std': cross_var})
    results.to_csv('/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/wooster/rois/sl_dyloc_regions_final/he_func_space_rois/hef_edits_1027/trainshape_noNIF_csv.csv')

