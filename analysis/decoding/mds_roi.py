import os
import pickle

import scipy.stats
from scipy import ndimage
from sklearn.feature_selection import f_regression
import torch
from neurotools.geometry import dissimilarity_from_supervised
from neurotools.embed import MDScale
import numpy as np
from dataloader import TrialDataLoader
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from decode_policy import MapDecode, LinearDecode
import nibabel as nib
import datetime
import pickle as pk
from scipy.stats import f


def get_decoding_rois(acc_map, roi_dir, roi_keys, thresh=0.21):
    """
    Goes through rois, returns index in each roi where decoding is significant
    """
    acc_map = nib.load(acc_map).get_fdata()
    rois = [f for f in os.listdir(roi_dir) if ".nii.gz" in f and "~" not in f]
    out_rois = [None] * len(roi_keys)
    for i, key in enumerate(roi_keys):
        for f in rois:
            if key in f:
                path = os.path.join(roi_dir, f)
                roi = nib.load(path).get_fdata()
                ind = roi > .5
                decode_roi_ind = np.logical_and(ind, acc_map > thresh)
                if out_rois[i] is None:
                    out_rois[i] = decode_roi_ind
                else:
                    out_rois[i] = np.logical_and(out_rois[i], decode_roi_ind)
    return out_rois


if __name__ == "__main__":
    SUBJECT = "wooster"
    ROI_DIR = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/wooster/rois/IT-subdivisions/auto_func_space_rois"
    # ROI_DIR = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/jeeves/rois/IT-subdivisions/auto_func_space_rois"

    S2C_ACC_MAP = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/wooster/analysis/global_decode/shape2color_all_searchlight_ACC.nii.gz"
    # S2C_ACC_MAP = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/jeeves/analysis/global_decode/shape2color_all_searchlight_ACC.nii.gz"

    C2S_ACC_MAP = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/wooster/analysis/global_decode/color2shape_all_searchlight_ACC.nii.gz"
    # C2S_ACC_MAP = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/jeeves/analysis/global_decode/color2shape_all_searchlight_ACC.nii.gz"

    ROI_KEYS = ["V4", "post_IT", "mid_IT", "ant_IT", "TP", "peri_ento"]

    decode_rois = get_decoding_rois(S2C_ACC_MAP, ROI_DIR, ROI_KEYS)

    EPOCHS = 20

    # USE_CLASSES = [2, 3, 6, 7, 10, 11]
    USE_CLASSES = [1, 4, 5, 8, 9, 12]
    c_colors = ["red", "yellow", "green", "cyan", "blue", "purple"]
    CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]
    CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]

    DATA_KEY_PATH = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/" + SUBJECT + "/analysis/shape_color_attention_decode_stimulus_response_data_key.csv"
    COTENT_ROOT = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1"

    all_classes = set(range(1, 15))
    ignore_classes = all_classes - set(USE_CLASSES)

    if SUBJECT == 'wooster':
        crop = CROP_WOOSTER
    elif SUBJECT == 'jeeves':
        crop = CROP_JEEVES
    else:
        exit(1)

    MTurk1 = TrialDataLoader(
        DATA_KEY_PATH,
        40,
        content_root=COTENT_ROOT, ignore_class=ignore_classes, crop=crop)

    s_set = MTurk1.batch_iterator("uncolored_shape_all", resample=False,
                                  standardize=True, num_train_batches=EPOCHS,
                                  n_workers=2)
    c_set = MTurk1.batch_iterator("color_all", resample=False,
                                  standardize=True, num_train_batches=EPOCHS,
                                  n_workers=2)


    def get_full_dset(dloader, n_classes):
        shape = list(MTurk1.shape) + [n_classes]
        # class_avg_map = np.zeros(shape, dtype=float)
        class_data = []
        targets = []
        for stim, target in dloader:
            # stim = ndimage.gaussian_filter(stim, 1.0, axes=(2, 3, 4))
            # class_avg_map[:, :, :, target] += stim.mean(axis=1).transpose((1, 2, 3, 0))  # add stim avg across channels to avg map
            class_data.append(stim)
            targets.append(target)

        class_data = np.concatenate(class_data, axis=0)
        data_shape = class_data.shape
        # class_data = class_data.reshape((data_shape[0], -1))
        targets = np.concatenate(targets).squeeze()
        return class_data, targets


    full_s_data, s_targets = get_full_dset(c_set, len(USE_CLASSES))
    # c_data, c_targets = get_full_dset(c_set, len(USE_CLASSES), roi_path=ROI_PATH)

    print("computing s2s pairwise dissimilarity matrix")


    def spatial_stats(data, targets):
        """
        Parameters
        ----------
        data: array, shape batch, x, y, z
        targets: array, batch
        kernel: the dimension of the spatial kernel. int
        Returns
        -------
        """
        # group_labels = list(np.unique(targets))
        # df_n = len(group_labels) - 1
        # df_d = len(data) - len(group_labels)
        examples = data.shape[0]
        data = torch.from_numpy(data).float()
        targets = torch.from_numpy(targets).int()
        data = data.reshape((1, examples, -1))
        matrix = dissimilarity_from_supervised(data, targets, metric="dot")
        return matrix.squeeze()


    for roi_data in decode_rois:
        s_data = full_s_data.reshape((full_s_data.shape[0], full_s_data.shape[1], -1))
        roi_data = np.tile(MTurk1.crop_volume(roi_data, cube=True).reshape((1, 1, -1)), (s_data.shape[0], s_data.shape[1], 1))
        s_data = s_data[roi_data > .5].reshape((s_data.shape[0], -1))

        n_classes = len(USE_CLASSES)
        dissim = spatial_stats(s_data,
                               s_targets)  # probability that all shape groups were NOT drawn from the same distribution

        dim = 2
        md_scaler = MDScale(n=n_classes, embed_dims=dim)

        latent = md_scaler.fit_transform(dissim, max_iter=100000, tol=.001)

        fig = plt.figure()
        if dim == 2:
            ax = fig.add_subplot()
            ax.scatter(latent[:, 0], latent[:, 1], color=c_colors)
        else:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2],
                       color=c_colors)
        plt.show()

