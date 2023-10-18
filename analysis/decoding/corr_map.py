import os
import pickle

import scipy.stats
from scipy import ndimage
from sklearn.feature_selection import f_regression
import torch
import neurotools
import numpy as np
from dataloader import TrialDataLoader
from decode_policy import MapDecode, LinearDecode
import nibabel as nib
import datetime
import pickle as pk
from scipy.stats import f

if __name__=="__main__":
    SUBJECT = "jeeves"
    EPOCHS = 20

    USE_CLASSES = [2, 3, 6, 7, 10, 11]
    CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]
    CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]

    DATA_KEY_PATH = "/home/bizon/Projects/MTurk1/MTurk1/subjects/" + SUBJECT + "/analysis/shape_color_attention_decode_stimulus_response_data_key.csv"
    COTENT_ROOT = "/home/bizon/Projects/MTurk1/MTurk1"

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

    s_set = MTurk1.batch_iterator("uncolored_shape_all", resample=False, standardize=True, num_train_batches=EPOCHS, n_workers=2)
    c_set = MTurk1.batch_iterator("color_all", resample=False, standardize=True, num_train_batches=EPOCHS, n_workers=2)


    def f_stat_from_wilks_trace(wilks, p, df_h, df_e):
        """
        Wilks -> F-stat approx From Rencher 2002, Methods of Multivariate Analysis, p.163
        Parameters
        ----------
        wilks: Tensor   shape (batch, wilks_scores)
        p: int   number of manova dimensions
        df_h: int   degrees of freedom for the hypothesis (number of groups - 1 )
        df_e: int   degrees of freedom for the error (number of examples - number of classes)

        Returns
        -------
        Tensor   The approximate F statistics for these wilks traces
        """
        # Below
        t = ((p**2 * df_h**2 - 4) / (p**2 + df_h**2 - 5))**(1/2)
        w = df_e + df_h - (.5 * (p + df_h + 1))
        f_df_1 = p * df_h
        f_p = w * t - .5*(p*df_h - 2)
        f_stat_approx = ((1 - torch.pow(wilks, 1 / t)) / torch.pow(wilks, 1 / t)) * (f_p / f_df_1)
        return f_stat_approx

    def sl_trace(eig_vals_a, eig_vals_b, eig_vecs_a, eig_vecs_b):
        # TODO Directional mode?
        # each arg has leading batch or spatial dimension
        eig_vals_a = eig_vals_a / (eig_vals_a + 1)
        eig_vals_b = eig_vals_b / (eig_vals_b + 1)
        print("shape a", eig_vals_a.shape, "shape b", eig_vals_b.shape)
        eig_a = eig_vals_a[:, :, None] * eig_vecs_a
        eig_b = eig_vals_b[:, :, None] * eig_vecs_b
        trace = torch.sum(torch.abs(eig_a * eig_b), dim=(1, 2))
        # trace = torch.sum(torch.abs(combined), dim=(1, 2))
        return trace

    def pillai_trace(eig_vals):
        # each arg has leading batch or spatial dimension
        trace = torch.zeros(eig_vals.shape[0])
        for e in eig_vals.T:
            trace += e / (e + 1)
        return trace


    def batch_mancova(data, targets):
        """
        Follows algorithm spec in Rencher 2002, Methods of Multivariate Analysis
        Compute the mancova for each group present in targets over data.

        Parameters
        ----------
        data: FloatTensor   shape (batch, examples, features). The input data.
        targets: IntTensor   shape (examples). The group that each example belongs to.

        Returns
        -------
        FloatTensor    the wilks trace for each item in batch
        """
        group_labels = list(np.unique(targets))
        group_data = []
        group_means = []
        p = data.shape[-1]
        for t in group_labels:
            g = data[:, targets == t, :]
            group_mean = torch.mean(g, dim=1)  # spatial, kernel
            group_data.append(g - group_mean.unsqueeze(1))  # (centered) spatial, batch, kernel
            group_means.append(group_mean)

        # center the unfolded data  across batch for each feature of each kernel
        unfolded_data = data - data.mean(dim=1).unsqueeze(1)

        # total sum of squares and cross-products matrices
        SSCP_T = unfolded_data.transpose(1, 2) @ unfolded_data  # spatial, kernel, kernel
        SSCP_W = torch.zeros(unfolded_data.shape[0], p, p, dtype=torch.float)

        for i, g in enumerate(group_data):
            # withing group sum of squares and cross-products matrices
            SSCP_W += g.transpose(1, 2) @ g  # spatial kernel kernel

        # between groups sum of squares and cross-products matrices
        SSCP_B = SSCP_T - SSCP_W

        # compute multivarte seperation between groupd
        S = torch.linalg.pinv(SSCP_W) @ SSCP_B  # spatial, kernel, kernel
        eig_vals, eig_vecs = torch.linalg.eig(S)  # (S, kernel), (S, kernel, kernel)
        return torch.abs(eig_vals), torch.real(eig_vecs)


    def spatial_stats(data, targets, kernel):
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
        data = torch.from_numpy(data).float()
        targets = torch.from_numpy(targets).int()
        unfolded_data: torch.Tensor = neurotools.util.unfold_nd(data, kernel_size=kernel, padding=1, spatial_dims=3) # batch, kernel**3, spatial
        unfolded_data = unfolded_data.permute((2, 0, 1))  # spatial, batch, kernel**3

        # Mancova adapted from Rencher 2002, Methods of Multivariate Analysis
        eig_vals, eig_vecs = batch_mancova(unfolded_data, targets)

        return eig_vals, eig_vecs

    def resize_to_in(data, shape):
        # deal with data resizing
        dim = round(data.shape[0] ** (1 / 3))
        stat = data.reshape((1, 1, dim, dim, dim))
        stat = torch.nn.functional.interpolate(stat, size=shape, mode="trilinear").squeeze().detach().numpy()
        stat = ndimage.gaussian_filter(stat, sigma=.5)
        return stat

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

    s_data, s_targets = get_full_dset(s_set, len(USE_CLASSES))
    c_data, c_targets = get_full_dset(c_set, len(USE_CLASSES))

    print("computing shape 2 shape spatial class decodability...")
    s_eig_vals, s_eig_vecs = spatial_stats(s_data, s_targets, kernel=2)  # probability that all shape groups were NOT drawn from the same distribution
    s_p = pillai_trace(s_eig_vals)

    print("computing color 2 color spatial class decodability...")
    c_eig_vals, c_eig_vecs = spatial_stats(c_data, c_targets, kernel=2)  # probability that all color groups were NOT drawn from the same distribution
    c_p = pillai_trace(c_eig_vals)

    print("computing inter-modality decodability...")
    s_joint = sl_trace(s_eig_vals, c_eig_vals, s_eig_vecs, c_eig_vecs)

    s_joint = resize_to_in(s_joint, MTurk1.shape)
    # c_joint = resize_to_in(c_joint, MTurk1.shape)
    s_p = resize_to_in(s_p, MTurk1.shape)
    c_p = resize_to_in(c_p, MTurk1.shape)

    # all_data = np.concatenate([s_data, c_data], axis=0)
    # mod_targets = np.array([0]*len(s_data) + [1] * len(c_data), dtype=int)  # group is zero for all shape data and 1 for all color data
    # x_p = compute_f_map(all_data, mod_targets, kernel=2)  # probability that shape and color data are both drawn from the same distribution
    #
    # joint = np.power((1 / x_p) * s_p * c_p, 1 / 3)  # geometric mean of pillai scores
    #
    #c_joint_map = MTurk1.to_full(c_joint)
    s_joint_map = MTurk1.to_full(s_joint)
    s_p_map = MTurk1.to_full(s_p)
    c_p_map = MTurk1.to_full(c_p)

    #c_joint_nii = nib.Nifti1Image(c_joint_map, affine=MTurk1.affine, header=MTurk1.header)
    s_joint_nii = nib.Nifti1Image(s_joint_map, affine=MTurk1.affine, header=MTurk1.header)
    sp_nii = nib.Nifti1Image(s_p_map, affine=MTurk1.affine, header=MTurk1.header)
    cp_nii = nib.Nifti1Image(c_p_map, affine=MTurk1.affine, header=MTurk1.header)
    # xp_nii = nib.Nifti1Image(cross_p_map, affine=MTurk1.affine, header=MTurk1.header)

    # nib.save(s_joint_nii, os.path.join(COTENT_ROOT, "analysis", "decoding", "models", SUBJECT + "s2c_joint_f_map.nii.gz"))
    nib.save(s_joint_nii, os.path.join(COTENT_ROOT, "analysis", "decoding", "models", SUBJECT + "joint_f_map.nii.gz"))
    nib.save(sp_nii, os.path.join(COTENT_ROOT, "analysis", "decoding", "models", SUBJECT + "_shape_f_map.nii.gz"))
    nib.save(cp_nii, os.path.join(COTENT_ROOT, "analysis", "decoding", "models", SUBJECT + "_color_f_map.nii.gz"))
   # nib.save(xp_nii, os.path.join(COTENT_ROOT, "analysis", "decoding", "models", SUBJECT + "_cross_p_map.nii.gz"))


