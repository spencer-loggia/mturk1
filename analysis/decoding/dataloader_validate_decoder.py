import math
import queue
import random

import torch
import neurotools
import numpy as np
import nibabel as nib
import pandas as pd
import pickle
import os
from neurotools import util
import multiprocessing as mp
from multiprocessing import queues

# val_map = {1:2, 2:4, 3:6, 4:8, 5:10, 6:12, 7:14,
#            8:2, 9:4, 10:6, 11:8, 12:10, 13:12, 14:14,
#            15:1, 16:3, 17:5, 18:7, 19:9, 20:11, 21:13,
#            22:1, 23:3, 24:5, 25:7, 26:9, 27:11, 28:13}
val_map = {1:1, 2:2, 3:1, 4:2}


class TrialDataLoader:
    """
    Designed to load batches of MTurk1 Decoding data from a data key csv file generated by "get_run_betas" subject level
    commands in parallel. That is, multiple batches are prepared and processed (lin combined, translated, etc.), while
    one is served.
    """
    def __init__(self, key_path, batch_size, crop=((9, 73), (0, 40), (13, 77)), content_root="./", device="cuda",
                 ignore_sessions=(), ignore_class=(), seed=4242, binary_target=None, folds=1, use_behavior=False, verbose=False):
        self.content_root = content_root
        data = pd.read_csv(key_path)
        data.drop("Unnamed: 0", axis=1, inplace=True)
        for ignore_sess in ignore_sessions:
            data = data[data["session"] != ignore_sess]
        self.data = data.sample(frac=1., ignore_index=True, random_state=seed)
        self.batch_size = batch_size
        self.full_size = None
        self.affine = None
        self.header = None
        self.one_v_rest_target = binary_target
        self.ignore_class = ignore_class
        self.use_behavior = use_behavior
        self.verbose = verbose
        
        self.current_fold = 0
        self.folds = 1

        color_set = self._get_set("colored_blobs", correct_only=use_behavior)
        self.color_all = color_set
        self.color_train = color_set[:int(85 * len(color_set) / 100)]
        self.color_test = color_set[int(85 * len(color_set) / 100):]

        uncolored_shape_set = self._get_set("uncolored_shapes", correct_only=use_behavior)
        self.uncolored_shape_all = uncolored_shape_set
        self.uncolored_shape_train = uncolored_shape_set[:int(85 * len(uncolored_shape_set) / 100)]
        self.uncolored_shape_test = uncolored_shape_set[int(85 * len(uncolored_shape_set) / 100):]

        self.crop = crop
        size = max([t[1] - t[0] for t in crop])
        self.shape = (size, size, size)
        self.device = device
        self._processed_ = 0
        self._max_q_size_ = 20

        self.set_mem = {}

    def _get_set(self, id, correct_only=True):
        data = self.data[self.data["condition_group"] == id]
        print("")
        if correct_only:
            data = data[data["correct"] == 1].reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)
        # vals = sorted(pd.unique(data["condition_integer"]))
        for i, item in enumerate(data["condition_integer"]):
            data.loc[i, "condition_integer"] = val_map[item]
        nii = nib.load(os.path.join(self.content_root, eval(data["beta_path"][0])[0]))
        self.full_size = nii.get_fdata().shape
        self.affine = nii.affine
        self.header = nii.header
        return data

    def to_full(self, data: np.array):
        full = np.zeros(self.full_size, dtype=float)
        crop_size = self.shape[0]
        crop = []
        for i, c in enumerate(self.crop):
            dim = c[1] - c[0]
            diff = crop_size - dim
            pad_l = int(math.floor(diff / 2))
            pad_r = int(math.ceil(diff / 2))
            crop.append((pad_l, crop_size - pad_r))
        full[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1], self.crop[2][0]:self.crop[2][1]] = \
            data[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
        return full

    def get_pop_stats(self, data_type):

        try:
            data = eval("self." + data_type.strip())
        except AttributeError:
            raise ValueError("This dataset was not defined.")

        beta_coef = []
        for paths_str in data["beta_path"]:
            paths = eval(paths_str)
            chan_beta = []
            for path in paths[1:2]:
                beta_nii = nib.load(os.path.join(self.content_root, path))
                betas = beta_nii.get_fdata()[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1], self.crop[2][0]:self.crop[2][1]]
                chan_beta.append(betas)
            chan_beta = np.stack(chan_beta, axis=0)
            beta_coef.append(chan_beta)
        beta_coef = np.stack(beta_coef, axis=0)
        mean = beta_coef.mean(axis=(0, 2, 3, 4))  # mean across each channel
        std = beta_coef.std(axis=(0, 2, 3, 4))  # std across ech channel
        return mean, std

    def crop_volume(self, input, cube=True):
        out = input[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1],
                self.crop[2][0]:self.crop[2][1]]
        if cube:
            out = util._pad_to_cube(out)
        return out

    def translate_volume(self, volume, x, y, z):
        """
        Translates a (width, height, depth, channels) volume by (x, y, z) pixels.
        """
        # Get the shape of the input volume
        c, w, h, d = volume.shape

        # Create a new volume with the same shape as the input volume
        translated_volume = np.zeros_like(volume)

        # Create new index arrays for the translated volume
        i_t = np.arange(w) - x
        j_t = np.arange(h) - y
        k_t = np.arange(d) - z

        # Adjust index arrays for negative translations
        if x < 0:
            i_t = np.arange(-x, w - x)
        if y < 0:
            j_t = np.arange(-y, h - y)
        if z < 0:
            k_t = np.arange(-z, d - z)

        # Clip the indices to the range [0, w-1], [0, h-1], [0, d-1]
        i_t = np.clip(i_t, 0, w - 1)
        j_t = np.clip(j_t, 0, h - 1)
        k_t = np.clip(k_t, 0, d - 1)

        # Use the new index arrays to index into the original volume and assign to the translated volume
        translated_volume[:, i_t[:, None, None], j_t[None, :, None], k_t[None, None, :, ]] = volume

        return translated_volume

    def data_generator(self, dset, noise_frac_var=.2, spatial_translation_var=.67, max_base_examples=3, cube=True):
        """
        Combines some random number of examples from one random class with random weight with some amount of random translation and noise
        Parameters
        ----------
        dset
        noise_frac_var
        spatial_translation_var
        max_base_examples

        Returns
        -------

        """
        options = sorted(list(set(range(1, 3)) - set(self.ignore_class)))
        if self.one_v_rest_target is not None:
            if random.choice([True, False]):
                example_class = self.one_v_rest_target
                target = 0
            else:
                options = set(options) - {self.one_v_rest_target}
                example_class = int(random.choice(list(options)))
                target = 1
        else:
            example_class = int(random.choice(options))  # which class will this be?
            target = options.index(example_class)
        basis_dim = np.random.randint(1, max_base_examples + 1)  # how many real examples to use as basis?
        trajectory = np.random.random((basis_dim, 1, 1, 1, 1))  # weight vector for combining basis
        trajectory = trajectory / np.sum(trajectory)
        class_dset = dset[dset['condition_integer'] == example_class]
        names = class_dset['condition_name']
        basis_idxs = np.random.randint(0, len(class_dset), size=(basis_dim,))
        if self.verbose:
            print("chose class", example_class, "desc", names.iloc[int(basis_idxs[0])], "basis examples", len(class_dset))

        data = class_dset.iloc[basis_idxs]
        beta_coef = []
        for z, path_str in enumerate(data["beta_path"]):
            chan_beta = []
            paths = eval(path_str)
            # ignore the final fir
            for path in paths[1:2]:
                if path not in self.set_mem:
                    beta_nii = nib.load(os.path.join(self.content_root, path))
                    betas = self.crop_volume(beta_nii.get_fdata(), cube=cube)
                    if np.count_nonzero(np.isnan(betas)) > 0:
                        print("WARNING: NaN values encountered in", data["session"].iloc[z], "IMA", data["ima"].iloc[z])
                    self.set_mem[path] = betas
                else:
                    betas = self.set_mem[path]
                chan_beta.append(betas)
            chan_beta = np.stack(chan_beta, axis=0)  # construct channel dimension of delays
            beta_coef.append(chan_beta)
        beta_coef = np.stack(beta_coef, axis=0)
        beta_coef = beta_coef * trajectory  # weight basis
        beta = beta_coef.sum(axis=0)
        if spatial_translation_var > 0.:
            trans_xyz = np.round(np.random.normal(loc=0, scale=spatial_translation_var, size=(3,)))
            beta = self.translate_volume(beta, int(trans_xyz[0]), int(trans_xyz[1]), int(trans_xyz[2]))
        var = np.std(beta)
        noise = np.random.normal(0, var * noise_frac_var, beta.shape)
        beta = beta + noise
        return beta, target

    def get_batch(self, bs, dset, resample=False, cube=True):
        beta_coef = []
        targets = []
        for _ in range(int(bs)):
            if resample:
                beta, target = self.data_generator(dset, noise_frac_var=.2, spatial_translation_var=0.67,
                                                         max_base_examples=3, cube=cube)
            else:
                beta, target = self.data_generator(dset, noise_frac_var=0.0, spatial_translation_var=0.0,
                                                         max_base_examples=3, cube=cube)
            beta_coef.append(beta)
            targets.append(target)
        targets = np.array(targets, dtype=int)
        beta_coef = np.stack(beta_coef, axis=0)
        return beta_coef, targets

    def data_queuer(self, dset, bs, num_batches, resample, standardize, mean, std, q, cube=True):
        while self._processed_ < num_batches:
            beta_coef, targets = self.get_batch(bs, dset, resample=resample, cube=cube)
            if standardize:
                beta_coef = (beta_coef - mean[None, :, None, None, None]) / std[None, :, None, None, None]
            try:
                q.put((beta_coef, targets), block=True, timeout=120)
            except queue.Full:
                print("The data queue is full and does not appear to be emptying.")
                del beta_coef
                del targets
                return

    def batch_iterator(self, data_type, num_train_batches=1000, return_all=False, standardize=False, resample=True, n_workers=16, cube=True,):
        try:
            dset = eval("self." + data_type.strip())
        except AttributeError:
            raise ValueError("This dataset was not defined.")
        mean = 0
        std = None
        if standardize:
            mean, std = self.get_pop_stats(data_type)
        dset = dset.sample(frac=1., ignore_index=True)
        if return_all:
            bs = len(dset)
        else:
            bs = self.batch_size
        self._processed_ = 0
        context = mp.get_context("fork")
        q = context.Queue(maxsize=self._max_q_size_)
        workers = []
        use_mp = n_workers > 1
        if use_mp:
            for i in range(n_workers):
                p = context.Process(target=self.data_queuer, args=(dset, bs, num_train_batches, resample, standardize, mean, std, q, cube))
                p.start()
                workers.append(p)

        for i in range(num_train_batches):
            if use_mp:
                try:
                    res = q.get(block=True, timeout=120)
                except queue.Full:
                    print("The data queue is empty and does not appear to be populating.")
                    break
                self._processed_ += 1
                beta_coef, targets = res
            else:
                beta_coef, targets = self.get_batch(bs, dset, resample=resample, cube=cube)
            yield beta_coef, targets

        if use_mp:
            print("killing...")
            q.close()
            for i in range(n_workers):
                workers[i].terminate()
                workers[i].kill()

