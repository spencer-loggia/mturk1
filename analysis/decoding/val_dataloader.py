import math
import queue
import random

import matplotlib.pyplot as plt
import torch
import neurotools
import numpy as np
import nibabel as nib
import pandas as pd
import pickle
import os

import torchvision
from neurotools import util
import multiprocessing as mp
from multiprocessing import queues


class SimpleValDataLoader:
    """
    generates cross decodable and non-cross docadable 2d rois. Only tests the simple case where cross decodable rois
    directly overlap with their template ROIs
    """

    def __init__(self, spatial=32, epochs=500, mode="simple", motion=True, structured_noise=True, n_classes=6):
        self.spatial = spatial
        # a only, b patterns that appear on a trials, a patterns that appear on b trial, b only
        self.roi_n = [2, 1, 1, 2]
        self.classes_per_modality = n_classes
        self.batch_size = 30
        self.epochs = epochs
        self.warp = motion

        # self.origins = [np.random.randint(low=0, high=32 - 5, size=(n, 2)) for n in self.roi_n]
        self.origins = [[[1, 1], [3, 21]],
                        [[10, 8]],
                        [[20, 18]],
                        [[27,16], [21, 24]]]
        seeds = [1, 2, 3, 4]
        self.template_patterns = []
        for s, n in enumerate(self.roi_n):
            np.random.seed(s * 3 + 2)  # Using deprecated np syntax cause I don't want to learn about random generator things
            self.template_patterns.append(np.random.random(size=(n, self.classes_per_modality, 4, 5)))
        np.random.seed(s + 2)
        self.a_noise_template = np.random.random((spatial, spatial))
        np.random.seed(s + 3)
        self.b_noise_template = np.random.random((spatial, spatial))

    def __getitem__(self, item: int):
        if item == 0:
            return self.batch_iterator(batch_size=self.batch_size, num_batches=self.epochs, modality="a")
        if item == 1:
            return self.batch_iterator(batch_size=self.batch_size, num_batches=self.epochs, modality="b")
        else:
            raise IndexError

    def generate_example(self, batch_size, noise: float, stim_type:str, use_pattern=True, warp=True):
        template = np.zeros((batch_size, self.spatial, self.spatial), dtype=float)
        class_labels = np.random.randint(low=0, high=self.classes_per_modality, size=(batch_size,))

        for i, type_origins in enumerate(self.origins):
            if stim_type == "a":
                warper = torchvision.transforms.RandomAffine(degrees=(0, 0),
                                                             translate=(.07, .02),
                                                             scale=(.80, .90),
                                                             shear=(-2, 3, -2, 4))
                template += self.a_noise_template * .3
                if i not in [0, 1, 2]:
                    continue
            elif stim_type == "b":
                warper = torchvision.transforms.RandomAffine(degrees=(0, 0),
                                                             translate=(.0, .1),
                                                             scale=(.85, .93),
                                                             shear=(-4, 0, -4, 0))
                template += self.b_noise_template * .2
                if i not in [1, 2, 3]:
                    continue
            else:
                def warper(x):
                    return x
            for j, origin in enumerate(type_origins):
                pattern = self.template_patterns[i][j, class_labels, :, :] * 2
                if use_pattern:
                    code = pattern
                    if noise > 0:
                        code += (np.random.normal(0., noise, pattern.shape))
                else:
                    code = i + 1
                template[:, origin[0]:origin[0] + pattern.shape[1],
                            origin[1]:origin[1] + pattern.shape[2]] += code
        np.random.seed(None)
        if noise > 0.:
            template = template + np.random.normal(0., noise, size=template.shape) # + np.random.random(size=template.shape)
        if warp:
            template = warper(torch.from_numpy(template)).numpy()
        if stim_type == "a" or stim_type == "b":
            print(template.std().max())
            template = (template - np.expand_dims(template.mean(axis=0), 0)) / np.expand_dims(template.std(axis=0), 0)
            template[np.isnan(template)] = 0
        return template.squeeze(), class_labels

    def plot_template_pattern(self):
        fig, ax = plt.subplots(1)
        template, _ = self.generate_example(1, 0., "all", use_pattern=False)
        template = template.squeeze()
        plt.imshow(template)
        fig.show()

    def data_generator(self, modality:str, noise=.1):
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
        options = ["a", "b"]
        choice = random.choice(options)
        data = self.generate_example(.1, modality, choice)
        return data, choice

    def batch_iterator(self, modality:str, batch_size=60, num_batches=500):
        for epoch in range(num_batches):
            data, targets = self.generate_example(batch_size=batch_size,
                                                  noise=.6,
                                                  stim_type=modality)
            data = data[:, None, :, :]
            print(data.shape)
            yield data, targets

