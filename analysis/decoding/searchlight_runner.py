import os
import pickle

from scipy.ndimage import gaussian_filter
import torch
import neurotools
import numpy as np
from dataloader import TrialDataLoader
from val_dataloader import SimpleValDataLoader
import nibabel as nib
import datetime
import pickle as pk

from matplotlib import pyplot as plt


if __name__ == '__main__':

    # exp setup
    SUBJECTS = ["jeeves", "wooster"]
    SET = "ALL"
    IN_SETS = ['shape', 'color']
    X_SETS = ['color', 'shape']

    # model hyperparams
    EPOCHS = 4500
    LR = .01
    KERNEL = 3  # true rf = kernel + (n_layers - 1)
    REG = .02
    MODE = "linear"
    N_LAYERS = 3
    STRATEGY = "multiclass"
    BATCH_SIZE = 15
    SPATIAL_NORMALIZATION = None

    if STRATEGY not in ["multiclass", "binary", "pairwise"]:
        raise ValueError

    COTENT_ROOT = "/home/bizon/Projects/MTurk1/MTurk1"
    USE_CLASSES = [2, 3, 6, 7, 10, 11]
    DSET = "MTURK1"
    LOAD_MODEL = None # "/home/bizon/Projects/MTurk1/MTurk1/analysis/decoding/models/jeeves_searchlight_linear_color_2_shape_pairwise_r0.075_l3"

    for SUBJECT in SUBJECTS:
        DATA_KEY_PATH = "/home/bizon/Projects/MTurk1/MTurk1/subjects/" + SUBJECT + "/analysis/shape_color_attention_decode_stimulus_response_data_key.csv"
        for s in range(len(IN_SETS)):
            IN_SET = IN_SETS[s]
            X_SET = X_SETS[s]
            CROP_WOOSTER = [(37, 93), (15, 79), (0, 42)]
            CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]
            if SUBJECT == 'wooster':
                crop = CROP_WOOSTER
            elif SUBJECT == 'jeeves':
                crop = CROP_JEEVES
            else:
                exit(1)

            all_classes = set(range(1, 15))
            ignore_classes = all_classes - set(USE_CLASSES)

            dev = 'cuda'

            bootstrap_accuracies = np.empty((2, 2, 0))
            bootstrap_masks = [[list() for _ in range(2)] for _ in range(2)]
            bootstrap_sal = [[list() for _ in range(2)] for _ in range(2)]

            if SPATIAL_NORMALIZATION is None:
                name = SUBJECT + "_searchlight_" + MODE + "_" + IN_SET + "_2_" + X_SET + "_" + STRATEGY + "_r" + str(REG) + "_l" + str(N_LAYERS)
            else:
                name = SUBJECT + "_searchlight_" + MODE + "_" + IN_SET + "_2_" + X_SET + "_" + STRATEGY + "_r" + str(REG) + "_l" + str(N_LAYERS) + "_spatial_norm"


            if LOAD_MODEL is None:
                boot_dir = os.path.join(COTENT_ROOT, "analysis", "decoding", "models", name)
            else:
                boot_dir = LOAD_MODEL

            if not os.path.isdir(boot_dir):
                os.mkdir(boot_dir)

            total_in_ce = None
            total_in_acc = None
            total_x_ce = None
            total_x_acc = None

            if STRATEGY == "binary":
                for c in USE_CLASSES:
                    MTurk1 = TrialDataLoader(
                        DATA_KEY_PATH,
                        50,
                        content_root=COTENT_ROOT, ignore_class=ignore_classes, crop=crop, binary_target=c)

                    x_decoder = neurotools.decoding.SearchlightDecoder(KERNEL, pad=1, stride=1, lr=LR, reg=REG, device="cuda",
                                                                       num_layer=N_LAYERS, n_classes=2, hidden_channels=1, channels=2)

                    abv_map = {"color": "color_all",
                               "shape": "uncolored_shape_all"}

                    train_set = MTurk1.batch_iterator(abv_map["shape"], resample=True, num_train_batches=EPOCHS, n_workers=12, standardize=True)

                    # train
                    x_decoder.fit(train_set)

                    # get in modality maps
                    train_set = MTurk1.batch_iterator(abv_map["shape"], resample=False, num_train_batches=10, n_workers=2, standardize=True)
                    in_acc_map, in_ce_map = x_decoder.evaluate(train_set)

                    # get cross modality maps
                    test_set = MTurk1.batch_iterator(abv_map["color"], resample=False, num_train_batches=10, n_workers=2, standardize=True)
                    x_acc_map, x_ce_map = x_decoder.evaluate(test_set)

                    if total_x_acc is None:
                        total_x_ce = x_ce_map
                        total_x_acc = x_acc_map
                        total_in_ce = in_ce_map
                        total_in_acc = in_acc_map
                    else:
                        total_x_ce += x_ce_map
                        total_x_acc += x_acc_map
                        total_in_ce += in_ce_map
                        total_in_acc += in_acc_map

                    def map_2_nifti(data, fname):
                        data = data.numpy()
                        data = MTurk1.to_full(data)
                        # data = gaussian_filter(data, sigma=1)
                        nii = nib.Nifti1Image(data, header=MTurk1.header, affine=MTurk1.affine)
                        nib.save(nii, os.path.join(boot_dir, fname))

                    map_2_nifti(in_ce_map, IN_SET + "2" + IN_SET + "_c" + str(c) + "_searchlight_CE.nii.gz")
                    map_2_nifti(in_acc_map, IN_SET + "2" + IN_SET + "_c" + str(c) + "_searchlight_ACC.nii.gz")

                    map_2_nifti(x_ce_map, IN_SET + "2" + X_SET + "_c" + str(c) + "_searchlight_CE.nii.gz")
                    map_2_nifti(x_acc_map, IN_SET + "2" + X_SET + "_c" + str(c) + "_searchlight_ACC.nii.gz")
                total_x_ce /= len(USE_CLASSES)
                total_in_acc /= len(USE_CLASSES)
                total_in_ce /= len(USE_CLASSES)
                total_x_acc /= len(USE_CLASSES)

            elif STRATEGY == "pairwise":
                classes = sorted(list(USE_CLASSES))
                n_comparisons = int(((len(classes) - 1) * (len(classes) - 2)) / 2)
                print("running", n_comparisons, "models")
                acc_x_pairwise_triu = []
                ce_x_pairwise_triu = []
                acc_in_pairwise_triu = []
                ce_in_pairwise_triu = []
                count = 0
                for i, c1 in enumerate(classes):
                    for j, c2 in enumerate(classes[i + 1:]):
                        MTurk1 = TrialDataLoader(
                            DATA_KEY_PATH,
                            25,
                            content_root=COTENT_ROOT, ignore_class=ignore_classes, crop=crop,
                            binary_target=c1, binary_other=c2)
                        if LOAD_MODEL is None:

                            print("***************************************")
                            print("RUNNING", IN_SET, "->", X_SET, "class", c1, "vs.", c2)
                            print("***************************************")
                            x_decoder = neurotools.decoding.SearchlightDecoder(KERNEL, pad=1, stride=1, lr=LR, reg=REG, device="cuda",
                                                                               num_layer=N_LAYERS, n_classes=2, hidden_channels=1, channels=2,
                                                                               standardization_mode=SPATIAL_NORMALIZATION)

                            abv_map = {"color": "color_all",
                                       "color_train": "color_train",
                                       "color_test": "color_test",
                                       "shape": "uncolored_shape_all"}

                            train_set = MTurk1.batch_iterator(abv_map[IN_SET], resample=True, num_train_batches=EPOCHS, n_workers=8,
                                                              standardize=True)

                            # Fit on all in data and predict on all test data
                            # train
                            x_decoder.fit(train_set)

                            # get cross modality maps
                            test_set = MTurk1.batch_iterator(abv_map[X_SET], resample=False, num_train_batches=100, n_workers=2,
                                                             standardize=True)
                            x_acc_map, x_ce_map = x_decoder.evaluate(test_set)


                            # get in modality maps
                            train_set = MTurk1.batch_iterator(abv_map[IN_SET + "_train"], resample=True, num_train_batches=EPOCHS, n_workers=2,
                                                              standardize=True)
                            x_decoder.fit(train_set)

                            test_set = MTurk1.batch_iterator(abv_map[IN_SET + "_test"], resample=False, num_train_batches=100, n_workers=2,
                                                             standardize=True)

                            in_acc_map, in_ce_map = x_decoder.evaluate(test_set)

                        else:
                            in_ce_map = MTurk1.crop_volume(nib.load(os.path.join(boot_dir,
                                             IN_SET + "2" + IN_SET + "_c" + str(c1) + "_o" + str(c2) + "_searchlight_CE.nii.gz")).get_fdata())
                            in_acc_map = MTurk1.crop_volume(nib.load(os.path.join(boot_dir,
                                             IN_SET + "2" + IN_SET + "_c" + str(c1) + "_o" + str(c2) + "_searchlight_ACC.nii.gz")).get_fdata())

                            x_ce_map = MTurk1.crop_volume(nib.load(os.path.join(boot_dir,
                                             IN_SET + "2" + X_SET + "_c" + str(c1) + "_o" + str(c2) + "_searchlight_CE.nii.gz")).get_fdata())
                            x_acc_map = MTurk1.crop_volume(nib.load(os.path.join(boot_dir,
                                             IN_SET + "2" + X_SET + "_c" + str(c1) + "_o" + str(c2) + "_searchlight_ACC.nii.gz")).get_fdata())

                        count = count + 1
                        # need to eliminate negative ce regions so smoothing doesn't obscure small positive areas.
                        # really only an issue for x-decode, there are no negative regions in in-set.
                        in_ce_map[in_ce_map < 0] = 0
                        x_ce_map[x_ce_map < 0] = 0

                        in_ce_map = gaussian_filter(in_ce_map, .5)
                        in_acc_map = gaussian_filter(in_acc_map, .5)
                        x_ce_map = gaussian_filter(x_ce_map, .5)
                        x_acc_map = gaussian_filter(x_acc_map, .5)

                        if total_x_acc is None:
                            total_x_ce = x_ce_map
                            total_x_acc = x_acc_map
                            total_in_ce = in_ce_map
                            total_in_acc = in_acc_map
                        else:
                            total_x_ce += x_ce_map
                            total_x_acc += x_acc_map
                            total_in_ce += in_ce_map
                            total_in_acc += in_acc_map

                        def map_2_nifti(data, fname):
                            # saves nifti for data and returns the max 5 avg.
                            if type(data) != np.ndarray:
                                data = data.numpy()
                            sdata = np.sort(data.flatten())[-5:]
                            metric = sdata.mean()
                            data = MTurk1.to_full(data)
                            # data = gaussian_filter(data, sigma=1)
                            nii = nib.Nifti1Image(data, header=MTurk1.header, affine=MTurk1.affine)
                            nib.save(nii, os.path.join(boot_dir, fname))
                            return metric


                        ce_in_pairwise_triu.append(map_2_nifti(in_ce_map, IN_SET + "2" + IN_SET + "_c" + str(c1) + "_o" + str(c2) + "_searchlight_CE.nii.gz"))
                        acc_in_pairwise_triu.append(map_2_nifti(in_acc_map, IN_SET + "2" + IN_SET + "_c" + str(c1) + "_o" + str(c2) + "_searchlight_ACC.nii.gz"))

                        ce_x_pairwise_triu.append(map_2_nifti(x_ce_map, IN_SET + "2" + X_SET + "_c" + str(c1) + "_o" + str(c2) + "_searchlight_CE.nii.gz"))
                        acc_x_pairwise_triu.append(map_2_nifti(x_acc_map, IN_SET + "2" + X_SET + "_c" + str(c1) + "_o" + str(c2) + "_searchlight_ACC.nii.gz"))

                def plot_pairwise_map(triu_data, ax, name, min=0.):
                    mat = np.zeros((len(classes), len(classes)))
                    triu_ind = np.triu_indices(len(classes), 1)
                    mat[triu_ind] = np.array(triu_data)
                    mat = mat + mat.T
                    ax.imshow(mat.squeeze(), vmin=min)
                    ax.set_title(name)

                acc_maps = [[None, None],
                           [None, None]]

                fig, axs = plt.subplots(2, 2)
                fig.set_size_inches(12, 12)
                plot_pairwise_map(ce_in_pairwise_triu, axs[0, 0], IN_SET + "2" + IN_SET + "_CE")
                plot_pairwise_map(ce_x_pairwise_triu, axs[0, 1], IN_SET + "2" + X_SET + "_CE")
                plot_pairwise_map(acc_in_pairwise_triu, axs[1, 0], IN_SET + "2" + IN_SET + "_ACC", min=.5)
                plot_pairwise_map(acc_x_pairwise_triu, axs[1, 1], IN_SET + "2" + X_SET + "_ACC", min=.5)
                plt.show()

                total_x_ce =  total_x_ce / count
                total_in_acc = total_in_acc / count
                total_in_ce = total_in_ce / count
                total_x_acc = total_x_acc / count

            elif STRATEGY == "multiclass":
                reweight = True

                if SUBJECT == "jeeves":
                    MTurk1 = TrialDataLoader(
                        DATA_KEY_PATH,
                        BATCH_SIZE,
                        content_root=COTENT_ROOT, ignore_class=ignore_classes, crop=crop,
                        ignore_sessions=[])
                else:
                    MTurk1 = TrialDataLoader(
                        DATA_KEY_PATH,
                        BATCH_SIZE,
                        content_root=COTENT_ROOT, ignore_class=ignore_classes, crop=crop,
                        ignore_sessions=["scd_20230804", "scd_20230806", "scd_20230824"])

                print("***************************************")
                print("RUNNING", IN_SET, "->", X_SET, "multiclass")
                print("***************************************")

                x_decoder = neurotools.decoding.SearchlightDecoder(KERNEL, pad=1, stride=1, lr=LR, reg=REG, device="cuda",
                                                                   num_layer=N_LAYERS, n_classes=len(USE_CLASSES), hidden_channels=2,
                                                                   channels=2, reweight=reweight)

                abv_map = {"color": "color_all",
                           "color_train": "color_train",
                           "color_test": "color_test",
                           "shape": "uncolored_shape_all"}

                train_set = MTurk1.batch_iterator(abv_map[IN_SET], resample=True, num_train_batches=EPOCHS, n_workers=8,
                                                  standardize=True)

                # Fit on all in data and predict on all test data
                # train
                x_decoder.fit(train_set)

                # get cross modality maps
                test_set = MTurk1.batch_iterator(abv_map[X_SET], resample=False, num_train_batches=100, n_workers=2,
                                                 standardize=True)
                x_acc_map, x_ce_map = x_decoder.evaluate(test_set)

                # get in modality maps
                train_set = MTurk1.batch_iterator(abv_map[IN_SET + "_train"], resample=True, num_train_batches=EPOCHS,
                                                  n_workers=2,
                                                  standardize=True)
                x_decoder.fit(train_set)

                test_set = MTurk1.batch_iterator(abv_map[IN_SET + "_test"], resample=False, num_train_batches=100, n_workers=2,
                                                 standardize=True)

                in_acc_map, in_ce_map = x_decoder.evaluate(test_set)

                print("Min class weights", x_decoder.class_weights.detach().cpu().min(dim=1))
                print("Max class weights", x_decoder.class_weights.detach().cpu().max(dim=1))
                print("AVG class weights", x_decoder.class_weights.detach().cpu().mean(dim=1))

                out = os.path.join(boot_dir, "model_binary.pkl")
                with open(out, "wb") as f:
                    pickle.dump(x_decoder, f)
                print("model saved to ", out)

                total_x_ce = x_ce_map
                total_in_acc = in_acc_map
                total_in_ce = in_ce_map
                total_x_acc = x_acc_map

            def map_2_nifti(data, fname):
                # saves nifti for data and returns the max 5 avg.
                if type(data) != np.ndarray:
                    data = data.numpy()
                sdata = np.sort(data.flatten())[-5:]
                metric = sdata.mean()
                data = MTurk1.to_full(data)
                #data = gaussian_filter(data, sigma=.5)
                nii = nib.Nifti1Image(data, header=MTurk1.header, affine=MTurk1.affine)
                nib.save(nii, os.path.join(boot_dir, fname))
                return metric


            print("***************************************")
            print("FINISHED", SUBJECT, IN_SET, "->", X_SET, STRATEGY)
            print("***************************************")


            map_2_nifti(total_in_ce, IN_SET + "2" + IN_SET + "_all" + "_searchlight_CE.nii.gz")
            map_2_nifti(total_in_acc, IN_SET + "2" + IN_SET + "_all" + "_searchlight_ACC.nii.gz")

            map_2_nifti(total_x_ce, IN_SET + "2" + X_SET + "_all" + "_searchlight_CE.nii.gz")
            map_2_nifti(total_x_acc, IN_SET + "2" + X_SET + "_all" + "_searchlight_ACC.nii.gz")