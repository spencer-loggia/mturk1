import os
import pickle

import torch
import neurotools
import numpy as np
from dataloader import TrialDataLoader
from decode_policy import MapDecode, LinearDecode
import nibabel as nib
import datetime
import pickle as pk


class NTCompliantWrap:

    def __init__(self, mt1: TrialDataLoader, num_epochs, resample=True):
        self.mt1 = mt1
        self.epochs = num_epochs
        self.abv_map = {"color": "color_all",
                        "shape": "uncolored_shape_all"}
        self.resample = resample

    def __getitem__(self, item: int):
        if item == 0:
            return self.mt1.batch_iterator(self.abv_map["shape"], resample=self.resample,
                                           num_train_batches=self.epochs, n_workers=5)
        if item == 1:
            return self.mt1.batch_iterator(self.abv_map["color"], resample=self.resample,
                                           num_train_batches=self.epochs, n_workers=5)
        else:
            raise IndexError


if __name__ == '__main__':
    LOAD_MODEL = False
    models_to_load = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/analysis/decoding/models/lh_20230822_jeeves_complete"
    BOOTSTRAP_ITER = 25
    LINEAR = True
    CONV = True
    SET = "4"

    if LOAD_MODEL:
        if os.path.isfile(models_to_load):
            boot_dir = os.path.dirname(models_to_load)
            models_to_load = [models_to_load]
        elif os.path.isdir(models_to_load):
            boot_dir = models_to_load
            models_to_load = [os.path.join(models_to_load, f) for f in os.listdir(models_to_load) if
                              "~" not in f and f[0] != "." and ".pkl" in f and "end_epoch" in f]
        else:
            raise RuntimeError
        BOOTSTRAP_ITER = len(models_to_load)

    FIT = True
    MASK_ONLY = False

    LR = .01

    KERNEL = 9


    COTENT_ROOT = "/home/bizon/Projects/MTurk1/MTurk1"
    USE_CLASSES = [2, 3, 6, 7, 10, 11]
    SUBJECT = "jeeves"
    FUNC_WM_PATH = "/home/bizon/Projects/MTurk1/MTurk1/subjects/" + SUBJECT + "/mri/func_wm.nii"
    BRAIN_MASK = "/home/bizon/Projects/MTurk1/MTurk1/subjects/" + SUBJECT + "/mri/no_cereb_decode_mask.nii.gz"
    DATA_KEY_PATH = "/home/bizon/Projects/MTurk1/MTurk1/subjects/" + SUBJECT + "/analysis/shape_color_attention_decode_stimulus_response_data_key.csv"
    HEMI = 'lh'
    IN_SET = 'shape'
    X_SET = 'color'
    EPOCHS = 2000
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

    if LINEAR and not CONV:
        mode = "linear"
    elif not LINEAR and CONV:
        mode = "convnet"
    elif LINEAR and CONV:
        mode = "linconv"
    else:
        raise ValueError

    name = SUBJECT + "_" + HEMI + "_unified_" + mode + "_set_" + str(SET)

    if not LOAD_MODEL:
        date = str(datetime.datetime.now().date())
        boot_name = date + "_" + name + "_" + str(BOOTSTRAP_ITER)
        boot_dir = os.path.join("models", boot_name)
        if not os.path.isdir(boot_dir):
            os.mkdir(boot_dir)

    for boot in range(BOOTSTRAP_ITER):
        print(BOOTSTRAP_ITER)
        MTurk1 = TrialDataLoader(
            DATA_KEY_PATH,
            150,
            content_root=COTENT_ROOT, ignore_class=ignore_classes, crop=crop)

        func_wm = nib.load(FUNC_WM_PATH).get_fdata()
        func_wm = torch.from_numpy(MTurk1.crop_volume(func_wm, cube=True)).float()
        force_mask = torch.zeros_like(func_wm)
        force_mask[func_wm < .75] = 1

        brain_mask = nib.load(BRAIN_MASK).get_fdata()
        brain_mask = torch.from_numpy(MTurk1.crop_volume(brain_mask, cube=True)).float()
        force_mask2 = brain_mask > .25

        force_mask = torch.logical_and(force_mask2, force_mask)

        if HEMI == "rh":
            force_mask[:32, :, :] = 0
        elif HEMI == 'lh':
            force_mask[32:, :, :] = 0

        force_mask_full = MTurk1.to_full(force_mask)
        mask_nii = nib.Nifti1Image(force_mask_full, header=MTurk1.header, affine=MTurk1.affine)
        nib.save(mask_nii, os.path.join(boot_dir, "default_mask.nii.gz"))

        if LINEAR and not CONV:
            policies = LinearDecode(force_mask, in_channels=2, out=len(USE_CLASSES))
        elif LINEAR and CONV:
            policies = tuple([MapDecode(shape=(64, 64, 64), dev=dev, out=len(USE_CLASSES), nonlinear=False) for _ in range(2)])
        elif not LINEAR and CONV:
            policies = tuple([MapDecode(shape=(64, 64, 64), dev=dev, out=len(USE_CLASSES)) for _ in range(2)])
        else:
            # not linear and not conv
            raise NotImplementedError

        if LOAD_MODEL:
            out_path = models_to_load[boot]
            with open(models_to_load[boot], "rb") as f:
                x_decoder = pickle.load(f)
                if FIT:
                    with torch.no_grad():
                        for i in range(2):
                            for j in range(2):
                                x_decoder.mask_base[i][j] += torch.normal(0, .001, size=x_decoder.mask_base[i][j].shape, device=x_decoder.device)
                                if i != j:
                                    x_decoder.mask_optim[i][j].lr = LR
                            x_decoder.decode_optim[i].lr = LR
                    x_decoder.smooth_kernel_size = KERNEL
                    x_decoder.smooth_kernel = x_decoder._create_smoothing_kernel(kernel_size=KERNEL)
        else:
            x_decoder = neurotools.decoding.GlobalMultiStepCrossDecoder(decoder=policies, smooth_kernel_size=KERNEL, input_spatial=(64, 64, 64),
                                                               input_channels=2, force_mask=force_mask, name=name,
                                                               save_dir=boot_dir, lr=LR, n_sets=2)

        gens = NTCompliantWrap(MTurk1, EPOCHS, resample=True)

        if FIT:
            x_decoder.restart_marks.append(len(x_decoder.loss_histories[0][0]))
            if MASK_ONLY:
                x_decoder.fit(gens, iters=EPOCHS, mask_only=True)
            else:
                x_decoder.fit(gens, iters=EPOCHS, mask_only=False)

        accs = np.array(x_decoder.accuracies)
        bootstrap_accuracies = np.concatenate([bootstrap_accuracies, accs[:, :, -10:]], axis=2)


       # x_decoder.plot_loss_curves()

        sal_gens = NTCompliantWrap(MTurk1, 10, resample=True)

        if x_decoder.sal_map is None:
            sal_map = x_decoder.compute_saliancy(sal_gens)
            x_decoder.sal_map = sal_map
        else:
            sal_map = x_decoder.sal_map

        for i in range(2):
            for j in range(2):
                mask = x_decoder.get_mask(x_decoder.mask_base[i][j], noise=False).detach().cpu().squeeze().numpy()
                bootstrap_masks[i][j].append(mask)
                sal_m = sal_map[i][j]
                bootstrap_sal[i][j].append(sal_m)

        if LOAD_MODEL:
            with open(out_path, "wb") as out:
                pickle.dump(x_decoder, out)
        else:
            x_decoder.save("end_epoch_" + str(len(x_decoder.loss_histories[0][0])))

        del x_decoder

    def generate_avg_nifti(batched_data, out_dir, tag):
        from scipy.ndimage import gaussian_filter
        # expects len(boot_iter) list of (x, y, z) arrays
        img = np.stack(batched_data)
        norm_effect_size = img / np.std(img)
        norm_effect_size = gaussian_filter(norm_effect_size, sigma=1)
        var = np.std(norm_effect_size, axis=0)
        mean_effect_size = np.mean(norm_effect_size, axis=0)
        zmap = (mean_effect_size) / var
        zmap[mean_effect_size < .25] = 0
        effect_size = MTurk1.to_full(mean_effect_size)
        zmap = MTurk1.to_full(zmap)
        effect_size_nii = nib.Nifti1Image(effect_size, header=MTurk1.header, affine=MTurk1.affine)
        zmap_nii = nib.Nifti1Image(zmap, header=MTurk1.header, affine=MTurk1.affine)
        out_path = os.path.join(out_dir, name + "_effect_" + tag + ".nii.gz")
        nib.save(effect_size_nii, out_path)
        out_path = os.path.join(out_dir, name + "_zmap_" + tag + ".nii.gz")
        nib.save(zmap_nii, out_path)

    set_names = ["shape", "color"]
    accs = np.array(bootstrap_accuracies)
    mean = np.mean(accs, axis=2)
    std = np.std(accs, axis=2)
    for i in range(2):
        for j in range(2):
            # compute confidence maps
            print(set_names[i] + " -> " + set_names[j] + " In ACC:", mean[i, j], "+/-", std[i, j] * 2)

            tag = set_names[i] + "2" + set_names[j]

            generate_avg_nifti(bootstrap_masks[i][j], boot_dir, tag + "_mask")
            generate_avg_nifti(bootstrap_sal[i][j], boot_dir, tag + "_sal")
