from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
import torch

EXTS = ("png", "tiff", "tif")


class WoundHealingDataset(Dataset):
    def __init__(self, data_path, masks_path, target_size: tuple, device, augment=False, add_noise=False,
                 random_crop=False, wound_only=False, check_data_pair_names=False):
        self.data_path = data_path
        self.masks_path = masks_path
        self.target_size = target_size
        self.data_ids, self.gt_mask_ids = get_wound_healing_datasets(self.data_path, self.masks_path,
                                                                     check_data_pair_names)
        self.length = np.concatenate(self.gt_mask_ids).size
        self.device = device
        self.augment = augment
        self.add_noise = add_noise
        self.random_crop = random_crop
        self._augment_p = 0.5  # Augmentation probability.
        self.wound_only = wound_only

        self._dataset_ranges = np.array([0])
        for id_set in self.gt_mask_ids:
            self._dataset_ranges = np.append(self._dataset_ranges,
                                             self._dataset_ranges[-1] + len(id_set))
        self._set_count = len(self.gt_mask_ids)

    def set_augment(self, do_augment):
        self.augment = do_augment

    def get_new_paths_dataset(self):
        return WoundHealingPathsDataset(self.data_path, self.masks_path, self.target_size, self.device)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Calculate corresponding dataset and item ids.
        dataset_idx = np.searchsorted(self._dataset_ranges, idx, side='right') - 1
        item_idx = idx - self._dataset_ranges[dataset_idx]

        real_img = Image.open(self.data_ids[dataset_idx][item_idx])
        cond_img = Image.open(self.gt_mask_ids[dataset_idx][item_idx])

        if self.random_crop:
            crop_pars = transforms.RandomCrop.get_params(real_img, self.target_size)
            real_img = transforms.functional.crop(real_img, crop_pars[0], crop_pars[1], crop_pars[2],
                                                  crop_pars[3]).convert('RGB')
            cond_img = transforms.functional.crop(cond_img, crop_pars[0], crop_pars[1], crop_pars[2],
                                                  crop_pars[3]).convert('RGB')
        else:
            # TODO: Resize as tensor using GPU using Albumentations or Kornia libraries.?
            real_img = real_img.resize(self.target_size, resample=Image.BICUBIC).convert('RGB')
            cond_img = cond_img.resize(self.target_size, resample=Image.NEAREST).convert('RGB')

        if self.augment and random.random() < self._augment_p:
            # real_img = self._transform(real_img)
            # cond_img = self._transform(cond_img)
            real_img = transforms.functional.hflip(real_img)
            cond_img = transforms.functional.hflip(cond_img)
        if self.augment and random.random() < self._augment_p:
            # real_img = transforms.functional.rotate(real_img, int(90), fill=(0,))
            real_img = real_img.transpose(Image.ROTATE_90)
            cond_img = cond_img.transpose(Image.ROTATE_90)

        real_img = transforms.functional.to_tensor(real_img)
        cond_img = transforms.functional.to_tensor(cond_img)

        if self.wound_only:
            cond_img = 1 - cond_img
            real_img = real_img * cond_img

        if self.add_noise:
            cond_img = cond_img * torch.empty_like(cond_img).uniform_(0.1, 1)
            # cond_img = cond_img + ((1-cond_img) * torch.empty_like(cond_img).uniform_(0.05, 0.1))  # TODO: Test

        return real_img, cond_img


class WoundHealingPathsDataset(WoundHealingDataset):
    def __init__(self, data_path, masks_path, target_size: tuple, device, augment=False, add_noise=False,
                 random_crop=False, wound_only=False, check_data_pair_names=False):
        super().__init__(data_path, masks_path, target_size, device, augment, add_noise, random_crop,
                         wound_only, check_data_pair_names)

    def __getitem__(self, idx):
        # Calculate corresponding dataset and item ids.
        dataset_idx = np.searchsorted(self._dataset_ranges, idx, side='right') - 1
        item_idx = idx - self._dataset_ranges[dataset_idx]

        real_img_path = self.data_ids[dataset_idx][item_idx]
        cond_img_path = self.gt_mask_ids[dataset_idx][item_idx]

        return real_img_path, cond_img_path


def get_item_paths(folder_path):
    file_names = sorted(os.listdir(folder_path))
    item_paths = []
    for i in range(len(file_names)):
        if file_names[i] == ".directory":
            continue
        item_paths.append(os.path.join(folder_path, file_names[i]))
    return item_paths


def get_corresponding_item_paths(data_folder, mask_folder, check_data_pair_names=False):
    data_files = sorted(os.listdir(data_folder))
    mask_files = sorted(os.listdir(mask_folder))
    data_paths = []
    mask_paths = []
    data_ind = 0
    mask_ind = 0
    while data_ind < len(data_files) and mask_ind < len(mask_files):
        if not data_files[data_ind].endswith(EXTS):
            data_ind += 1
            continue
        if not mask_files[mask_ind].endswith(EXTS):
            mask_ind += 1
            continue
        if check_data_pair_names:
            # mask_name = os.path.splitext(mask_files[mask_ind])[0]  # Exclude extension
            # mask_name = mask_name.replace("_mask", "")
            mask_name = mask_files[mask_ind]
            mask_name = mask_name[mask_name.find("_t") : mask_name.find("_t")+4]  # get id substring
            while mask_name not in data_files[data_ind]:
                data_ind += 1
                if data_ind >= len(data_files):
                    raise Exception("Mask file name did not match with any data file:\n\t" + mask_name)
        data_paths.append(os.path.join(data_folder, data_files[data_ind]))
        mask_paths.append(os.path.join(mask_folder, mask_files[mask_ind]))
        data_ind += 1
        mask_ind += 1
    return data_paths, mask_paths


def _get_wound_healing_assay_paths(data_folder_path, masks_folder_path, exts=None, chk_mask_folder_empty=True):
    # Corresponding shared paths in masks and data folders.
    #     -/data/corresponding_path.../data_file
    #     -/masks/corresponding_path.../mask_file
    if exts is None and chk_mask_folder_empty:
        exts = EXTS  # file extensions
    data_folder_path = os.path.normpath(data_folder_path)
    masks_folder_path = os.path.normpath(masks_folder_path)

    # Get corresponding data and masks folder paths...
    data_folder_list = []
    mask_folder_list = []
    for root, dirs, _ in os.walk(masks_folder_path):
        if not dirs:  # if leaf folder...
            for i in os.listdir(root):
                if not chk_mask_folder_empty or (root + os.path.sep + i).split('.')[-1] in exts:
                    mask_folder = os.path.abspath(root)
                    mask_folder_list.append(mask_folder)
                    data_folder_list.append(mask_folder.replace(masks_folder_path, data_folder_path))
                    break

    return data_folder_list, mask_folder_list


# Get data-mask pairs. Raises exception if there is no data.
def get_wound_healing_datasets(data_path, gt_masks_path, check_data_pair_names=False):
    data_path = os.path.normpath(data_path)
    gt_masks_path = os.path.normpath(gt_masks_path)
    data_folder_list, mask_folder_list = _get_wound_healing_assay_paths(data_path, gt_masks_path)
    data_paths_list = []
    mask_paths_list = []
    for data_folder, masks_folder in zip(data_folder_list, mask_folder_list):
        data_path_list, mask_path_list = get_corresponding_item_paths(data_folder, masks_folder,
                                                                      check_data_pair_names)
        data_paths_list.append(data_path_list)
        mask_paths_list.append(mask_path_list)
    if len(data_paths_list) < 1 or len(mask_paths_list) < 1:
        raise FileNotFoundError("\n\t{}\n\t{}".format(data_path, gt_masks_path))
    return data_paths_list, mask_paths_list
