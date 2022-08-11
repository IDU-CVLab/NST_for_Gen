import numpy as np
import skimage.morphology
import matplotlib.pyplot as plt
import PIL.Image as Image
from pathlib import Path
from sklearn.model_selection import KFold


def combine_by_mask(mask_path, ones_img_path, zeros_img_path, verbose=False):
    ones_img = Image.open(ones_img_path)
    zeros_img = Image.open(zeros_img_path).resize(ones_img.size, Image.BICUBIC)
    mask = Image.open(mask_path).resize(ones_img.size, Image.NEAREST)

    ones_img = np.asarray(ones_img)
    mask = np.asarray(mask) / 255
    zeros_img = np.asarray(zeros_img)

    mask = skimage.morphology.dilation(mask)
    comb = ones_img * mask
    comb += zeros_img * (1-mask)
    comb = np.rint(comb).astype(np.uint8)
    if verbose:
        plt.imshow(comb, cmap="gray")
        plt.show()
    return comb


def combine_by_mask_dirs(mask_dir, ones_imgs_dir, zeros_imgs_dir):
    """
    Combines two images by two regions from the mask image. Takes pixels from an image for ones and takes
    pixels from the other image for zeros in the mask. Saves results to a new file next to @zeros_imgs_dir.
    """
    # TODO: Delete comments...
    # mask_dir = "output/results(k=0)-cell/mask"
    # ones_imgs_dir = "output/results(k=0)-cell/generated"
    # zeros_imgs_dir = "output/results(k=0)-wound/generated"

    out_dir = Path(str(Path(zeros_imgs_dir))+"(ones_added)")
    out_dir.mkdir(exist_ok=True, parents=True)

    mask_dir = Path(mask_dir)
    zeros_imgs_dir = Path(zeros_imgs_dir)
    ones_imgs_dir = Path(ones_imgs_dir)
    exts = [".png", ".tif", ".tiff"]
    mask_path_list = sorted([str(f) for f in mask_dir.glob('*') if f.suffix in exts])
    zeros_path_list = sorted([str(f) for f in zeros_imgs_dir.glob('*') if f.suffix in exts])
    ones_path_list = sorted([str(f) for f in ones_imgs_dir.glob('*') if f.suffix in exts])
    assert len(ones_path_list) == len(zeros_path_list) and len(mask_path_list) == len(zeros_path_list),\
        "Number of files do not match in: \n\tmask_dir: {} \n\tzeros_imgs_dir: {} \n\tones_imgs_dir: {}"\
        .format(mask_dir, zeros_imgs_dir, ones_imgs_dir)

    print("Combining images and saving to:\n\t"+str(out_dir))
    for mask_path, ones_img_path, zeros_img_path in zip(mask_path_list, ones_path_list, zeros_path_list):
        comb_img = combine_by_mask(mask_path, ones_img_path, zeros_img_path)
        img_path = Path.joinpath(out_dir, Path(zeros_img_path).name)
        Image.fromarray(comb_img).save(img_path)

    print("Finished.")


def get_k_fold_ind(item_cnt, n_splits, split_num, shuffle=False):
    train_ind_list = []
    test_ind_list = []
    if shuffle:
        np.random.seed(0)
        indexes = np.random.permutation(item_cnt)
    else:
        indexes = np.array(range(0, item_cnt))

    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_index, test_index in kf.split(indexes):
        test_ind_list.append(indexes[test_index])
        train_ind_list.append(indexes[train_index])
    return train_ind_list[split_num], test_ind_list[split_num]


def bin_lists_nearest_neighbor(lists_to_bin, best_bin_size):
    last_bin_ids = []
    cur_size = 0
    prev_size = 0
    data_list_id = -1

    # Determine last list indexes on bins.
    for data_list_id, data_list in enumerate(lists_to_bin):
        cur_size += len(data_list)
        if cur_size >= best_bin_size:
            if best_bin_size - prev_size < cur_size - best_bin_size:
                last_bin_ids.append(data_list_id - 1)
                cur_size = len(data_list)
                prev_size = len(data_list)
            else:
                prev_size = cur_size
        else:
            prev_size = cur_size
    last_bin_ids.append(data_list_id)

    # Get all list indexes on bins.
    bin_ids = []
    cur_bin = []
    for data_list_id, data_list in enumerate(lists_to_bin):
        if data_list_id not in last_bin_ids:
            cur_bin.append(data_list_id)
        else:
            cur_bin.append(data_list_id)
            bin_ids.append(cur_bin.copy())
            cur_bin = []

    return bin_ids


def get_k_fold_ind_from_bins(source_lists, n_splits, split_index):
    """
    Calculates indexes for k-fold cross validation. Calculates folds by number of elements in list of lists
    but does not split lists. Gets nearest possible size for each group.
    :param source_lists: Source data in form of list of lists.
    :param n_splits: Number of folds
    :param split_index: Index of current fold
    :return: List pair of train and test indexes
    """
    dataset_size = 0
    for data_list in source_lists:
        dataset_size += len(data_list)
    best_bin_size = dataset_size / n_splits
    bin_list = bin_lists_nearest_neighbor(source_lists, best_bin_size)

    # bin ids to bin lengths
    bin_list_lengths = []
    for bin in bin_list:
        bin_length = 0
        for list_id in bin:
            bin_length += len(source_lists[list_id])
        bin_list_lengths.append(bin_length)

    # Lengths to indexes
    indexes = []
    prev_index = 0
    for bin_length in bin_list_lengths:
        indexes.append(np.array(range(prev_index, prev_index + bin_length)))
        prev_index = prev_index + bin_length

    train_ind_list = []
    test_ind_list = []
    for idx, index_list in enumerate(indexes):
        if idx == split_index:
            test_ind_list.extend(index_list)
        else:
            train_ind_list.extend(index_list)

    return train_ind_list, test_ind_list


def get_k_fold_style_ind_from_bins(source_lists, n_splits, split_index):
    """
    Calculates style input indexes for k-fold cross validation. Calculates folds by number of elements in
     list of lists but does not split lists. Gets nearest possible size for each group. Calculates style
     input indexes as being only the first elements of each list in the :param source_lists.
    :param source_lists: Source data in form of list of lists.
    :param n_splits: Number of folds
    :param split_index: Index of current fold
    :return: List pair of train and test style input indices as first frame indices of each assay.
    """
    dataset_size = 0
    for data_list in source_lists:
        dataset_size += len(data_list)
    best_bin_size = dataset_size / n_splits
    bin_list = bin_lists_nearest_neighbor(source_lists, best_bin_size)

    # bin ids to bin style indexes using assay lengths
    bin_style_ids = []
    last_ind = 0
    for bin in bin_list:
        cur_bin = []
        for list_id in bin:
            cur_bin.append(last_ind)
            last_ind += len(source_lists[list_id])
        bin_style_ids.append(cur_bin)

    # # Lengths to style indexes
    # style_indexes = []
    # prev_index = 0
    # for bin in bin_style_ids:
    #     cur_style_bin_indexes = []
    #     for assay_length in bin:
    #         cur_style_bin_indexes.extend([prev_index for _ in range(prev_index, assay_length)])
    #         prev_index = assay_length
    #     style_indexes.append(cur_style_bin_indexes)
    # bin_style_ids = style_indexes

    train_ind_list = []
    test_ind_list = []
    for idx, index_list in enumerate(bin_style_ids):
        if idx == split_index:
            test_ind_list.extend(index_list)
        else:
            train_ind_list.extend(index_list)

    return train_ind_list, test_ind_list
