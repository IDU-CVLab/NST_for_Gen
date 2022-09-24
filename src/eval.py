# "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization" model implementation is
# adopted from: https://github.com/naoto0804/pytorch-AdaIN?ref=pythonrepo.com

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from skimage.exposure import match_histograms
import skimage.filters
import numpy as np
from datetime import datetime
import shutil
import random

from tools.datasets import WoundHealingDataset
from tools import net
from tools.function import adaptive_instance_normalization
from tools.utils import get_k_fold_ind_from_bins, get_k_fold_style_ind_from_bins


def test_transform(size):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    # TODO: Compute 1 by 1 for lower memory consumption when "interpolation_weights" is set.

    content_f = vgg(content).detach()
    # torch.cuda.synchronize()
    torch.cuda.empty_cache()

    style_f = vgg(style).detach()
    # torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def masked_hist_match(dest_img, dest_mask, ref_img, ref_mask=None):
    # img_2d -> masked_1d
    dest_img_masked = dest_img[dest_mask != 0]
    if ref_mask:
        ref_img = ref_img[ref_mask != 0]
    # mask_1d -> hist_matched
    dest_img_masked = match_histograms(dest_img_masked, ref_img.ravel())
    # hist_matched -> img_2d
    dest_img[dest_mask != 0] = dest_img_masked
    dest_img = np.rint(dest_img).astype(np.uint8)

    return dest_img


def metric_psnr(true_img: torch.Tensor, test_img: torch.Tensor, data_range=255.0):
    mean_sqr_err = torch.mean(torch.pow(true_img - test_img, 2))
    return 10. * torch.log10(data_range / torch.sqrt(mean_sqr_err))


def style_transfer_separate(style, content, style_mask, device="cpu", interpolation_weights=None,
                            wound_only=False):
    content = content.to(device)
    style = style.to(device)
    style_mask = style_mask.to(device)

    # Calculate for foreground...
    if wound_only:
        content = 1 - content
        style_mask = 1 - style_mask
        style = style * style_mask
    style_fg = style * style_mask
    # style_fg = style * content  # Apply mask
    content_fg = content * torch.empty_like(content).uniform_(0.1, 1)  # adding masked noise
    if len(style_fg.shape) == 3:
        style_fg = style_fg.unsqueeze(0)
    if len(content_fg.shape) == 3:
        content_fg = content_fg.unsqueeze(0)
    output_fg = style_transfer(vgg, decoder, content_fg, style_fg,
                               interpolation_weights=interpolation_weights)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    output_fg = torch.mean(output_fg.detach(), dim=1)  # to gray scale

    # Calculate for background...
    # style_bg = style * (1 - content)  # Apply mask
    style_bg = style * (1 - style_mask)  # Apply mask
    content_bg = (1 - content) * torch.empty_like(content).uniform_(0.1, 1)  # adding masked noise
    if len(style_bg.shape) == 3:
        style_bg = style_bg.unsqueeze(0)
    if len(content_bg.shape) == 3:
        content_bg = content_bg.unsqueeze(0)
    output_bg = style_transfer(vgg, decoder, content_bg, style_bg,
                               interpolation_weights=interpolation_weights)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    output_bg = torch.mean(output_bg.detach(), dim=1)  # to gray scale

    # Combine foreground and background...
    content_1ch = content[0][0] if len(content.shape)==4 else content[0]
    content_blurred = skimage.filters.gaussian(content_1ch.cpu().numpy(), sigma=3, truncate=1)  # sigma=3 -> 7x7 kernel
    content_blurred = torch.from_numpy(content_blurred)
    output_fg = output_fg.cpu() * content_blurred
    output_bg = output_bg.cpu() * (1-content_blurred)
    output = output_fg + output_bg
    # output = (output_fg * content_1ch) + (output_bg * (1-content_1ch))

    output = output.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy().squeeze()
    torch.cuda.empty_cache()
    return output


parser = argparse.ArgumentParser()
# Basic options
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('--content_dir', type=str, required=True,
                           help='Directory path to a batch of content images')
requiredNamed.add_argument('--style_dir', type=str, required=True,
                           help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='assets/models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='assets/experiments/decoder_epoch_5.pt')
parser.add_argument('--output_size', type=int, nargs=2, default=(1440, 1920),
                    help='Output size as "height width" (e.g.: --output_size 1440 1920)')
parser.add_argument('--wound_only', type=int,
                    help='Set 1 if the input model is trained for wound regions only or set to 0. '
                         'The wound regions are regions marked as 0 at the mask inputs.',
                    default=0)
parser.add_argument('--n_splits', type=int, default=5, help="Number of splits for k-fold  cross-validation."
                                                            " Set below 2 for not splitting.")
parser.add_argument('--split_num', type=int, default=0)

# Additional options
parser.add_argument('--save_ext', default='.png',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='assets/output',
                    help='Directory to save the output image(s)')

parser.print_help()
print()
args = parser.parse_args()

n_splits = args.n_splits
split_num = args.split_num
wound_only = args.wound_only

output_size_w_h = args.output_size[1], args.output_size[0]

# device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: "+str(device.type))
exts = [".png", ".tif", ".tiff"]

output_dir = Path(args.output)
file_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S-results')
output_dir = output_dir / file_name
output_dir.mkdir(exist_ok=True, parents=True)
cont_dir = output_dir / "mask"
cont_dir.mkdir(exist_ok=True, parents=True)
real_dir = output_dir / "real"
real_dir.mkdir(exist_ok=True, parents=True)
gen_dir = output_dir / "generated"
gen_dir.mkdir(exist_ok=True, parents=True)

dataset = WoundHealingDataset(args.style_dir, args.content_dir, output_size_w_h, device, add_noise=False,
                              random_crop=False, check_data_pair_names=True)
# Create data indexes for training and validation splits...
if n_splits > 1:
    train_indices, val_indices = get_k_fold_ind_from_bins(dataset.data_ids, n_splits=n_splits,
                                                          split_index=split_num)
    train_style_indices, _ = get_k_fold_style_ind_from_bins(dataset.data_ids,
                                                            n_splits=n_splits, split_index=split_num)

    # Get random style input indices for validation inputs from training data...
    random.shuffle(train_style_indices)
    val_style_indices = []
    for val_id, idx in enumerate(val_indices):
        random_syle_id = train_style_indices[idx % len(train_style_indices)]
        val_style_indices.append(random_syle_id)
else:
    val_indices = list(range(len(dataset)))
    val_style_indices = val_indices

val_loader = DataLoader(Subset(dataset, val_indices), batch_size=1, num_workers=0)
paths_dataset = dataset.get_new_paths_dataset()
val_path_iter = iter(DataLoader(Subset(paths_dataset, val_indices), batch_size=1, num_workers=0))

val_style_iter = iter(DataLoader(Subset(dataset, val_style_indices), batch_size=1, num_workers=0))
val_style_path_iter = iter(DataLoader(Subset(paths_dataset, val_style_indices), batch_size=1, num_workers=0))

# style_path =
# style = # Style image tensor
# style_content = # Style image mask tensor

# Prepare models...
decoder = net.decoder
vgg = net.vgg
decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

print("Starting to generate...")
for i, (_, content) in enumerate(val_loader):
    torch.cuda.empty_cache()
    style, style_content = next(val_style_iter)
    style_real_path, _style_content_path = next(val_style_path_iter)
    transfer_separate = True
    if transfer_separate:
        with torch.no_grad():
            output = style_transfer_separate(style, content, style_content, device, wound_only=wound_only)
    else:
        raise NotImplementedError
        content = content * torch.empty_like(content).uniform_(0.1, 1)  # adding masked noise
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        output = torch.mean(output.detach().data[0], dim=0)  # to gray scale
        output = output.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()

    real_path, content_path = next(val_path_iter)
    gen_path = gen_dir / (f'{i:03d}-' + '{:s}--stylized--{:s}{:s}'.format(
        Path(content_path[0]).stem, Path(style_real_path[0]).stem, args.save_ext))
    Image.fromarray(output).save(str(gen_path))
    # Copy source image files...
    shutil.copy(content_path[0], str((cont_dir / Path(f'{i:03d}-' + Path(content_path[0]).name)).absolute()))
    shutil.copy(real_path[0], str((real_dir / Path(f'{i:03d}-' + Path(real_path[0]).name)).absolute()))
print("Output dir: "+str(output_dir.absolute()))
