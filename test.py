# "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization" model implementation is
# adopted from: https://github.com/naoto0804/pytorch-AdaIN?ref=pythonrepo.com

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from skimage.exposure import match_histograms
import skimage.filters
import numpy as np
from datetime import datetime

from tools import net
from tools.function import adaptive_instance_normalization


def test_transform(size):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size, interpolation=Image.BICUBIC))
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
    output_fg = style_transfer(vgg, decoder, content_fg, style_fg, args.alpha, interpolation_weights)
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
    output_bg = style_transfer(vgg, decoder, content_bg, style_bg, args.alpha, interpolation_weights)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    output_bg = torch.mean(output_bg.detach(), dim=1)  # to gray scale

    # Combine foreground and background...
    content_1ch = content[0][0] if len(content.shape)==4 else content[0]
    content_blurred = skimage.filters.gaussian(content_1ch.cpu().numpy(), sigma=3, truncate=1)  # sigma=3 -> 7x7 kernel
    output_fg = output_fg.cpu() * content_blurred
    output_bg = output_bg.cpu() * (1-content_blurred)
    output = output_fg + output_bg
    # output = (output_fg * content_1ch) + (output_bg * (1-content_1ch))

    output = output.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy().squeeze()
    torch.cuda.empty_cache()
    return output


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--style_mask', type=str,
                    help='File path to the style mask image')
parser.add_argument('--style_mask_dir', type=str,
                    help='Directory path to a batch of style mask images')

parser.add_argument('--output_size', type=int, nargs=2, default=(1440, 1920),
                    help='Output size as "height width" (e.g.: --output_size 1440 1920)')
parser.add_argument('--wound_only', type=int,
                    help='Set 1 if the input model is trained for wound regions only or set to 0. '
                         'The wound regions are regions marked as 0 at the mask inputs.',
                    default=0)
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder_epoch_5-wound_healing.pt')

# Additional options
parser.add_argument('--save_ext', default='.png',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

parser.print_help()
print()
args = parser.parse_args()

wound_only = args.wound_only

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exts = [".png", ".tif", ".tiff"]

output_dir = Path(args.output)
file_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S-results')
output_dir = output_dir / file_name
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --content_dir should be given.
assert (args.content and args.style and args.style_mask) or \
       (args.style_dir and args.content_dir and args.style_mask_dir) \
       , "Either --content, --style and --style_mask or --content_dir, --style_dir and" \
         " --style_mask_dir parameters should be given!"
if args.content_dir:
    content_dir = Path(args.content_dir)
    content_paths = sorted([f for f in content_dir.glob('*') if f.suffix in exts])
else:
    content_paths = [Path(args.content)]

if args.style_dir:
    style_dir = Path(args.style_dir)
    style_paths = sorted([f for f in style_dir.glob('*') if f.suffix in exts])
else:
    style_paths = [Path(args.style)]

if args.style_mask_dir:
    style_mask_dir = Path(args.style_mask_dir)
    style_mask_paths = sorted([f for f in style_mask_dir.glob('*') if f.suffix in exts])
else:
    style_mask_paths = [Path(args.style_mask)]

decoder = net.decoder
vgg = net.vgg
decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)
output_size = args.output_size
content_tf = test_transform(output_size)
style_tf = test_transform(output_size)

print("Starting to generate...")
for content_path, style_path, style_mask_path in zip(content_paths, style_paths, style_mask_paths):
    torch.cuda.empty_cache()
    content_img = Image.open(str(content_path)).convert('RGB')
    style_img = Image.open(str(style_path)).convert('RGB')
    content = content_tf(content_img)
    style = style_tf(style_img)
    style_mask = content_tf(Image.open(style_mask_path).convert('RGB'))

    with torch.no_grad():
        output = style_transfer_separate(style, content, style_mask, device, wound_only=wound_only)

    # output = masked_hist_match(output, np.asarray(content_img.convert('L')),
    #                            np.asarray(style_img.convert('L')))

    output = Image.fromarray(output)

    output_name = output_dir / '{:s}--stylized--{:s}{:s}'.format(
        content_path.stem, style_path.stem, args.save_ext)
    output.save(str(output_name))
print("Output dir: "+str(output_dir.absolute()))
