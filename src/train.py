# "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization" model implementation is
# adopted from: https://github.com/naoto0804/pytorch-AdaIN?ref=pythonrepo.com


import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.utils import get_k_fold_ind_from_bins
from tools import net
from tools.datasets import WoundHealingDataset

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # Due to error: "RuntimeError: Cannot re-initialize
    # CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"

    parser = argparse.ArgumentParser()
    # Basic options
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--content_dir', type=str, required=True, default="assets/dataset/masks",
                        help='Directory path to a batch of content images')
    requiredNamed.add_argument('--style_dir', type=str, required=True, default="assets/dataset/data",
                        help='Directory path to a batch of style images')
    parser.add_argument('--target_size', type=int, nargs=2, default=(256, 256),
                    help='Target training patch size as "height width" (e.g.: --target_size 256 256)')
    parser.add_argument('--vgg', type=str, default='assets/models/vgg_normalised.pth')
    parser.add_argument('--wound_only', type=int,
                        help='Set 0 or 1 for wound regions only training or not. The wound regions are '
                             'regions marked as 0 at the mask inputs.',
                        default=0)

    # training options
    parser.add_argument('--n_splits', type=int, default=5, help="Number of splits for k-fold  "
                                                                "cross-validation.")
    parser.add_argument('--split_num', type=int, default=0)
    parser.add_argument('--save_dir', default='./assets/experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./assets/logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--epoch_count', type=int, default=5)  # default=5
    parser.add_argument('--batch_size', type=int, default=4)  # default=8
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--save_interval_epoch', type=int, default=5)  # default=25
    parser.add_argument('--pre_decoder', type=str,
                        help='Directory to pre-trained model file for initialization.',
                        default='assets/models/decoder.pth')

    parser.print_help()
    print()
    args = parser.parse_args()

    # args.content_dir = "/mnt/72821AE3821AAC1B/WH_dataset_with_structure(new)/masks"
    # args.style_dir = "/mnt/72821AE3821AAC1B/WH_dataset_with_structure(new)/data"
    target_size = args.target_size
    wound_only = args.wound_only
    n_splits = args.n_splits
    split_num = args.split_num

    add_noise = True
    random_crop = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: "+str(device.type))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    decoder = net.decoder
    vgg = net.vgg

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    if len(args.pre_decoder) > 0:
        decoder.load_state_dict(torch.load(args.pre_decoder))
    network = net.Net(vgg, decoder)
    network.train()
    network.to(device)

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    dataset = WoundHealingDataset(args.style_dir, args.content_dir, target_size, device, add_noise=add_noise,
                                  random_crop=random_crop, wound_only=wound_only, check_data_pair_names=True)

    # Create data indexes for training and validation splits...
    dataset_size = len(dataset)
    if n_splits > 1:
        # train_indices, val_indices = get_k_fold_ind(dataset_size, n_splits=n_splits, split_num=split_num)
        train_indices, _val_indices = get_k_fold_ind_from_bins(dataset.data_ids, n_splits=n_splits,
                                                               split_index=split_num)
    else:
        train_indices = list(range(dataset_size))
        _val_indices = []
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    # for i in tqdm(range(args.epoch_count)):
    i = 0
    for epoch in tqdm(range(args.epoch_count)):
        for style_images, content_images in train_loader:
            adjust_learning_rate(optimizer, iteration_count=i)
            style_images = style_images.to(device)
            content_images = content_images.to(device)
            loss_c, loss_s = network(content_images, style_images)
            loss_c = args.content_weight * loss_c
            loss_s = args.style_weight * loss_s
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss_content', loss_c.item(), i + 1)
            writer.add_scalar('loss_style', loss_s.item(), i + 1)
            i += 1

        if (epoch + 1) % args.save_interval_epoch == 0 or (epoch + 1) == args.epoch_count:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            model_name = 'decoder_epoch_{:d}(k={:d}).pt'.format(epoch + 1, split_num)
            torch.save(state_dict, save_dir / model_name)
            print("Model saved: \n\t" + str((save_dir / model_name).absolute()))
    print("Saved models dir: "+str(save_dir.absolute()))
    writer.close()
