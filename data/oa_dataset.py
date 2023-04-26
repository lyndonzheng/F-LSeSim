"""A modification of previous implementations to support optoacoustic image reconstructions.
"""

import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import h5py
import numpy as np
import torchvision.transforms as transforms


class OADAT_Dataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.current_epoch = 0
        if 'scd_vcBP_swfd_scBP' in opt.name:
            self.fname_h5 = opt.dataroot
            self.data_key1 = 'SCD'
            self.data_key2 = 'SWFD_sc'
            self.key1 = 'vc_BP'
            self.key2 = 'sc_BP'
            self.key1_pp = self._scaleclip_normalize_fn
            self.key2_pp = self._scaleclip_normalize_fn
        elif 'scd_lbls_swfd_scBP' in opt.name:
            self.fname_h5 = opt.dataroot
            self.data_key1 = 'SCD'
            self.data_key2 = 'SWFD_sc'
            self.key1 = 'labels'
            self.key2 = 'sc_BP'
            self.key1_pp = self._scale_labels_to_255_fn
            self.key2_pp = self._scaleclip_normalize_fn
        else:
            raise ValueError("Unknown dataset name: {}".format(opt.name))
        self.len1, self.len2 = None, None
        self.check_data()
        self.transform = get_transform(opt, convert=False, grayscale=True)
        if self.opt.isTrain and opt.augment:
            self.transform_aug = transforms.Compose(
                [transforms.RandomAdjustSharpness(sharpness_factor=2., p=0.2),
                 transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                 transforms.ToTensor(), 
                 transforms.Normalize((0.5), (0.5))
                ]
            )
        else:
            self.transform_aug = None
        self.transform_tensor = transforms.Compose([
            transforms.ToTensor(), ## convert [0, 255] to [0, 1]
            transforms.Normalize((0.5), (0.5)), ## convert to [-1, 1]
            ])

    def check_data(self):
        with h5py.File(self.fname_h5, 'r') as fh:
            self.len1 = fh[self.data_key1][self.key1].shape[0]
            self.len2 = fh[self.data_key2][self.key2].shape[0]

    def _scale_labels_to_255_fn(self, x, max_lbl=2):
        '''Scale labels to [0, 255] '''
        x = np.asarray(x, dtype=np.float32)
        x /= float(max_lbl)
        x *= 255
        return x

    def _scaleclip_normalize_fn(self, x):
        '''Apply scaleclip, then normalize to [0, 255] '''
        x = np.clip(x/np.max(x), a_min=-0.2, a_max=None)
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        idx1 = index % self.len1
        if self.opt.serial_batches:   # make sure index is within then range
            idx2 = index % self.len2
        else: ## randomize just to make sure sample pairs are not aligned
            idx2 = random.randint(0, self.len2 - 1)
        
        with h5py.File(self.fname_h5, 'r') as fh:
            A_img = fh[self.data_key1][self.key1][idx1,...]
            B_img = fh[self.data_key2][self.key2][idx2,...]
        
        A_img = self.key1_pp(A_img)
        B_img = self.key2_pp(B_img)
        A_pil = Image.fromarray(A_img, mode="F")
        B_pil = Image.fromarray(B_img, mode="F")

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        transform = self.transform
        A_pil = transform(A_pil)
        B_pil = transform(B_pil)
        A = self.transform_tensor(A_pil)
        B = self.transform_tensor(B_pil)

        A_paths = f'[{self.data_key1}][{self.key1}][{idx1},...]'
        B_paths = f'[{self.data_key2}][{self.key2}][{idx2},...]'

        if self.opt.isTrain and self.transform_aug is not None:
            A_aug = self.transform_aug(A_pil)
            B_aug = self.transform_aug(B_pil)
            return {'A': A, 'B': B, 'A_paths': A_paths, 'B_paths': B_paths, 'A_aug': A_aug, 'B_aug': B_aug}
        else:
            return {'A': A, 'B': B, 'A_paths': A_paths, 'B_paths': B_paths}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.len1, self.len2)
