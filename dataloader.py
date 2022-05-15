import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from scipy import fft
from torch.utils.data import Dataset
import torchvision
import cv2

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from create_dataset.file_io import read_disp, write_pfm

class DispDataset(Dataset):

    def __init__(self, args):
        self.source_dir = args.source_dir
        self.left_meas_dir = os.path.join(self.source_dir, 'left_meas')
        self.right_meas_dir = os.path.join(self.source_dir, 'right_meas')
        self.disp_dir = os.path.join(self.source_dir, 'disp')

        self.fnames = os.listdir(self.left_meas_dir)
        self.fnames = [Path(x).stem for x in self.fnames] # Remove extensions

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        left_meas_path = os.path.join(self.left_meas_dir, self.fnames[idx]+'.png')
        left_meas = np.array(Image.open(left_meas_path))/255.0
        left_meas = np.transpose(left_meas, (2, 0, 1))
        left_meas = torch.from_numpy(left_meas)
        left_meas = torchvision.transforms.CenterCrop(320)(left_meas)
        left_meas = torchvision.transforms.Pad((64,0,64,0), fill=0, padding_mode='constant')(left_meas)

        right_meas_path = os.path.join(self.right_meas_dir, self.fnames[idx]+'.png')
        right_meas = np.array(Image.open(right_meas_path))/255.0
        right_meas = np.transpose(right_meas, (2, 0, 1))
        right_meas = torch.from_numpy(right_meas)
        right_meas = torchvision.transforms.CenterCrop(320)(right_meas)
        right_meas = torchvision.transforms.Pad((64,0,64,0), fill=0, padding_mode='constant')(right_meas)

        disp_path = os.path.join(self.disp_dir, self.fnames[idx]+'.pfm')
        disp = read_disp(disp_path)
        disp = torch.from_numpy(disp)

        return left_meas, right_meas, disp


def create_dataloaders(args, dataset):
    """
    Returns train, val and test dataloaders for given splits and Dataset.
    """
    dataset_len = len(dataset)
    indices = list(range(dataset_len))

    val_split = int(np.floor(args.val_split * dataset_len))
    test_split = int(np.floor(args.test_split * dataset_len))

    if args.shuffle_data:
        np.random.seed(100)
        np.random.shuffle(indices)

    val_indices, test_indices, train_indices = indices[:val_split], \
        indices[val_split:val_split + test_split], \
        indices[val_split + test_split:]

    # Create Samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create Dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=val_sampler)
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=test_sampler)

    return train_loader, val_loader, test_loader

