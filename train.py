import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model import DispModel
from dataloader import DispDataset, create_dataloaders

def zero_pad(psf, shp):
    top_pad = int((shp[1]-psf.shape[1])/2)
    bottom_pad = (shp[1]-psf.shape[1]) - top_pad
    left_pad = int((shp[0]-psf.shape[0])/2)
    right_pad = (shp[0]-psf.shape[0]) - left_pad
    psfR = np.pad(psf[:,:,0], ((top_pad, bottom_pad),(left_pad, right_pad)), 'constant')
    psfG = np.pad(psf[:,:,1], ((top_pad, bottom_pad),(left_pad, right_pad)), 'constant')
    psfB = np.pad(psf[:,:,2], ((top_pad, bottom_pad),(left_pad, right_pad)), 'constant')
    psf = np.stack([psfR, psfG, psfB], axis = 2)
    
    return psf

def train(model, train_loader, val_loader, args, device):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    writer = SummaryWriter(os.path.join('checkpoints', args.exp_name))

    for i in range(args.num_epochs):

        epoch_loss = 0
        for b, (left_meas, right_meas, disp) in tqdm(enumerate(train_loader)):

            left_meas = left_meas.to(device).float()
            right_meas = right_meas.to(device).float()
            disp = disp.to(device).float()

            pred_disp = model(left_meas, right_meas)

            loss = criterion(pred_disp, disp)
            epoch_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def getArgs():
    pass

def main():
    
    args = getArgs()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    psf = loadPSF(args.psf_path)
    psf = zero_pad(psf, (639, 767))
    psf = np.transpose(psf, (2, 0, 1))[np.newaxis,:,:,:]
    psf = torch.from_numpy(psf)
    model = DispModel(args, psf)
    model = model.to(device)

    train_loader, val_loader, _ = create_dataloaders(args, DispDataset(args))

    train(model, train_loader, val_loader, args, device)

if __name__=="__main__":
    main()