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
from ruamel.yaml import YAML

def zero_pad(psf, shp):
    top_pad = int((shp[1]-psf.shape[0])/2)
    bottom_pad = (shp[1]-psf.shape[0]) - top_pad
    left_pad = int((shp[0]-psf.shape[1])/2)
    right_pad = (shp[0]-psf.shape[1]) - left_pad
    psfR = np.pad(psf[:,:,0], ((top_pad, bottom_pad),(left_pad, right_pad)), 'constant')
    psfG = np.pad(psf[:,:,1], ((top_pad, bottom_pad),(left_pad, right_pad)), 'constant')
    psfB = np.pad(psf[:,:,2], ((top_pad, bottom_pad),(left_pad, right_pad)), 'constant')
    psf = np.stack([psfR, psfG, psfB], axis = 2)
    
    return psf

def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg

def train(model, train_loader, val_loader, args, device):

    cfg = load_configs(args.cfg_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    writer = SummaryWriter(os.path.join('checkpoints', args.exp_name))

    for i in range(args.num_epochs):

        epoch_loss = 0
        for b, (left_meas, right_meas, disp_true) in tqdm(enumerate(train_loader)):

            left_meas = left_meas.to(device).float()
            right_meas = right_meas.to(device).float()
            disp_true = -disp_true.to(device).float()

            disp_pred, _ = model(left_meas, right_meas, training = True)

            losses = []
            train_weights = cfg['training']['training_scales_weighting']
            for disp_pred_ in disp_pred[:-1]:

                disp_pred_ = F.interpolate(
                    disp_pred_.unsqueeze(1),
                    size=(disp_true.shape[1], disp_true.shape[2]),
                    mode='bilinear').squeeze(1)
                # ---------
                
                mask = torch.logical_and(
                    disp_true <= cfg['model']['stereo']['max_disparity'],
                    disp_true > 0)
                mask.detach_()
                # ----
                
                losses.append(F.smooth_l1_loss(
                    disp_pred_[mask], disp_true[mask], reduction='mean'))
                
            
            loss = sum([losses[i] * train_weights[i] for i in range(len(disp_pred[:-1]))]) /\
                sum([1 * train_weights[i] for i in range(len(disp_pred[:-1]))])
            


            #loss = criterion(pred_disp, disp)
            epoch_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('Train Epoch Loss', epoch_loss/(b*args.batch_size), i+1)

        val_epoch_loss = 0
        for b, (left_meas, right_meas, disp) in tqdm(enumerate(val_loader)):

            left_meas = left_meas.to(device).float()
            right_meas = right_meas.to(device).float()
            disp = -disp.to(device).float()

            pred_disp, left_inv = model(left_meas, right_meas, training = False)

            loss = criterion(pred_disp[0], disp)
            val_epoch_loss+=loss.item()

            grid = torchvision.utils.make_grid(pred_disp[0]/192.0)
            writer.add_image(f"Epoch - {i} | Validation - {b+1} | Predicted", grid)
            grid = torchvision.utils.make_grid(disp/192.0)
            writer.add_image(f"Epoch - {i} | Validation - {b+1} | Ground Truth", grid)
            grid = torchvision.utils.make_grid(left_inv)
            writer.add_image(f"Epoch - {i} | Validation - {b+1} | Wiener Output", grid)

        writer.add_scalar('Validation Epoch Loss', val_epoch_loss/b, i+1)

        torch.save(model.state_dict(), os.path.join('checkpoints/'+ args.exp_name, 'latest_weights.pt'))

def getArgs():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type = str, default = "cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--psf_path', type = str, default = "create_dataset/basler_phlatcam_psf_binned2x_320x448_rgb.npy")
    parser.add_argument('--reg_param_init', type = int, default = 100)
    parser.add_argument('--cfg_path', type = str, default = 'configs/stereo/cfg_coex.yaml')

    # Data Args
    parser.add_argument('--source_dir', type = str, default = 'create_dataset/lensless_disp_dataset')
    parser.add_argument('--val_split', type = float, default = 0.005)
    parser.add_argument('--test_split', type = float, default = 0.005)
    parser.add_argument('--shuffle_data', type = bool, default = True)
    parser.add_argument('--batch_size', type = int, default = 4)

    # Train Args
    parser.add_argument('--lr', type = float, default = 3e-4)
    parser.add_argument('--num_epochs', type = int, default = 40)
    parser.add_argument('--exp_name', type = str, required = True)

    args = parser.parse_args()

    return args

def main():
    
    args = getArgs()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    psf = np.load(args.psf_path)
    #psf = zero_pad(psf, (767,639))
    psf = np.transpose(psf, (2, 0, 1))[np.newaxis,:,:,:]
    psf = torch.from_numpy(psf)
    model = DispModel(args, psf)
    model = model.to(device)

    train_loader, val_loader, _ = create_dataloaders(args, DispDataset(args))

    train(model, train_loader, val_loader, args, device)

if __name__=="__main__":
    main()