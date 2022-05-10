import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import fft
import cv2

from file_io import read_disp

def channel2Lensless(imgC, psfC):
    """
    Convolution of imgC and psfC in the frequency domain.

    Arguments:
    imgC - Single Channel of image
    psfC - Single Channel of psf

    Returns:
    lenslessC - Lensless Measurement (Single Channel)
    """

    shape = [
        imgC.shape[i] +
        psfC.shape[i] -
        1 for i in range(
            imgC.ndim)]  # Target Shape
    imgC_dft = fft.fftn(imgC, shape)  # DFT of Image
    psfC_dft = fft.fftn(psfC, shape)  # DFT of PSF
    # Convolution is Multiplication in Frequency Domain
    lenslessC = fft.ifftn(imgC_dft * psfC_dft, shape)

    return lenslessC

def RGB2Lensless(img, psf):
    """
    Convolution of img (RGB) and psf in the frequency domain.

    Arguments:
    img - RGB Input Image to be convolved (channel dim last)
    psf - Convolution Kernel

    Returns:
    lenslessRGB - Lensless Measurement
    """

    lenslessRGB = [channel2Lensless(img[:, :, i], psf[:, :, i]) for i in range(3)]
    lenslessRGB = np.stack(lenslessRGB, axis=2)
    lenslessRGB = np.abs(lenslessRGB)
    lenslessRGB = np.clip(lenslessRGB, a_min = 0, a_max = np.max(lenslessRGB))
    max_val = np.max(lenslessRGB)
    lenslessRGB = lenslessRGB/max_val

    noise = np.random.normal(0, 0.01, (lenslessRGB.shape[0], lenslessRGB.shape[1]))
    noise = np.stack([noise]*3, axis = 2)

    lensless_noisy = lenslessRGB+noise
    #print(cv2.PSNR(lensless_noisy*255, lenslessRGB*255))
    lensless_noisy = np.clip(lensless_noisy, a_min=0, a_max=1)

    return lensless_noisy

def crop(left_img, right_img, disp, size=(320, 320)):
    """
    Returns Random Cropped image given target size.
    """
    dim1, dim2, _ = img.shape
    c1 = np.random.choice(dim1 - size[0])
    c2 = np.random.choice(dim2 - size[1])

    return left_img[c1:c1 + size[0], c2:c2 + size[1]], right_img[c1:c1 + size[0], c2:c2 + size[1]], disp[c1:c1 + size[0], c2:c2 + size[1]]

def loadCropImg(left_path, right_path, disp_path):
    """
    Loads and Crops Image
    """
    left_img = np.array(Image.open(left_path))
    left_img = left_img / 255.0
    
    right_img = np.array(Image.open(right_path))
    right_img = right_img / 255.0

    disp = read_disp(disp_path)

    left_img, right_img, disp = crop(left_img, right_img, disp)
    
    return left_img, right_img, disp

def getArgs():
    pass

def main():
    args = getArgs()

    left_paths = os.listdir(args.left_dir)
    right_paths = os.listdir(args.right_dir)
    disp_paths = os.listdir(args.disp_dir)

    common_fnames = [Path(x).stem for x in left_paths if x in right_paths]
    common_fnames = [Path(x).stem for x in disp_paths if Path(x).stem in common_fnames]

    



if __name__=="__main__":
    main()