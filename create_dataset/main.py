import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import fft
import cv2
import imageio
from tqdm import tqdm

from file_io import read_disp, write_pfm


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

    lenslessRGB = [channel2Lensless(
        img[:, :, i], psf[:, :, i]) for i in range(3)]
    lenslessRGB = np.stack(lenslessRGB, axis=2)
    lenslessRGB = np.abs(lenslessRGB)
    lenslessRGB = np.clip(lenslessRGB, a_min=0, a_max=np.max(lenslessRGB))
    max_val = np.max(lenslessRGB)
    lenslessRGB = lenslessRGB / max_val

    noise = np.random.normal(
        0, 0.01, (lenslessRGB.shape[0], lenslessRGB.shape[1]))
    noise = np.stack([noise] * 3, axis=2)

    lensless_noisy = lenslessRGB + noise
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

    return left_img[c1:c1 +
                    size[0], c2:c2 +
                    size[1]], right_img[c1:c1 +
                                        size[0], c2:c2 +
                                        size[1]], disp[c1:c1 +
                                                       size[0], c2:c2 +
                                                       size[1]]


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

    parser = argparse.ArgumentParser()

    parser.add_argument('--left_dir', type=str, default='image_clean/left')
    parser.add_argument('--right_dir', type=str, default='image_clean/right')
    parser.add_argument(
        '--disp_dir',
        type=str,
        default='image_clean/disparity')
    parser.add_argument(
        '--target_dir',
        type=str,
        default='lensless_disp_dataset')
    parser.add_argument('--psf_path', type=str, default='')
    parser.add_argument('--num_passes', type=int, default=3)

    args = parser.parse_args()

    return args


def main():
    args = getArgs()

    left_paths = os.listdir(args.left_dir)
    right_paths = os.listdir(args.right_dir)
    disp_paths = os.listdir(args.disp_dir)

    common_fnames = [Path(x).stem for x in left_paths if x in right_paths]
    common_fnames = [
        Path(x).stem for x in disp_paths if Path(x).stem in common_fnames]

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    os.makedirs(os.path.join(args.target_dir, 'left'))
    os.makedirs(os.path.join(args.target_dir, 'right'))
    os.makedirs(os.path.join(args.target_dir, 'disp'))
    os.makedirs(os.path.join(args.target_dir, 'left_meas'))
    os.makedirs(os.path.join(args.target_dir, 'right_meas'))

    psf = np.load(args.psf_path)

    target_fnum = 0
    for i in args.num_passes:
        for fname in tqdm(common_fnames):
            left_path = os.path.join(args.left_dir, fname + '.png')
            right_path = os.path.join(args.right_dir, fname + '.png')
            disp_path = os.path.join(args.disp_dir, fname + '.pfm')

            left_img, right_img, disp = loadCropImg(
                left_path, right_path, disp_path)

            left_meas = RGB2Lensless(left_img, psf)
            right_meas = RGB2Lensless(right_img, psf)

            ### Save Dataset ###

            imageio.imwrite(
                os.path.join(
                    args.target_dir,
                    'left') +
                target_fnum +
                '.png',
                (left_img *
                 255).astype(
                    np.uint8))
            imageio.imwrite(
                os.path.join(
                    args.target_dir,
                    'right') +
                target_fnum +
                '.png',
                (right_img *
                 255).astype(
                    np.uint8))
            imageio.imwrite(
                os.path.join(
                    args.target_dir,
                    'left_meas') +
                target_fnum +
                '.png',
                (left_meas *
                 255).astype(
                    np.uint8))
            imageio.imwrite(
                os.path.join(
                    args.target_dir,
                    'right_meas') +
                target_fnum +
                '.png',
                (right_meas *
                 255).astype(
                    np.uint8))
            write_pfm(
                os.path.join(
                    args.target_dir,
                    'disp') +
                target_fnum +
                '.pfm',
                disp)


if __name__ == "__main__":
    main()
