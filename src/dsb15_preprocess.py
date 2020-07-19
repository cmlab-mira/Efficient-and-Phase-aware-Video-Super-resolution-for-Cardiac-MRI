import logging
import argparse
import random
import functools
import cv2
import numpy as np
import nibabel as nib
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from pathlib import Path


def main(args):
    for type_ in ['train', 'valid', 'test']:
        paths = sorted([dir_ for dir_ in (args.data_dir / type_).iterdir() if dir_.is_dir()])
        logging.info(f'Process the {type_} data.')

        sum_, square_sum, num = 0, 0, 0
        for path in paths:
            patient_name = path.parts[-1]
            logging.info(f'Process {patient_name}.')

            # Read the MRI data.
            sequence_paths = sorted(list(path.glob('*/*.nii.gz')))
            for j, sequence_path in enumerate(sequence_paths):
                data = nib.load(str(sequence_path)).get_data() # (H, W, 1, T)

                # If the data format is wrong, skip
                if data.shape[2] != 1 or len(data.shape) != 4 or data.shape[-1] < 30:
                    continue

                # If the data type is 'int16', remove the outlier and then apply the min-max normalization.
                if data.dtype == 'int16':
                    hist, _ = np.histogram(data.ravel(), bins=range(int(data.max()) + 1), density=True)
                    cdf = np.cumsum(hist)
                    idx = (np.abs(cdf - 0.995)).argmin()
                    data[data > idx] = idx
                    data = ((data - data.min()) / (data.max() - data.min()) * 255.0).round()
                data = data.astype(np.float32)

                # Make the image size divisible by 12.
                h, w, r = data.shape[0], data.shape[1], 12
                h0, hn = (h % r) // 2, h - ((h % r) - (h % r) // 2)
                w0, wn = (w % r) // 2, w - ((w % r) - (w % r) // 2)

                # Accumulate the pixel values, the square of the pixel values and the number of the pixels.
                sum_ += data[h0:hn, w0:wn, ...].sum()
                square_sum += (data[h0:hn, w0:wn, ...] ** 2).sum()
                num += np.prod(data[h0:hn, w0:wn, ...].shape)

                for i, downscale_factor in enumerate([2, 3, 4]):
                    # Create the output directories.
                    if i == 0:
                        hr_imgs_dir = args.output_dir / 'imgs' / type_ / 'HR' / patient_name
                        hr_videos_dir = args.output_dir / 'videos' / type_ / 'HR' / patient_name
                        for dir_ in [hr_imgs_dir, hr_videos_dir]:
                            if not dir_.is_dir():
                                dir_.mkdir(parents=True)
                    lr_imgs_dir = args.output_dir / 'imgs' / type_ / 'LR' / f'X{downscale_factor}' / patient_name
                    lr_videos_dir = args.output_dir / 'videos' / type_ / 'LR' / f'X{downscale_factor}' / patient_name
                    for dir_ in [lr_imgs_dir, lr_videos_dir]:
                        if not dir_.is_dir():
                            dir_.mkdir(parents=True)

                    # Define the downscaled function.
                    downscale_fn = Downscale(downscale_factor)

                    # Save the processed images and videos.
                    hr_video = data[h0:hn, w0:wn] # (H, W, C, T)
                    lr_video = np.stack(downscale_fn(*[hr_video[..., t] for t in range(hr_video.shape[-1])]), axis=-1) # (H, W, C, T)
                    if i == 0:
                        nib.save(nib.Nifti1Image(hr_video, np.eye(4)),
                                 str(hr_videos_dir / f'{patient_name}_2d+1d_sequence{j+1:0>2d}.nii.gz'))
                    nib.save(nib.Nifti1Image(lr_video, np.eye(4)),
                             str(lr_videos_dir / f'{patient_name}_2d+1d_sequence{j+1:0>2d}.nii.gz'))
                    for t in range(data.shape[-1]):
                        if i == 0:
                            hr_img = hr_video[..., t] # (H, W, C)
                            nib.save(nib.Nifti1Image(hr_img, np.eye(4)),
                                     str(hr_imgs_dir / f'{patient_name}_2d_slice{j+1:0>2d}_frame{t+1:0>2d}.nii.gz'))
                        lr_img = lr_video[..., t] # (H, W, C)
                        nib.save(nib.Nifti1Image(lr_img, np.eye(4)),
                                 str(lr_imgs_dir / f'{patient_name}_2d_slice{j+1:0>2d}_frame{t+1:0>2d}.nii.gz'))

        # Calculate the mean and the standard deviation.
        mean = sum_ / num
        square_mean = square_sum / num
        std = np.sqrt(square_mean - mean ** 2)
        logging.info(f'The mean and the standard deviation of the {type_} data is {mean:.4f} and {std:.4f}.')

def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('data_dir', type=Path, help='The directory of the data.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the processed data.')
    args = parser.parse_args()
    return args


class Downscale:
    """Downscale the HR images to the LR images by using the Fourier Transform and the bicubic interpolation.
    Args:
        downscale_factor (int): The downscale factor.
    """
    def __init__(self, downscale_factor):
        self.downscale_factor = downscale_factor
        self._truncate_kspace = functools.partial(self._truncate_kspace, downscale_factor=downscale_factor)

    def __call__(self, *imgs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be downscaled.

        Returns:
            imgs (tuple of numpy.ndarray): The downscaled images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images).")

        _imgs = []
        for img in imgs:
            kspace = self._transform_img_to_kspace(img)
            truncated_kspace = self._truncate_kspace(kspace)
            img = self._transform_truncated_kspace_to_img(truncated_kspace)
            h, w, c = img.shape
            _h, _w = h // self.downscale_factor, w // self.downscale_factor
            img = cv2.resize(img, (_w, _h), interpolation=cv2.INTER_CUBIC)[..., np.newaxis]
            img = np.clip(img.round(), 0, 255)
            _imgs.append(img)
        imgs = tuple(_imgs)
        return imgs

    @staticmethod
    def _transform_img_to_kspace(img):
        """Transform the spatial domain image data to the frequency domain kspace data.
        Args:
            img (numpy.ndarray): The spatial domain image data.

        Returns:
            kspace (numpy.ndarray): The frequency domain kspace data.
        """
        kspace = fftshift(fftn(ifftshift(img), norm='ortho'))
        return kspace

    @staticmethod
    def _truncate_kspace(kspace, downscale_factor):
        """Truncate the frequency domain kspace data according to the downscale factor.
        Args:
            kspace (numpy.ndarray): The frequency domain kspace data.

        Returns:
            truncated_kspace (numpy.ndarray): The truncated frequency domain kspace data.
        """
        rect_fn = np.zeros_like(kspace)
        kx_max = kspace.shape[0] // 2
        ky_max = kspace.shape[1] // 2
        lx = kspace.shape[0] // downscale_factor
        ly = kspace.shape[1] // downscale_factor
        rect_fn[kx_max - lx // 2 : kx_max + (lx - (lx // 2)),
                ky_max - ly // 2 : ky_max + (ly - (ly // 2))] = 1
        truncated_kspace = rect_fn * kspace
        return truncated_kspace

    @staticmethod
    def _transform_truncated_kspace_to_img(truncated_kspace):
        """Transform the truncated frequency domain kspace data to the spatial domain image data.
        Args:
            truncated_kspace (numpy.ndarray): The truncated frequency domain kspace data.

        Returns:
            img (numpy.ndarray): The spatial domain image data.
        """
        img = fftshift(ifftn(ifftshift(truncated_kspace), norm='ortho'))
        img = np.around(np.abs(img))
        return img


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
