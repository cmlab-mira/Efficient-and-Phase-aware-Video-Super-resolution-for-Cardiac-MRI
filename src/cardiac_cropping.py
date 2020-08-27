import logging
import argparse
import cv2
import imageio
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path


def main(args):
    coordinates = {}
    patient_dirs = sorted(list(args.data_dir.glob('**/HR/*')))
    for patient_dir in patient_dirs:
        # Create the output directories.
        patient_name = patient_dir.parts[-1]
        logging.info(f'Process {patient_name}.')
        videos_dir = args.output_dir / patient_name
        if not videos_dir.is_dir():
            videos_dir.mkdir(parents=True)

        # Compute the bbox and save the cropped patches.
        data_paths = sorted(patient_dir.glob('**/*2d+1d*.nii.gz'))
        data = nib.load(str(data_paths[0])).get_data()
        h0, hn, w0, wn = find_bbox(data)
        coordinates[patient_name] = (h0, hn, w0, wn)
        for data_path in data_paths:
            data = nib.load(str(data_path)).get_data().squeeze().transpose([2, 0, 1]).astype(np.uint8)
            imgs = [img[h0:hn, w0:wn] for img in data]
            dump_video(videos_dir / data_path.parts[-1].replace('.nii.gz', '.gif'), imgs)

    with open(args.output_dir / 'coordinates.pkl', 'wb') as f:
        pickle.dump(coordinates, f)

def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('data_dir', type=Path, help='The directory of the data.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the processed data.')
    args = parser.parse_args()
    return args


def find_bbox(data):
    hmax, wmax, _, num_frames = data.shape
    img1, img2 = data[..., 0].squeeze(), data[..., num_frames // 2].squeeze()

    smoothed_img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    smoothed_img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    diff = np.abs(smoothed_img1 - smoothed_img2).astype(np.uint8)
    _, mask = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)))

    xsets, ysets = np.where(opened)
    xsets, ysets = xsets[int(len(xsets) * .05):int(len(xsets) * .95)], ysets[int(len(ysets) * .05):int(len(ysets) * .95)]

    height, width = int(np.std(xsets).round() * 5), int(np.std(ysets).round() * 5)
    hc, wc = int(np.mean(xsets).round()), int(np.mean(ysets).round())
    h0, hn = max(0, hc - height // 2), min(hc + (height - height // 2), hmax)
    w0, wn = max(0, wc - width // 2), min(wc + (width - width // 2), wmax)
    return h0, hn, w0, wn


def dump_video(path, imgs):
    """To dump the video by concatenate the images.
    Args:
        path (Path): The path to save the video.
        imgs (list): The images to form the video.
    """
    with imageio.get_writer(path) as writer:
        for img in imgs:
            writer.append_data(img)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
