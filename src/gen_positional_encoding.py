import logging
import argparse
import cv2
import imageio
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path


def main(args):
    patient_dirs = sorted(list(args.data_dir.glob('**/HR/*')))
    with open(args.coordinate_path, 'rb') as f:
        coordinates = pickle.load(f)
    pos_codes = {}
    
    for patient_dir in patient_dirs:
        patient_name = patient_dir.parts[-1]
        logging.info(f'Process {patient_name}.')
        
        h0, hn, w0, wn = coordinates[patient_name]
        data_paths = sorted(patient_dir.glob('**/*2d+1d*.nii.gz'))
        data = nib.load(str(data_paths[0])).get_data()
        smoothed_img1 = cv2.GaussianBlur((data[h0:hn, w0:wn, 0, 0]).astype(np.uint8), (5, 5), 0)
        _, mask1 = cv2.threshold(smoothed_img1, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        diffs = []
        for i in range(int(np.floor(data.shape[-1] * 0.25)), int(np.ceil(data.shape[-1] * 0.6))):
            smoothed_img2 = cv2.GaussianBlur((data[h0:hn, w0:wn, 0, i]).astype(np.uint8), (5, 5), 0)
            _, mask2 = cv2.threshold(smoothed_img2, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            diff = np.abs(mask1 - mask2)
            diffs.append(np.sum(diff))
        start, end = 0, np.argmax(diffs) + int(np.floor(data.shape[-1] * 0.25))
        
        y1 = np.cos(np.linspace(0, np.pi, end-start, endpoint=False))
        y2 = np.cos(np.linspace(np.pi, np.pi*2, data.shape[-1]-y1.shape[0], endpoint=False))
        pos_code = np.concatenate((y1, y2))
        pos_codes[patient_name] = np.concatenate((pos_code[-start:], pos_code[:-start]))
        
    with open(args.output_dir / 'position_code.pkl', 'wb') as f:
        pickle.dump(pos_codes, f)
        
def _parse_args():
    parser = argparse.ArgumentParser(description="The generation of the positional encoding.")
    parser.add_argument('data_dir', type=Path, help='The directory of the preprocessed data.')
    parser.add_argument('coordinate_path', type=Path, help='The path of the cardiac cropping coordinates pickle file.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the processed data.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)