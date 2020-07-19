import os, sys, glob, re
import logging
import argparse
import random
import numpy as np
import nibabel as nib
from pathlib import Path


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    for type_ in ['train', 'validate', 'test']:
        path = data_dir / type_
        patient_dirs = sorted([dir_ for dir_ in path.iterdir() if dir_.is_dir()])
        
        for dir_ in patient_dirs:
            patient_id = dir_.parts[-1]
            case_paths = list(dir_.glob("*/sax*"))
            
            for case in case_paths:
                case_id = case.parts[-1]
                if type_ == 'validate':
                    output_path = output_dir / 'valid' / patient_id / case_id
                else:
                    output_path = output_dir / type_ / patient_id / case_id
                
                if not output_path.is_dir():
                    output_path.mkdir(parents=True)
                
                os.system('dcm2niix -o %s -t y -s n -m y -b y -ba n -z y -f %%d %s' % (str(output_path), str(case)))

def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('data_dir', type=Path, help='The directory of the dataset.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
