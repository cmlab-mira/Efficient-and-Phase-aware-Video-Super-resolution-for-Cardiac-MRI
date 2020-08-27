import nibabel as nib

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class Dsb15SISRDataset(BaseDataset):
    """The dataset of the 2015 Data Science Bowl challenge for the Single-Image Super-Resolution.
    
    Ref: https://www.kaggle.com/c/second-annual-data-science-bowl
    
    Args:
        downscale_factor (int): The downscale factor (2, 3, 4).
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, downscale_factor, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)
        if downscale_factor not in [2, 3, 4]:
            raise ValueError(f'The downscale factor should be 2, 3, 4. Got {downscale_factor}.')
        self.downscale_factor = downscale_factor

        self.transforms = compose(transforms)
        self.augments = compose(augments)
        
        lr_paths = sorted((self.data_dir / self.type / 'LR' / f'X{downscale_factor}').glob('**/*2d*.nii.gz'))
        hr_paths = sorted((self.data_dir / self.type / 'HR').glob('**/*2d*.nii.gz'))
        self.data = [(lr_path, hr_path) for lr_path, hr_path in zip(lr_paths, hr_paths)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        lr_path, hr_path = self.data[index]
        lr_img = nib.load(str(lr_path)).get_data() # (H, W, C)
        hr_img = nib.load(str(hr_path)).get_data() # (H, W, C)

        if self.type == 'train':
            lr_img, hr_img = self.augments(lr_img, hr_img)
        lr_img = self.transforms(lr_img).permute(2, 0, 1).contiguous()
        hr_img = self.transforms(hr_img).permute(2, 0, 1).contiguous()
        return {'lr_img': lr_img, 'hr_img': hr_img, 'index': index}
    