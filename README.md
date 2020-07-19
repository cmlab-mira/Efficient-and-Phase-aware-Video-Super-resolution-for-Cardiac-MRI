# Efficient-and-Phase-aware-Video-Super-resolution-for-Cardiac-MRI
Official Pytorch implementation of "[Efficient and Phase-aware Video Super-resolution for Cardiac MRI](https://arxiv.org/abs/2005.10626). Lin et al. MICCAI 2020."

# Datasets
We establish two datasets named **ACDCSR** and **DSB15SR** based on the public MRI datasets.

## ACDCSR (from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html))
1. Please download the original ACDC dataset from [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html) and save it at ACDC_DIR.
2. Run
```
python3 src/acdc_preprocess.py ACDC_DIR ACDCSR_DIR
```

## DSB15SR (from [DSB15](https://www.kaggle.com/c/second-annual-data-science-bowl))
1. Please download the original DSB15 dataset from [here](https://www.kaggle.com/c/second-annual-data-science-bowl) and save it at DSB15_DICOM_DIR.
2. Run
```
python3 src/dsb15_dicom2nifty DSB15_DICOM_DIR DSB15_NIFTI_DIR
python3 src/dsb15_preprocess.py DSB15_NIFTI_DIR DSB15SR_DIR
```

# Acknowledgement
This work was supported in part by the Ministry of Science and Technology, Taiwan, under Grant MOST 109-2634-F-002-032 and Microsoft Research Asia. We are grateful to the NVIDIA grants and the DGX-1 AI Supercomputer and the National Center for High-performance Computing. We thank Dr. Chih-Kuo Lee, National Taiwan University Hospital, for the early discussions.
