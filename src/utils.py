def denormalize(imgs, dataset):
    """Denormalize the images.
    Args:
        imgs (torch.Tensor) (N, C, H, W): The images to be denormalized.
        dataset (str): The name of the dataset.

    Returns:
        imgs (torch.Tensor) (N, C, H, W): The denormalized images.
    """
    if dataset not in ['acdc', 'dsb15']:
        raise ValueError(f"The name of the dataset should be 'acdc' or 'dsb15'. Got {dataset}.")

    if dataset == 'acdc':
        mean, std = 54.089, 48.084
    elif dataset == 'dsb15':
        mean, std = 51.193, 52.671

    imgs = imgs.clone()
    imgs = (imgs * std + mean).round().clamp(0, 255)
    return imgs
