import torch
from torch.utils.data import DataLoader
import numpy as np


class Dataloader(DataLoader):
    """The modified class of ``torch.utils.data.DataLoader`` with default ``collate_fn`` and ``worker_init_fn``.
    Args:
        dataset (Dataset): Dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load (default: ``1``).
        shuffle (bool, optional): Set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
        sampler (Sampler, optional): Defines the strategy to draw samples from the dataset. If specified, ``shuffle`` must be False (default: ``None``).
        batch_sampler (Sampler, optional): Like ``sampler``, but returns a batch of indices at a time. Mutually exclusive with ``batch_size``, ``shuffle``, ``sampler``, and ``drop_last`` (default: ``None``).
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: ``0``)
        collate_fn (callable, optional): Merges a list of samples to form a mini-batch (default: ``default_collate``).
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them. If your data elements are a custom type, or your ``collate_fn`` returns a batch that is a custom type see the example below (default: ``False``).
        drop_last (bool, optional): Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then the last batch will be smaller (default: ``False``).
        timeout (numeric, optional): If positive, the timeout value for collecting a batch from workers. Should always be non-negative (default: ``0``).
        worker_init_fn (callable, optional): If not ``None``, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: ``_default_worker_init_fn``)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        if worker_init_fn is None:
            worker_init_fn = self._default_worker_init_fn

        if collate_fn is None:
            super().__init__(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=sampler,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=drop_last,
                             timeout=timeout,
                             worker_init_fn=worker_init_fn)
        else:
            super().__init__(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=sampler,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             pin_memory=pin_memory,
                             drop_last=drop_last,
                             timeout=timeout,
                             worker_init_fn=worker_init_fn)

    @staticmethod
    def _default_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
