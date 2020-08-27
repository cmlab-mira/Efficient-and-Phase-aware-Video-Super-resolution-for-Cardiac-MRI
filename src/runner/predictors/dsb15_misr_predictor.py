import csv
import torch
import logging
import imageio
import numpy as np
import functools
from scipy.misc import imsave
from tqdm import tqdm
from pathlib import Path

from src.runner.predictors.base_predictor import BasePredictor
from src.utils import denormalize


class Dsb15MISRPredictor(BasePredictor):
    """The DSB15 predictor for the Multi-Images Super-Resolution.
    Args:
        saved_dir (str): The directory to save the predicted videos, images and metrics (default: None).
        exported (bool): Whether to export the predicted video, images and metrics (default: False).
    """
    def __init__(self, saved_dir=None, exported=False, **kwargs):
        super().__init__(**kwargs)
        if self.test_dataloader.batch_size != 1:
            raise ValueError(f'The testing batch size should be 1. Got {self.test_dataloader.batch_size}.')

        if exported:
            self.saved_dir = Path(saved_dir)
        self.exported = exported
        self._denormalize = functools.partial(denormalize, dataset='dsb15')
        
    def predict(self):
        """The testing process.
        """
        self.net.eval()
        trange = tqdm(self.test_dataloader,
                      total=len(self.test_dataloader),
                      desc='testing')

        if self.exported:
            videos_dir = self.saved_dir / 'videos'
            imgs_dir = self.saved_dir / 'imgs'
            csv_path = self.saved_dir / 'results.csv'

            sr_imgs = []
            tmp_sid = None
            header = ['name'] + \
                     [metric_fn.__class__.__name__ for metric_fn in self.metric_fns] + \
                     [loss_fns.__class__.__name__ for loss_fns in self.loss_fns]
            results = [header]

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            inputs, target, index = self._get_inputs_targets(batch)
            with torch.no_grad():
                lr_path, hr_path, t = self.test_dataloader.dataset.data[index]
                filename = lr_path.parts[-1].split('.')[0]
                patient, _, sid = filename.split('_')
                
                output = self.net(inputs)
                losses = self._compute_losses(output, target)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                metrics = self._compute_metrics(output, target, patient)

                if self.exported:
                    fid = f'frame{t+1:0>2d}'
                    _losses = [loss.item() for loss in losses]
                    _metrics = [metric.item() for metric in metrics]
                    filename = filename.replace('2d+1d', '2d').replace('sequence', 'slice') + f'_{fid}'
                    results.append([filename, *_metrics, *_losses])

                    # Save the video.
                    if sid != tmp_sid and index != 0:
                        output_dir = videos_dir / patient
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True)
                        self._dump_video(output_dir / f'{tmp_sid}.gif', sr_imgs)
                        sr_imgs = []

                    output = self._denormalize(output)
                    sr_img = output.squeeze().detach().cpu().numpy().astype(np.uint8)
                    sr_imgs.append(sr_img)
                    tmp_sid = sid

                    # Save the image.
                    output_dir = imgs_dir / patient
                    if not output_dir.is_dir():
                        output_dir.mkdir(parents=True)
                    img_name = sid.replace('sequence', 'slice') + f'_{fid}.png'
                    imsave(output_dir / img_name, sr_img)

            batch_size = self.test_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        # Save the results.
        if self.exported:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results)

        for key in log:
            log[key] /= count
        logging.info(f'Test log: {log}.')

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.

        Returns:
            inputs (list of torch.Tensor): The data inputs.
            target (torch.Tensor): The data target.
            index (int): The index of the target path in the `dataloder.data`.
        """
        return batch['lr_imgs'], batch['hr_img'], batch['index']

    def _compute_losses(self, output, target):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The data target.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss_fn(output, target) for loss_fn in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target, name):
        """Compute the metrics.
        Args:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The data target.
            name (str): The patient name.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        output, target = self._denormalize(output), self._denormalize(target)
        metrics = []
        for metric_fn in self.metric_fns:
            if 'Cardiac' in metric_fn.__class__.__name__:
                metrics.append(metric_fn(output, target, name))
            else:
                metrics.append(metric_fn(output, target))
        return metrics
    
    def _dump_video(self, path, imgs):
        """To dump the video by concatenate the images.
        Args:
            path (Path): The path to save the video.
            imgs (list): The images to form the video.
        """
        with imageio.get_writer(path) as writer:
            for img in imgs:
                writer.append_data(img)