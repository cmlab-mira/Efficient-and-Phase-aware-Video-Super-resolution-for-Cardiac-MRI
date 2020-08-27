import csv
import torch
import logging
import imageio
import numpy as np
from scipy.misc import imsave
from tqdm import tqdm
from pathlib import Path

from src.runner.predictors import AcdcSISRPredictor


class AcdcSISRSRFBPredictor(AcdcSISRPredictor):
    """The ACDC predictor for the Single-Image Super-Resolution using the SRFBNet.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            input, target, index = self._get_inputs_targets(batch)
            with torch.no_grad():
                lr_path, hr_path = self.test_dataloader.dataset.data[index]
                filename = lr_path.parts[-1].split('.')[0]
                patient, _, sid, fid = filename.split('_')
                
                outputs = self.net(input)
                losses = self._compute_losses(outputs, target)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                metrics = self._compute_metrics(outputs, target, patient)

                if self.exported:
                    _losses = [loss.item() for loss in losses]
                    _metrics = [metric.item() for metric in metrics]
                    results.append([filename, *_metrics, *_losses])

                    # Save the video.
                    if sid != tmp_sid and index != 0:
                        output_dir = videos_dir / patient
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True)
                        video_name = tmp_sid.replace('slice', 'sequence') + '.gif'
                        self._dump_video(output_dir / video_name, sr_imgs)
                        sr_imgs = []

                    output = self._denormalize(outputs[-1])
                    sr_img = output.squeeze().detach().cpu().numpy().astype(np.uint8)
                    sr_imgs.append(sr_img)
                    tmp_sid = sid

                    # Save the image.
                    output_dir = imgs_dir / patient
                    if not output_dir.is_dir():
                        output_dir.mkdir(parents=True)
                    imsave(output_dir / f'{sid}_{fid}.png', sr_img)

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

    def _compute_losses(self, outputs, target):
        """Compute the losses.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            target (torch.Tensor): The data target.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = []
        for loss_fn in self.loss_fns:
            # Average the losses computed at every time steps.
            loss = torch.stack([loss_fn(output, target) for output in outputs]).mean()
            losses.append(loss)
        return losses

    def _compute_metrics(self, outputs, target, name):
        """Compute the metrics.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            target (torch.Tensor): The data target.
            name (str): The patient name.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        output, target = self._denormalize(outputs[-1]), self._denormalize(target)
        metrics = []
        for metric_fn in self.metric_fns:
            if 'Cardiac' in metric_fn.__class__.__name__:
                metrics.append(metric_fn(output, target, name))
            else:
                metrics.append(metric_fn(output, target))
        return metrics