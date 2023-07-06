import os
import numpy as np
import torch
from base import BaseFuseNetTrainer
from matplotlib import pyplot as plt
from utils.util import ensure_dir
from torch.autograd import Variable

plt.switch_backend('agg')


class FuseNetTrainer(BaseFuseNetTrainer):
    """
    FuseNet Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, models, optimizers, loss, metrics, resume, config,
                 train_data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):

        super(FuseNetTrainer, self).__init__(
            models,
            optimizers,
            loss=loss,
            metrics=metrics,
            resume=resume,
            config=config,
            train_logger=train_logger)

        self.config = config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(train_data_loader.batch_size)) * 5
        self.lr_scheduler = lr_scheduler

        self.weight_sat = config["trainer"]["weight_sat"]
        self.weight_par = config["trainer"]["weight_par"]

    def _train_sat_epoch(self, epoch):
        """
        Pre training logic for an epoch
        :param epoch: Current training epoch
        :return: A log that contrains all information you want to save
        Note:
        If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The m
            etrics in log must have the key 'metrics'.
        """

        self.sat.train()
        total_loss = 0.
        total_bce_loss = 0.
        total_dice_loss = 0.
        for batch_idx, (sats, maps_partial, maps_complete, _) in enumerate(self.train_data_loader):
            sats, maps_partial, maps_complete = sats.to(self.device), maps_partial.to(self.device), maps_complete.to(
                self.device)

            self.optimizer.zero_grad()
            inputs = sats
            maps, encs, fms = self.sat(inputs)

            # BCE Dice Loss
            loss_sum, bce_loss, dice_loss = self.loss["BCE_Dice"](maps, maps_complete)
            loss_sum.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('sat_loss', loss_sum.item())
            self.writer.add_scalar('sat_bce_loss', bce_loss.item())
            self.writer.add_scalar('sat_dice_loss', dice_loss.item())
            total_loss += loss_sum.item()
            total_bce_loss += bce_loss.item()
            total_dice_loss += dice_loss.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] '
                    'Total:{:.6f} '
                    'FuseSat: [BCE:{:.6f}, Dice:{:.6f}'.format(
                        epoch,
                        batch_idx * self.train_data_loader.batch_size,
                        self.train_data_loader.n_samples,
                        100.0 * batch_idx / len(self.train_data_loader),
                        loss_sum.item(),
                        bce_loss.item(),
                        dice_loss.item()
                    ))

        log = {
            'sat_loss': total_loss / len(self.train_data_loader),
            'sat_bce_loss': total_bce_loss / len(self.train_data_loader),
            'sat_dice_loss': total_dice_loss / len(self.train_data_loader)
        }

        return log

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.sat.train()
        self.par.train()

        total_loss = 0.
        total_sat_loss = 0.
        total_sat_bce_loss = 0.
        total_sat_dice_loss = 0.
        total_par_loss = 0.
        total_par_bce_loss = 0.
        total_par_dice_loss = 0.
        for batch_idx, (sats, maps_partial, maps_complete, _) in enumerate(self.train_data_loader):
            sats, maps_partial, maps_complete = sats.to(self.device), maps_partial.to(self.device), maps_complete.to(
                self.device)

            self.optimizer.zero_grad()
            inputs = sats
            sat_maps, encs, fms = self.sat(inputs)
            sat_loss_sum, sat_bce_loss, sat_dice_loss = self.loss["BCE_Dice"](sat_maps, maps_complete)

            inputs = maps_partial
            par_maps, _ = self.par(inputs, encs, fms)
            par_loss_sum, par_bce_loss, par_dice_loss = self.loss["BCE_Dice"](par_maps, maps_complete)

            loss_sum = sat_loss_sum * self.weight_sat + par_loss_sum * self.weight_par
            torch.cuda.empty_cache()
            loss_sum.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('total_loss', sat_loss_sum.item() + par_loss_sum.item())
            self.writer.add_scalar('sat_loss', sat_loss_sum.item())
            self.writer.add_scalar('sat_bce_loss', sat_bce_loss.item())
            self.writer.add_scalar('sat_dice_loss', sat_dice_loss.item())
            self.writer.add_scalar('par_loss', par_loss_sum.item())
            self.writer.add_scalar('par_bce_loss', par_bce_loss.item())
            self.writer.add_scalar('par_dice_loss', par_dice_loss.item())
            total_loss += sat_loss_sum.item() + par_loss_sum.item()
            total_sat_loss += sat_loss_sum.item()
            total_sat_bce_loss += sat_bce_loss.item()
            total_sat_dice_loss += sat_dice_loss.item()
            total_par_loss += par_loss_sum.item()
            total_par_bce_loss += par_bce_loss.item()
            total_par_dice_loss += par_dice_loss.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] '
                                 'Total: {:.6f} '
                                 'FuseSat: [BCE:{:.6f}, Dice:{:.6f}] '
                                 'FusePar: [BCE:{:.6F}, Dice:{:.6f}]'.format(
                    epoch,
                    batch_idx * self.train_data_loader.batch_size,
                    self.train_data_loader.n_samples,
                    100.0 * batch_idx / len(self.train_data_loader),
                    sat_loss_sum.item() + par_loss_sum.item(),
                    sat_bce_loss.item(),
                    sat_dice_loss.item(),
                    par_bce_loss.item(),
                    par_dice_loss.item()
                ))
                self.save_images(sats, sat_maps, maps_partial, par_maps, maps_complete, epoch=epoch,
                                 batch_idx=batch_idx, r=2, c=5)

        log = {
            'total_loss': total_loss / len(self.train_data_loader),
            'sat_loss': total_sat_loss / len(self.train_data_loader),
            'sat_bce_loss': total_sat_bce_loss / len(self.train_data_loader),
            'sat_dice_loss': total_sat_dice_loss / len(self.train_data_loader),
            'par_loss': total_par_loss / len(self.train_data_loader),
            'par_bce_loss': total_par_bce_loss / len(self.train_data_loader),
            'par_dice_loss': total_par_dice_loss / len(self.train_data_loader)
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(total_loss)

        self.save_images(sats, sat_maps, maps_partial, par_maps, maps_complete, epoch=epoch, batch_idx=batch_idx, r=2,
                         c=5)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.sat.eval()
        self.par.eval()
        total_val_loss = 0.
        total_sat_loss = 0.
        total_sat_bce_loss = 0.
        total_sat_dice_loss = 0.
        total_par_loss = 0.
        total_par_bce_loss = 0.
        total_par_dice_loss = 0.
        self.metrics.reset()
        with torch.no_grad():
            for batch_idx, (sats, maps_partial, maps_complete, _) in enumerate(self.valid_data_loader):
                sats, maps_partial, maps_complete = sats.to(self.device), maps_partial.to(
                    self.device), maps_complete.to(self.device)

                inputs = sats
                sat_maps, encs, fms = self.sat(inputs)
                sat_loss_sum, sat_bce_loss, sat_dice_loss = self.loss["BCE_Dice"](sat_maps, maps_complete)

                inputs = maps_partial
                par_maps, _ = self.par(inputs, encs, fms)
                par_loss_sum, par_bce_loss, par_dice_loss = self.loss["BCE_Dice"](par_maps, maps_complete)

                self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
                self.writer.add_scalar('val_total_loss', sat_loss_sum.item() + par_loss_sum.item())
                self.writer.add_scalar('val_sat_loss', sat_loss_sum.item())
                self.writer.add_scalar('val_sat_bce_loss', sat_bce_loss.item())
                self.writer.add_scalar('val_sat_dice_loss', sat_dice_loss.item())
                self.writer.add_scalar('val_par_loss', par_loss_sum.item())
                self.writer.add_scalar('val_par_bce_loss', par_bce_loss.item())
                self.writer.add_scalar('val_par_dice_loss', par_dice_loss.item())
                total_val_loss += sat_loss_sum.item() + par_loss_sum.item()
                total_sat_loss += sat_loss_sum.item()
                total_sat_bce_loss += sat_bce_loss.item()
                total_sat_dice_loss += sat_dice_loss.item()
                total_par_loss += par_loss_sum.item()
                total_par_bce_loss += par_bce_loss.item()
                total_par_dice_loss += par_dice_loss.item()

                maps_complete = maps_complete.squeeze(1).cpu().numpy().astype(np.int32)
                par_maps = par_maps.squeeze(1).cpu().numpy()
                par_maps[par_maps >= 0.5] = 1
                par_maps[par_maps < 0.5] = 0
                par_maps = par_maps.astype(np.int32)
                self.metrics.add_batch(maps_complete, par_maps)

        return {
            'val_total_loss': total_val_loss / len(self.valid_data_loader),
            'val_sat_loss': total_sat_loss / len(self.train_data_loader),
            'val_sat_bce_loss': total_sat_bce_loss / len(self.train_data_loader),
            'val_sat_dice_loss': total_sat_dice_loss / len(self.train_data_loader),
            'val_par_loss': total_par_loss / len(self.train_data_loader),
            'val_par_bce_loss': total_par_bce_loss / len(self.train_data_loader),
            'val_par_dice_loss': total_par_dice_loss / len(self.train_data_loader),
            'val_metrics_acc': self.metrics.pixel_accuracy(),
            'val_metrics_acc_class': self.metrics.pixel_accuracy_class(),
            'val_metrics_IoU': self.metrics.intersection_over_union(),
            'val_metrics_mIoU': self.metrics.mean_intersection_over_union(),
            'val_metrics_FWIoU': self.metrics.frequency_weighted_intersection_over_union()
        }

    def save_images(self, sats, sat_maps, partials, par_maps, maps_complete, epoch, batch_idx, r=1, c=4):
        bs = sats.size(0)
        r = r if bs >= r else bs
        sats = sats.cpu().numpy()
        sat_maps = sat_maps.detach().squeeze(1).cpu().numpy()
        sat_maps[sat_maps >= 0.5] = 1
        sat_maps[sat_maps < 0.5] = 0
        partials = partials.squeeze(1).cpu().numpy()
        par_maps = par_maps.detach().squeeze(1).cpu().numpy()
        par_maps[par_maps >= 0.5] = 1
        par_maps[par_maps < 0.5] = 0
        maps_complete = maps_complete.squeeze(1).cpu().numpy()

        fig, axs = plt.subplots(r, c)

        if r == 1:
            axs[0].set_title('Sat')
            axs[0].imshow(sats[0].transpose(1, 2, 0))
            axs[0].axis('off')

            axs[1].set_title('Sat->Map')
            axs[1].imshow(sat_maps[0], cmap='gray')
            axs[1].axis('off')

            axs[2].set_title('Par')
            axs[2].imshow(partials[0], cmap='gray')
            axs[2].axis('off')

            axs[3].set_title('Par->Map')
            axs[3].imshow(par_maps[0], cmap='gray')
            axs[3].axis('off')

            axs[4].set_title('Complete Map')
            axs[4].imshow(maps_complete[0], cmap='gray')
            axs[4].axis('off')
        else:
            count = 0
            for row in range(r):
                if row == 0:
                    axs[row, 0].set_title('Sat')
                axs[row, 0].imshow(sats[count].transpose(1, 2, 0))
                axs[row, 0].axis('off')

                if row == 0:
                    axs[row, 1].set_title('Sat->Map')
                axs[row, 1].imshow(sat_maps[count], cmap='gray')
                axs[row, 1].axis('off')

                if row == 0:
                    axs[row, 2].set_title('Par')
                axs[row, 2].imshow(partials[count], cmap='gray')
                axs[row, 2].axis('off')

                if row == 0:
                    axs[row, 3].set_title('Par->Map')
                axs[row, 3].imshow(par_maps[count], cmap='gray')
                axs[row, 3].axis('off')

                if row == 0:
                    axs[row, 4].set_title('Complete Map')
                axs[row, 4].imshow(maps_complete[count], cmap='gray')
                axs[row, 4].axis('off')
                count += 1

        ensure_dir(os.path.join(self.checkpoint_dir, 'results', 'epoch_{}').format(epoch))
        fig.savefig('{0}/results/epoch_{1}/{2}.jpg'.format(self.checkpoint_dir, epoch, batch_idx))
        plt.close(fig)
