import os
import numpy as np
import torch
from base import BaseNetTrainer
from matplotlib import pyplot as plt
from utils.util import ensure_dir


plt.switch_backend('agg')


class NetTrainer(BaseNetTrainer):
    """
    GAN Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, models, optimizers, loss, metrics, resume, config,
                 train_data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):

        super(NetTrainer, self).__init__(
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

    def _train_net_epoch(self, epoch):
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

        self.model.train()
        total_loss = 0.
        total_bce_loss = 0.
        total_dice_loss = 0.
        for batch_idx, (sats, maps_partial, maps_complete, _) in enumerate(self.train_data_loader):
            sats, maps_partial, maps_complete = sats.to(self.device), maps_partial.to(self.device), maps_complete.to(self.device)

            self.optimizer.zero_grad()
            inputs = torch.cat((sats, maps_partial), dim=1)
            maps = self.model(inputs)

            # BCE Dice Loss
            loss_sum, bce_loss, dice_loss = self.loss["BCE_Dice"](maps, maps_complete)
            loss_sum.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss_sum.item())
            self.writer.add_scalar('bce_loss', bce_loss.item())
            self.writer.add_scalar('dice_loss', dice_loss.item())
            total_loss += loss_sum.item()
            total_bce_loss += bce_loss.item()
            total_dice_loss += dice_loss.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} BCE Loss: {:.6f} Dice Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.train_data_loader.batch_size,
                    self.train_data_loader.n_samples,
                    100.0 * batch_idx / len(self.train_data_loader),
                    loss_sum.item(),
                    bce_loss.item(),
                    dice_loss.item()
                    ))

        log = {
            'loss': total_loss / len(self.train_data_loader),
            'bce_loss': total_bce_loss / len(self.train_data_loader),
            'dice_loss': total_dice_loss / len(self.train_data_loader)
        }

        return log

    def forward(self, batch_x):
        maps = self.model(batch_x)
        return maps

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
        self.model.train()
        total_loss = 0.
        total_bce_loss = 0.
        total_dice_loss = 0.
        for batch_idx, (sats, maps_partial, maps_complete, _) in enumerate(self.train_data_loader):
            sats, maps_partial, maps_complete = sats.to(self.device), maps_partial.to(self.device), maps_complete.to(
                self.device)

            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            inputs = torch.cat((sats, maps_partial), dim=1)
            maps = self.model(inputs)

            # BCE Dice Loss
            loss_sum, bce_loss, dice_loss = self.loss["BCE_Dice"](maps, maps_complete)
            torch.cuda.empty_cache()
            del inputs
            loss_sum.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss_sum.item())
            self.writer.add_scalar('bce_loss', bce_loss.item())
            self.writer.add_scalar('dice_loss', dice_loss.item())
            total_loss += loss_sum.item()
            total_bce_loss += bce_loss.item()
            total_dice_loss += dice_loss.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} BCE Loss: {:.6f} Dice Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.train_data_loader.batch_size,
                        self.train_data_loader.n_samples,
                        100.0 * batch_idx / len(self.train_data_loader),
                        loss_sum.item(),
                        bce_loss.item(),
                        dice_loss.item()
                    ))
                self.save_images(sats, maps_partial, maps, maps_complete, epoch=epoch, batch_idx=batch_idx, r=2, c=4)

        log = {
            'loss': total_loss / len(self.train_data_loader),
            'bce_loss': total_bce_loss / len(self.train_data_loader),
            'dice_loss': total_dice_loss / len(self.train_data_loader)
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(total_loss)

        # Save images
        self.save_images(sats, maps_partial, maps, maps_complete, epoch=epoch, batch_idx=batch_idx, r=2, c=4)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_bce_loss = 0.
        total_dice_loss = 0.
        self.metrics.reset()
        with torch.no_grad():
            for batch_idx, (sats, maps_partial, maps_complete, _) in enumerate(self.valid_data_loader):
                sats, maps_partial, maps_complete = sats.to(self.device), maps_partial.to(self.device), maps_complete.to(self.device)
                inputs = torch.cat((sats, maps_partial), dim=1)
                maps = self.forward(inputs)

                # BCE Dice Loss
                loss_sum, bce_loss, dice_loss = self.loss["BCE_Dice"](maps, maps_complete)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)
                self.writer.add_scalar('val_loss', loss_sum.item())
                self.writer.add_scalar('val_bce_loss', bce_loss.item())
                self.writer.add_scalar('val_dice_loss', dice_loss.item())
                total_val_loss += loss_sum.item()
                total_bce_loss += bce_loss.item()
                total_dice_loss += dice_loss.item()

                maps_complete = maps_complete.squeeze(1).cpu().numpy().astype(np.int32)
                maps = maps.squeeze(1).cpu().numpy()
                maps[maps >= 0.5] = 1
                maps[maps < 0.5] = 0
                maps = maps.astype(np.int32)
                self.metrics.add_batch(maps_complete, maps)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_bce_loss': total_bce_loss / len(self.valid_data_loader),
            'val_dice_loss': total_dice_loss / len(self.valid_data_loader),
            'val_metrics_acc': self.metrics.pixel_accuracy(),
            'val_metrics_acc_class': self.metrics.pixel_accuracy_class(),
            'val_metrics_IoU': self.metrics.intersection_over_union(),
            'val_metrics_mIoU': self.metrics.mean_intersection_over_union(),
            'val_metrics_FWIoU': self.metrics.frequency_weighted_intersection_over_union()
        }

    def save_images(self, sats, partials, maps, maps_complete, epoch, batch_idx, r=1, c=4):
        bs = sats.size(0)
        r = r if bs >= r else bs
        sats = sats.cpu().numpy()
        partials = partials.squeeze(1).cpu().numpy()
        maps = maps.detach().squeeze(1).cpu().numpy()
        maps[maps >= 0.5] = 1
        maps[maps < 0.5] = 0
        maps_complete = maps_complete.squeeze(1).cpu().numpy()

        fig, axs = plt.subplots(r, c)

        if r == 1:
            axs[0].set_title('Satellite')
            axs[0].imshow(sats[0].transpose(1, 2, 0))
            axs[0].axis('off')

            axs[1].set_title('Partial Map')
            axs[1].imshow(partials[0], cmap='gray')
            axs[1].axis('off')

            axs[2].set_title('Generated Map')
            axs[2].imshow(maps[0], cmap='gray')
            axs[2].axis('off')

            axs[3].set_title('Complete Map')
            axs[3].imshow(maps_complete[0], cmap='gray')
            axs[3].axis('off')
        else:
            count = 0
            for row in range(r):
                axs[row, 0].set_title('Satellite - {}'.format(count))
                axs[row, 0].imshow(sats[count].transpose(1, 2, 0))
                axs[row, 0].axis('off')

                axs[row, 1].set_title('Partial Map - {}'.format(count))
                axs[row, 1].imshow(partials[count], cmap='gray')
                axs[row, 1].axis('off')

                axs[row, 2].set_title('Generated Map - {}'.format(count))
                axs[row, 2].imshow(maps[count], cmap='gray')
                axs[row, 2].axis('off')

                axs[row, 3].set_title('Complete Map - {}'.format(count))
                axs[row, 3].imshow(maps_complete[count], cmap='gray')
                axs[row, 3].axis('off')
                count += 1

        ensure_dir(os.path.join(self.checkpoint_dir, 'results', 'epoch_{}').format(epoch))
        fig.savefig('{0}/results/epoch_{1}/{2}.jpg'.format(self.checkpoint_dir, epoch, batch_idx))
        plt.close(fig)
