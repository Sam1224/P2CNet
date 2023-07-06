import os
import math
import json
import logging
import datetime
import torch
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX


class BaseFuseNetTrainer:
    """
    Base class for all FuseNet trainers
    """

    def __init__(self, models, optimizers, loss, metrics, resume, config, train_logger=None):
        """
        class initialization
        :param models: models dictionary contains generator model, discriminator model
        :param optimizers: optimizers dictionary contains generator optimizer and discriminator optimizers
        :param loss: loss dictionary contains the loss objectives
        :param metrics: other metrics except for the loss want to display during training
        :param resume: resume checkpoint
        :param config: config file
        :param train_logger: logger
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        self.sat = models["sat"].to(self.device)
        self.par = models["par"].to(self.device)

        # paralleling the models if multiple GPUs
        if len(self.device_ids) > 1:
            self.sat = torch.nn.DataParallel(self.sat, device_ids=self.device_ids)
            self.par = torch.nn.DataParallel(self.par, device_ids=self.device_ids)

        self.loss = loss
        self.metrics = metrics
        self.train_logger = train_logger
        self.optimizer = optimizers["optimizer"]

        # read training settings
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.verbosity = cfg_trainer['verbosity']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)

        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], config['name'], start_time)
        # setup visualization writer instance
        writer_dir = os.path.join(cfg_trainer['log_dir'], config['name'], start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, cfg_trainer['tensorboardX'])

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        self.config["checkpoint_dir"] = self.checkpoint_dir
        config_save_path = os.path.join(self.checkpoint_dir, '{}.json'.format(config['name']))
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        print("have {} gpu".format(n_gpu))
        print("use {} gpu".format(n_gpu_use))
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def pre_train(self):
        """
        pretrain the FuseNet
        :return:
        """
        if not os.path.exists(self.config["trainer"]["pretrain_sat_path"]):
            print("pre train fusesat")
            self.pre_train_sat()
            self._save_sat_checkpoint()
        else:
            print("load fusesat")
            checkpoint = torch.load(self.config["trainer"]["pretrain_sat_path"], map_location='cpu')
            self.sat.load_state_dict(checkpoint['sat_state_dict'])
            self._save_sat_checkpoint()

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            # save logged informations into log dict
            log = {'epoch': epoch}
            result = self._train_epoch(epoch)
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    self._save_checkpoint(epoch, save_best=best)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def pre_train_sat(self):
        """
        Pre FuseSat training logic
        """
        for epoch in range(self.config["trainer"]["pretrain_sat_epochs"]):

            # save logged informations into log dict
            log = {'pretrain fusesat epoch': epoch}
            result = self._train_sat_epoch(epoch)
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

    def _train_sat_epoch(self, epoch):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        sat_arch = type(self.sat).__name__
        par_arch = type(self.par).__name__
        state = {
            'sat_arch': sat_arch,
            'par_arch': par_arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'sat_state_dict': self.sat.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'par_state_dict': self.par.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _save_sat_checkpoint(self):
        """
        Saving pre-trained FuseSat checkpoints
        """
        sat_arch = type(self.sat).__name__
        state = {
            'sat_arch': sat_arch,
            'logger': self.train_logger,
            'sat_state_dict': self.sat.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'pretrain_sat.pth')
        torch.save(state, filename)
        self.logger.info("Saving fusesat checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        # self.mnt_best = checkpoint['monitor_best']
        self.mnt_best = -math.inf

        # load architecture params from checkpoint.
        if checkpoint['config']['par_arch'] != self.config['par_arch'] or \
                checkpoint['config']['sat_arch'] != self.config['sat_arch']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.')
        self.par.load_state_dict(checkpoint['par_state_dict'])
        self.sat.load_state_dict(checkpoint['sat_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
