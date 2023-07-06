import os
import json
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import torch.nn as nn
from parse_config import ConfigParser
from trainer import FuseNetMPSatTrainer
from utils import Logger
from model.metric import MixMetrics

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, config_log, resume):
    # ========================================
    # Logger
    # ========================================
    logger = Logger()

    # ========================================
    # DataLoaders
    # ========================================
    train_data_loader = config.init_obj('train_data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)

    # ========================================
    # Models
    # ========================================
    models = {}
    sat = config.init_obj('sat_arch', module_arch)
    par = config.init_obj('par_arch', module_arch)

    models["sat"] = sat
    models["par"] = par

    # pretrained sat branch
    if config["trainer"]["pretrained_sat"]:
        print("load pretrained sat branch from {}".format(config["trainer"]["pretrained_sat"]))
        checkpoint = torch.load(config["trainer"]["pretrained_sat"], map_location="cuda:1")

        sat_state_dict = checkpoint['sat_state_dict']
        new_sat_state_dict = sat.state_dict()
        sat_state_dict = {k: v for k, v in sat_state_dict.items() if k in new_sat_state_dict}
        new_sat_state_dict.update(sat_state_dict)

        par_state_dict = checkpoint['par_state_dict']
        new_par_state_dict = par.state_dict()
        par_state_dict = {k: v for k, v in par_state_dict.items() if k in new_par_state_dict}
        new_par_state_dict.update(par_state_dict)
        if config['n_gpu'] > 1:
            sat = torch.nn.DataParallel(sat)
            par = torch.nn.DataParallel(par)
        sat.load_state_dict(new_sat_state_dict)
        par.load_state_dict(new_par_state_dict)

        for p in sat.named_parameters():
            p[1].requires_grad = False

        del checkpoint, sat_state_dict, new_sat_state_dict, par_state_dict, new_par_state_dict

    # ========================================
    # Losses
    # ========================================
    loss = {}
    losses = [getattr(module_loss, met) for met in config['loss']]
    # BCE Dice Loss
    loss["BCE_Dice"] = losses[0]
    loss["MP"] = losses[1]

    # ========================================
    # Metrics
    # ========================================
    metrics = MixMetrics(num_class=2)

    # ========================================
    # Optimizers
    # ========================================
    optimizers = {}
    # params = [
    #     {"params": [p for n, p in sat.named_parameters() if p.requires_grad], "lr": config["optimizer"]["args"]["lr"] / 100},
    #     {"params": [p for n, p in par.named_parameters() if p.requires_grad and "adaptor" not in n], "lr": config["optimizer"]["args"]["lr"] / 100},
    #     {"params": [p for n, p in par.named_parameters() if p.requires_grad and "adaptor" in n], "lr": config["optimizer"]["args"]["lr"]}
    # ]
    # optimizer = torch.optim.Adam(
    #     params,
    #     lr=config["optimizer"]["args"]["lr"],
    #     weight_decay=config["optimizer"]["args"]["weight_decay"],
    #     betas=config["optimizer"]["args"]["betas"],
    #     amsgrad=config["optimizer"]["args"]["amsgrad"]
    # )
    sat_trainable_params = filter(lambda p: p.requires_grad, sat.parameters())
    par_trainable_params = filter(lambda p: p.requires_grad, par.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, list(sat_trainable_params) + list(par_trainable_params))
    optimizers["optimizer"] = optimizer

    # ========================================
    # LR Scheduler
    # ========================================
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # ========================================
    # Select Trainer and Start Training
    # ========================================
    trainer = FuseNetMPSatTrainer(models, optimizers, loss, metrics,
                                  resume=resume,
                                  config=config_log,
                                  train_data_loader=train_data_loader,
                                  valid_data_loader=valid_data_loader,
                                  lr_scheduler=lr_scheduler,
                                  train_logger=logger)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Train')
    args.add_argument('-c', '--config', default='./configs/deeplabv3plus_mix_mp_sat_gsam_spacenet_config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    args = args.parse_args()

    config_log = None
    if args.config:
        # load config file
        config_log = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config_log = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c gan_config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, config_log, args.resume)
