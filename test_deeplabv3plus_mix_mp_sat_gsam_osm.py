import os
import cv2
import argparse
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.metric import MixMetrics
from utils.util import *

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def save_image(sats, sat_maps, maps_partial, par_maps, maps_complete, save_dir, batch_idx):
    # remove batch_size, batch_size must be 1
    sats = sats.squeeze(0)
    sats = sats.transpose(1, 2, 0)
    sats = sats * 255.
    sats = cv2.cvtColor(sats, cv2.COLOR_RGB2BGR)
    sat_maps = sat_maps.squeeze(0)
    sat_maps = sat_maps * 255.
    maps_partial = maps_partial.squeeze(0)
    maps_partial = maps_partial * 255.
    par_maps = par_maps.squeeze(0)
    par_maps = par_maps * 255.
    maps_complete = maps_complete.squeeze(0)
    maps_complete = maps_complete * 255.

    save_dir = "{}{}/".format(save_dir, batch_idx)
    ensure_dir(save_dir)
    sats_name = "{}{}.png".format(save_dir, "sats")
    sat_maps_name = "{}{}.png".format(save_dir, "sat_maps")
    maps_partial_name = "{}{}.png".format(save_dir, "maps_partial")
    par_maps_name = "{}{}.png".format(save_dir, "maps")
    maps_complete_name = "{}{}.png".format(save_dir, "maps_complete")
    cv2.imwrite(sats_name, sats)
    cv2.imwrite(sat_maps_name, sat_maps)
    cv2.imwrite(maps_partial_name, maps_partial)
    cv2.imwrite(par_maps_name, par_maps)
    cv2.imwrite(maps_complete_name, maps_complete)


def main(config, args):
    # ========================================
    # Logger
    # ========================================
    logger = config.get_logger('test')

    # ========================================
    # DataLoader
    # ========================================
    valid_data_loader = config.init_obj('valid_data_loader', module_data)
    test_data_loader = config.init_obj('test_data_loader', module_data)

    # ========================================
    # Model
    # ========================================
    sat = config.init_obj('sat_arch', module_arch)
    par = config.init_obj('par_arch', module_arch)
    logger.info(sat)
    logger.info(par)

    # ========================================
    # Losses
    # ========================================
    losses = [getattr(module_loss, met) for met in config['loss']]
    bce_dice_loss = losses[0]
    mp_loss = losses[1]

    # ========================================
    # Metric
    # ========================================
    metrics_sat = MixMetrics(num_class=2)
    metrics_par = MixMetrics(num_class=2)
    metrics_avg = MixMetrics(num_class=2)

    # ========================================
    # Load Checkpoint
    # ========================================
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location="cuda:0")
    sat_state_dict = checkpoint['sat_state_dict']
    par_state_dict = checkpoint['par_state_dict']
    if config['n_gpu'] > 1:
        sat = torch.nn.DataParallel(sat)
        par = torch.nn.DataParallel(par)
    sat.load_state_dict(sat_state_dict)
    par.load_state_dict(par_state_dict)

    del checkpoint, sat_state_dict, par_state_dict

    # ========================================
    # Device
    # ========================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ========================================
    # Evaluation
    # ========================================
    sat = sat.to(device)
    par = par.to(device)

    log_step = int(np.sqrt(test_data_loader.batch_size)) * 5

    sat.eval()
    par.eval()
    metrics_sat.reset()
    metrics_par.reset()
    metrics_avg.reset()

    resume = args.resume
    resume_dir = resume.rsplit('/', 1)[0]
    save_dir = "{}/{}/".format(resume_dir, 'test')
    ensure_dir(save_dir)
    ratio = config["test_data_loader"]["args"]["ratio"]
    if ratio != "mix":
        ratio = int(ratio * 100)
    save_dir = "{}{}/".format(save_dir, ratio)
    ensure_dir(save_dir)

    with torch.no_grad():
        for batch_idx, (sats, maps_partial, maps_complete, _) in enumerate(test_data_loader):
            sats, maps_partial, maps_complete = sats.to(device), maps_partial.to(device), maps_complete.to(device)

            inputs = sats
            sat_maps, encs, fms = sat(inputs)

            inputs = maps_partial
            par_maps, _ = par(inputs, encs, fms)

            sats = sats.cpu().numpy()
            sat_maps = sat_maps.squeeze(1).cpu().numpy()
            sat_maps[sat_maps >= 0.5] = 1
            sat_maps[sat_maps < 0.5] = 0
            sat_maps = sat_maps.astype(np.int32)
            maps_partial = maps_partial.squeeze(1).cpu().numpy().astype(np.int32)
            maps_complete = maps_complete.squeeze(1).cpu().numpy().astype(np.int32)
            par_maps = par_maps.squeeze(1).cpu().numpy()
            par_maps[par_maps >= 0.5] = 1
            par_maps[par_maps < 0.5] = 0
            par_maps = par_maps.astype(np.int32)
            avg_maps = (sat_maps + par_maps) / 2
            avg_maps = avg_maps.astype(np.int32)
            metrics_sat.add_batch(maps_complete, sat_maps)
            metrics_par.add_batch(maps_complete, par_maps)
            metrics_avg.add_batch(maps_complete, avg_maps)

            save_image(sats, sat_maps, maps_partial, par_maps, maps_complete, save_dir, batch_idx)

            if batch_idx % log_step == 0:
                logger.info("[{}/{} ({:.0f}%)]".format(
                    batch_idx * test_data_loader.batch_size,
                    test_data_loader.n_samples,
                    100.0 * batch_idx / len(test_data_loader)
                    )
                )

    # sat
    precision = metrics_sat.precision()
    recall = metrics_sat.recall()
    f1score = metrics_sat.f1score()
    acc = metrics_sat.pixel_accuracy()
    acc_class = metrics_sat.pixel_accuracy_class()
    IoU = metrics_sat.intersection_over_union()
    mIoU = metrics_sat.mean_intersection_over_union()
    FWIoU = metrics_sat.frequency_weighted_intersection_over_union()
    log = {
        'test_metrics_precision': precision,
        'test_metrics_recall': recall,
        'test_metrics_f1score': f1score,
        'test_metrics_acc': acc,
        'test_metrics_acc_class': acc_class,
        'test_metrics_IoU': IoU,
        'test_metrics_mIoU': mIoU,
        'test_metrics_FWIoU': FWIoU
    }
    logger.info("Sat:")
    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(str(key), value))

    # par
    precision = metrics_par.precision()
    recall = metrics_par.recall()
    f1score = metrics_par.f1score()
    acc = metrics_par.pixel_accuracy()
    acc_class = metrics_par.pixel_accuracy_class()
    IoU = metrics_par.intersection_over_union()
    mIoU = metrics_par.mean_intersection_over_union()
    FWIoU = metrics_par.frequency_weighted_intersection_over_union()
    log = {
        'test_metrics_precision': precision,
        'test_metrics_recall': recall,
        'test_metrics_f1score': f1score,
        'test_metrics_acc': acc,
        'test_metrics_acc_class': acc_class,
        'test_metrics_IoU': IoU,
        'test_metrics_mIoU': mIoU,
        'test_metrics_FWIoU': FWIoU
    }
    logger.info("Par:")
    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(str(key), value))

    # avg
    precision = metrics_avg.precision()
    recall = metrics_avg.recall()
    f1score = metrics_avg.f1score()
    acc = metrics_avg.pixel_accuracy()
    acc_class = metrics_avg.pixel_accuracy_class()
    IoU = metrics_avg.intersection_over_union()
    mIoU = metrics_avg.mean_intersection_over_union()
    FWIoU = metrics_avg.frequency_weighted_intersection_over_union()
    log = {
        'test_metrics_precision': precision,
        'test_metrics_recall': recall,
        'test_metrics_f1score': f1score,
        'test_metrics_acc': acc,
        'test_metrics_acc_class': acc_class,
        'test_metrics_IoU': IoU,
        'test_metrics_mIoU': mIoU,
        'test_metrics_FWIoU': FWIoU
    }
    logger.info("Avg:")
    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Test')
    args.add_argument('-c', '--config', default='./configs/deeplabv3plus_mix_mp_sat_gsam_osm_config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume',
                      default='./saved/osm/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args)
