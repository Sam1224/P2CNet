import os
import cv2
import numpy as np
import torch
import random
import math
import json
import collections
dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)
from torch.utils.data import Dataset
from utils.util import get_files
from torchvision import transforms


# ========================================
# SpaceNet Road Dataset
# - Train: [sat, map]
# - Valid: [sat, map]
# - Test:  [sat, map]
# ========================================
class SpaceNetDataset(Dataset):
    def __init__(self, data_dir, mode="train", ratio=0.75, augmentation=None, transform=None):
        assert ratio in [0.0, 0.25, 0.50, 0.75, 'mix']
        if mode == 'train':
            self.training = True
        else:
            self.training = False
        self.ratio = ratio
        self.augmentation = augmentation
        self.transform = transform
        self.base_path = os.path.join(dirname, data_dir)

        vegas_path = os.path.join(self.base_path, 'Vegas/')
        paris_path = os.path.join(self.base_path, 'Paris/')
        shanghai_path = os.path.join(self.base_path, 'Shanghai/')
        khartoum_path = os.path.join(self.base_path, 'Khartoum/')
        sat_vegas = get_files(os.path.join(vegas_path, 'sats/'), format='png')
        sat_paris = get_files(os.path.join(paris_path, 'sats/'), format='png')
        sat_shanghai = get_files(os.path.join(shanghai_path, 'sats/'), format='png')
        sat_khartoum = get_files(os.path.join(khartoum_path, 'sats/'), format='png')
        map_vegas = get_files(os.path.join(vegas_path, 'maps/'), format='png')
        map_paris = get_files(os.path.join(paris_path, 'maps/'), format='png')
        map_shanghai = get_files(os.path.join(shanghai_path, 'maps/'), format='png')
        map_khartoum = get_files(os.path.join(khartoum_path, 'maps/'), format='png')
        self.sat_ids = sat_vegas
        self.sat_ids.extend(sat_paris)
        self.sat_ids.extend(sat_shanghai)
        self.sat_ids.extend(sat_khartoum)
        self.map_ids = map_vegas
        self.map_ids.extend(map_paris)
        self.map_ids.extend(map_shanghai)
        self.map_ids.extend(map_khartoum)

        # never mind we set ratio to 0.75 here, when ratio is 0, we would create an empty mask later
        if ratio == 'mix':
            self.partial_ids = []
            par25_vegas = get_files(os.path.join(vegas_path, 'maps_{}/'.format(int(100 * 0.25))), format='png')
            par25_paris = get_files(os.path.join(paris_path, 'maps_{}/'.format(int(100 * 0.25))), format='png')
            par25_shanghai = get_files(os.path.join(shanghai_path, 'maps_{}/'.format(int(100 * 0.25))), format='png')
            par25_khartoum = get_files(os.path.join(khartoum_path, 'maps_{}/'.format(int(100 * 0.25))), format='png')
            par50_vegas = get_files(os.path.join(vegas_path, 'maps_{}/'.format(int(100 * 0.5))), format='png')
            par50_paris = get_files(os.path.join(paris_path, 'maps_{}/'.format(int(100 * 0.5))), format='png')
            par50_shanghai = get_files(os.path.join(shanghai_path, 'maps_{}/'.format(int(100 * 0.5))), format='png')
            par50_khartoum = get_files(os.path.join(khartoum_path, 'maps_{}/'.format(int(100 * 0.5))), format='png')
            par75_vegas = get_files(os.path.join(vegas_path, 'maps_{}/'.format(int(100 * 0.75))), format='png')
            par75_paris = get_files(os.path.join(paris_path, 'maps_{}/'.format(int(100 * 0.75))), format='png')
            par75_shanghai = get_files(os.path.join(shanghai_path, 'maps_{}/'.format(int(100 * 0.75))), format='png')
            par75_khartoum = get_files(os.path.join(khartoum_path, 'maps_{}/'.format(int(100 * 0.75))), format='png')
            for i in range(len(par25_vegas)):
                self.partial_ids.append([par25_vegas[i], par50_vegas[i], par75_vegas[i]])
            for i in range(len(par25_paris)):
                self.partial_ids.append([par25_paris[i], par50_paris[i], par75_paris[i]])
            for i in range(len(par25_shanghai)):
                self.partial_ids.append([par25_shanghai[i], par50_shanghai[i], par75_shanghai[i]])
            for i in range(len(par25_khartoum)):
                self.partial_ids.append([par25_khartoum[i], par50_khartoum[i], par75_khartoum[i]])
            # 0 => 25%
            # 1 => 50%
            # 2 => 75%
            self.random_pars = np.random.randint(0, 3, len(self.partial_ids))

            # save the mix information
            mix_info_file = os.path.join(self.base_path, "mix_info.json")
            assert len(self.partial_ids) == len(self.random_pars)
            if not os.path.exists(mix_info_file):
                print("mix dataset information does not exist, create one...")
                info = {}
                for _, (partial_id_3, idx) in enumerate(zip(self.partial_ids, self.random_pars)):
                    file_name = partial_id_3[idx].split("spacenet\\")[-1].replace("/", "\\")
                    splits = file_name.split("\\")
                    file_name = "{}\\{}".format(splits[0], splits[2])
                    if file_name not in info:
                        partial = 25
                        if idx == 0:
                            partial = 25
                        elif idx == 1:
                            partial = 50
                        elif idx == 2:
                            partial = 75
                        info[file_name] = partial
                    else:
                        print("Duplicate...")
                with open(mix_info_file, "w") as f:
                    json.dump(info, f)
        else:
            if ratio == 0.0:
                ratio = 0.75
            par_vegas = get_files(os.path.join(vegas_path, 'maps_{}/'.format(int(100 * ratio))), format='png')
            par_paris = get_files(os.path.join(paris_path, 'maps_{}/'.format(int(100 * ratio))), format='png')
            par_shanghai = get_files(os.path.join(shanghai_path, 'maps_{}/'.format(int(100 * ratio))), format='png')
            par_khartoum = get_files(os.path.join(khartoum_path, 'maps_{}/'.format(int(100 * ratio))), format='png')
            self.partial_ids = par_vegas
            self.partial_ids.extend(par_paris)
            self.partial_ids.extend(par_shanghai)
            self.partial_ids.extend(par_khartoum)
        assert len(self.sat_ids) == len(self.map_ids) and len(self.sat_ids) == len(self.partial_ids), "lengths of satellite and map images are different"

    def __len__(self):
        return len(self.sat_ids)

    def __getitem__(self, index):
        sat_id = self.sat_ids[index]
        map_id = self.map_ids[index]
        partial_id = self.partial_ids[index]

        # load image
        img_sat = cv2.imread(sat_id, cv2.IMREAD_COLOR)
        img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)

        img_map = cv2.imread(map_id, 0)
        _, img_map = cv2.threshold(img_map, 127, 255, cv2.THRESH_BINARY)

        # create an all-zero partial map when the ratio is 0
        if self.ratio == 0.0:
            img_partial = np.zeros(img_map.shape, dtype=np.float32)
        elif self.ratio == 'mix':
            partial_id = partial_id[self.random_pars[index]]
            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)
        else:
            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)

        if self.augmentation:
            sample = self.augmentation(image=img_sat, mask=img_map, mask_partial=img_partial)
            img_sat, img_map, img_partial = sample['image'], sample['mask'], sample['mask_partial']

        if self.transform:
            img_sat = self.transform(img_sat)
            img_map = self.transform(img_map)
            img_partial = self.transform(img_partial)
        return img_sat, img_partial, img_map, sat_id


# ========================================
# OSM Road Dataset
# - Train: [sat, map]
# - Valid: [sat, map]
# - Test:  [sat, map]
# ========================================
class OSMDataset(Dataset):
    def __init__(self, data_dir, mode="train", file_list="../data/osm/train.txt", ratio=0.75, augmentation=None, transform=None):
        assert ratio in [0.0, 0.25, 0.50, 0.75, 'mix']
        if mode == 'train':
            self.training = True
        else:
            self.training = False
        self.ratio = ratio
        self.augmentation = augmentation
        self.transform = transform
        self.base_path = os.path.join(dirname, data_dir)
        mix_info_file = os.path.join(self.base_path, "mix_info.json")

        self.file_list = file_list
        if not file_list:
            sat_ids = get_files(os.path.join(self.base_path, 'imagery/'), format='png')
            map_ids = get_files(os.path.join(self.base_path, 'masks/'), format='png')
            self.sat_ids = sat_ids
            self.map_ids = map_ids

            # never mind we set ratio to 0.75 here, when ratio is 0, we would create an empty mask later
            if ratio == 'mix':
                self.partial_ids = []
                par25_ids = get_files(os.path.join(self.base_path, 'masks_{}/'.format(int(100 * 0.25))), format='png')
                par50_ids = get_files(os.path.join(self.base_path, 'masks_{}/'.format(int(100 * 0.5))), format='png')
                par75_ids = get_files(os.path.join(self.base_path, 'masks_{}/'.format(int(100 * 0.75))), format='png')
                for i in range(len(par25_ids)):
                    self.partial_ids.append([par25_ids[i], par50_ids[i], par75_ids[i]])
                # 0 => 25%
                # 1 => 50%
                # 2 => 75%
                self.random_pars = np.random.randint(0, 3, len(self.partial_ids))

                # save the mix information
                assert len(self.partial_ids) == len(self.random_pars)
                if not os.path.exists(mix_info_file):
                    print("mix dataset information does not exist, create one...")
                    info = {}
                    for _, (partial_id_3, idx) in enumerate(zip(self.partial_ids, self.random_pars)):
                        file_name = partial_id_3[idx].split("osm\\")[-1].replace("/", "\\")
                        splits = file_name.split("\\")
                        file_name = "{}\\{}".format(splits[0], splits[2])
                        if file_name not in info:
                            partial = 25
                            if idx == 0:
                                partial = 25
                            elif idx == 1:
                                partial = 50
                            elif idx == 2:
                                partial = 75
                            info[file_name] = partial
                        else:
                            print("Duplicate...")
                    with open(mix_info_file, "w") as f:
                        json.dump(info, f)
            else:
                if ratio == 0.0:
                    ratio = 0.75
                par_ids = get_files(os.path.join(self.base_path, 'masks_{}/'.format(int(100 * ratio))), format='png')
                self.partial_ids = par_ids
        else:
            self.image_ids = [line.rstrip("\n") for line in open(file_list)]
            self.sat_ids = []
            self.map_ids = []
            self.partial_ids = []
            if ratio == "mix":
                with open(mix_info_file, "r") as f:
                    mix_infos = json.load(f)
                    for image in self.image_ids:
                        image = image.replace("\\", "/")
                        mask = image.replace("imagery", "masks")
                        self.sat_ids.append(os.path.join(self.base_path, image))
                        self.map_ids.append(os.path.join(self.base_path, mask))

                        image_name = image.replace("imagery/", "")
                        partial = mix_infos[image_name]
                        par = image.replace("imagery", "masks_{}".format(partial))
                        self.partial_ids.append(os.path.join(self.base_path, par))
            else:
                if ratio == 0.0:
                    ratio = 0.75
                for image in self.image_ids:
                    image = image.replace("\\", "/")
                    mask = image.replace("imagery", "masks")
                    par = image.replace("imagery", "masks_{}".format(int(100 * ratio)))
                    self.sat_ids.append(os.path.join(self.base_path, image))
                    self.map_ids.append(os.path.join(self.base_path, mask))
                    self.partial_ids.append(os.path.join(self.base_path, par))

        assert len(self.sat_ids) == len(self.map_ids) and len(self.sat_ids) == len(self.partial_ids), "lengths of satellite and map images are different"

    def __len__(self):
        return len(self.sat_ids)

    def __getitem__(self, index):
        sat_id = self.sat_ids[index]
        map_id = self.map_ids[index]
        partial_id = self.partial_ids[index]

        # load image
        img_sat = cv2.imread(sat_id, cv2.IMREAD_COLOR)
        img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)

        img_map = cv2.imread(map_id, 0)
        _, img_map = cv2.threshold(img_map, 127, 255, cv2.THRESH_BINARY)

        # create an all-zero partial map when the ratio is 0
        if self.ratio == 0.0:
            img_partial = np.zeros(img_map.shape, dtype=np.float32)
        elif self.ratio == 'mix':
            if not self.file_list:
                partial_id = partial_id[self.random_pars[index]]
            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)
        else:
            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)

        if self.augmentation:
            sample = self.augmentation(image=img_sat, mask=img_map, mask_partial=img_partial)
            img_sat, img_map, img_partial = sample['image'], sample['mask'], sample['mask_partial']

        if self.transform:
            img_sat = self.transform(img_sat)
            img_map = self.transform(img_map)
            img_partial = self.transform(img_partial)
        return img_sat, img_partial, img_map, sat_id
