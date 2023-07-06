import torch
from torchvision import transforms
from base import BaseDataLoader
from data_loader.datasets import *
import albumentations as albu


# ========================================
# SpaceNet Road DataLoader
# ========================================
class SpaceNetDataLoader(BaseDataLoader):
    def __init__(self, data_dir, mode="train", batch_size=1, size=1024, ratio=0.75, shuffle=True, validation_split=0.0, num_workers=0):
        torch.manual_seed(123)
        np.random.seed(123)
        if mode == "train":
            train_augmentation = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                albu.PadIfNeeded(min_height=size, min_width=size, always_apply=True, border_mode=0),
                albu.RandomCrop(height=size, width=size, always_apply=True),
                albu.IAAAdditiveGaussianNoise(p=0.2),
                albu.IAAPerspective(p=0.5),
                albu.OneOf(
                    [
                        albu.CLAHE(p=1),
                        albu.RandomBrightness(p=1),
                        albu.RandomGamma(p=1),
                    ],
                    p=0.9,
                ),
                albu.OneOf(
                    [
                        albu.IAASharpen(p=1),
                        albu.Blur(blur_limit=3, p=1),
                        albu.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.9,
                ),
                albu.OneOf(
                    [
                        albu.RandomContrast(p=1),
                        albu.HueSaturationValue(p=1),
                    ],
                    p=0.9,
                )
            ]
            augmentation = albu.Compose(train_augmentation, additional_targets={
                'mask_partial': 'mask',
                'mask_edge': 'mask'
            })
        else:
            # when do the validation and test, we only use the center 1024x1024 patch
            test_augmentation = [
                albu.PadIfNeeded(size, size),
                albu.CenterCrop(height=size, width=size, always_apply=True)
            ]
            augmentation = albu.Compose(test_augmentation, additional_targets={
                'mask_partial': 'mask',
                'mask_edge': 'mask'
            })
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        dataset = SpaceNetDataset(data_dir, mode, ratio=ratio, augmentation=augmentation, transform=transform)
        total_num = len(dataset)
        val_ratio = 0.1
        test_ratio = 0.1
        val_num = int(total_num * val_ratio)
        test_num = int(total_num * test_ratio)
        train_num = total_num - val_num - test_num
        train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num])
        if mode == 'train':
            self.dataset = train_set
        elif mode == 'valid':
            self.dataset = valid_set
        elif mode == 'test':
            self.dataset = test_set
        else:
            self.dataset = train_set
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, mode=mode)


# ========================================
# OSM DataLoader
# ========================================
class OSMDataLoader(BaseDataLoader):
    def __init__(self, data_dir, mode="train", file_list="../data/osm/train.txt", batch_size=1, size=1024, ratio=0.75, shuffle=True, validation_split=0.0, num_workers=0):
        torch.manual_seed(123)
        np.random.seed(123)
        if mode == "train":
            train_augmentation = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                albu.PadIfNeeded(min_height=size, min_width=size, always_apply=True, border_mode=0),
                albu.RandomCrop(height=size, width=size, always_apply=True),
                albu.IAAAdditiveGaussianNoise(p=0.2),
                albu.IAAPerspective(p=0.5),
                albu.OneOf(
                    [
                        albu.CLAHE(p=1),
                        albu.RandomBrightness(p=1),
                        albu.RandomGamma(p=1),
                    ],
                    p=0.9,
                ),
                albu.OneOf(
                    [
                        albu.IAASharpen(p=1),
                        albu.Blur(blur_limit=3, p=1),
                        albu.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.9,
                ),
                albu.OneOf(
                    [
                        albu.RandomContrast(p=1),
                        albu.HueSaturationValue(p=1),
                    ],
                    p=0.9,
                )
            ]
            augmentation = albu.Compose(train_augmentation, additional_targets={
                'mask_partial': 'mask',
                'mask_edge': 'mask'
            })
        else:
            # when do the validation and test, we only use the center 1024x1024 patch
            test_augmentation = [
                albu.PadIfNeeded(size, size),
                albu.CenterCrop(height=size, width=size, always_apply=True)
            ]
            augmentation = albu.Compose(test_augmentation, additional_targets={
                'mask_partial': 'mask',
                'mask_edge': 'mask'
            })
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        dataset = OSMDataset(data_dir, mode, file_list=file_list, ratio=ratio, augmentation=augmentation, transform=transform)
        if not file_list:
            total_num = len(dataset)
            val_ratio = 0.1
            test_ratio = 0.1
            val_num = int(total_num * val_ratio)
            test_num = int(total_num * test_ratio)
            train_num = total_num - val_num - test_num
            train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num])
            if mode == 'train':
                self.dataset = train_set
            elif mode == 'valid':
                self.dataset = valid_set
            elif mode == 'test':
                self.dataset = test_set
            else:
                self.dataset = train_set
        else:
            self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, mode=mode)
