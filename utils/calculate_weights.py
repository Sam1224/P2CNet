import os
from tqdm import tqdm
import numpy as np


def calculate_weights_labels(path, dataloader, num_classes=2):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    print('Calculating classes weights')
    tqdm_batch = tqdm(dataloader)
    for sample in tqdm_batch:
        y = sample[2]
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = path
    np.save(classes_weights_path, ret)

    return ret
