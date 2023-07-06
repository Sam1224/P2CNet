import torch
from matplotlib import pyplot as plt
plt.switch_backend('agg')
DEFAULT_SEED = 1212


def extract_patch_from_tensor(tensor_X, patch_size):
    """
    Function for patch discriminator
    """
    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(tensor_X.size(2) // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(tensor_X.size(3) // patch_size[1])]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            patches = tensor_X[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]]
            patches = torch.where(patches >= 0.5, torch.ones_like(patches), torch.zeros_like(patches))
            list_X.append(patches)
    return list_X