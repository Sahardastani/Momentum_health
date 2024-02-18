import os 
import numpy as np
import torch
import torch.nn as nn

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def save_model(cfg, model, optimizer, scheduler, current_epoch, name):
    out = os.path.join(f'{cfg.DATA.current_dir}/save_models/',name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)

def set_seed(seed = 16):
    np.random.seed(16)
    torch.manual_seed(16)