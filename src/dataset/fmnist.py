import torch
import random
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class FMNIST(Dataset):
    def __init__(self, cfg, phase, img):
        self.phase = phase
        self.img = img
        self.cfg = cfg
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(self.cfg.DATA.HorizontalFlip),
                                              transforms.RandomResizedCrop(self.cfg.DATA.ResizedCrop,(self.cfg.DATA.Cropheight, self.cfg.DATA.Cropwidth))])

    def __len__(self):
        return len(self.img)

    def __getitem__(self,idx):
        
        x, label = self.img[idx]

        x1 = self.augment(x)
        x2 = self.augment(x)
        
        return x1, x2

    def augment(self, frame):
        if self.phase == 'train':
            frame = self.transforms(frame)
            return frame
        else:
            return frame
        
class Downstream_FMNIST(Dataset):
    def __init__(self, cfg, phase, img, num_classes):

        self.cfg = cfg
        self.phase = phase
        self.num_classes = num_classes
        self.img = img

        self.randomcrop = transforms.RandomResizedCrop(self.cfg.DATA.ResizedCrop,(self.cfg.DATA.Cropheight, self.cfg.DATA.Cropwidth))

    def __len__(self):
        return len(self.img)

    def __getitem__(self,idx):

        x, label = self.img[idx]

        img = x.float()

        if self.phase == 'train':
            img  = self.randomcrop(img)

        return img, label

