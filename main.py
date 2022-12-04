import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse

from matplotlib import pyplot as plt
from torchvision.io import read_image

from anon_models.anon_resnet import AnonResNet18
from anon_models.anon_vgg import VGG16, AnonVGG16, AnonVGG, OriginalVGG
from aug_utils.param_count_table import count_total_params, show_parameters

from aug_utils.print_nonzero import print_nonzeros
from models import *
from utils import progress_bar

from ptflops import get_model_complexity_info
from datetime import datetime


class AugDataset2(torch.utils.data.Dataset):
    def __init__(self, aug_dataset, original_dataset):
        # self.x = torch.from_numpy(np.array([data.numpy().astype(np.float16) for data in aug_dataset[:, 0]]))
        self.x = aug_dataset
        self.x = self.x.float()

        print("set:", self.x.shape)

        # self.y = torch.from_numpy(original_dataset[:, 1].astype('int'))
        self.y = torch.IntTensor([data[1] for data in original_dataset])
        self.y = self.y.long()

        self.n_samples = aug_dataset.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]