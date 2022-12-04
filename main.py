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