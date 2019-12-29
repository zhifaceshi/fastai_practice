"""
@Time    :2019/12/14 20:06
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
from tqdm import tqdm
import random
from pprint import pprint
import os
import collections
from typing import List, Dict, Tuple
import logging
from fastai import *
from fastai.vision import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

defaults.device = torch.device('cpu')
path = '../data/proteins_imgs/U_RISC_OPEN_DATA_SIMPLE'
path = Path(path)
# print(path.ls())

fnames = get_image_files(path / 'train')
# print(fnames)
src = ImageImageList.from_folder(path, exclude=['labels', 'val'], )
src = src.split_none()
src = src.label_from_func(lambda x: path / 'labels'/ 'train' / f'{x.stem}.tiff' if int(str(x.stem).split('_')[0])>31 else path / 'labels'/ 'train' / f'{x.stem}_m.tiff')
data = src.databunch(bs = 2)
data = data.normalize(imagenet_stats)

import torch
class Mymodel(torch.nn.Module):
    def __init__(self, base):
        super(Mymodel, self).__init__()

        del base.avgpool
        del base.fc
        self.base = base
        self.decoder = torch.nn.ModuleList(
            [conv2d_trans(512, 1024), torch.nn.Sigmoid()]
        )
    def forward(self, x):
        x = self.base(x)
        print(x.shape)
        for layer in self.decoder:
            x = layer(x)
            print(x.shape)
        return x

learn = cnn_learner(data, Mymodel(create_body(models.resnet18, True, None)), loss_func = F.l1_loss)
learn.lr_find()