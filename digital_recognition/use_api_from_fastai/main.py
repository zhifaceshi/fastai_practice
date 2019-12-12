"""
@Time    :2019/12/10 18:04
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
from fastai import *
from fastai.vision import *
from fastai.vision import ImageList, Image, CategoryList
from pandas import DataFrame
from tqdm import tqdm
import random
from pprint import pprint
import os
import collections
from typing import List, Dict, Tuple
from fastai.vision.gan import basic_critic, AvgFlatten
import logging
import torch
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MyImageList(ImageList):
    @classmethod
    def from_df(cls, df:DataFrame, path, cols=0, folder=None, suffix:str='', **kwargs)->'ItemList':
        "Get the filenames in `cols` of `df` with `folder` in front of them, `suffix` at the end."

        values = df.iloc[:, 1: ].values
        labels = df.iloc[:, 0].values
        values = torch.tensor(values).float()
        items = []
        for y, v in zip(labels, values):
            t = Image(v.view(28, 28).unsqueeze(0))
            t.y = y
            items.append(t)
        # res = cls(items, label_cls=CategoryList)
        res = cls(items)
        return res

    @classmethod
    def from_test_data(cls, pth):
        # 从测试集合上读取
        df = pd.read_csv(pth)
        values = df.values
        indexs = df.index
        values = torch.tensor(values).float()
        items = []
        for i, v in zip(indexs, values):
            t = Image(v.view(28, 28).unsqueeze(0))
            items.append(t)
            items.index = i
        # res = cls(items, label_cls=CategoryList)
        res = cls(items)
        return res

    def get(self, i):
        res = self.items[i]
        return res

pth = r'/home/liangjiaxi/TMP_PROJECT/fastaipractice/data/数字图像识别'
data = MyImageList.from_csv(pth, 'fake.csv', cols= list(range(1, 785)))
data = data.split_by_rand_pct(0.2, 0)
data = data.label_from_func(lambda x: x.y)
data = data.databunch(device = 'cuda')

assert isinstance(data, DataBunch)

class MyCNN(torch.nn.Module):
    def __init__(self, in_size:int, n_channels:int, n_features:int=64, n_extra_layers:int=0, **conv_kwargs):
        super(MyCNN, self).__init__()
        "A basic critic for images `n_channels` x `in_size` x `in_size`."
        layers = [
            conv_layer(n_channels, n_features, 3, 2, 1, leaky=0.2, norm_type=None, **conv_kwargs)]  # norm_type=None?
        cur_size, cur_ftrs = in_size // 2, n_features
        layers.append(nn.Sequential(
            *[conv_layer(cur_ftrs, cur_ftrs, 3, 1, leaky=0.2, **conv_kwargs) for _ in range(n_extra_layers)]))
        while cur_size > 4:
            layers.append(conv_layer(cur_ftrs, cur_ftrs * 2, 4, 2, 1, leaky=0.2, **conv_kwargs))
            cur_ftrs *= 2
            cur_size //= 2
        # layers += [conv2d(cur_ftrs, 1, 3, padding=0)]

        layers += [ResizeBatch( -1), torch.nn.Linear(256*3*3, 10) ]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, 10)
        return x

model = MyCNN(28, 1)
learn = Learner(data, model, metrics = accuracy)

learn.fit_one_cycle(1)

datalist = MyImageList.from_test_data('/home/liangjiaxi/TMP_PROJECT/fastaipractice/data/数字图像识别/test.csv')\
    .split_none()\
    .label_from_func(noop)\
    .databunch()

learn.predict(datalist)