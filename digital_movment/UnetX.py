"""

@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import random
from pprint import pprint
import os
from itertools import chain
import collections
from typing import List, Dict, Tuple
import logging
from collections import Iterable
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from fastai import *
from fastai.vision import *


class ConsImage(ItemBase):
    def __init__(self, imgs_x: List[Image], imgs_y: List[Image]):
        assert isinstance(imgs_x[0], Image)
        self.imgs_x = imgs_x
        self.imgs_y = imgs_y

    def apply_tfms(self, tfms:Collection, **kwargs):
        for img in chain(self.imgs_x, self.imgs_y):
            img.apply_tfms(tfms, **kwargs)

    def show(self, ax:plt.Axes, **kwargs):
        raise NotImplemented
    def __str__(self):
        return f"{len(self.imgs_x)} {len(self.imgs_y)}"

def generate(data: Tensor, winsize):
    assert winsize <= data.shape[0]
    for d in data.permute(1,0,2,3):
        assert len(d.shape) == 3
        for i in range(d.shape[0] - winsize + 1):
            yield d[i: i + winsize - 1, :, :,], d[i+1: i + winsize, :, :] # 错位

class TMPImageList(ItemList):
    def reconstruct(self, t:Tensor, x:Tensor=None):
        return Image(torch.cat([w for w in t], -1).unsqueeze(0))

class MyImageList(ItemList):
    _bunch = DataBunch
    _label_cls = TMPImageList

    @classmethod
    def from_npy(cls, file: Path, winsize = 4):
        # create ItemList, which contains objects we want
        data = np.load(file)
        if not isinstance(data, Tensor):
            data = torch.from_numpy(data)
        assert len(data.shape) == 4 # some tensor like: 20, 50, 64, 64
        items = []
        for x, y in tqdm(generate(data, winsize), desc="from npy"):
            x = [Image(w.unsqueeze(0)) for w in x]
            y = [Image(w.unsqueeze(0)) for w in y]
            items.append(ConsImage(x, y))
        res = cls(items)
        return res

    def reconstruct(self, t:Tensor, x:Tensor=None):
        return Image(torch.cat([w for w in t], -1).unsqueeze(0))

    # def get(self, i)->Any:
    #     "Subclass if you want to customize how to create item `i` from `self.items`."
    #     return self.items[i]
    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
        rows = len(xs)

        axs = subplots(rows, 2, imgsize=imgsize, figsize=figsize)
        for i in range(0, len(axs.flatten()),  2):
            ax = axs.flatten()[i]
            index = i // 2
            x = xs[index]
            y = ys[index]
            x.show(ax = ax, **kwargs)
            y.show(ax = axs.flatten()[i+1], **kwargs)
#         for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        if self._square_show_res:
            title = 'Ground truth\nPredictions'
            rows = int(np.ceil(math.sqrt(len(xs))))
            axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=12)
            for x,y,z,ax in zip(xs,ys,zs,axs.flatten()): x.show(ax=ax, title=f'{str(y)}\n{str(z)}', **kwargs)
            for ax in axs.flatten()[len(xs):]: ax.axis('off')
        else:
            title = 'Ground truth/Predictions'
            axs = subplots(len(xs), 2, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
            for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
                x.show(ax=axs[i,0], y=y, **kwargs)
                x.show(ax=axs[i,1], y=z, **kwargs)

def label_func(input):
    # print(type(x))
    # print(x)
    return input.imgs_y

def collate(samples):
    x, y = zip(*samples)
    tensor_x = torch.stack([torch.cat([w.data for w in c.imgs_x]) for c in x])
    tensor_y = torch.stack([torch.cat([w.data for w in c.imgs_y]) for c in x])
    assert len(tensor_x.shape) == len(tensor_y.shape) == 4
    return tensor_x, tensor_y

BATCHSIZE = 32
WINSIZE = 4

data_pth = Path('./data_samples.npy')
data = MyImageList.from_npy(data_pth, winsize=WINSIZE) # construct MyImageList which contains itembase
data = data.split_by_rand_pct(0.15) # ItemLists which contains trains and valids
data = data.label_from_func(label_func)
data = data.databunch(bs = BATCHSIZE, collate_fn = collate, num_workers = 0) # to debug, we set

data.show_batch(title = "Preview / Next")



from convlstm import *
def get_model():
    encoder = UnetConvLSTM(input_size=(64, 64),
                 input_dim=1,
                 hidden_dim=[64, 64, 1],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

    model = Encorder(encoder)
    return model

def loss_func(x, y):
    assert x.shape == y.shape
    x = x.view(-1).float()
    y = y.view(-1).float()
    assert len(x) == len(y), f'{len(x)} {len(y)}'
    return F.l1_loss(x, y)

model = get_model()
learner = Learner(data, model, loss_func = loss_func)

if __name__ == '__main__':
    learner.fit(1)
    learner.show_results()