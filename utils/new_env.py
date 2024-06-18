#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
import os
import cv2
import torch

from calculate_IoU import SAM
matplotlib.use('Agg')
import calculate_IoU
from utils.filters import *
import random

device = torch.device("cuda:0")
def split_list(index_list, fraction=0.2):
    # 计算要分配到第一部分的元素数量

    # 从index_list中随机抽取part1_count个元素作为第一部分
    part1 = random.sample(index_list, int(len(index_list) * fraction))

    # 剩余的元素构成第二部分
    part2 = [item for item in index_list if item not in part1]

    return part1, part2


class fed:
    def __init__(self,pic,train=1):
        self.filter=1
        self.train=train
        self.action_space=9
        self.action_space_low=-1
        self.action_space_high = 1
        self.filter_size=5
        self.state=None
        self.Image_list = []
        self.GT_list = []
        self.Nd_list=[]
        self.observation_space=pic.shape
    def step(self,action,observation,GT):
        done=None
        if self.filter:
            observation_=mix_filters(observation,action=action,filter_size=self.filter_size)
        reward=SAM(observation_,GT)
        # if reward<0.6:
        #     done=True
        return observation_, reward, done
