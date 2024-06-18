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
    def __init__(self,train=1):
        self.filter=0
        self.train=train
        self.action_space=9
        self.action_space_low=-1
        self.action_space_high = 1
        self.filter_size=5
        self.state=None
        self.Image_list = []
        self.GT_list = []
        self.Nd_list=[]
        path = 'FSSD-12'


        for i in os.listdir(path):
            path1 = os.path.join(path, i)
            for j in os.listdir(path1):
                if j == 'GT':
                    path2 = os.path.join(path1, j)
                    list_temp = [0 for i in range(50)]
                    for k in os.listdir(path2):
                        n=0
                        for x in k:
                            if str.isdigit(x):
                                n=n*10+int(x)
                        path3 = os.path.join(path2, k)
                        list_temp[n-1]=path3

                    self.GT_list=self.GT_list+list_temp

        for i in os.listdir(path):
            path1 = os.path.join(path, i)
            for j in os.listdir(path1):
                if j == 'Images':
                    path2 = os.path.join(path1, j)
                    list_temp = [0 for i in range(50)]
                    for k in os.listdir(path2):
                        n = 0
                        for x in k:
                            if str.isdigit(x):
                                n = n * 10 + int(x)
                        path3 = os.path.join(path2, k)
                        list_temp[n - 1] = path3
                    self.Image_list = self.Image_list + list_temp


        for i in os.listdir(path):
            path1 = os.path.join(path, i)
            for j in os.listdir(path1):
                if j == 'Nd':
                    path2 = os.path.join(path1, j)
                    list_temp = [0 for i in range(50)]
                    for k in os.listdir(path2):
                        n=0
                        for x in k:
                            if str.isdigit(x):
                                n=n*10+int(x)
                        path3 = os.path.join(path2, k)
                        list_temp[n-1]=path3
                    self.Nd_list=self.Nd_list+list_temp

        # for i in os.listdir(path):
        #     path1 = os.path.join(path, i)
        #     for j in os.listdir(path1):
        #         if j == 'GT':
        #             path2 = os.path.join(path1, j)
        #             files=os.listdir(path2)
        #             num_png=len(files)
        #             list_temp = [0 for i in range(num_png)]
        #             for k in os.listdir(path2):
        #                 n=0
        #                 for x in k:
        #                     if str.isdigit(x):
        #                         n=n*10+int(x)
        #                 path3 = os.path.join(path2, k)
        #                 list_temp[n-1]=path3
        #
        #             self.GT_list=self.GT_list+list_temp
        #
        # for i in os.listdir(path):
        #     path1 = os.path.join(path, i)
        #     for j in os.listdir(path1):
        #         if j == 'Images':
        #             path2 = os.path.join(path1, j)
        #             files = os.listdir(path2)
        #             num_png = len(files)
        #             print(num_png)
        #             list_temp = [0 for i in range(num_png)]
        #             for k in os.listdir(path2):
        #                 n = 0
        #                 for x in k:
        #                     if str.isdigit(x):
        #                         n = n * 10 + int(x)
        #                 path3 = os.path.join(path2, k)
        #                 list_temp[n - 1] = path3
        #             self.Image_list = self.Image_list + list_temp


        # for i in os.listdir(path):
        #     path1 = os.path.join(path, i)
        #     for j in os.listdir(path1):
        #         if j == 'Nd':
        #             path2 = os.path.join(path1, j)
        #             files = os.listdir(path2)
        #             num_png = len(files)
        #
        #             list_temp = [0 for i in range(num_png)]
        #             for k in os.listdir(path2):
        #                 n=0
        #                 for x in k:
        #                     if str.isdigit(x):
        #                         n=n*10+int(x)
        #                 path3 = os.path.join(path2, k)
        #                 list_temp[n-1]=path3
        #             self.Nd_list=self.Nd_list+list_temp


        self.index_list=[i for i in range(600)]
        self.test_index,self.train_index=split_list(self.index_list,fraction=0.2)
        i = random.choice(self.train_index)
        observation = cv2.imread(self.Image_list[i], cv2.IMREAD_GRAYSCALE)
        self.state = observation
        self.GT = self.GT_list[i]

        self.observation_space = observation.shape
    def reset(self):
        if self.train:
            i=random.choice(self.train_index)
        else:
            i = random.choice(self.test_index)

        observation=cv2.imread(self.Image_list[i],cv2.IMREAD_GRAYSCALE)

        self.state=observation

        self.GT=self.GT_list[i]
        reward = SAM(self.state, GT=cv2.imread(self.GT,cv2.IMREAD_GRAYSCALE))
        return observation,reward
    def step(self,action):
        done=None
        if self.filter:
            observation_=mix_filters(self.state,action=action,filter_size=self.filter_size)
            self.state=observation_
        else:
            observation_=self.state
        print(self.GT)
        GT=cv2.imread(self.GT,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite("Enhancement.jpg",observation_)
        reward=SAM(observation_,GT)
        # if reward<0.6:
        #     done=True
        return observation_, reward, done
