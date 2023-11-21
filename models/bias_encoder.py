# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""


import torch
import torch.nn as nn
from .lightgcn import LightGCN


class SemiGCN(nn.Module):
    def __init__(self, n_users, n_items, norm_adj, emb_size, n_layers, device, nb_classes):
        """

        :param n_users:
        :param n_items:
        :param norm_adj:
        :param emb_size:
        :param n_layers:
        :param device:
        :param nb_classes: 敏感属性的类别数，比如性别有两个类别{男, 女}
        """
        super(SemiGCN, self).__init__()
        self.body = LightGCN(n_users, n_items, norm_adj, emb_size, n_layers, device)
        self.fc = nn.Linear(emb_size, nb_classes)
        self.to(device)

    def forward(self, ):
        e_su, e_si = self.body()
        su = self.fc(e_su)  # 把用户按照购买习惯等等embedding里有的信息分为两类
        si = self.fc(e_si)  # 把item按照bias属性分为两类
        return e_su, e_si, su, si  # su代表，预测出来的结果，用户们属于男或者女的概率，size: 6040*2

