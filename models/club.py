# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""

import torch
import torch.nn as nn


class CLUBSample(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, device='cpu'):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())
        self.to(device)

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        """ 采样数据符合均值为mu，方差为sigma^2的正态分布"""
        mu, logvar = self.get_mu_logvar(x_samples)  # 计算并返回mu与sigma^2
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)  # 为机器学习函数q(X|Y)
        negative = (-(mu - y_samples[random_index]) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        bound = (positive - negative).mean()
        return torch.clamp(bound / 2., min=0.0)  # 将参数值限制在bound / 2.到0.0之间

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        llh = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1).mean()  # log likelihood，即对数似然
        return llh

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)
    