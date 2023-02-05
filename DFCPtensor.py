# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:03:03 2021

@author: Jiping
"""

import numpy as np
import torch
import tensorly as tl
import pandas as pd
from tensorly.decomposition import parafac
import copy
from collections import Counter
from torch.autograd import Variable



class DFCP(object):
    def __init__(self, column, rank, device, n_iter, lr, penalty, lambda_1,lambda_2 ,lambda_3,lambda_4, df_path):
        """"定义参数和优化器"""
        self.rank = rank
        self.device = device
        self.n_iter = n_iter
        self.lr = lr
        self.penalty = penalty
        self.df_path = df_path
        self.column = column
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        print('rank:', self.rank, 'penalty:', self.penalty, 'lambda_1:', self.lambda_1,
              'lambda_2:', self.lambda_2, 'lambda_3:', self.lambda_3, 'lambda_4:', self.lambda_4)

    """归一化"""
    def maxminnorm_np(self, array):
        max = np.max(array, axis=0)
        min = np.min(array, axis=0)
        t = (array - min) / (max - min)
        return t

    def maxminnorm_torch(self, array):
        max = torch.max(array)
        min = torch.min(array)
        t = (array - min) / (max - min)
        return t

       """提取link信息"""
    def extract_info(self):
        """"读取交通数据"""
        df = pd.read_csv(self.df_path)

        """构建link信息特征（1537，）"""
        info = df['speed_class'].loc[(df['day_id'] == 1) & (df['time_id'] == 1)]
        info = np.array(info)
        info = np.expand_dims(info, 1)
        info = np.tile(info, (1, self.rank))
        return info

    def val(self):
        """"读取交通数据"""
        df = pd.read_csv(self.df_path)
        # df.fillna(0, inplace=True)

        """构建四维数组"""
        obs = []

        for i in range(7):
            obs.append([])
            for j in range(96):
                temp = df['aveSpeed'].loc[(df['day_id'] == i) & (df['time_id'] == j )]
                arr = np.array(temp)
                obs[i].append(arr)

        """obs_or部分值设为空值用于训练，list_obs为人工设置的空值位置，obs_data为原始数组"""
        obs = np.array(obs)
        obs_or = copy.deepcopy(obs)
        obs_data = copy.deepcopy(obs)

        """计算非空值的数量为num_val"""
        loc_nan = np.isnan(obs)
        obs[loc_nan] = -1
        loc_obs = np.where(obs != -1)
        num_val = len(loc_obs[0])
        obs[loc_nan] = np.nan

        """将有值的固定比例调整为空值，并在list_obs记录这些位置的信息"""
        rate = 0.8
        rand_obs = np.random.randint(0, num_val, size=int(rate * num_val))
        list_obs = []
        for i in range(len(loc_obs)):
            list_obs.append(loc_obs[i][rand_obs])
        list_obs = tuple(list_obs)

        """将指定观测值位置变为空值"""
        obs_or[list_obs] = np.nan
#         obs_data.to_csv('sparse_v.csv')

        return obs_data, obs_or, list_obs

    def construction(self, obs_or):
        """获取不同时间、空间、天数特征的速度分布"""
        obs = np.array(obs_or)
        print(np.shape(obs))
        loc_nan = np.isnan(obs)
        obs[loc_nan] = -1
        loc_obs = np.where(obs != -1)
        obs[loc_nan] = 0

        m, n, p = np.shape(obs)
        """week特征"""
        W = []
        for i in range(m):
            counter = 0
            for j in range(n):
                counter += Counter(loc_nan[i, j, :])[False]
            num_sum = np.sum(obs[i, :, :])
            W.append(round(num_sum / counter, 2))
        speed_avg = np.mean(W)
        W = np.expand_dims(W, 1)
        W = np.tile(W, (1, self.rank))

        W = torch.from_numpy(np.array(W))

        """Time特征"""
        T = []
        for i in range(n):
            counter = 0
            for j in range(m):
                counter += Counter(loc_nan[j, i, :])[False]
            num_sum = np.sum(obs[:, i, :])
            T.append(round(num_sum / counter, 2))
        T = np.expand_dims(T, 1)
        T = np.tile(T, (1, self.rank))
        T = torch.from_numpy(np.array(T))
        
        """data type特征"""
        T = []
        for i in range(n):
            counter = 0
            for j in range(m):
                counter += Counter(loc_nan[j, i, :])[False]
            num_sum = np.sum(obs[:, i, :])
            T.append(round(num_sum / counter, 2))
        T = np.expand_dims(D, 1)
        T = np.tile(D, (1, self.rank))
        T = torch.from_numpy(np.array(D))
       
        """link特征"""
        L = []
        for i in range(p):
            counter = 0
            for j in range(m):
                counter += Counter(loc_nan[j, :, i])[False]
            num_sum = np.sum(obs[:, :, i])
            """无流量的网格用所有网格的均值填补"""
            if counter != 0:
                temp = num_sum / counter
                L.append(round(temp, 2))
            else:
                L.append(speed_avg)
        L = np.expand_dims(L, 1)
        L = np.tile(L, (1, self.rank))
        L = torch.from_numpy(np.array(L))

        tl.set_backend('pytorch')
        obs[loc_nan] = speed_avg
    
        "构建张量以及cp分解，这里假设rank = 1"
        tensor = tl.tensor(obs, device=self.device)
        factors_list = parafac(tensor, rank=self.rank)
   
        """构建分解后的张量列表"""
        factors = []
        for i in factors_list[1]:
#             n = factors_list[i].shape[1]
            factors.append(tl.tensor(i,
                                     device=self.device,
                                     requires_grad=True))

        "定义优化器"
        optimizer = torch.optim.Adam(factors, lr=self.lr)

        "提前计算Y和r"
        Y = copy.deepcopy(obs)
        Y = tl.tensor(Y, device=self.device, requires_grad=True)
        return factors, Y, W, T, L, loc_obs, optimizer

    def run_epoch(self, factors, Y,  W, T, L, info, loc_obs, obs_data, list_obs, optimizer):
        obs_data = tl.tensor(obs_data, device=self.device)
        info = tl.tensor(info, device=self.device)
        tl.set_backend('pytorch')
        for i in range(self.n_iter):
            optimizer.zero_grad()
            """rec：恢复后的张量"""
            rec = tl.kruskal_to_tensor((tl.tensor([1]*self.rank),factors))

            """计算损失函数"""
            loss_f = tl.norm(Y[loc_obs] - rec[loc_obs], 2)

            factor_origin = [W, T, L]
            factors_weight = [self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4]

            """建立损失函数"""
            for f in range(len(factors)):
                factor_origin[f] = factor_origin[f].type(torch.FloatTensor)
                if f == 2:
                    loss_f = loss_f + self.penalty * factors[f].pow(2).sum()
                    + factors_weight[f] * tl.norm(self.maxminnorm_torch(factor_origin[f]) - self.maxminnorm_torch(factors[f]), 2)
                    + factors_weight[-1] * tl.norm(self.maxminnorm_torch(info) - self.maxminnorm_torch(factors[f]), 2)
                else:
                    loss_f = loss_f + self.penalty * factors[f].pow(2).sum() + factors_weight[f] * \
                             tl.norm(self.maxminnorm_torch(factor_origin[f]) - self.maxminnorm_torch(factors[f]), 2)

            """更新迭代"""
            loss_f.backward()
            optimizer.step()

            if i % 100 == 0:
                rec_error = tl.norm(Y[loc_obs].data - rec[loc_obs].data, 2) / tl.norm(Y[loc_obs].data, 2)
                val_error = tl.norm(obs_data[list_obs].data - rec[list_obs].data, 2)/tl.norm(obs_data[list_obs].data, 2)
                print("Epoch {}, Rec. error: {}, Val. error: {}".format(i, rec_error, val_error))
        rec_error = tl.norm(Y[loc_obs].data - rec[loc_obs].data, 2) / tl.norm(Y[loc_obs].data, 2)
        rec[loc_obs] = obs_data[loc_obs]
        rec[list_obs] = obs_data[list_obs]
        return rec, rec_error

    def write_file(self, rec):
        rec = rec.detach()
        rec = np.around(np.array(rec), decimals=1)
        num = np.shape(rec)[2]
        num_day = np.shape(rec)[0]
        temp = []
        for i in range(len(rec)):
            a = np.transpose(rec[i, :, :])
            if i == 0:
                temp = a
            else:
                temp = np.vstack((a, temp))

        s = []
        for i in range(num_day):
            t = np.transpose(temp[num * i:num * i + num])
            if i == 0:
                s = t
            else:
                s = np.vstack((s, t))

        """写入文件"""
        pd_data = pd.DataFrame(s)
        return pd_data


if __name__ == '__main__':
#     {'lambda_1': 36.80231717425032, 'lambda_2': 9.51482038281362, 'lambda_3': 56.052746769948044, 'lambda_4': 38.636146341490225, 'penalty': 0.11721488665013455, 'rank': 5}
    device = 'cpu'
    n_iter = 2000
    lr = 0.0005
    penalty = 0.11721488665013455
    lambda_1 = 36.80231717425032
    lambda_2 = 9.51482038281362
    lambda_3 = 56.052746769948044
    lambda_4 = 38.636146341490225
    rank =5
    df_path = 'Feature_all_null.csv'
    columns_list = ['aveSpeed']
    output_path = ['completed_v.csv']
    for i in range(1):
        column = columns_list[i]
        results = DFCP(column, rank, device, n_iter, lr, penalty, lambda_1, lambda_2, lambda_3, lambda_4, df_path)
        obs_data, obs_or, list_obs = results.val()
        info = results.extract_info()
        factors, Y, W, T, L, pos_obs, optimizer = results.construction(obs_or)
        rec = results.run_epoch(factors, Y,  W, T, L, info, pos_obs, obs_data, list_obs, optimizer)[0]
        data = results.write_file(rec)
        data.to_csv(output_path[i])
        print('已完成补全：' + column)
