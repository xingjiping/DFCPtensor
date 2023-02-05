from DFCP_2 import DFCP
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

"""定义超参数"""
space4cp = {
    'penalty': hp.uniform('penalty', 0, 1),
    'lambda_1': hp.uniform('lambda_1', 0, 100),
    'lambda_2': hp.uniform('lambda_2', 0, 100),
    'lambda_3': hp.uniform('lambda_3', 0, 100),
    'lambda_4': hp.uniform('lambda_4', 0, 100),
    'rank': hp.choice('rank', [1, 2, 3, 4, 5, 6])
}

"""调用函数，以迭代500次之后的损失值为目标，此处以link_speed为补全对象，如有变化可直接改变i的实际值"""
def f(params):
    device = 'cpu'
    n_iter = 1000
    lr = 0.00005
    penalty = params['penalty']
    lambda_1 = params['lambda_1']
    lambda_2 = params['lambda_2']
    lambda_3 = params['lambda_3']
    lambda_4 = params['lambda_4']
    rank = params['rank']
    df_path = 'Feature_all_null.csv'
    columns_list = ['aveSpeed']
    i = 0
    column = columns_list[i]
    results = DFCP(column, rank, device, n_iter, lr, penalty, lambda_1, lambda_2, lambda_3, lambda_4, df_path)
    obs_data, obs_or, list_obs = results.val()
    info = results.extract_info()
    factors, Y, W, T, L, pos_obs, optimizer = results.construction(obs_or)
    loss = results.run_epoch(factors, Y, W, T, L, info, pos_obs, obs_data, list_obs, optimizer)[1]

    return {'loss': loss, 'status': STATUS_OK}


trials = Trials()

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import lhsmdu
import random


    
def objective_func (x):
    return np.sin(x)

# stochastic griewank function with 30 eval per point
def stochastic_griewank_30(x, dim=1):
    partA = 0
    partB = 1
    for i in range(dim):
        if not isinstance(x, (list, np.ndarray)):
            partA += x**2
            partB *= np.cos(x / np.sqrt(i + 1))
        else:    
            partA += x[i]**2
            partB *= np.cos(x[i] / np.sqrt(i + 1))
    func_eval_ = 1 + partA / 4000 - partB
    func_eval = [func_eval_ + np.random.normal(0, 0.1) for num in range(30)]
    return sum(func_eval) / len(func_eval)
    
##高斯核函数
def G_kernel(x1, x2, l):
    len1 = len(x1)
    len2 = len(x2)
    K = np.zeros([len1, len2])
    for i in range(len(x1)):
        for j in range(len(x2)):
            dis = np.linalg.norm(np.array(x1[i]) - np.array(x2[j]))
            K[i][j] = dis
    return np.exp(-0.5 * K ** 2 / l ** 2)

def estimation(x_t, x, y, l):
    x_t = [x_t]
    len1 = len(x)
    K_yy = G_kernel(x, x, l) + np.eye(len(x)) * 10**(-3)
    K_yy_ = G_kernel(x, x_t, l)
    K_yy_inv = np.linalg.inv(K_yy)

    mean1 = np.dot(K_yy_.reshape(1, len1), K_yy_inv)
    mean = np.dot(mean1.reshape(1, len1), y.reshape(len1, 1))

    variance1 = np.dot(K_yy_.reshape(1, len1), K_yy_inv)
    variance = 1 - np.dot(variance1.reshape(1, len1), K_yy_.reshape(len1, 1))
    factor = np.exp(-len(x)/300)

    return (mean - factor * variance)[0][0]#3.6版本，3.7版本去掉[0]

def find_l(x, y):

    len1 = len(x)
    min_v = 10 ** 9
    opt_l = 10

    for l in range(1, 400):
        l = l * 0.01
        K_yy = G_kernel(x, x, l)
        K_yy_inv = np.linalg.inv(K_yy)
        a1 = np.dot(y.reshape(1, len1), K_yy_inv)
        a = np.dot(a1.reshape(1, len1), y.reshape(len1, 1))
        b = np.log(np.linalg.det(K_yy))

        if a + b < min_v:
            min_v = a + b
            opt_l = l
    return opt_l

#批量产生约束
def cons1(p):

    a = p[0]
    lower_bound = p[1]
    return {'type': 'ineq', 'fun': lambda x: x[a]-lower_bound}  
  
def cons2(p):

    a = p[0]
    upper_bound = p[1]
    return {'type': 'ineq', 'fun': lambda x: upper_bound-x[a]}  

def random_selection (x,y,l,problem_dimension, upper_bound, lower_bound): ##防止求解采集函数得到的解聚集在一个位置，增加一个随机搜索的步骤
    indicator = 0
    while indicator == 0:
        Dis = []
        lhsmdu.setRandomSeed(random.randint(1,10000))
        random_selct_point = lhsmdu.sample(1,problem_dimension) * (upper_bound - lower_bound) + lower_bound
        f_x_ad = list(np.array(random_selct_point[0])[0]) ##一个随机选取的解，防止后面搜索到的解不可行
        for i in range(len(x)):
            dis_i = np.linalg.norm(np.array(f_x_ad) - x[i])
            Dis.append(dis_i)
        if (min(Dis) < 0.0001):
            indicator = 0
        else:
            indicator = 1
    return f_x_ad
        
         

def multi_start_opt(x,y,l,problem_dimension, upper_bound, lower_bound):
    P1 = []
    P2 = []
    S = 5 #搜索过程中采用多起点搜索，S表示生成初始采样点的数目
    lhsmdu.setRandomSeed(random.randint(1,10000))
    Initial_Points = lhsmdu.sample(S,problem_dimension) * (upper_bound - lower_bound) + lower_bound
    for a in range(problem_dimension):
        P1.append((a, lower_bound))
        P2.append((a, upper_bound))
    con1=list(map(cons1,P1))   
    con2=list(map(cons2,P2))
    f_x_ad = None
    Min_est = 10**100
    for idx in range(S):
        C = con1 + con2
        x_t_c = np.array(Initial_Points[idx])
        x_t = list(x_t_c[0])
        res = minimize(estimation, x_t, (x,y,l), constraints=C)
        x_ad = res.x
        est_y = estimation(x_ad,x,y,l)
        Dis = []
        for i in range(len(x)):
            dis_i = np.linalg.norm(np.array(x_ad) - x[i])
            Dis.append(dis_i)
        if (min(Dis) > 0.00000000001):
            if est_y < Min_est:
                Min_est = est_y
                f_x_ad = x_ad
    if f_x_ad == None :
        f_x_ad = random_selection (x,y,l,problem_dimension, upper_bound, lower_bound)
    return f_x_ad

def get_stopping_criteria(X,Y,l):
    covariance_matrix = G_kernel(X, X, l)
    if (np.linalg.cond(covariance_matrix) > 10**16):
        return 1
    else:
        return 0


class BO:
    def __init__ (self, problem_dimension, upper_bound, lower_bound, init_sample_num):
        self.problem_dimension = problem_dimension ##这个是决策变量的维度，整数类型
        self.upper_bound = upper_bound ##决策变量的上限约束，整数类型（这个代码比较基础，默认所有维度的决策变量的上限约束是相同的，如果需要对不同维度设置不同的约束，我可以过几天做一下）
        self.lower_bound = lower_bound ##决策变量的下限约束，整数类型（这个代码比较基础，默认所有维度的决策变量的下限约束是相同的）
        self.init_sample_num = init_sample_num ## 算法的初始采样点数目，整数类型
        self.X = []
        self.Y = []
        self.l = 10
    
    def get_opt_solution(self):
        min_y = 10**20
        opt_x = self.X[0]
        for i in range(len(self.Y)):
            yi = self.Y[i]
            xi = self.X[i]
            if yi < min_y:
                min_y = yi
                opt_x = xi
        return opt_x
    
    def minimization (self, func):
        '''
        

        Parameters
        ----------
        func : TYPE
            DESCRIPTION.func是需要优化的目标函数，现在的代码比较基础，只支持func包含一个输入参数，就是决策变量，其他参数需要事先给定

        Returns
        -------
        opt_x : TYPE
            DESCRIPTION. 最优目标函数值

        '''
        lhsmdu.setRandomSeed(random.randint(1,1000))
        Initial_Points = lhsmdu.sample(self.init_sample_num, self.problem_dimension) * self.upper_bound
        for idx in range(self.init_sample_num):
            x_t_c = np.array(Initial_Points[idx])
            x_t = list(x_t_c[0])
            y_t = func (x_t)
            self.X.append(x_t)
            self.Y.append(y_t)
        self.Y = np.array(self.Y)
        #self.l = find_l(self.X, self.Y)
        stopping_criteria = get_stopping_criteria(self.X,self.Y,self.l)
        while (stopping_criteria == 0):
            covariance_matrix = G_kernel(self.X, self.X, self.l)
            #print("------the condition number is:")
            #print(np.linalg.cond(covariance_matrix))
            
            new_sample = multi_start_opt(self.X,self.Y,self.l,self.problem_dimension, self.upper_bound, self.lower_bound)
            new_y      = func (new_sample)
            #print("----- objective function is",new_y, "--------")
            self.X.append(list(new_sample))
            self.Y = list(self.Y)
            self.Y.append(new_y)
            self.Y = np.array(self.Y)
            self.l = find_l(self.X, self.Y)
            stopping_criteria = get_stopping_criteria(self.X,self.Y,self.l)
        opt_x = self.get_opt_solution()
        return opt_x
        
    def plot_approximate_func (self):
        X_seed = np.arange(self.lower_bound, self.upper_bound, 0.01)
        X_test = []
        Y_test = []
        for xi in X_seed:
            X_test.append(np.ones(self.problem_dimension) * xi)
        for x_t in X_test:
            y_t = estimation(x_t, self.X, self.Y, self.l)
            Y_test.append(y_t)
        plt.plot(X_test, Y_test)
        
            

bo = BO(1, 15, 0.1, 5)
re = bo.minimization(stochastic_griewank_30)
# r = stochastic_griewank_30([1])
# bo.X可以查看历史采样点


"""优化超参程序，迭代次数为1200，可以适量增加"""
best = fmin(f, space=space4cp, algo=tpe.suggest, max_evals=1200, trials=trials)
print('best:')
print(best)
print('trials:')
for trial in trials.trials:
    print(trial)


