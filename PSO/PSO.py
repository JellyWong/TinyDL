# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 15:01:17 2017

@author: wanggd
"""

import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    x = np.atleast_2d(x)
    return np.cos(x[:,0]-x[:,1]**2) + np.log(x[:, 2]**2+1) + x[:, 3]**2
    
def init_pos():
    pos = np.random.uniform(x_min, x_max, size=(partical_nums, ndim))
    v = np.random.uniform(v_min, v_max, size = (partical_nums, ndim))    
    lbest_pos = pos.copy()
    eval_pos = fun(pos)
    gbest_pos = np.atleast_2d(pos[np.argmax(eval_pos)])
    center = np.zeros(pos.shape[0], dtype='int')
    return pos, v, lbest_pos, gbest_pos, center

def is_out_bound(pos):
    for dim in pos:
        if dim < x_min or dim > x_max:
            return True
    return False
    
def next_pos(pre_pos, v, lbest_pos, gbest_pos, center, truncated=False, multi=True):
    lbest_pos_eval = fun(lbest_pos).reshape((lbest_pos.shape[0], 1))
    pos = pre_pos + v

    if truncated:
        for i in range(pos.shape[0]):
            for j in range(pos[i].shape[0]):
                if pos[i][j] >= x_max:
                    pos[i][j] = x_max
                elif pos[i][j] <= x_min:
                    pos[i][j] = x_min
                    
    eval = fun(pos).reshape((pos.shape[0], 1))
    for i in range(pre_pos.shape[0]):
        if eval[i] > lbest_pos_eval[i]:
            lbest_pos[i] = pos[i]
    max_eval_idx = np.argmax(eval) 
    if multi:
        if np.min(fun(gbest_pos)) < eval[max_eval_idx]:
            for i in range(gbest_pos.shape[0]):
                dist = np.sum((gbest_pos[i] - pos[max_eval_idx])**2)
                if dist < threshold_x:
                    if fun(gbest_pos[i]) < eval[max_eval_idx]:
                        gbest_pos[i] = pos[max_eval_idx]
                    break
            else:
                gbest_pos = np.vstack((gbest_pos, pos[max_eval_idx]))
                gbest_pos_eval = fun(gbest_pos)
                if np.ptp(gbest_pos_eval) > threshold_y:
                    gbest_pos = np.delete(gbest_pos, np.argmin(gbest_pos_eval), 0)
                n_classes = gbest_pos.shape[0]
                center = np.random.randint(0, n_classes, pos.shape[0])
    else:
        if fun(gbest_pos) < eval[max_eval_idx]:
            gbest_pos = pos[max_eval_idx]
    return pos, lbest_pos, gbest_pos, center
    
def next_v(pos, v, lbest_pos, gbest_pos, center, multi=True):
    c1 = 0.001
    c2 = 0.001
    r1 = np.random.rand(pos.shape[0])
    r2 = np.random.rand(pos.shape[0])
    gbest_pos = np.atleast_2d(gbest_pos)
    if multi:
        for i in range(v.shape[0]):
            v[i] = v[i] + c1*r1[i]*(lbest_pos[i]-pos[i]) + \
                    c2*r2[i]*(gbest_pos[center[i]]-pos[i])
    else:
        r1 = r1.repeat(pos.shape[1]).reshape(pos.shape[0], pos.shape[1])
        r2 = r2.repeat(pos.shape[1]).reshape(pos.shape[0], pos.shape[1])
        v = v + c1*r1*(lbest_pos-pos) - c2*r2*(gbest_pos[0]-pos)
    return v

x_max = np.pi/2
x_min = -np.pi/2

v_max = 0.001
v_min = -v_max

threshold_y = 1e-2
threshold_x = 1e-2

multi = True

partical_nums = 400
iteration_nums = 2000
ndim = 4

data = []
max_data = []
for i in range(iteration_nums):
    if i == 0:
        pos, v, lbest_pos, gbest_pos, center= init_pos()
        data.append(np.mean(fun(lbest_pos)))
        max_data.append(np.max(fun(gbest_pos)))
        continue
    pos, lbest_pos, gbest_pos, center= \
        next_pos(pos, v, lbest_pos, gbest_pos, center, truncated=True, multi=multi)
    v = next_v(pos, v, lbest_pos, gbest_pos, center, multi=multi)
    data.append(np.mean(fun(lbest_pos)))
    max_data.append(np.max(fun(gbest_pos)))

'''
plt.plot(data)
plt.plot(max_data)
'''
data = gbest_pos[:,[0,1]]
data = data[data[:,0].argsort()]
plt.scatter(data[:,0], data[:,1])
plt.grid()
print(max_data[-1])









