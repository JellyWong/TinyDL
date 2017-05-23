# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:55:58 2017

@author: wanggd
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def fun(x):
    '''
    y = []
    x = np.atleast_1d(x)
    for i in range(x.shape[0]):
        if x[i] > 1 and x[i] < 2*np.pi-1:
            y.append(10*np.cos(1))
            continue
        y.append(10*np.cos(x[i]))
    return np.asarray(y)
    '''
    return np.cos(4*x)
    
def next_pos(pre_pos, v, lbest_pos, gbest_pos, center, truncated=True, multi=True):
    lbest_pos_eval = fun(lbest_pos).reshape((lbest_pos.shape[0], 1))
    pos = pre_pos + v
    if truncated:
        for i in range(pos.shape[0]):
            if pos[i] >= x_right_lim:
                pos[i] = x_right_lim
            elif pos[i] <= x_left_lim:
                pos[i] = x_left_lim
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
                gbest_pos = np.append(gbest_pos, pos[max_eval_idx])
                gbest_pos_eval = fun(gbest_pos)
                if np.ptp(gbest_pos_eval) > threshold_y:
                    gbest_pos = np.delete(gbest_pos, np.argmin(gbest_pos_eval), 0)
                center = np.random.randint(0, gbest_pos.shape[0], pos.shape[0])
    else:
        if fun(gbest_pos) < eval[max_eval_idx]:
            gbest_pos[0] = pos[max_eval_idx]
    #print center
    print gbest_pos
    solution.append(gbest_pos[0])
    return pos, lbest_pos, gbest_pos, center
    
def next_v(pos, v, lbest_pos, gbest_pos, center, multi=True):
    c1 = 0.001
    c2 = 0.001
    r1 = np.random.rand(pos.shape[0])
    r2 = np.random.rand(pos.shape[0])
    if multi:
        for i in range(v.shape[0]):
            v[i] = v[i] + c1*r1[i]*(lbest_pos[i]-pos[i]) + \
                    c2*r2[i]*(gbest_pos[center[i]]-pos[i])
    else:
        v = v + c1*r1*(lbest_pos-pos) + c2*r2*(gbest_pos[0]-pos)
    return v

def generator():
    pos = np.random.uniform(x_left_lim, x_right_lim, size = partical_nums)
    v = np.random.uniform(v_min, v_max, size = partical_nums)    
    lbest_pos = pos.copy()
    eval_pos = fun(pos)
    global gbest_pos
    gbest_pos = np.array([pos[np.argmax(eval_pos)]])
    solution.append(gbest_pos[0])
    center = np.zeros(pos.shape[0], dtype='int')
    cnt = 0
    while cnt < iteration_nums:
        cnt += 1
        pos, lbest_pos, gbest_pos, center= next_pos(pos, v, lbest_pos, gbest_pos, center, multi=multi)
        v = next_v(pos, v, lbest_pos, gbest_pos, center, multi=multi)
        eval_pos = fun(pos)
        yield pos, eval_pos

def update(data):
    pos, eval_pos = data
    pos = pos.reshape((pos.shape[0],1))
    eval_pos = eval_pos.reshape((eval_pos.shape[0],1))
    points.set_offsets(np.concatenate((pos,eval_pos), axis=1))
    return points,


v_max = 0.001
v_min = -v_max

partical_nums = 50
iteration_nums = 500

x_left_lim = -1
x_right_lim = 2*np.pi+1

multi = True
threshold_y = 1e-4
threshold_x = 1e-2

x = np.arange(x_left_lim, x_right_lim, 0.0001)
y = fun(x)

solution = []

figure, ax = plt.subplots()  
plt.plot(x, y)
plt.xlim(-2,8)
plt.grid()
points = plt.scatter([], [], c=u'r')
ani = animation.FuncAnimation(figure, update, generator,
                              interval=50, blit=False, repeat=False) 
#ani.save('pso.gif', dpi=80)
plt.show() 














