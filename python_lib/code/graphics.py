# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:17:16 2021

@author: pablo
"""

from random import randint
from math import sqrt

import matplotlib.pyplot as plt
# plt.plot([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 1, 2, 1, 0, -1], 'b')
# plt.plot([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 1, 2, 1, 0, -1], 'bo')
# plt.ylabel('$M_k$')
# plt.xlabel('$k$')
# plt.text(-0.2, 0.1, '$M_0$')
# plt.text(-0.45+1, 1, '$M_1$')
# plt.text(-0.45+2, 2, '$M_2$')
# plt.text(-0.45+3, 1, '$M_3$')
# plt.text(-0.45+4, 2, '$M_4$')
# plt.text(-0.45+5, 1, '$M_5$')
# plt.text(-0.45+6, 0, '$M_6$')
# plt.text(-0.45+7, -1, '$M_7$')
# plt.show()

def paso():
    return randint(0, 1)*2 - 1

def simetric_path(k):
    result = [0]
    for i in range(k):
        result.append(paso()+result[i])
    return result

def scaled_simetric_path(n, t):
    sim = simetric_path(n*t)
    return [elem/sqrt(n) for elem in sim]

def plot_scaled(n, t, color='b', label='graph'):
    graph = scaled_simetric_path(n, t)
    steps = range(0, len(graph))
    time = [elem/n for elem in steps]
    
    plt.plot(time, graph, color, label=label)
    plt.ylabel('$W^{(n)}(t)$')
    plt.xlabel('$t$')
    plt.show()
    
plot_scaled(100, 5, 'b', '$n=100$')
plot_scaled(20, 5, 'r', '$n=20$')
plot_scaled(5, 5, 'g', '$n=5$')
plt.legend()