# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:25:36 2021

@author: pablo
"""


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot([0, 1, 2], [1, 0, 0], 'b')
ax.spines['bottom'].set_position('center')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(-1,1)

plt.text(0, 1, 'Payoff')
plt.text(2, -0.15, '$S_T$')
plt.text(1, -0.15, '$K$')
plt.text(-0.2, 0, '0')

plt.show()