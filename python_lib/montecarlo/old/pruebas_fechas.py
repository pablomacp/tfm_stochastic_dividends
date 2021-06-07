# -*- coding: utf-8 -*-
"""
Created on Mon May 18 02:08:42 2020

@author: pablo
"""


import numpy as np

t1 = '2020-04-14'
t2 = '2021-04-14'
T0 = '2020-01-01'
T1 = '2021-01-01'
T2 = '2022-01-01'

tiempo = np.busday_count(t1,T1)/np.busday_count(T0,T1) + np.busday_count(T1,t2)/np.busday_count(T1,T2)
print(tiempo)