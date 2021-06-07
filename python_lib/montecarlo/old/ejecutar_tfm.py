# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:37:59 2020

@author: pablo
"""

import numpy as np
from montecarlo import *
from math import exp, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt

# Datos del problema
r = 0.01
St = np.array([6,4])
sigma = np.array([0.3,0.5])
N = 365
M = 1000
T = 1
t = 0
rho = 0.5

# Tamaño del paso
h = (T-t)/N

# Matriz de covarianzas
Cov = np.array([[1, rho], [rho, 1]])

# Cálculos usando normrnd y correlacionando variables después
ST = SimulationAssetDividendsMSamples(r,St,sigma,N,M,h,Cov)

for i in range(30):
    plt.plot(range(366), ST[i,0,:],'b')
    plt.plot(range(366), ST[i,1,:],'r')
plt.show()