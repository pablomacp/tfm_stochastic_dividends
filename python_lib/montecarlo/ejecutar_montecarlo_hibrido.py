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

# Fechas futuros dividendos eurostox
# Tfutdiv0 = [ql.Date(18, 12, 2020),
#            ql.Date(17, 12, 2020),
#            ql.Date(16, 12, 2020),
#            ql.Date(15, 12, 2020)]
T_futdiv = ['2020-12-18',
            '2021-12-17',
            '2022-12-16',
            '2023-12-15']
T_endyear = ['2020-01-01',
             '2021-01-01',
             '2022-01-01',
             '2023-01-01',
             '2024-01-01']
t0 = '2020-04-01'
days_futdiv = [np.busday_count(t0, fecha) for fecha in T_futdiv]

# Year fractions maturities
yf_futdiv = list()
yf_futdiv.append(np.busday_count(t0, T_futdiv[0])/np.busday_count(T_endyear[0], T_endyear[1]))
yf_futdiv.append(np.busday_count(t0, T_endyear[1])/np.busday_count(T_endyear[0], T_endyear[1]) + np.busday_count(T_endyear[1], T_futdiv[1])/np.busday_count(T_endyear[1], T_endyear[2]))
yf_futdiv.append(np.busday_count(t0, T_endyear[1])/np.busday_count(T_endyear[0], T_endyear[1]) + 1 + np.busday_count(T_endyear[2], T_futdiv[2])/np.busday_count(T_endyear[2], T_endyear[3]))
yf_futdiv.append(np.busday_count(t0, T_endyear[1])/np.busday_count(T_endyear[0], T_endyear[1]) + 2 + np.busday_count(T_endyear[3], T_futdiv[3])/np.busday_count(T_endyear[3], T_endyear[4]))

# np.concatenate([np.ones(3),np.ones(6)])
incr_t0t1 = np.ones(np.busday_count(t0, T_futdiv[0]))/np.busday_count(T_endyear[0], T_endyear[1])
incr_t1t2 = np.concatenate([np.ones(np.busday_count(T_futdiv[0], T_endyear[1]))/np.busday_count(T_endyear[0], T_endyear[1]), np.ones(np.busday_count(T_endyear[1], T_futdiv[1]))/np.busday_count(T_endyear[1], T_endyear[2])])
incr_t2t3 = np.concatenate([np.ones(np.busday_count(T_futdiv[1], T_endyear[2]))/np.busday_count(T_endyear[1], T_endyear[2]), np.ones(np.busday_count(T_endyear[2], T_futdiv[2]))/np.busday_count(T_endyear[2], T_endyear[3])])
incr_t3t4 = np.concatenate([np.ones(np.busday_count(T_futdiv[2], T_endyear[3]))/np.busday_count(T_endyear[2], T_endyear[3]), np.ones(np.busday_count(T_endyear[3], T_futdiv[3]))/np.busday_count(T_endyear[3], T_endyear[4])])

h1 = incr_t0t1
h2 = np.concatenate([h1,incr_t1t2])
h3 = np.concatenate([h2,incr_t1t2])
h4 = np.concatenate([h3,incr_t1t2])

# Datos del problema
S0 = 2680.3
q0 = 0.022794603
volS = 0.3
volq = 0.1
M = 10000

# Parametros Hull-White
b = 0.001 # Criterio experto
a = [0.01, 0.02, 0.03, 0.04]
print(2*min(a)*b>=volq**2)

rho = -0.088195234
r = -0.00168

a_pasos = np.concatenate([a[0]*np.ones(np.busday_count(t0,T_futdiv[0])), a[1]*np.ones(np.busday_count(T_futdiv[0],T_futdiv[1])), a[2]*np.ones(np.busday_count(T_futdiv[1],T_futdiv[2])), a[3]*np.ones(np.busday_count(T_futdiv[2],T_futdiv[3]))])

# Número de pasos N en la simulación
N = len(h1)

# Cálculos usando normrnd y correlacionando variables después
S, q = HybridStockDividendsMSamples(S0,q0,r,a_pasos[:N],b,volS,volq,rho,M,N,h1)

print('divs_min:', min([min(elementos) for elementos in q]), 'divs_max:', max([max(elementos) for elementos in q]))

for i in range(30):
    plt.plot(range(N+1), S[i],'b')
    plt.plot(range(N+1), q[i],'r')
plt.show()

payoffs = [PayoffDivFut(index[1:], divs[1:], h1) for index, divs in zip(S,q)]
precio = np.mean(payoffs)
error = np.std(payoffs)/sqrt(M)
alpha = 0.05
IC = [precio - norm.ppf(1-alpha/2)*error, precio + norm.ppf(1-alpha/2)*error]
print('Precio MC:', precio)
print('IC 95%:', IC)
