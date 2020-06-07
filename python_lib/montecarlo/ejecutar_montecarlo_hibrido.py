# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:37:59 2020

@author: pablo
"""

import numpy as np
from montecarlo import *
from funciones import cal_yf_from_mat, days_yf
from math import exp, sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt

# Fechas futuros dividendos eurostox
# Tfutdiv0 = [ql.Date(18, 12, 2020),
#            ql.Date(17, 12, 2020),
#            ql.Date(16, 12, 2020),
#            ql.Date(15, 12, 2020)]
T_opt_sx5e = ['2020-04-17',
              '2020-05-15',
              '2020-06-19',
              '2020-07-17',
              '2020-08-21',
              '2020-09-18',
              '2020-12-18', #
              '2021-03-19',
              '2021-06-18',
              '2021-12-17', #
              '2022-06-17',
              '2022-12-16', #
              '2023-12-15'] #
T_futdiv = ['2020-12-18',
            '2021-12-17',
            '2022-12-16',
            '2023-12-15']
T_endyear = ['2020-01-01',
             '2021-01-01',
             '2022-01-01',
             '2023-01-01',
             '2024-01-01']
T_trf = ['2020-06-19',
         '2020-09-18',
         '2020-12-18',
         '2021-03-19',
         '2021-06-18',
         '2021-09-17',
         '2021-12-17',
         '2022-03-18',
         '2022-06-17',
         '2022-09-16',
         '2022-12-16',
         '2023-03-17',
         '2023-06-16',
         '2023-09-15',
         '2023-12-15']
t0 = '2020-04-01'
days_futdiv = [np.busday_count(t0, fecha) for fecha in T_futdiv]
days_opt_sx5e = [np.busday_count(t0, fecha) for fecha in T_opt_sx5e]

# Year fractions maturities
yf_futdiv = cal_yf_from_mat(t0, T_futdiv, T_endyear)

# Steps for each maturity.
days_yf, h = days_yf(t0, T_futdiv, T_endyear)

h1 = h[0]
h2 = h[1]
h3 = h[2]
h4 = h[3]

div_fut_prices = [53.1, 49.8, 63.6, 67.4]
trf_prices = [3108.96,
                3101.94,
                3102.94,
                3108.79,
                3115.24,
                3115.88,
                3115.98,
                3119.81,
                3126.00,
                3127.76,
                3128.66,
                3134.92,
                3144.48,
                3147.60,
                3150.72]
div_call_opt_prices = [5.60, 4.19, 11.02, 16.13] # Excel. Tomo el strike 65, que es el más líquido.
eurostoxx_call_opt_prices = [97.3592,
                             145.1804,
                             175.7592,
                             189.5804,
                             206.1592,
                             214.5804,
                             239.0016,
                             269.5972,
                             270.8668,
                             292.9,
                             290.2604,
                             312.5604,
                             336.7692]


# Datos del problema
S0 = 2680.3 # También es el strike de las call sobre el eurostoxx
q0 = 0.022794603
M = 2**12
K_div = 65 # Strike dividend call options
rho = -0.088195234
r = -0.00168

# Parametros a calibrar
b = 0.001 # Criterio experto
a = [0.01, 0.02, 0.03, 0.04]
volS = 0.3
volq = 0.1
# print(2*min(a)*b>=volq**2) #Condition for q = 0 possibility

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
