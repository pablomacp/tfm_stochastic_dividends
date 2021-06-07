# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:37:59 2020

@author: pablo
"""

import numpy as np
from funciones import *
from math import exp, sqrt
from scipy.stats import norm

# Datos del problema
r = 0.01
St = np.array([6,4])
sigma = np.array([0.3,0.5])
N = 365
M = 10000
T = 1
t = 0
rho = 0.5

# Tamaño del paso
h = (T-t)/N

# Matriz de covarianzas
Cov = np.array([[1, rho], [rho, 1]])

# Cálculos usando normrnd y correlacionando variables después
ST = BSMultiAssetMSamples(r,St,sigma,N,M,h,Cov)

# Calculo del payoff según la muestra
payoff = np.maximum(ST[:,0]-ST[:,1], np.zeros(2*M))

# Precio obtenido por el montecarlo, desviación típica e intervalo de
# confianza del alpha#
priceMC = np.mean(exp(-r*(T-t))*payoff)
errorMC = np.std(payoff)/sqrt(2*M)
alpha = 0.01
IC = [priceMC-norm.ppf(1-alpha/2)*errorMC, priceMC+norm.ppf(1-alpha/2)*errorMC]

#Precio obtenido mediante la fórmula analítica
priceAnalytic = BSExOptionAnalytic(r,St,sigma,T,t,rho)

print('Precio montecarlo:', priceMC)
print('IC', IC)
print('Precio analítico:', priceAnalytic)

#Comprobación del precio analítico dentro del intervalo de confianza
if IC[0]<priceAnalytic and priceAnalytic<IC[1]:
    print('Precio analítico dentro del intervalo de confianza')
else:
    print('Precio analítico fuera del intervalo de confianza')

porcentaje_error = abs(priceMC-priceAnalytic)*100/priceAnalytic