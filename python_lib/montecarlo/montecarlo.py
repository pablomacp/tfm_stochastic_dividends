# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:43:14 2020

@author: pablo
"""

import time
import numpy as np
from math import log, sqrt
from scipy.stats import norm

# Sustitución de columnas en numpy: a[:,0]=np.array([5,5])
# Las tienes que meter en filas !!!!!!!!!!!!!!!!!
# Solo da problemas al bajar a dimensión 1, si no, funciona igual que matlab.

# np.random.normal(0,1,(3,2,2)) matrices dos x dos. El orden de las
# dimensiones va al revés que matlab

def HybridStockDividendsMSamples(S0,q0,r,a_pasos,b,volS_pasos,volq_pasos,rho,M,N,h):
    # M number of paths.
    # N number of step of each path.
    
    a = a_pasos[:N]
    volq = volq_pasos[:N]
    volS = volS_pasos[:N]
    
    S=[[]] * M * 2
    q=[[]] * M * 2
    
    if len(a)!=N:
        print('a está mal')
    if len(h)!=N:
        print('h está mal')

    t0 = time.time()
    # Calculo de la mtriz aleatoria    
    Cov = np.array([[1, rho], [rho, 1]])
        # Descomposición de Cholesky. Diferente que en matlab, python da TriInferior.
    L = np.linalg.cholesky(Cov).T
        # Generación de normales independientes
    Z = np.random.normal(0,1,(N,2,M))
        # Transformación para correlacionarlas
    random_walk = np.zeros((N,2,M))
    for k in range(M):
        random_walk[:,:,k] = np.dot(Z[:,:,k],L)

    t1 = time.time()
    print('Generar números:', t1-t0)
    for path in range(M):
        Saux = np.ones(N+1)
        qaux = np.ones(N+1)
        Saux[0] = log(S0)
        qaux[0] = q0

        Sauxneg = np.ones(N + 1)
        qauxneg = np.ones(N + 1)
        Sauxneg[0] = log(S0)
        qauxneg[0] = q0
        for i in range(N):
            Saux[i + 1] = Saux[i] + h[i] * (r - qaux[i] - 0.5 * volS[i] ** 2) + sqrt(h[i]) * volS[i] * random_walk[
                i, 0, path]
            # Modelo Cox-Ingerson-Ross
            qaux[i + 1] = qaux[i] + h[i] * (a[i] - b * qaux[i]) + sqrt(max(qaux[i], 0) * h[i]) * volq[i] * random_walk[
                i, 1, path]
            # Modelo Hull-White
            # qaux[i+1] = qaux[i] + h[i]*(a[i]-b*qaux[i]) + sqrt(h[i])*volq*random_walk[i,1,path]
            Sauxneg[i + 1] = Sauxneg[i] + h[i] * (r - qauxneg[i] - 0.5 * volS[i] ** 2) - sqrt(h[i]) * volS[i] * random_walk[
                i, 0, path]
            qauxneg[i + 1] = qauxneg[i] + h[i] * (a[i] - b * qauxneg[i]) - sqrt(max(qauxneg[i], 0) * h[i]) * volq[i] * random_walk[
                i, 1, path]

        S[path] = np.exp(Saux)
        q[path] = qaux
        S[path+M] = np.exp(Sauxneg)
        q[path+M] = qauxneg

    t2 = time.time()
    print('Cálculo caminos:', t2 - t1)
    return S, q

def PayoffDivFut(S,q,h):
    return np.sum(np.multiply(np.multiply(S,q),h))