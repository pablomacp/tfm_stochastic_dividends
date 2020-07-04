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
    
    S=[[]] * (N + 1)
    q=[[]] * (N + 1)
    
    if len(a)!=N:
        print('a está mal')
    if len(h)!=N:
        print('h está mal')

    t0 = time.time()
    np.random.seed(140494)
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


    Saux = np.ones(2*M) * log(S0)
    q[0] = np.ones(2*M) * q0
    S[0] = np.exp(Saux)
    for dia in range(N):
        Saux += h[dia] * (np.ones(2*M) * (r - 0.5 * volS[dia] ** 2) - q[dia]) + np.sqrt(h[dia]) * volS[dia] * np.concatenate((random_walk[dia,0,:], -random_walk[dia,0,:]))
        # qaux = np.maximum(np.zeros(2*M), q[dia]) # q se hace negativo por la discretización del problema
        # q[dia+1] = q[dia] + h[dia] * (a[dia] * np.ones(2*M) - b * q[dia]) + np.multiply(np.sqrt(qaux * h[dia]) * volq[dia], np.concatenate((random_walk[dia,1,:], -random_walk[dia,1,:])))
        q[dia + 1] = q[dia] + b * h[dia] * (a[dia] * np.ones(2 * M) -  q[dia]) + np.multiply(
            np.sqrt(h[dia]) * volq[dia], np.concatenate((random_walk[dia, 1, :], -random_walk[dia, 1, :])))
        S[dia+1] = np.exp(Saux)

    t2 = time.time()
    print('Cálculo caminos:', t2 - t1)
    return S, q

def PayoffDivFut(S,q,h,hstart):
    result = np.zeros(len(S[0]))
    for i in range(hstart, len(h)):
        result += np.multiply(S[i],q[i])*h[i]
    return result

def PayoffOptCall(S, K):
    return np.maximum(S - K * np.ones(len(S)), np.zeros(len(S)))