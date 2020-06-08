# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:43:14 2020

@author: pablo
"""

import numpy as np
from math import log, sqrt
from scipy.stats import norm

# Sustitución de columnas en numpy: a[:,0]=np.array([5,5])
# Las tienes que meter en filas !!!!!!!!!!!!!!!!!
# Solo da problemas al bajar a dimensión 1, si no, funciona igual que matlab.

# np.random.normal(0,1,(3,2,2)) matrices dos x dos. El orden de las
# dimensiones va al revés que matlab

def HybridStockDividendsMSamples(S0,q0,r,a_pasos,b,volS,volq_pasos,rho,M,N,h):
    # M number of paths.
    # N number of step of each path.
    
    a = a_pasos[:N]
    volq = volq_pasos[:N]
    
    S=list()
    q=list()
    
    if len(a)!=N:
        print('a está mal')
    if len(h)!=N:
        print('h está mal')
    
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
    
    for path in range(M):
        Saux = np.ones(N+1)
        qaux = np.ones(N+1)
        Saux[0] = log(S0)
        qaux[0] = q0
        for i in range(N):
            # Probar el efecto de quitar los dividendos al stock.
            Saux[i+1] = Saux[i] + h[i]*(r-qaux[i]-0.5*volS**2) + sqrt(h[i])*volS*random_walk[i,0,path]
            # Modelo Cox-Ingerson-Ross
            qaux[i+1] = qaux[i] + h[i]*(a[i]-b*qaux[i]) + sqrt(qaux[i]*h[i])*volq[i]*random_walk[i,1,path]
            # Modelo Hull-White
            # qaux[i+1] = qaux[i] + h[i]*(a[i]-b*qaux[i]) + sqrt(h[i])*volq*random_walk[i,1,path]
        S.append(np.exp(Saux))
        q.append(qaux)
        
    return S, q

def PayoffDivFut(S,q,h):
    return np.sum(np.multiply(np.multiply(S,q),h))