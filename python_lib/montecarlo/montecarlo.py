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

def SimulationAssetDividendsMSamples(r,St,sigma,N,M,h,Cov):
    A = np.ones((1, N+1))
    B = np.log(St)  + (r - np.multiply(sigma, sigma) / 2) * h * N
    C = np.ones((N,1)) * sigma
    R = MultiNormCorr(h*Cov,N,M)
    
    ST=np.zeros((2*M,2,N+1))
    
    for i in range(M):
        E = np.multiply(C, R[:,:,i])
        D = np.append([B], E, axis=0)
        for dia in range(N+1):
            ST[i,:,dia]=np.exp(np.dot(A[:,:dia+1], D[:dia+1,:]))
            ST[i+M,:,dia] = np.exp(np.dot(A[:,:dia+1], np.append([B],-E, axis = 0)[:dia+1,:]))
        
    return ST

def MultiNormCorr(sigma,N,M):
    # N es el número de pasos de cada camino
    # M es el número de caminos
    
    # dim es el número de assets
    dim = len(sigma)
    
    # Descomposición de Cholesky. Diferente que en matlab, python da TriInferior.
    L = np.linalg.cholesky(sigma).T
    
    # Generación de normales independientes
    Z = np.random.normal(0,1,(N,dim,M))
    
    # Transformación para correlacionarlas
    result = np.zeros((N,dim,M))
    for k in range(M):
        result[:,:,k] = np.dot(Z[:,:,k],L)
    
    return result

def BSExOptionAnalytic(r,St,sigma,T,t,rho):
    tau = T - t
    beta = sqrt( np.dot(sigma,sigma.T) - 2*rho*sigma[0]*sigma[1])
    d = log(St[0]/St[1])/(sqrt(tau)*beta) + sqrt(tau)*beta/2;
    return St[0]*norm.cdf(d) - St[1]*norm.cdf(d-sqrt(tau)*beta);