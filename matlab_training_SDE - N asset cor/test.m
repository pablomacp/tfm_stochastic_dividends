N=365;
M=100000;
sigma=[1 -0.6 0.3;-0.6 1 0.4; 0.3 0.4 1];
[result]=MultiNormCorr(sigma,N,M);