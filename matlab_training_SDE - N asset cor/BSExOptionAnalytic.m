function [price]=BSExOptionAnalytic(r,St,sigma,T,t,rho)
tau=T-t;
beta=sqrt(sigma*transpose(sigma)-2*rho*sigma(1)*sigma(2));
d=log(St(1)/St(2))/(sqrt(tau)*beta)+sqrt(tau)*beta/2;
price=St(1)*normcdf(d)-St(2)*normcdf(d-sqrt(tau)*beta);