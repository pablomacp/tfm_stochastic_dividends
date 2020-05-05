clear all;

% Datos del problema
r=0.01; St=[6,4,2]; sigma=[0.3,0.5,0.15]; N=365; M=10000; T=1; t=0; rho=0.5;

% Tama�o del paso
h=(T-t)/N;

% Matriz de covarianzas
Cov=[1 0.5 0.2; 0.5 1 -0.4; 0.2 -0.4 1];

%%%%%%%%%%%%% C�lculos usando mvnrnd de MATLAB %%%%%%%%%%%%%%%%%%%%%%
% tic
% ST=zeros(M,2);
% for i=1:M
%     ST(i,:)=BSAssetMSamples(r,St,sigma,N,M,T,t,Cov);
% end
% toc

% C�lculos usando normrnd y correlacionando variables despu�s
tic
[ST]=BSMultiAssetMSamples(r,St,sigma,N,M,h,Cov);
toc

% Calculo del payoff seg�n la muestra
payoff=max(ST(:,1)-ST(:,2)-ST(:,3),zeros(2*M,1));

% Precio obtenido por el montecarlo, desviaci�n t�pica e intervalo de
% confianza del alpha%
priceMC=mean(exp(-r*(T-t))*payoff)
errorMC=std(payoff)/sqrt(M);
alpha=0.01;
IC=[priceMC-norminv(1-alpha/2)*errorMC,priceMC+norminv(1-alpha/2)*errorMC]

%Precio obtenido mediante la f�rmula anal�tica
[priceAnalytic]=BSExOptionAnalytic(r,St,sigma,T,t,rho)

%Comprobaci�n del precio anal�tico dentro del intervalo de confianza
if IC(1)<priceMC<IC(2)
    disp('Precio anal�tico dentro del intervalo de confianza')
else
    disp('Precio anal�tico fuera del intervalo de confianza')
end

porcentaje_error=abs(priceMC-priceAnalytic)*100/priceAnalytic