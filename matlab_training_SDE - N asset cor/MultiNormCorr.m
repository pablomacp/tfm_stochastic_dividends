function [result]=MultiNormCorr(sigma,N,M)
% N es el n�mero de pasos de cada camino
% M es el n�mero de caminos

% dim es el n�mero de assets
dim=length(sigma);

% Descomposici�n de Cholesky
L=chol(sigma);

% Generaci�n de normales independientes
Z=normrnd(0,1,N,dim,M);

% Transformaci�n para correlacionarlas
result=zeros(N,dim,M);
for k=1:M
    result(:,:,k)=Z(:,:,k)*L;
end


%%%%% Bucle for para multiplicar matrices 3D de internet. Es m�s lento %%%%
% tic
% Z2 = reshape(reshape(permute(Z, [2 1 3]), [N dim*M]), [dim N*M])' * L;
% Z2 = permute(reshape(Z2',[dim N M]),[2 1 3]);
% toc