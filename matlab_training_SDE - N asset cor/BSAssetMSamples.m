function [ST]=BSAssetSample(r,St,sigma,N,M,T,t,Cov)
h=(T-t)/N;
% lnSt=log(St);
%     for i=1:N
%         lnSt(i+1,:)=lnSt(i,:)+(r-sigma.*sigma/2)*h+sigma.*mvnrnd([0 0], h*[1 rho; rho 1]);
%     end
% ST=exp(lnSt(N+1,:));

A=ones(1,N+2);
B=[log(St);(r-sigma.*sigma/2)*h*N;ones(N,1)*sigma.*mvnrnd(zeros(1,length(St)), h*Cov,N)];

ST=exp(A*B);

% tic
% aux=zeros(M,N*M);
% for i=1:M
%     for j=1+N*(i-1):N*i
%         aux(i,j)=1;
%     end 
% end
% toc
% tic
% A=[ones(M,2),aux];
% B=[log(St);(r-sigma.*sigma/2)*h*N;ones(N*M,1)*sigma.*mvnrnd([0 0], h*[1 rho; rho 1],N*M)];
% ST=exp(A*B);
% toc
% 
% 
% 
% auxSt=zeros(1,2*M);
% auxSigma=zeros(1,2*M);
% auxSig=(r-sigma.*sigma/2)*h*N;
% for i=1:M
%    auxSt(1,i)=log(St(1)) ;
%    auxSt(1,M+i)=log(St(2));
%    auxSigma(1,i)=auxSig(1) ;
%    auxSigma(1,M+i)=auxSig(2);
% end
% 
% B=[auxSt;auxSigma;ones(N*M,1)*sigma.*mvnrnd([0 0], h*[1 rho; rho 1],N*M)];
