function [ST]=BSMultiAssetMSamples(r,St,sigma,N,M,h,Cov)

A=ones(1,N+1);
B=log(St)+(r-sigma.*sigma/2)*h*N;
C=ones(N,1)*sigma;
R=MultiNormCorr(h*Cov,N,M);

ST=zeros(2*M,length(sigma));
for i=1:M
    ST(i,:)=exp(A*[B;C.*R(:,:,i)]);
    ST(i+M,:)=exp(A*[B;-1*C.*R(:,:,i)]);
end