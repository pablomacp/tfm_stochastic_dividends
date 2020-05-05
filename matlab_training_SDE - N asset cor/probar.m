clear
M=1000000; N=365;
tic
aux1=zeros(M,N*M);
for i=1:M
    for j=1+N*(i-1):N*i
        aux1(i,j)=1;
    end 
end
toc

tic
aux2=zeros(M,N*M);
j=0;
for i=1:M
    for k=1:N
        j=j+1;
        aux2(i,j)=1;
    end 
end
toc

tic
aux3=zeros(M,N*M);
for i=1:M
    aux3(i,:)=ones(1,N*M)-[zeros(1,i*N),ones(1,N*M-i*N)]-[ones(1,(i-1)*N),zeros(1,M*N-(i-1)*N)];
end
toc