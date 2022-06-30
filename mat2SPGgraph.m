function [C, CNorm]=mat2SPGgraph(W)
% W is a p*p weight matrix, where only the upper triangle part will be used and 
% each entry is the weight \tau(m,l) as in the SPG paper, of each edge (diagnal and lower trangle are zero)

% output
% C   a #edge by p sparse matrix, as defined in the SPG paper 
% CNorm   the norm of C, as defined in the SPG paper

% Note: the weight W does not include the tuning parameter for the penalty.
% In most cases, just use W = ones(p,p).

nV=size(W,2); % number of predictors, p 
weight=abs(triu(W,1)); %upper triangluar of C
nzUR=find(weight~=0);
[E1,E2]=ind2sub([nV,nV],nzUR);
E=[E1,E2]; %indices in E(i,1) E(i,2) forms an edge
nE=size(E,1); % should be smaller or equal to p(p-1)/2
Ecoef=weight(nzUR);
C_I=[(1:nE)';(1:nE)'];
C_J=[E1;E2];
C_S=[Ecoef, -Ecoef];
C=sparse(C_I, C_J, C_S, nE, nV);         
 
CNorm=2*max(sum(C.^2,1)); 
          
   