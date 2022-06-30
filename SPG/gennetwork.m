function [C, CNorm, E, Ecoef, Esign, R]=gennetwork(X, option)
% C: is the E by V incident matrix as in the paper
% CNorm in Proposition 2 in the paper
% E is E by 2 matrix E(i,1) E(i,2) forms an edge
% Ecoef:  \tau(r_{ml}) for each edge e=(m,l)
% Esign:  sign(r_{ml}) for each edge e=(m,l)
% R: correlation matrix


          if isfield(option,'cortype')
              cortype=option.cortype;  %cortype=1 \tau(x)=|x|;  cortype=2: \tau(x)=x^2
          else
              cortype=1;
          end
          
           if isfield(option,'corthreshold')
               corthreshold=option.corthreshold;  % correlation threshold to obtain the graph
           else
               corthreshold=0.5;
           end   
           
          [nV]=size(X,2);
          R=fastCorr(X);
          
          if isfield(option,'nE') && ~isempty(option.nE) % Instead of threshold correlation to obtain the graph, one can set the number of edges
              nE=option.nE;
              [tmp, idx]=sort(abs(R(:)));
              R(idx(1:end-2*nE-nV))=0;
          else
              R(abs(R)<corthreshold)=0;
          end
          
          UR=triu(R,1); %upper triangluar of C
          if (cortype==1)
              W=abs(UR);
          else
              W=UR.^2;
          end
          
          nzUR=find(UR~=0);
          [E1,E2]=ind2sub([nV,nV],nzUR);
          E=[E1,E2];
          
          nE=size(E,1);
          Ecoef=W(nzUR);
          Esign=sign(R(nzUR));
          
          C_I=[(1:nE)';(1:nE)'];
          C_J=[E1;E2];
          C_S=[Ecoef, -Ecoef.*Esign];
          C=sparse(C_I, C_J, C_S, nE, nV);         
          
          CNorm=2*max(sum(C.^2,1)); 
          
end