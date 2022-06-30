function C=fastCorr(A)
       n=size(A,1);
       if ~(all(abs(mean(A))<1e-5) && all(abs(std(A)-1)<1e-5))
          A=zscore(A);
       end 
       C=A'*A/(n-1);       
end 