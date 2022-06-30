function X_thres=Threshold(X, threshold)
% 
% input: 
%
%   X       n*p compositional data matrix, where each row sums to 1
%
%   thres   scalar in [0,1], any value below thres is set to 0
%
%
%
% Output: 
%
%   X_thres  n*p compositional data with excessive zeros (may contain value below thres because of rescaling) 
% 
%
% 4/15/2020 by Gen Li





[n,p]=size(X);
if min(X(:))<0 
    error("Input is not non-negative! Terminated...")
elseif max(sum(X,2))>1+1E-9 || min(sum(X,2))<1-1E-9
    warning("Input is not a compositional matrix! Converted...")
    X=diag(1./sum(X,2))*X;
end

if threshold>min(max(X,[],2)) % some rows will be truncated to a zero row
    error("Threshold too large! Terminated...")
end
    
X_thres=X.*double(X>=threshold);
X_thres=diag(1./sum(X_thres,2))*X_thres;


