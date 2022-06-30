function [beta_opt,beta_cvrt,gamma_opt,CV] = cv_SPG_cvrt(prob,Y, X, Z, C, CNorm, option,g_idx) 
% this function is for cross validation of relative shift model with
% clustered lasso penalty (or complete-graph-guided lasso penalty).
% Input:
%   prob    "graph" or "group"
%   Y       n*1 training response
%   X       n*p training compositional matrix (for tree, it's the expanded matrix)
%   Z       n*q covariate matrix (if no covariate, just input an empty set [])
%
%   C, CNorm and option are all inputs for SPG('graph'), ******specific to
%   compositions, exactly the same as the input for cv_SPG
%   
%     option.nfold is the number of CV folds, default=5       
%     option.gammarange is the range of tuning parameters, default=exp(-5:0.1:0)
%   g_idx   optional, only required for "group"
%
% Output:
%   beta_opt    p*1 optimal coefficient for compositions in the RS model
%   beta_cvrt   q*1 coefficient for cvrt
%   gamma_opt   optimal tuning parameter for RS model
%   CV      1*#tuning CV scores
%
% by Gen Li, 5/31/2020

if isfield(option, 'nfold')
    nfold=option.nfold;
else
    nfold=5;
end  

if isfield(option, 'gammarange')
    gammarange=option.gammarange;
else
    gammarange=exp(-5:0.1:0);
end  

nocvrt=isempty(Z);
[n,p]=size(X);
[~,q]=size(Z);
% process C (this is the key to cvrts adjustment)
C=[C,zeros(size(C,1),q)];
%
CV_score=zeros(nfold,length(gammarange));
index=randsample(n,n);
foldsize=floor(n/nfold);

for ifold=1:nfold
    index_test=index(((ifold-1)*foldsize+1):(ifold*foldsize));
    index_train=setdiff(index,index_test);
    
    Xtrain=X(index_train,:);
    Ytrain=Y(index_train,:);
    Xtest=X(index_test,:);
    Ytest=Y(index_test,:);
    if nocvrt
        Ztrain=[];
        Ztest=[];
    else
        Ztrain=Z(index_train,:);
        Ztest=Z(index_test,:);
    end

    
    
    for itune=1:length(gammarange)
        gamma=gammarange(itune);
        option.verbose=false;
        if (strcmpi(prob, 'group'))
            [beta,~,~,~,~] = SPG(prob, Ytrain, [Xtrain,Ztrain], gamma, 0, C, CNorm, option,g_idx); 
        else
            [beta,~,~,~,~] = SPG(prob, Ytrain, [Xtrain,Ztrain], gamma, 0, C, CNorm, option);
        end
        
        cvscore=sum((Ytest-[Xtest,Ztest]*beta).^2)/length(Ytest); % MSE
        
        CV_score(ifold,itune)=cvscore;
    end
end
    
CV=mean(CV_score,1);
[~,ind]=min(CV);
gamma_opt=gammarange(ind);
if (strcmpi(prob, 'group'))
    [beta_final,~,~,~,~] = SPG(prob, Y, [X,Z], gamma_opt, 0, C, CNorm, option,g_idx); 
else
    [beta_final,~,~,~,~] = SPG(prob, Y, [X,Z], gamma_opt, 0, C, CNorm, option);
end
beta_opt=beta_final(1:p);
beta_cvrt=beta_final((p+1):end);
figure(100);clf
plot(gammarange,CV,'o-');
xlabel('gamma value');
ylabel('CV score');

