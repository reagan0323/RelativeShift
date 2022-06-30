%% simulation studies
% This file contains examples about how to use the functions to fit the
% relative shift models (RS-ES, RS-L1, RS-CL2, RS-DL2) and reproduce simulation results in the manuscript
%
% Full comparison with competiting methods (LC-Lasso, Lin et al., 2014; KPR
% Randolph et al., 2018) requires exporting simulated data into R and
% implementing the methods in R

% by Gen Li

clear all;
close all;
clc;
rng(20200630)
addpath(genpath(['.',filesep]))
%% Study 1: Equi-Sparsity Setting
% 1. RS-ES
% 2. CLR+lasso
% 3. (in R) LR+Lasso, CLR+ridge, CLR+ridge/UniFrac
rng(20200630)

n=500; % 100 training, 400 testing
p=100;

% true and obs compositional matrix
temp=randn(n,p-1);
X_true=bsxfun(@rdivide, [exp(temp),ones(n,1)],sum(exp(temp),2)+1); % logistic normal distribuition, no zero
thres=0.005;
X_obs=Threshold(X_true,thres); % get observed X with excessive zeros
disp(['Avg zero proportion in the observed data is ', num2str(mean(sum(X_obs==0,2)/p))]); % avg zero percentage in each sample, about 25%



% true coefficient (with equi-sparsity)
beta=[-1*ones(20,1);2*ones(10,1);zeros(70,1)];
figure(1);clf;
plot(beta)


% parameters that can be tweaked (SNR, eps, thres)
SNR=1;sigma=sqrt(var(X_true*beta)/SNR)
% sigma=0.1; SNR=var(X_true*beta)/sigma^2
eps=1e-3; % surrogate of zeros for LC




% repeat 100 times
nsim=100;
rec_RSS=zeros(nsim,2); % in-sample prediction
rec_PMSE=zeros(nsim,2); % out-sample prediction
rec_time=zeros(nsim,2); % computing time
% data for R
Ytrain_forR=zeros(100,nsim);
Xtrain_forR=zeros(100,p,nsim);
Ytest_forR=zeros(400,nsim);
Xtest_forR=zeros(400,p,nsim);
for isim=1:nsim
    disp(['Running sim #', num2str(isim)]); 
    E=randn(n,1)*sigma;
    Y=X_true*beta+E; % relative shift model, from true X

    % separate into 100 training and 400 testing (using obs X)
    Ytrain=Y(1:100);
    Xtrain=X_obs(1:100,:);
    Ytest=Y(101:n);
    Xtest=X_obs(101:n,:);
    % for LR and CLR methods, replace 0 by eps and renormalize
    Xtrain_nozero=max(Xtrain,eps);
    Xtrain_nozero=Xtrain_nozero./sum(Xtrain_nozero,2);
    Xtest_nozero=max(Xtest,eps);
    Xtest_nozero=Xtest_nozero./sum(Xtest_nozero,2);

    
    
    % RS with equi-sparsity (5-fold CV for tuning selection)
    Tstart=tic;
    W=ones(p,p); % weight matrix for graph (complete graph without edge-specific weights)
    [C, CNorm]=mat2SPGgraph(W);
    option.mu=1e-02;   % Smoothing Parameter, fixed
    option.gammarange=(5.5:0.5:15)*1E-4; % tuning parameter
    option.fig=1;
%     option.gammarange=0; % no penalty -> vanilla relative shift model
%     option.gammarange=0.1; % too much penalty -> beta all equal
    [beta_RS,~,gamma_opt,CV] = cv_SPG_cvrt('graph',Ytrain, Xtrain,[], C, CNorm, option); 
    T1=toc(Tstart);
%     figure(1); clf;
%     plot(beta,'k'); hold on;
%     plot(beta_RS,'r')    
    
    
    
    
    % CLR regression with lasso (5-fold CV tuning selection)
    % (a naive competitor created by myself)
    Tstart=tic;
    Xtrain_CLR=log(Xtrain_nozero)-mean(log(Xtrain_nozero),2); % CLR 
    [beta_naive,info]=lasso(Xtrain_CLR,Ytrain,'cv',5); % lasso with CV for tuning selection
    beta_Lasso=beta_naive(:,info.IndexMinMSE);
    beta_Lasso0=info.Intercept(info.IndexMinMSE);
    T2=toc(Tstart);
    
    % implementation of LC with lasso -- R package CVS (Lin et al, 2014)
    % implementation of KPR -- R package KPR (Randolph et al, 2018)
    % Use *5-fold CV* and *10-value tuning range* if possible.

    
    
    
    
    % compare
    % in-sample residual
    res_RS=Ytrain-Xtrain*beta_RS;
    res_CLRlasso=Ytrain-beta_Lasso0-Xtrain_CLR*beta_Lasso;
    rec_RSS(isim,:)=[sum(res_RS.^2),sum(res_CLRlasso.^2)];
    % out-sample prediction
    Xtest_CLR=log(Xtest_nozero)-mean(log(Xtest_nozero),2); % CLR
    rec_PMSE(isim,:)=[sum((Ytest-Xtest*beta_RS).^2),sum((Ytest-Xtest_CLR*beta_Lasso-beta_Lasso0).^2)];
    % time
    rec_time(isim,:)=[T1,T2];

    % save data for R (no-zero compositional data)
    Xtrain_forR(:,:,isim)=Xtrain_nozero;
    Xtest_forR(:,:,isim)=Xtest_nozero;
    Ytrain_forR(:,isim)=Ytrain;
    Ytest_forR(:,isim)=Ytest;
end
save('Sim1.mat','Xtrain_forR','Xtest_forR','Ytrain_forR','Ytest_forR','rec_RSS','rec_PMSE','rec_time');


%% Study 2: Tree-Guided Equi-Sparsity Setting

% 1. RS-DL2
% 2. RS-CL2
% 3. RS-L1
% 4. RS-ES 
% 5. CLR+lasso
% 6. (in R) LR+Lasso, CLR+ridge, CLR+ridge/UniFrac, CLR+tree (patristic kernel), CLR+tree/UniFrac

rng(20200630)

n=500; % 100 training, 400 testing
p=100;

% true and obs compositional matrix
temp=randn(n,p-1);
X_true=bsxfun(@rdivide, [exp(temp),ones(n,1)],sum(exp(temp),2)+1); % logistic normal distribuition, no zero
thres=0.005;
X_obs=Threshold(X_true,thres); % get observed X with excessive zeros
disp(['Avg zero proportion in the observed data is ', num2str(mean(sum(X_obs==0,2)/p))]); % avg zero percentage in each sample, about 25%




% true coefficient (tree-guided)
taxonomy=[(1:100)',kron((1:10)',ones(10,1)),kron((1:5)',ones(20,1)), [ones(40,1);2*ones(40,1);3*ones(20,1)]];
beta=[ones(20,1);-2*ones(10,1);0.5*ones(10,1);2*ones(40,1);randn(20,1)]; % some grouping structure according to the tree
beta_group=[20,10,10,40,ones(1,20)];


% rootless reparameterization (remember to center Y and cvrt)
[A_tree,C_tree1,CNorm_tree1,g_idx_tree1,node,M2, ~,~]=mat2SPGtree_rootless(taxonomy,1); % tree-derived parameters for SPG
    figure();
    treeplot(node);
    title('Trimmed tree (without redundant nodes) -- ready for analysis')
    [x,y] = treelayout(node);
    for i=1:length(x)
        text(x(i),y(i),num2str(i))
    end
% coeff for node111, node103, node104, node117, node81-100
figure();clf;
plot(beta)
[~,C_tree2,CNorm_tree2,g_idx_tree2,~,~, ~,~]=mat2SPGtree_rootless(taxonomy,2); 
[~,C_tree3,CNorm_tree3,g_idx_tree3,~,~, ~,~]=mat2SPGtree_rootless(taxonomy,3); 


% graph parameters for SPG
W=ones(p,p); % weight matrix for graph (complete graph without edge-specific weights)
[C_graph, CNorm_graph]=mat2SPGgraph(W);




% parameters that can be tweaked (SNR, eps, thres)
SNR=1; sigma=sqrt(var(X_true*beta)/SNR)
% sigma=0.1; SNR=var(X_true*beta)/sigma^2;  % SNR=2.8
eps=1e-3; % surrogate of zeros for LC




% repeat 100 times
nsim=100;
rec_RSS=zeros(nsim,5);
rec_PMSE=zeros(nsim,5);
rec_time=zeros(nsim,5); % computing time
rec_es=zeros(nsim,4); % equi-sparsity measure
% data for R
Ytrain_forR=zeros(100,nsim);
Xtrain_forR=zeros(100,p,nsim);
Ytest_forR=zeros(400,nsim);
Xtest_forR=zeros(400,p,nsim);
for isim=1:nsim
    disp(['Running sim #', num2str(isim)]);
    E=randn(n,1)*sigma;
    Y=X_true*beta+E; % relative shift model, from true X

    % separate into 100 training and 400 testing (using obs X)
    Ytrain=Y(1:100);
    Xtrain=X_obs(1:100,:);
    Ytest=Y(101:n);
    Xtest=X_obs(101:n,:);
    % for LR and CLR methods, replace 0 by eps and renormalize
    Xtrain_nozero=max(Xtrain,eps);
    Xtrain_nozero=Xtrain_nozero./sum(Xtrain_nozero,2);
    Xtest_nozero=max(Xtest,eps);
    Xtest_nozero=Xtest_nozero./sum(Xtest_nozero,2);

    
    
    % NOTE: remember to center Y since we have removed the root node!!!
    Ymean=mean(Ytrain);
    Ytrain_c=Ytrain-Ymean;
    
 
    % RS-DL2
    Tstart=tic;
    option.mu=1e-02;   % Smoothing Parameter, fixed
    option.gammarange=(0.5:.5:10)*1E-2;
    option.fig=1;
    [coeff_RS1,~,gamma_opt1,CV1] = cv_SPG_cvrt('group',Ytrain_c, Xtrain*A_tree, [], C_tree1, CNorm_tree1, option,g_idx_tree1); 
    beta_RS_tree1=A_tree*coeff_RS1;
    ES1=beta_group(1)*var(beta_RS_tree1(1:20),1)+...
        beta_group(2)*var(beta_RS_tree1(21:30),1)+...
        beta_group(3)*var(beta_RS_tree1(31:40),1)+...
        beta_group(4)*var(beta_RS_tree1(41:80),1);

    T1_1=toc(Tstart);
    %check equi-sparsity
    figure(1);clf;subplot(2,2,1);plot(beta_RS_tree1);title('Descendant L2')

    % RS-CL2
    Tstart=tic;
    option.mu=1e-02;   % Smoothing Parameter, fixed
    option.gammarange=(0.5:.5:10)*1E-2;
    option.fig=1;
    [coeff_RS2,~,gamma_opt2,CV2] = cv_SPG_cvrt('group',Ytrain_c, Xtrain*A_tree,[], C_tree2, CNorm_tree2, option,g_idx_tree2); 
    beta_RS_tree2=A_tree*coeff_RS2;
    ES2=beta_group(1)*var(beta_RS_tree2(1:20),1)+...
        beta_group(2)*var(beta_RS_tree2(21:30),1)+...
        beta_group(3)*var(beta_RS_tree2(31:40),1)+...
        beta_group(4)*var(beta_RS_tree2(41:80),1);    
    T1_2=toc(Tstart);
    %check equi-sparsity
    figure(1);subplot(2,2,2);plot(beta_RS_tree2);title('Child L2')
    
    % RS-L1
    Tstart=tic;
    option.mu=1e-02;   % Smoothing Parameter, fixed
    option.gammarange=(0.5:.5:10)*1E-2;
    option.fig=1;
    [coeff_RS3,~,gamma_opt3,CV3] = cv_SPG_cvrt('group',Ytrain_c, Xtrain*A_tree,[], C_tree3, CNorm_tree3, option,g_idx_tree3); 
    beta_RS_tree3=A_tree*coeff_RS3;
    ES3=beta_group(1)*var(beta_RS_tree3(1:20),1)+...
        beta_group(2)*var(beta_RS_tree3(21:30),1)+...
        beta_group(3)*var(beta_RS_tree3(31:40),1)+...
        beta_group(4)*var(beta_RS_tree3(41:80),1);    
    T1_3=toc(Tstart);
    %check equi-sparsity
    figure(1);subplot(2,2,3);plot(beta_RS_tree3);title('Node L1')
    
    
    % RS-ES
    Tstart=tic;
    option.mu=1e-02;   % Smoothing Parameter, Carefully choose mu, a larger mu may not converge while a smaller one  leads to slow convergence
    option.gammarange=(0.25:0.25:5)*1E-3;
    option.fig=1;
    [beta_RS_graph,~,gamma_opt,CV] = cv_SPG_cvrt('graph',Ytrain, Xtrain,[], C_graph, CNorm_graph, option); 
    ES4=beta_group(1)*var(beta_RS_graph(1:20),1)+...
        beta_group(2)*var(beta_RS_graph(21:30),1)+...
        beta_group(3)*var(beta_RS_graph(31:40),1)+...
        beta_group(4)*var(beta_RS_graph(41:80),1);    
    T2=toc(Tstart);
    %check equi-sparsity
    figure(1);subplot(2,2,4);plot(beta_RS_graph);title('Equi-Sparsity')

    
    
    
    % CLR regression with lasso (with CV tuning selection)
    % (a naive competitor created by myself)
    Tstart=tic;
    Xtrain_CLR=log(Xtrain_nozero)-mean(log(Xtrain_nozero),2); % CLR 
    [beta_naive,info]=lasso(Xtrain_CLR,Ytrain,'cv',5); % lasso with CV for tuning selection
    beta_Lasso=beta_naive(:,info.IndexMinMSE);
    beta_Lasso0=info.Intercept(info.IndexMinMSE);
    T3=toc(Tstart);
    
    % implementation of LC with lasso -- R package CVS (Lin et al, 2014)
    % implementation of KPR -- R package KPR (Randolph et al, 2018)
    % Use *5-fold CV* and *10-value tuning range* if possible.

    
    
    
    
    % compare
    % in-sample residual
    res_RS_tree1=Ytrain_c-Xtrain*beta_RS_tree1;
    res_RS_tree2=Ytrain_c-Xtrain*beta_RS_tree2;
    res_RS_tree3=Ytrain_c-Xtrain*beta_RS_tree3;
    res_RS_graph=Ytrain-Xtrain*beta_RS_graph;
    res_Lasso=Ytrain-beta_Lasso0-Xtrain_CLR*beta_Lasso;
    rec_RSS(isim,:)=[sum(res_RS_tree1.^2),sum(res_RS_tree2.^2),sum(res_RS_tree3.^2),sum(res_RS_graph.^2),sum(res_Lasso.^2)];
    % out-sample prediction
    Xtest_CLR=log(Xtest_nozero)-mean(log(Xtest_nozero),2); % CLR
    rec_PMSE(isim,:)=[sum((Ytest-Ymean-Xtest*beta_RS_tree1).^2),sum((Ytest-Ymean-Xtest*beta_RS_tree2).^2),sum((Ytest-Ymean-Xtest*beta_RS_tree3).^2),...
        sum((Ytest-Xtest*beta_RS_graph).^2),sum((Ytest-Xtest_CLR*beta_Lasso-beta_Lasso0).^2)];
    % time
    rec_time(isim,:)=[T1_1,T1_2,T1_3,T2,T3];
    % equi-sparsity
    rec_es(isim,:)=[ES1,ES2,ES3,ES4];
        
    % save data for R (no-zero compositional data)
    Xtrain_forR(:,:,isim)=Xtrain_nozero;
    Xtest_forR(:,:,isim)=Xtest_nozero;
    Ytrain_forR(:,isim)=Ytrain;
    Ytest_forR(:,isim)=Ytest;
end
save('Sim2.mat','Xtrain_forR','Xtest_forR','Ytrain_forR','Ytest_forR','rec_RSS','rec_PMSE','rec_es', 'rec_time','taxonomy');
