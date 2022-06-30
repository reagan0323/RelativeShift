clear;

% overlapping group lasso structure 

ng=50;  % number of groups 
g_size=100;  % group size
overlap=10;  % number of variables overlapped between two consecutive groups
n=1000;      % sample size
p=ng*(g_size-overlap)+overlap;   % totoal number of variables 

option.maxiter=10000;  
% max iteration 

option.mu=1e-3;   
% Smoothing Parameter
% Carefully choose mu, a larger mu may not converge while a smaller one 
% leads to slow convergence

option.verbose=true;    
option.display_iter=10;   
option.tol=1e-8;          % tolerance

[X, Y, T, Tw, w]=gentoy_group(n, ng, g_size, overlap); 
% generate toy data
% X: design matrix
% Y: output
        % T: 0/1 sparse matrix: # of groups by # of features,
        % indicate each group contains which variables 
        % Tw: # of groups by 1 column vector,  weight for each group
% w: true regression coefficients

[C, g_idx, CNorm] = pre_group(T, Tw); 
% C: C matrix 
% g_idx: #groups*3 matrix, group index for the rows of C, i.e., [start ind, end ind, group size]
% CNorm: 

gamma=ng/10;   % regularization parameter for group penalty 
lambda=ng/10;  % regularization parameter for L1-norm

prob='group';
      
[grad_beta,grad_obj,grad_density,grad_iter,grad_time] = ...
             SPG(prob, Y, X, gamma, lambda, C, CNorm, option, g_idx);  
% SPG with a pre-compuated Lipschitz constant for medium-scale problems

[grad_beta,grad_obj,grad_density,grad_iter,grad_time, L] = ...
              SPG_linesearch(prob, Y, X, gamma, lambda, C,option, g_idx);  
% SPG with line search on Lipschitz constant for large-scale problems
