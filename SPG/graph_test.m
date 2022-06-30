clear;

N=1000;      % sample size
J=500;

option.maxiter=10000;  
% max iteration 

option.mu=1e-1;   
% Smoothing Parameter
% Carefully choose mu, a larger mu may not converge while a smaller one 
% leads to slow convergence

option.verbose=true;    
option.display_iter=100;   
option.tol=1e-8;          % tolerance

[X, Y, w]=gentoy_graph(N, J); 
% generate toy data
% X: design matrix
% Y: output
% w: true regression coefficients

opts=struct('cortype', 1, 'corthreshold', 0.3);
[C, CNorm, E]=gennetwork(X,opts);
% C: C matrix 
% CNorm: spectral norm of C;

gamma=150;   % regularization parameter for group penalty 
lambda=150;  % regularization parameter for L1-norm

prob='graph';
      
[grad_beta,grad_obj,grad_density,grad_iter,grad_time] = ...
              SPG(prob, Y, X, gamma, lambda, C, CNorm, option);  
% SPG with a pre-compuated Lipschitz constant for medium-scale problems