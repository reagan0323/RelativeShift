function [beta, obj, density, iter,  time]=SPG(prob, Y, X, gamma, lambda, C, CNorm, option, g_idx)
% Smoothing Proximal Gradient based on FISTA for medium-scale problem
% with a pre-computed Lipschitz constant (without line-search

% problem to solve either 'group' or 'graph'
% Y output
% X inputs
% gamma:  regularization for group norm
% lambda: regularization for L1 penalty 
% C:  \sum_|g| by J matrix or |E| by J matrix
% CNorm: ||C||^2
% option: maxiter; tol, b0 (initial est) ; 
% g_idx: n_group by 2, group index

    [N,J] = size(X);
    
    if (~strcmpi(prob,'group') && ~strcmpi(prob, 'graph'))
        error('Please input a correct problem name for the solver, either group or graph!')
    end
    
    if isfield(option,'maxiter')
        maxiter=option.maxiter;
    else
        maxiter=10000;
    end
    
    if isfield(option, 'verbose')
        verbose=option.verbose;
    else
        verbose=true;
    end  

    if isfield(option, 'display_iter')
        display_iter=option.display_iter;
    else
        display_iter=true;
    end  
        
    if isfield(option, 'tol')
        tol=option.tol;
    else
        tol=1e-7;
    end
    
    if isfield(option, 'b_init')
        beta=option.b_init;
    else
        beta=zeros(J,1);
    end  
    
    if isfield(option, 'mu')
        mu=option.mu;
    else
        mu=1e-3;
    end 
    
    obj=zeros(maxiter, 1);
    density=zeros(maxiter, 1);  %ratio of nonzero elements
    
    C=C*gamma;
    w=beta;
    theta=1;
    
    XX=X'*X;
    XY=X'*Y;
    
    if (J<10000)
        L=eigs(XX,1)+gamma^2*CNorm/mu;
    else
        L=sum(XX(:).^2)+gamma^2*CNorm/mu;
        clear XX XY 
    end
    
    tic;  
    
    for iter=1:maxiter
        
        if (strcmpi(prob, 'group'))               % overlap group lasso
           A=shrink_group(C*w/mu, g_idx);
        else                                     % graph_guided fused lasso
           A=hard_threshold(full(C*w/mu), 1);
        end
     
        if (J<2*N && J<10000)
            grad=XX*w-XY+C'*A;
        else 
            grad=X'*(X*w-Y)+C'*A;
        end
        
        [beta_new, density(iter)]=soft_threshold(w-(1/L)*grad, lambda/L);
        density(iter)=density(iter)/J;
                
        theta_new=(sqrt(theta^4+4*theta^2)-theta^2)/2;
        
        w=beta_new+(1-theta)/theta*theta_new*(beta_new-beta);
        
        if (strcmpi(prob, 'group'))
            obj(iter)=sum((Y-X*beta_new).^2)/2+cal2norm(C*beta_new, g_idx)+lambda*sum(abs(beta_new(:)));
        else 
            obj(iter)=sum((Y-X*beta_new).^2)/2+ sum(abs(C*beta_new))+lambda*sum(abs(beta_new(:)));
        end
        
        if (iter>10 && (abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<tol))
              break;
        end
              
        if (verbose && (iter==1 || mod(iter, display_iter)==0))
            fprintf('Iter %d, Obj: %g, density: %f\n', iter, obj(iter), density(iter));
        end
        
        beta=beta_new;
        theta=theta_new;
        
    end    
    time=toc;
    
    if (verbose)
        fprintf('Iter %d, Obj: %g, density: %f\n', iter, obj(iter), density(iter));
    end
    
    obj=obj(1:iter);
    density=density(1:iter);
end 