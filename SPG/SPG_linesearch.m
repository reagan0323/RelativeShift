function [beta, obj, density, iter,  time, L]=SPG_linesearch(prob, Y, X, gamma, lambda, C, option, g_idx)
% Smoothing Proximal Gradient based on FISTA for large-scale problem
% with line search on Lipschitz constant
% For large-scale problems

% problem to solve either 'group' or 'graph'
% Y output
% X inputs
% gamma:  regularization for group norm
% lambda: regularization for L1 penalty 
% C:  \sum_|g| by J or |E| by J matrix where J is the number of features
% option: maxiter; tol, b0 ; 
% g_idx: n_group by 2, group index

    if (~strcmpi(prob,'group') && ~strcmpi(prob, 'graph'))
        error('Please input a correct problem name for the solver, either group or graph!')
    end

    [N,J] = size(X);
    
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
    L=10;    
    
    tic;  
    
    for iter=1:maxiter
        
        if (strcmpi(prob, 'group'))
            A=shrink_group(C*w/mu, g_idx);
            h_w=sum((Y-X*w).^2)/2+cal2norm(C*w, g_idx);
        else 
            A=hard_threshold(full(C*w/mu), 1);
            h_w=sum((Y-X*w).^2)/2+sum(abs(C*w));
        end 
        
        
        grad=X'*(X*w-Y)+C'*A;        
        
        while (true)       
            [beta_new, density(iter)]=soft_threshold(w-(1/L)*grad, lambda/L);
            density(iter)=density(iter)/J;
            if (strcmpi(prob, 'group'))
                h_beta=sum((Y-X*beta_new).^2)/2+cal2norm(C*beta_new, g_idx);
            else 
                h_beta=sum((Y-X*beta_new).^2)/2+sum(abs(C*beta_new));
            end 
            Q=h_w+(beta_new-w)'*grad+L/2*sum((beta_new-w).^2);
            if (h_beta<=Q)
                break;
            else 
                L=2*L;
            end
        end   
                
        theta_new=(sqrt(theta^4+4*theta^2)-theta^2)/2;
        
        w=beta_new+(1-theta)/theta*theta_new*(beta_new-beta);
        
        obj(iter)=h_beta+lambda*sum(abs(beta_new(:)));
        
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