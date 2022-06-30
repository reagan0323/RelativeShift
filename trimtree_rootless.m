function [beta_RS, coef_trim ,fusion]=trimtree_rootless(M2,coef,thres)
% This function further trim the result of RS-tree to get exact clustering
%
%
% Input
%    M2     standardized tree matrix from mat2SPGtree, matched with coef
%   coef    nodenum*1 coeffcient vector, output of SPG, the order of coef
%           matches with the node index in M2
%   thres   user-defined threshold for fusion. if not defined, will use
%           median of leaf node values as default
%    
% output
%     beta_RS    p*1 fused coefficient vector
%     coef_trim  nodenum*1 trimmed coefficient vector, with group zeros
%     fusion     q*p binary fusion index, each row indicate a fusion group,
%                each column corresponds to a variable in coef_RS
%                non-overlapping for different rows 
%
% Gen Li, 5/6/2020




[A,~,~,~,node,M2temp,T,~]=mat2SPGtree_rootless(M2,1);
% check
if prod(M2temp(:)==M2(:))~=1 
    error('The input tree matrix is not a standardized tree from mat2SPGtree! Terminated..')
end
p=size(M2,1);

numnode=max(M2(:));
if length(coef)~=numnode-1
    error('The tree matrix is not matched with input coefficient vector!')
end


if nargin == 2    % no thres defined
  thres = median(coef(1:p)); % use median of leaf nodes as basic threshold
end



% trim tree from bottom up
coef_trim=coef;
child_comb_ind=[ones(p,1);zeros(numnode-p,1)]; % for each node, whether its children are combined
for i=(p+1):numnode
    grp=(find(node==i)); % the direct children of node i
    if prod(child_comb_ind(grp))==0 % uncombinable
        child_comb_ind(i)=0;
    elseif norm(coef_trim(grp),'fro')>thres % only need to check direct children b/c other children are all zero by def
        child_comb_ind(i)=0;
    else % actual combination
        coef_trim(grp)=0; % combinable, shrink children node values to 0
        child_comb_ind(i)=1; 
    end
end

% trimmed length-p coefficient vector
beta_RS=A*coef_trim;


% fusion matrix 
fusion=[];
if child_comb_ind(numnode)==1 % all combined to the root
    fusion=ones(1,p);
else
    for i=(p+1):(numnode-1) % node index
        if child_comb_ind(i)==0 || child_comb_ind(node(i))==1
            continue
        else % this is the highest node whose children are combined
            fusion=[fusion;T(i-p,1:p)]; % all child leaf nodes
        end
    end
end
    
        
