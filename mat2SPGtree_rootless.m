function [A,    C,CNorm,g_idx,   node,M2,  T,Tw]=mat2SPGtree_rootless(M, penalty)
% This function translate a tree structure (parent node vector without
% redundant nodes from mat2node.m function) to two sets of parameters: the
% group membership parameter C, CNorm, g_idx (induced from T and Tw), and RS model expansion matrix A.
%
% Input
%    M      p*L matrix, each row corresponds to a variable at the
%           finest level, each column corresponds to an ordered taxonomic level; the
%           entry values in each column are the unique ID of the variable 
%           at that level. As we move to the right, the # of unique values become fewer
%
%   penalty     1 (default): overlapping group penalty, L2 of all descendant 
%                            nodes for each internal node
%               2: non-overlapping group penalty, L2 of direct child nodes
%                  for each internal node
%               3: lasso penalty for each node (excluding root node)
%    
% output
%     A         #leaf * (#node-1) binary expansion matrix (for tree regression reparameterization)
%
%     C         sum(group size) * (#node-1),   very tall, sparse matrix (for SPG)
%     CNorm     the norm of C, as defined in the SPG paper (for SPG)
%     g_idx     #groups*3 matrix, starting row in C of a group, end row of a
%               group, group size
%
%     node      1*#node vector, [output of mat2node], a parent node vector with p leaf nodes; NO redundant internal 
%               nodes; use treeplot to show the actural tree structure
%     M2        p*L or p*(L+1) node index matrix, [output of mat2node], similar to M, but cleaner,
%               with index going from 1 to #nodes (root node has index equal to #node). 
%
%     T         #groups * (#node-1) matrix, group index matrix, each row is
%               a group, the column order is the same as in node or M2
%               For instance, if penalty=1 or 2, T has size
%               (#node-#leaf)*(#node-1); if penalty=3, T has size
%               (#node-1)*(#node-1)
%     Tw        length(nrow(T)) vector,  weight for each group, default is
%               a vector of 1
%
% Adapted from mat2SPGtree by Gen Li, 4/30/2020
% Updated on 8/20/2020 by GL
% --- remove root node from parameterization
% --- add option of L1 penalty on each node (excluding root node)
% --- add option of non-overlapping group lasso penalty on direct child
%     nodes of each internal node



% call mat2node to get tree vector
[node, M2] = mat2node(M); % M2 is the legit tree structure matrix
p=size(M,1);


numall=length(node); % number of nodes in T
numint=numall-p; % number of internal nodes (including root)


% expansion matrix A
A=zeros(p,numall);
for j=1:p % leaf node/variable in model
    % first, get all parents of the leaf node
    parent=unique(M2(j,:));
    A(j,parent)=1;
end
% remove root node
A=A(:,1:(numall-1));



% membership matrix T 
if penalty==1 % overlapping group lasso of all descendant nodes
    T=zeros(numint,numall-1); 
    for j=1:numint % group index
        nodeindex=j+p; % index of internal nodes
        [ix,iy]=ind2sub(size(M2),find(M2==nodeindex));
        iy1=max(iy);
        ix1=ix(iy==iy1);
        child=setdiff(M2(ix1,1:iy1),nodeindex); % all descendants of nodeindex
        T(j,child)=1;
    end
elseif penalty==2 % group lasso of direct child nodes
    T=zeros(numint,numall-1);
    for j=1:numint
        nodeindex=j+p;
        [ix,iy]=ind2sub(size(M2),find(M2==nodeindex));
        iy1=min(iy);
        ix1=ix(iy==iy1);
        child=unique(M2(ix1,iy1-1)); % direct children of nodeindex
        T(j,child)=1;
    end
elseif penalty==3 % lasso of non-root node (may not be most efficient to use SPG)
    T=eye(numall-1);
end
    
    
    
    
% weight for different node groups
Tw=ones(size(T,1),1); % default is a vector of 1


% SPG required input 
[C, g_idx, CNorm] = pre_group(T, Tw); % from SPG

end








function [node, M2] = mat2node(M) 
% this function converts a matrix M representing tree structure to a tree
% parent-node vector and a standarded matrix M2 (with node index from 1 to numnode)
% 
% Input:
%   M       p*L matrix, each row corresponds to a variable at the
%           finest level, each column corresponds to an ordered taxonomic level; the
%           entry values in each column are the unique ID of the variable 
%           at that level. As we move to the right, the # of unique values become fewer
%
% Output
%   node     a parent node vector with p leaf nodes; NO redundant internal 
%            nodes; use treeplot to show the actural tree structure
%
%   M2       p*L node index matrix, standardized, with index going from 1 to # of nodes. 
%
% Note: to use the function, first convert the p*L matrix of taxa names at
% different taxonomic levels to a column-by-column unique ID matrix. For
% example, 
%
%   A  AB  ABC  ABCDE       1  1  1  1
%   B  AB  ABC  ABCDE       2  1  1  1
%   C  CC  ABC  ABCDE  -->  3  2  1  1
%   D  DD  DDE  ABCDE       4  3  2  1
%   E  EE  DDE  ABCDE       5  4  2  1
%   
% By Gen Li, 4/30/2020

[p,L]=size(M);

% check if the finest level has unique taxa
if length(unique(M(:,1)))<p
    error('The first level has overlapping taxa! Terminated..');
end
% check if M is really a tree
for j=1:(L-1)
    uniqid=unique(M(:,j));
    for k=1:length(uniqid)
        if length(unique(M(M(:,j)==uniqid(k),j+1)))>1 % check if nested
            error('The input matrix does not have a tree structure! Terminated..');
        end
    end
end
        


if length(unique(M(:,L)))~=1 % if the last level is not all equal, add a root node
    warning('Adding a root node on top of the highest taxonomic level...')
    M=[M,ones(p,1)];
    L=L+1;
end


% Parent node vector with redundant nodes in the tree
M1=M; % node index matrix, index goes from 1 to # of nodes
M1(:,1)=1:p;
for j=2:L
    [~,M1(:,j)]=ismember(M(:,j),unique(M(:,j))); % each column contains 1,2,3,... # unique groups; 1st column has 1~p, last column is all 1
    M1(:,j)=M1(:,j)+max(M1(:,j-1)); % now the numbers are unique node indices, the last column values should be the total number of nodes in the tree
end
total_redun=M1(1,L); % total number of nodes in the redundant tree
% convert to a parent node vector
node_redun=zeros(1,total_redun);
for i=1:(total_redun-1)
    [ind_i,ind_j] = ind2sub(size(M1),find(M1==i,1));
    node_redun(i)=M1(ind_i,ind_j+1); % the parent of node i is the node index to its right
end

% figure(1);clf;
% treeplot(node_redun);
% title('Original tree (with nodes at each taxonomic level)')


% A trimmed/standardized version, parent node vector without redundancy
M2=M; % node index matrix, index goes from 1 to # of nodes
M2(:,1)=1:p;
for j=2:L
    [~,M2(:,j)]=ismember(M(:,j),unique(M(:,j))); % each column contains 1,2,3,... # unique groups; 1st column has 1~p, last column is all 1
    tempseq=M2(:,j);
    for k=max(tempseq):(-1):1
        if length(unique(M2(find(tempseq==k),j-1)))==1 % no new node joining
            tempseq(find(tempseq==k))=0; 
            tempseq(tempseq>k)=tempseq(tempseq>k)-1;
        end
    end
    tempseq(tempseq~=0)=tempseq(tempseq~=0)+max(M2(:,j-1)); % unique new node index
    tempseq(tempseq==0)=M2(tempseq==0,j-1); % carry over
    M2(:,j)=tempseq;
end
total=M2(1,L); % total number of nodes
% convert to a parent node vector
node=zeros(1,total);
for i=1:(total-1)
    [ind_i,ind_j] = ind2sub(size(M2),find(M2==i,1,'last'));
    node(i)=M2(ind_i,ind_j+1); % the parent of node i is the node index to its right
end

% figure(2);clf;
% treeplot(node);
% title('Trimmed tree (without redundant nodes) -- ready for analysis')

end




function [C, g_idx, TauNorm] = pre_group(T, Tw)

    [V,K] = size(T);       
    sum_col_T=full(sum(T,2));
    SV=sum(sum_col_T);
    csum=cumsum(sum_col_T);
    g_idx=[[1;csum(1:end-1)+1], csum, sum_col_T]; %each row is the range of the group
    
    J=zeros(SV,1);
    W=zeros(SV,1);
    for v=1:V
       J(g_idx(v,1):g_idx(v,2))=find(T(v,:));
       W(g_idx(v,1):g_idx(v,2))=Tw(v);
    end 

    C=sparse(1:SV, J, W, SV, K); 
    
    TauNorm=spdiags(Tw(:), 0, V, V)*T;
    TauNorm=full(max(sum(TauNorm.^2)));      

end

