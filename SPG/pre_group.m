function [C, g_idx, TauNorm] = pre_group(T, Tw)

    [V K] = size(T);       
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

