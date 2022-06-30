function [X,Y,w]=gentoy_graph(N,J,s)

    if ~exist('s', 'var')
        s=0.7;
    end
    
    Sigma=s.^(abs(repmat(1:J, J, 1)-repmat((1:J)', 1, J)));
    X=zscore(mvnrnd(zeros(1,J), Sigma, N));
    
    w=randn(J,1);
    w(w(:)<median(w(:)))=0;
    
    sn2 = 1;  % signal to noise ratio
    Y=X*w+sn2*rand(N,1);     

end
