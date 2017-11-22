function [Y_pred] = FMR_predict_softmax_giv_rho(X,W,Phi,rho,pi,Omega,opts)
 
k = opts.k;

[n,d] = size(X);
[~,m,~] = size(W);
G = zeros([n,m,k]);

for r = 1:k
G(:,:,r) = X*W(:,:,r); 
end
 
Y_pred = zeros(n,m);
 
for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    mu = link_fun_mu(G(:,idx,:),opts.task_type_name{label_type});
    mu = permute(mu,[1 3 2]);
    mu = bsxfun(@times,mu,rho);
    mu = sum(mu,2);
    mu = squeeze(mu);
    
    Y_pred(:,idx) = mu;
    
end

end