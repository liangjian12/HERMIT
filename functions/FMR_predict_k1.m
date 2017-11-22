function [Y_pred] = FMR_predict_k1(X,W,Phi,pi,Omega,opts)
[n,d] = size(X);
[~,m] = size(W);

G = X*W(:,:,1);
n = size(X,1);
Y_pred = zeros(n,m); 
for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    mu = link_fun_mu(G(:,idx),opts.task_type_name{label_type});
%     y_pred = link_fun_rnd_mean_gauss_no_err(mu,Phi{1}(idx,idx),opts.task_type_name{label_type});
    Y_pred(:,idx) = mu;
     

end