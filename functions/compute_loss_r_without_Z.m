function [L] = compute_loss_r_without_Z(X,Y,W,Phi,rho,Omega,opts)

G = X*W;

L = zeros(size(Y));

for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end    
    [L(:,idx)] = link_fun(Y(:,idx),G(:,idx),Phi(idx,idx),opts.task_type_name{label_type});    
end

L = L.*Omega;

L = sum(L,2);
L = sum(L.*rho);

 



end