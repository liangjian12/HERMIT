function [sum_L,L] = compute_rho_r_only_Z(X,Y,W,Z,Phi,Omega,opts)

G =  Z;

L = zeros(size(Y));

for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end    
    [L(:,idx)] = link_fun(Y(:,idx),G(:,idx),Phi(idx,idx),Omega,opts.task_type_name{label_type});    
end

L = L.*Omega;

sum_L = sum(L,2);



end