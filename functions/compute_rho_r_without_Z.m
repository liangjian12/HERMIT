function [sum_L,L] = compute_rho_r_without_Z(X,Y,W,Phi,Omega,opts)

G = X*W;

% G = minmaxbound(G,'user',1e-40,1e40);

L = zeros(size(Y));

for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end    
    [L(:,idx)] = link_fun(Y(:,idx),G(:,idx),Phi(idx,idx),Omega(:,idx),opts.task_type_name{label_type});    
end

L = L.*Omega;

% L = L/size(Y,2);
 
sum_L = sum(L,2); 


end