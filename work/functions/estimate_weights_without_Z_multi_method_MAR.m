function [alpha] = estimate_weights_without_Z_multi_method_MAR(X,Y,W,Phi,pi,pi_each,rho,Omega,opts,method_name)
% 
    k = opts.k;
    m = size(Y,2);
   
    n = size(X,1);
  
    L = zeros(n,m,k);
    sum_L = zeros(n,k);
    for r = 1:k    
        [sum_L(:,r),L(:,:,r)] = compute_rho_r_without_Z(X,Y,W(:,:,r),Phi(:,:,r),Omega,opts);        
    end
    L = permute(L,[1 3 2]);
    alpha = zeros(m);
    for i = 1:m
        for j = 1:m
            rho_i  = loss2rho(L(:,:,i),pi_each{i},k);
            rho_j  = loss2rho(L(:,:,j),pi_each{j},k);
            alpha(i,j) = - prob_distance(rho_i,rho_j,method_name);
        end    
    end
    
      
   
end