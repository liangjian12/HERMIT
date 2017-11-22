function [alpha] = estimate_weights_without_Z(X,Y,W,Phi,pi,rho,Omega,opts)
% 
    k = opts.k;
    m = size(Y,2);
   
    n = size(X,1);
  
    L = zeros(n,m,k);
    sum_L = zeros(size(rho));
    for r = 1:k    
        [sum_L(:,r),L(:,:,r)] = compute_rho_r_without_Z(X,Y,W(:,:,r),Phi(:,:,r),Omega,opts);        
    end
    L = permute(L,[1 3 2]);
    alpha = zeros(1,m);
    for j = 1:m
       alpha(j) =  sum(sum(rho.*L(:,:,j)))/n;
    end
    alpha = alpha - mean(alpha);
    alpha = exp(alpha);
    alpha = alpha/sum(alpha);
     if any(isnan(alpha(:))) || any(isinf(alpha(:)))
        alpha(isnan(alpha)) = 1/m;
        alpha(isinf(alpha)) = 1/m;
        alpha = alpha/sum(alpha);
    end
    alpha = max(alpha,0);
    alpha = min(alpha,1);     
   
end