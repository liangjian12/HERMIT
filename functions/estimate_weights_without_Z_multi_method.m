function [alpha] = estimate_weights_without_Z_multi_method(X,Y,W,Phi,pi,rho,Omega,opts,method_name)
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
        rho_single  = loss2rho(L(:,:,j),pi,k);
%         alpha(j)=-sum((KLDiv(rho_single,rho)+ KLDiv(rho,rho_single))/2)/n;
        alpha(j) =-prob_distance(rho,rho_single,method_name);
    end    
    
%     alpha = softmax(alpha')';
    
%     alpha = alpha - mean(alpha);
%     alpha = exp(alpha);
%     alpha = alpha/sum(alpha);
%      if any(isnan(alpha(:))) || any(isinf(alpha(:)))
%         alpha(isnan(alpha)) = 1/m;
%         alpha(isinf(alpha)) = 1/m;
%         alpha = alpha/sum(alpha);
%     end
%     alpha = max(alpha,0);
%     alpha = min(alpha,1);     
   
end