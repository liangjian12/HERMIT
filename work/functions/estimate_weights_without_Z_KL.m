function [alpha] = estimate_weights_without_Z_KL(X,Y,W,Phi,pi,rho,Omega,opts)
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
        alpha(j)=-sum((KLDiv(rho_single,rho)+ KLDiv(rho,rho_single))/2)/n;
    end    
    
%     for j = 1:m
%         tmp = L ;
%         tmp(:,:,j) = [];
%         tmp = sum(tmp,3);
%         rho_single  = loss2rho(tmp,pi,k);
% %         alpha(j)=sum(  KLDiv(rho,rho_single))/n;
%         alpha(j)= -sum((KLDiv(rho_single,rho)+ KLDiv(rho,rho_single))/2)/n;
%     end    
     
   
end