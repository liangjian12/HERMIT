function [ fun] = estimate_fun_giv_rho_without_Z(X,Y,W,Phi,pi,rho,Omega,opts)

 
    k = opts.k;
    n = size(Y,1);
 
    L = {};
    for r = 1:k    
        [~,L{r}] = compute_rho_r_without_Z(X,Y,W(:,:,r),Phi(:,:,r),Omega,opts);        
    end
 
    tmp = []; 
    for r = 1:k           
        tmp(r) = sum(  (rho(:,r))'* L{r}  ) + log(pi(r)+eps) * sum(rho(:,r));
    end
    fun = sum(tmp)/n;
    fun = fun  - sum(sum(rho.*log(max(rho,eps))))/n;   


end