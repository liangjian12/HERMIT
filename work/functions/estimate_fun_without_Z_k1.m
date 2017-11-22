function [fun_c] = estimate_fun_without_Z_k1(X,Y,W,Phi,pi,Omega,opts)

    n = size(X,1);
    
    [sum_L,~] = compute_rho_r_without_Z(X,Y,W(:,:,1),Phi(:,:,1),Omega,opts);        
  
    fun_c = sum(sum_L(:))/n;

end