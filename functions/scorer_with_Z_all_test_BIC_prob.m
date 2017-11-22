function [fun_c] = scorer_with_Z_all_test_BIC_prob(X,Y,W,Z,Phi,pi,Omega,opts,prox_param)

    [fun_c,Z_idx,rho,Omega] = eval_with_Z_prob(X,Y,W,Phi,pi,Omega,opts);
    
    n = size(X,1);
    k = opts.k;
    de = 2*k-1 ;
   
    for r = 1:k
    tmp = Z{r};
    de = de + sum(abs(tmp(:))>eps);
    end
    
    fun_c = 2  * fun_c - log(n) * de/n;
    
    

end