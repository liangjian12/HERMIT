function [fun_c] = scorer_with_Z_all_test_BIC(X,Y,W,Phi,pi,Omega,opts,prox_param)

    [fun_c,Z,rho] = eval_with_Z(X,Y,W,Phi,pi,Omega,opts,prox_param);
    
    n = size(X,1);
    k = opts.k;
    de = 2*k-1 ;
    for r = 1:k
        tmp = abs(W{r})>eps;
        de = de + sum(tmp(:)); 
    end
    for r = 1:k
        tmp = abs(Z{r})>eps;
        de = de + sum(tmp(:)); 
    end
    
    fun_c = 2  * fun_c - log(n) * de /n;
    
    

end