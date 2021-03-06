function [fun,rho] = estimate_fun_only_Z(X,Y,W,Z,Phi,pi,Omega,opts)

    k = opts.k;
    n = size(X,1);
    m = size(Y,2);
    if k > 1
        rho = zeros(n,k);
        L = {};

        sum_L = zeros(size(rho));
        for r = 1:k    
            [sum_L(:,r),L{r}] = compute_rho_r_only_Z(X,Y,W(:,:,r),Z(:,:,r),Phi(:,:,r),Omega,opts);        
        end

        mean_L =  mean(sum_L,2);
        sum_L = bsxfun(@minus,sum_L,mean_L);

        rho = exp(sum_L);
        rho = bsxfun(@times,rho,pi);    

        rho = bsxfun(@rdivide,rho,sum(rho,2)); 

        if any(isnan(rho(:))) || any(isinf(rho(:)))
            rho(isnan(rho)) = 1/m;
            rho(isinf(rho)) = 1/m;
            rho = bsxfun(@rdivide,rho,sum(rho,2)); 
        end

        rho = max(rho,0);
        rho = min(rho,1);

        tmp = []; 
        for r = 1:k           
            tmp(r) = sum(  (rho(:,r))'* L{r}  ) + log(pi(r)+eps) * sum(rho(:,r));
        end
        fun = sum(tmp)/n;
        fun = fun  - sum(sum(rho.*log(max(rho,eps))))/n;
    else
        n = size(X,1);   
        [sum_L,~] =  compute_rho_r_only_Z(X,Y,W(:,:,1),Z(:,:,1),Phi(:,:,1),Omega,opts);   
        fun = sum(sum_L(:))/n;
        rho = ones(n,1);      
    end

end