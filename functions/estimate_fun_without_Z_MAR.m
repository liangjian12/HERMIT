function [fun,rho] = estimate_fun_without_Z_MAR(X,Y,W,Phi,pi,Omega,opts)
% 
    k = opts.k;
  
    if k > 1
        n = size(X,1);
        rho = zeros(n,k);
        L = {};
        sum_L = zeros(size(rho));
        for r = 1:k    
            [sum_L(:,r),L{r}] = compute_rho_r_without_Z_MAR(X,Y,W(:,r),Phi(r),Omega,opts);        
        end
%         likelihood = exp(sum_L);
%         likelihood = bsxfun(@times,likelihood,pi);    
%         fun  = sum(log(sum(likelihood,2)))/n;      
        
        if false %isfield(opts,'iter_out_gem') && opts.iter_out_gem < 25
            [~,idx_max] = max(sum_L,[],2);
            rho = full(sparse([1:n]',idx_max,1));
        else
            sum_L = bsxfun(@plus,sum_L,log(pi+eps));
            mean_L =  mean(sum_L,2);
            sum_L = bsxfun(@minus,sum_L,mean_L);
            rho = softmax(sum_L')';
            
%             rho = exp(sum_L);
%             rho = bsxfun(@times,rho,pi);    
%             rho = bsxfun(@rdivide,rho,sum(rho,2)); 
            if any(isnan(rho(:))) || any(isinf(rho(:)))
                rho(isnan(rho)) = 1/k;
                rho(isinf(rho)) = 1/k;
                rho = bsxfun(@rdivide,rho,sum(rho,2)); 
            end
            rho = max(rho,0);
            rho = min(rho,1);        
        end

        tmp = []; 
        for r = 1:k           
            tmp(r) = sum(  (rho(:,r))'* L{r}  ) + log(pi(r)+eps) * sum(rho(:,r));
        end
        fun = sum(tmp)/n;
        fun = fun  - sum(sum(rho.*log(max(rho,eps))))/n;   
    else
        n = size(X,1);   
        [sum_L,~] = compute_rho_r_without_Z(X,Y,W(:,1),Phi(1),Omega,opts);        
        fun = sum(sum_L(:))/n;
        rho = ones(n,1);      
    end

end