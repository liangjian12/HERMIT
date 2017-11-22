function [fun,rho] = estimate_fun_without_Z_MOE(X,Y,W,Phi,W_gate,Omega,opts)
% 
    k = opts.k;
    m = size(Y,2);
    if k > 1
        n = size(X,1);
        rho = zeros(n,k);
        L = {};
        sum_L = zeros(size(rho));
        for r = 1:k    
            [sum_L(:,r),L{r}] = compute_rho_r_without_Z(X,Y,W(:,:,r),Phi(:,:,r),Omega,opts);        
        end
         
        
        G = X*W_gate;
        mean_G  =  mean(G ,2);
        G  = bsxfun(@minus,G ,mean_G );
        rho_gate = softmax(G')';
        
        if false %isfield(opts,'iter_out_gem') && opts.iter_out_gem < 25
            [~,idx_max] = max(sum_L,[],2);
            rho = full(sparse([1:n]',idx_max,1));
        else
%             mean_L =  mean(sum_L,2);
%             sum_L = bsxfun(@minus,sum_L,mean_L);
%             rho = exp(sum_L);
%             rho = rho.*rho_gate;   
%             rho = bsxfun(@rdivide,rho,sum(rho,2)); 
            
            
            sum_L = sum_L + log(max(rho_gate,eps));
            mean_L =  mean(sum_L,2);
            sum_L = bsxfun(@minus,sum_L,mean_L);
            rho = softmax(sum_L')';
            
            
            if any(isnan(rho(:))) || any(isinf(rho(:)))
                rho(isnan(rho)) = 1/k;
                rho(isinf(rho)) = 1/k;
                rho = bsxfun(@rdivide,rho,sum(rho,2)); 
            end
            rho = max(rho,0);
            rho = min(rho,1);        
        end
        
        pi = sum(rho_gate);
        pi = pi/sum(pi);

        tmp = []; 
        for r = 1:k           
            tmp(r) = sum(  (rho(:,r))'* L{r}  ) + log(pi(r)+eps) * sum(rho(:,r));
        end
        fun = sum(tmp)/n;
        fun = fun  - sum(sum(rho.*log(max(rho,eps))))/n;   
    else
        n = size(X,1);   
        [sum_L,~] = compute_rho_r_without_Z(X,Y,W(:,:,1),Phi(:,:,1),Omega,opts);        
        fun = sum(sum_L(:))/n;
        rho = ones(n,1);      
    end

end