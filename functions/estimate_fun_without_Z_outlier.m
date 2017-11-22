function [fun,rho,u,v] = estimate_fun_without_Z_outlier(X,Y,W,Phi,W_k1,Phi_k1,pi_outlier,pi,Omega,opts)
% 
    k = opts.k;
    m = size(Y,2);
    if k > 1
        n = size(X,1);
        rho = zeros(n,k);
        L = zeros(n,m,k);
 
        for r = 1:k    
            [~,L(:,:,r)] = compute_rho_r_without_Z(X,Y,W(:,:,r),Phi(:,:,r),Omega,opts);        
        end
          
        [~,L_outlier] = compute_rho_r_without_Z(X,Y,W_k1,Phi_k1,Omega,opts);    
        L_outlier = repmat(L_outlier,[1 1 k]);
        
        L = bsxfun(@plus,L,log(pi_outlier+eps));
        L_outlier = bsxfun(@times,L_outlier,log(1 - pi_outlier+eps));
        mean_L = (L + L_outlier)/2;
        L = L - mean_L;
        L_outlier = L_outlier - mean_L;
        a = exp(L);
        b = exp(L_outlier);
        
%         a = bsxfun(@times,exp(L),pi_outlier);
%         b = bsxfun(@times,exp(L_outlier),1 - pi_outlier);
        c = a + b;
        L = log(c);
        sum_L = squeeze(sum(L,2));
        
        likelihood = exp(sum_L);
        likelihood = bsxfun(@times,likelihood,pi);    
%         fun  = sum(log(sum(likelihood,2)))/n;      
        
        if false %isfield(opts,'iter_out_gem') && opts.iter_out_gem < 25
            [~,idx_max] = max(sum_L,[],2);
            rho = full(sparse([1:n]',idx_max,1));
        else
            mean_L =  mean(sum_L,2);
            sum_L = bsxfun(@minus,sum_L,mean_L);
            rho = exp(sum_L);
            rho = bsxfun(@times,rho,pi);    
            rho = bsxfun(@rdivide,rho,sum(rho,2)); 
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
            tmp(r) = sum(  (rho(:,r))'* L(:,:,r)  ) + log(pi(r)+eps) * sum(rho(:,r));
        end
        fun = sum(tmp)/n;
        fun = fun  - sum(sum(rho.*log(max(rho,eps))))/n;   
        
        w = permute(rho,[1,3,2]);
        u = bsxfun(@times,a./c,w);
        v = bsxfun(@plus,-u,w);
        
    else
        n = size(X,1);   
        [sum_L,~] = compute_rho_r_without_Z(X,Y,W(:,:,1),Phi(:,:,1),Omega,opts);        
        fun = sum(sum_L(:))/n;
        rho = ones(n,1);      
        u = rho;
        v = rho;
    end

end