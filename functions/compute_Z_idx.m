function [Z_idx] = compute_Z_idx(X,Y,W,Phi,Omega,opts)

     
 
 
th_prob = opts.th_prob;
Y = Y.* Omega;
Z_idx = ones(size(Y));
Prob_all = zeros(size(Z_idx));
for r = 1:opts.k
    G = X * W{r};
    Prob = zeros(size(Z_idx));
    for i_type = 1:3
        idx =  opts.task_type == i_type;
        if sum(idx) == 0
            continue
        end        
        mu = link_fun_mu(G(:,idx), opts.task_type_name{i_type});
        Y_tmp = Y(:,idx);
        if i_type == 1
            sigma =  ones(size(Y_tmp))/Phi{r}(idx,idx);
            mu = mu/Phi{r}(idx,idx);
            prob = normcdf(Y_tmp,mu,sigma);
            prob = 0.5 - abs(0.5-prob);
        elseif i_type == 2
            prob = mu.^(Y_tmp) .* (1-mu).^(1-Y_tmp);
            prob = 0.5 - abs(0.5-prob);
        else
            prob = poisscdf(Y_tmp,mu);
            prob = 0.5 - abs(0.5-prob);
        end
        Prob(:,idx) = prob;

    end
   

    Z_idx  = Z_idx.*(Prob<th_prob+eps);
end 

Z_idx = Z_idx.*Omega;



end