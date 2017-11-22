function [Z_idx,Y_outlier] = compute_outlier(X,W,Z,Phi,rho,opts)

    [~,max_idx] = max(rho,[],2);
    Y_outlier = zeros(size(Z{1}));
    Z_idx = ones(size(Z{1}));
    for r = 1:opts.k
        G = X * W(:,:,r) + Z(:,:,r);
        idx_r = max_idx == r;
        for i_type = 1:3
            idx =  opts.task_type == i_type;
            if sum(idx) == 0
                continue
            end        
            mu = link_fun_mu(G(:,idx), opts.task_type_name{i_type});
            Y_outlier(idx_r,idx) = link_fun_rnd_mean_gauss_no_err(mu(idx_r,:),Phi(idx,idx,r), opts.task_type_name{i_type});
             
        end
        Y_outlier(idx_r,:) = Y_outlier(idx_r,:).*(abs(Z(idx_r,:,r))>eps);
        Z_idx(idx_r,:) = Z_idx(idx_r,:).*(abs(Z(idx_r,:,r))>eps);
    end

end