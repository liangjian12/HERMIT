function [best_param,score,best_score] = choose_param_on_valid_with_Z_rank(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param_range,solver_with_Z,solver_without_Z,scorer,parfor_flag,iter_num,th)  
n = length(param_range);
score = zeros(iter_num,n);
if parfor_flag
parfor i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    opts.tau = 0.01;
    for iter = 1:iter_num
        [W,Z,Phi,pi] = solver_with_Z(train_X,train_Y,train_Omega,k,opts,prox_param);
        score_outlier = sum(sum(abs(Z),3),2);
        [~,idx_tmp] = sort(score_outlier);
        [W,Phi,pi] = solver_without_Z(train_X(idx_tmp(1:round(size(train_X,1)*0.8)),:),train_Y(idx_tmp(1:round(size(train_X,1)*0.8)),:),train_Omega(idx_tmp(1:round(size(train_X,1)*0.8)),:),k,opts,prox_param);   
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
else
for i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    opts.tau = 0.01;
    for iter = 1:iter_num
        [W,Z,Phi,pi] = solver_with_Z(train_X,train_Y,train_Omega,k,opts,prox_param);
        score_outlier = sum(sum(abs(Z),3),2);
        [~,idx_tmp] = sort(score_outlier);
        [W,Phi,pi] = solver_without_Z(train_X(idx_tmp(1:round(size(train_X,1)*0.8)),:),train_Y(idx_tmp(1:round(size(train_X,1)*0.8)),:),train_Omega(idx_tmp(1:round(size(train_X,1)*0.8)),:),k,opts,prox_param);   
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
end
score(isnan(score)) = -inf;
score = max(score,[],1); 
s = score;
s = s - median(s);
idx = find(s>=th*max(s),1);
best_param = param_range(idx);
best_score = score(idx); 


end