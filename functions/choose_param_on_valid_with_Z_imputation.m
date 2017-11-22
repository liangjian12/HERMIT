function [best_param,score,best_score] = choose_param_on_valid_with_Z_imputation(train_X,train_Y,train_Omega,valid_X,valid_Y,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param_range,solver,scorer,parfor_flag,iter_num,th)  
n = length(param_range);
score = zeros(iter_num,n);
if parfor_flag
parfor i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    for iter = 1:iter_num
        [W,Z,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);     
        Z = Z(1:size(valid_Y,1),1:size(valid_Y,2));
        score_tmp = scorer(valid_X,valid_Y,W,Z,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
else
for i = 1:n
%     disp(i)
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    for iter = 1:iter_num
        [W,Z,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);    
        Z = Z(1:size(valid_Y,1),1:size(valid_Y,2));
        score_tmp = scorer(valid_X,valid_Y,W,Z,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
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