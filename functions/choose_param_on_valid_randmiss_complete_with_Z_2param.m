function [best_param_1,best_param_2,score,best_score] = choose_param_on_valid_randmiss_complete_with_Z_2param(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param_range_1,param_range_2,solver,scorer,parfor_flag,iter_num,th)

 

n = length(param_range_1);
score = zeros(iter_num,n);

if parfor_flag
parfor i = 1:n
    param_1 = param_range_1(i);
    param_2 = param_range_2(i);
    opts = opts_save;
    opts.lambda = param_1;
    opts.tau = param_2;
    for iter = 1:iter_num
        [W,~,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
else
for i = 1:n
    param_1 = param_range_1(i);
    param_2 = param_range_2(i);
    opts = opts_save;
    opts.lambda = param_1;
    opts.tau = param_2;
    for iter = 1:iter_num
        [W,~,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
end

score(isnan(score)) = -inf;

score = max(score,[],1);
  
score(isnan(score)) = -inf;
score = max(score,[],1); 
s = score;
s = s - median(s);
idx = find(s>=th*max(s),1);
best_param_1 = param_range_1(idx);
best_param_2 = param_range_2(idx);
best_score = score(idx); 
 

end