function [best_param1,best_param2,score,best_score] = choose_param_on_valid_2param_MC(valid_X,valid_Y,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param1_range,param2_range,solver,scorer,parfor_flag,iter_num)

  
n = length(param1_range);
score = zeros(iter_num,n);

if parfor_flag
parfor i = 1:n
    param1 = param1_range(i);
    param2 = param2_range(i);
    opts = opts_save;
    opts.lambda = param1;
    opts.lambda2 = param2;
    for iter = 1:iter_num  
        [W,Phi,pi] = solver(valid_X,valid_Y,valid_Omega_c.*valid_Omega_b,k,opts,prox_param);         
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
else
for i = 1:n
    param1 = param1_range(i);
    param2 = param2_range(i);
    opts = opts_save;
    opts.lambda = param1;
    opts.lambda2 = param2;
    for iter = 1:iter_num  
        [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);         
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
end

score(isnan(score)) = -inf;
score = max(score,[],1);  

s = score;
s = s - median(s);
idx = find(s>=1*max(s),1);
best_param1 = param1_range(idx);
best_param2 = param2_range(idx);
best_score = score(idx); 


end