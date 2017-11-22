function [best_param,score,best_score] = choose_param_on_valid_randmiss_complete_real_capL1(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param_range,solver,scorer,parfor_flag,iter_num,th)  
n = length(param_range);
score = zeros(iter_num,n);
if parfor_flag
parfor i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.theta = param;
    for iter = 1:iter_num
        [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);        
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
else
for i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.theta = param;
    for iter = 1:iter_num
        [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);        
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
end
score(isnan(score)) = -1000;
score = mean(score,1); 
s = score;
s = s - median(s);
idx = find(s>=th*max(s),1);
best_param = param_range(idx);
best_score = score(idx); 


end