function [best_param,score,best_score] = choose_param_on_valid_randmiss_complete_real_MC(valid_X,valid_Y,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param_range,solver,scorer,parfor_flag,iter_num,th)  
n = length(param_range);
score = zeros(iter_num,n);
if parfor_flag
parfor i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    for iter = 1:iter_num
        [W,Phi,pi] = solver(valid_X,valid_Y,valid_Omega_c.*valid_Omega_b,k,opts,prox_param);        
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
    end
end
else
for i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    for iter = 1:iter_num
        [W,Phi,pi] = solver(valid_X,valid_Y,valid_Omega_c.*valid_Omega_b,k,opts,prox_param);        
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