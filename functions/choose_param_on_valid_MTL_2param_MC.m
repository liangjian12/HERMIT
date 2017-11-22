function [best_param1,best_param2,score,best_score] = choose_param_on_valid_MTL_2param_MC(valid_X,valid_Y,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param1_range,param2_range,solver,scorer,sep_num,parfor_flag,iter_num)
 
n=length(param1_range); 
score_record = zeros(n,sep_num,iter_num);
for iter = 1:iter_num

n = length(param1_range);
score = zeros(n,sep_num);
 
if parfor_flag
    parfor i = 1:n
        param1 = param1_range(i);
        param2 = param2_range(i);
        opts = opts_save;
        opts.lambda_set = param1*ones(1,sep_num);
        opts.lambda2_set = param2*ones(1,sep_num);
        [W,Phi,pi,pi_each] = solver(valid_X,valid_Y,valid_Omega_b.*valid_Omega_c,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
    end
else
    for i = 1:n
        param1 = param1_range(i);
        param2 = param2_range(i);
        opts = opts_save;
        opts.lambda_set = param1*ones(1,sep_num);
        opts.lambda2_set = param2*ones(1,sep_num);
        [W,Phi,pi,pi_each] = solver(valid_X,valid_Y,valid_Omega_b.*valid_Omega_c,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
    end
end

score_record(:,:,iter) = score;

end
score_record(isnan(score_record)) = -inf;
score = max(score_record,[],3);
  
best_param1 = [];
best_param2 = [];
best_idx = [];
best_score = [];
for j = 1:sep_num
    s = score(:,j);
    s = s - median(s);
    idx = find(s>=1*max(s),1); 
    best_idx(j) = idx(1);
    best_param1(j) = param1_range(idx(1));
    best_param2(j) = param2_range(idx(1));
    best_score(j) = score(idx(1),j);
end
best_score = sum(best_score); 
 

end
 