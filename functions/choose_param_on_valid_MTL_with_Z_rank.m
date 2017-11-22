function [best_param,score,best_score] = choose_param_on_valid_MTL_with_Z(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param_range,solver_with_Z,solver_without_Z,scorer,sep_num,parfor_flag,iter_num,th)
 %afasf
n=length(param_range); 
score_record = zeros(n,sep_num,iter_num);
for iter = 1:iter_num
n = length(param_range);
score = zeros(n,sep_num);
if parfor_flag
     parfor i = 1:n
        param = param_range(i);
        opts = opts_save;
        opts.lambda_set = param*ones(1,sep_num);
        opts.tau_set = 0.01*ones(1,sep_num);
        [W,Z,Phi,pi,pi_each] = solver_with_Z(train_X,train_Y,train_Omega,k,opts,prox_param);
        score_outlier = sum(sum(abs(Z),3),2);
        [~,idx_tmp] = sort(score_outlier);
        [W,Phi,pi] = solver_without_Z(train_X(idx_tmp(1:round(size(train_X,1)*0.8)),:),train_Y(idx_tmp(1:round(size(train_X,1)*0.8)),:),train_Omega(idx_tmp(1:round(size(train_X,1)*0.8)),:),k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
    end
else
    for i = 1:n
        param = param_range(i);
        opts = opts_save;
        opts.lambda_set = param*ones(1,sep_num);
        opts.tau_set = 0.01*ones(1,sep_num);
        [W,Z,Phi,pi,pi_each] = solver_with_Z(train_X,train_Y,train_Omega,k,opts,prox_param);
        score_outlier = sum(sum(abs(Z),3),2);
        [~,idx_tmp] = sort(score_outlier);
        [W,Phi,pi] = solver_without_Z(train_X(idx_tmp(1:round(size(train_X,1)*0.8)),:),train_Y(idx_tmp(1:round(size(train_X,1)*0.8)),:),train_Omega(idx_tmp(1:round(size(train_X,1)*0.8)),:),k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
    end
end
score_record(:,:,iter) = score;
end
score_record(isnan(score_record)) = -inf;
score = max(score_record,[],3);
best_param = [];
best_idx = [];
best_score = [];
for j = 1:sep_num
    s = score(:,j);
    s = s - median(s); %min(s(s>-inf));%
    idx = find(s>=th*max(s),1); 
    best_idx(j) = idx(1);
    best_param(j) = param_range(idx(1));
    best_score(j) = score(idx(1),j);
end
best_score = sum(best_score); 
 

end
 