function [best_param_1,best_param_2,score] = choose_param_on_valid_MTL_randmiss_complete_with_Z_2param(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param_range_1,param_range_2,solver,scorer,sep_num,parfor_flag,iter_num,th)

n=length(param_range_1); 
score_record = zeros(n,sep_num,iter_num);
best_param_1 = zeros(1,sep_num);
best_param_2 = zeros(1,sep_num);
for iter = 1:iter_num
n = length(param_range_1);
score = zeros(n,sep_num);

if parfor_flag
    parfor i = 1:n
        param_1 = param_range_1(i);
        param_2 = param_range_2(i);
        opts = opts_save;
        opts.lambda_set = param_1*ones(1,sep_num);
        opts.tau_set = param_2*ones(1,sep_num);
        [W,~,Phi,pi,pi_each] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
    end
else

    for i = 1:n
        param_1 = param_range_1(i);
        param_2 = param_range_2(i);
        opts = opts_save;
        opts.lambda_set = param_1*ones(1,sep_num);
        opts.tau_set = param_2*ones(1,sep_num);
        [W,~,Phi,pi,pi_each] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
    end

end

score_record(:,:,iter) = score;

end

score_record(isnan(score_record)) = -inf;

score = max(score_record,[],3);
 
  
% [~,idx] = max(score,[],1);

% for j = 1:sep_num
%     best_param_1(j) = param_range_1(idx(j));
% end
% 
% for j = 1:sep_num
%     best_param_2(j) = param_range_2(idx(j));
% end

 
best_param_1 = [];
best_idx_1 = [];
best_score_1 = [];
best_param_2 = [];
best_idx_2 = [];
best_score_2 = [];
for j = 1:sep_num
    s = score(:,j);
    s = s - median(s);
    idx = find(s>=th*max(s),1); 
    best_idx_1(j) = idx(1);
    best_param_1(j) = param_range_1(idx(1));
    best_score_1(j) = score(idx(1),j);
    best_idx_2(j) = idx(1);
    best_param_2(j) = param_range_2(idx(1));
    best_score_2(j) = score(idx(1),j);
end
best_score_1 = sum(best_score_1); 
best_score_2 = sum(best_score_2); 

 

 

end