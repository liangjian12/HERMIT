function [best_param1,best_param2,score] = choose_param_on_valid_MTL_2param_1m2o(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
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
        opts.sep_pen.lambda_set = param1*ones(1,sep_num);
        opts.theta = param2 ;
        [W,Phi,pi,pi_each] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
    end
else
    for i = 1:n
        param1 = param1_range(i);
        param2 = param2_range(i);
        opts = opts_save;
        opts.lambda_set = param1*ones(1,sep_num);
        opts.sep_pen.lambda_set = param1*ones(1,sep_num);
        opts.theta = param2 ;
        [W,Phi,pi,pi_each] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
    end
end

score_record(:,:,iter) = score;

end
score_record(isnan(score_record)) = -inf;
score = max(score_record,[],3);
  
best_param1 = [];
best_param2 = [];
best_idx1 = [];
best_idx2 = [];
 
for j = 1:sep_num
    s_save = score(:,j);

    [~,idx_sort] = sort(param1_range,'descend');
    s = s_save(idx_sort);
    s = s - median(s);
    idx_max = find(s>=1*max(s),1); 
    
    best_idx1(j) = idx_sort(idx_max(1));
    best_param1(j) = param1_range(best_idx1(j));
end

s_save = sum(score,2);
[~,idx_sort] = sort(param2_range,'descend');
s = s_save(idx_sort);
s = s - median(s);
idx_max = find(s>=1*max(s),1); 

best_idx2 = idx_sort(idx_max(1));
best_param2 = param2_range(best_idx2);
     
 

end
 