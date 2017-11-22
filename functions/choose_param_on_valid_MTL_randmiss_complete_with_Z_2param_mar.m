function [best_param_1,best_param_2,score] = choose_param_on_valid_MTL_randmiss_complete_with_Z_2param_mar(train_X,train_Y,valid_X,valid_Y,observe_rate_train,observe_rate_c,observe_rate_b,...
    k,opts_save,prox_param,param_range_1,param_range_2,solver_1,solver_2,solver_3,scorer,sep_num,parfor_flag,iter_num)

n=length(param_range_1); 
score_record = zeros(n,sep_num,iter_num);
best_param_1 = zeros(1,sep_num);
best_param_2 = zeros(1,sep_num);
for iter = 1:iter_num

n = length(param_range_1);
score = zeros(n,sep_num);
 
train_Omega = rand(size(train_Y))<observe_rate_train;
valid_Omega_c_save = double(rand(size(train_Y))<observe_rate_c);
valid_Omega_b_save = double(rand(size(train_Y))<observe_rate_b);


if parfor_flag
    parfor i = 1:n
        param_1 = param_range_1(i);
        param_2 = param_range_2(i);
        opts = opts_save;
        opts.lambda_set = param_1*ones(1,sep_num);
        opts.tau_set = param_2*ones(1,sep_num);
        [W,Z,Phi,pi] = solver_1(train_X,train_Y,train_Omega,k,opts,prox_param);
        [W_all,Phi_all,pi_all] = solver_2(train_X,train_Y,train_Omega,k,opts,prox_param);
        [W_k1,Phi_k1,pi_k1] = solver_3(train_X,train_Y,train_Omega,1,opts,prox_param);
        
        z = sum(Z.*Z,1);
        col_idx = z == 0 ;
        if sum(col_idx)==0
            score(i,:) = -inf * ones(1,sep_num);
        else
            valid_Omega_c = valid_Omega_c_save;
            valid_Omega_b = valid_Omega_b_save;
  
            valid_Omega_c(:,~col_idx) = 0;
            valid_Omega_b(:,~col_idx) = 0;
             score_find_outlier = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
            score_all = scorer(valid_X,valid_Y,W_all,Phi_all,pi_all,[],valid_Omega_c,valid_Omega_b,opts);
            opts_tmp = opts; opts_tmp.k=1;
            score_k1 = scorer(valid_X,valid_Y,W_k1,Phi_k1,pi_k1,[],valid_Omega_c,valid_Omega_b,opts_tmp);
            score(i,:) = score_find_outlier - max(score_all,score_k1);
        end
    end
else

    for i = 1:n
        param_1 = param_range_1(i);
        param_2 = param_range_2(i);
        opts = opts_save;
        opts.lambda_set = param_1*ones(1,sep_num);
        opts.tau_set = param_2*ones(1,sep_num);
        [W,Z,Phi,pi] = solver_1(train_X,train_Y,train_Omega,k,opts,prox_param);
        [W_all,Z_all,Phi_all,pi_all] = solver_2(train_X,train_Y,train_Omega,k,opts,prox_param);
        [W_k1,Z_k1,Phi_k1,pi_k1] = solver_3(train_X,train_Y,train_Omega,1,opts,prox_param);
        
        z = sum(Z.*Z,1);
        col_idx = z == 0 ;
        if sum(col_idx)==0
            score(i,:) = -inf * ones(1,sep_num);
        else
            valid_Omega_c = valid_Omega_c_save;
            valid_Omega_b = valid_Omega_b_save;
  
            valid_Omega_c(:,~col_idx) = 0;
            valid_Omega_b(:,~col_idx) = 0;
             score_find_outlier = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
            score_all = scorer(valid_X,valid_Y,W_all,Phi_all,pi_all,[],valid_Omega_c,valid_Omega_b,opts);
            opts_tmp = opts; opts_tmp.k=1;
            score_k1 = scorer(valid_X,valid_Y,W_k1,Phi_k1,pi_k1,[],valid_Omega_c,valid_Omega_b,opts_tmp);
            score(i,:) = score_find_outlier - max(score_all,score_k1);
        end
    end

end

score_record(:,:,iter) = score;

end

score_record(isnan(score_record)) = -inf;

score = max(score_record,[],3);

% score = median(score_record,3);
  
[~,idx] = max(score,[],1);

for j = 1:sep_num
    best_param_1(j) = param_range_1(idx(j));
end

for j = 1:sep_num
    best_param_2(j) = param_range_2(idx(j));
end

% best_param = [];
% for j = 1:sep_num
%     [~,idx] = findpeaks(score(:,j));
%     best_param(j) = param_range(idx(1));
% end
 


end