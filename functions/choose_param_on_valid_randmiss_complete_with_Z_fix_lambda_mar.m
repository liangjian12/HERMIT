function [best_param,score] = choose_param_on_valid_randmiss_complete_with_Z_fix_lambda_mar(train_X,train_Y,valid_X,valid_Y,observe_rate_train,observe_rate_c,observe_rate_b,...
    k,opts_save,prox_param,param_range,solver_1,solver_2,solver_3,scorer,parfor_flag,iter_num)

 

n = length(param_range);
score = zeros(iter_num,n);

if parfor_flag
parfor i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.tau = param;
    for iter = 1:iter_num
        train_Omega = rand(size(train_Y))<observe_rate_train;
        valid_Omega_c_save = double(rand(size(train_Y))<observe_rate_c);
        valid_Omega_b_save = double(rand(size(train_Y))<observe_rate_b);
        [W,Z,Phi,pi] = solver_1(train_X,train_Y,train_Omega,k,opts,prox_param);
        [W_all,Phi_all,pi_all] = solver_2(train_X,train_Y,train_Omega,k,opts,prox_param);
        opts_tmp = opts; opts_tmp.k=1;
        [W_k1,Phi_k1,pi_k1] = solver_3(train_X,train_Y,train_Omega,1,opts_tmp,prox_param);
%         score(iter,i) = scorer(valid_X,valid_Y,W,Phi,pi,valid_Omega_c,opts);
        
 
        z = sum(sum(Z.*Z,3),1);
        col_idx = abs(z) < 1e-6 ;
        if sum(col_idx)==0
            score(iter,i) = -inf;
        else
            valid_Omega_c = valid_Omega_c_save;
            valid_Omega_b = valid_Omega_b_save;
  
            valid_Omega_c(:,~col_idx) = 0;
            valid_Omega_b(:,~col_idx) = 0;
            score_find_outlier = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);score_find_outlier = sum(score_find_outlier);
            score_all = scorer(valid_X,valid_Y,W_all,Phi_all,pi_all,[],valid_Omega_c,valid_Omega_b,opts);score_all = sum(score_all);
            opts_tmp = opts; opts_tmp.k=1;
            score_k1 = scorer(valid_X,valid_Y,W_k1,Phi_k1,pi_k1,[],valid_Omega_c,valid_Omega_b,opts_tmp);score_k1 = sum(score_k1);
            score(iter,i) = score_find_outlier - max(score_all,score_k1);
            score(iter,i) = score(iter,i)*sum(col_idx);
        end
    end
end
else
for i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.tau = param;
    for iter = 1:iter_num
        train_Omega = rand(size(train_Y))<observe_rate_train;
        valid_Omega_c_save = double(rand(size(train_Y))<observe_rate_c);
        valid_Omega_b_save = double(rand(size(train_Y))<observe_rate_b);
        [W,Z,Phi,pi] = solver_1(train_X,train_Y,train_Omega,k,opts,prox_param);
        [W_all,Z_all,Phi_all,pi_all] = solver_2(train_X,train_Y,train_Omega,k,opts,prox_param);
        [W_k1,Z_k1,Phi_k1,pi_k1] = solver_3(train_X,train_Y,train_Omega,1,opts,prox_param);
%         score(iter,i) = scorer(valid_X,valid_Y,W,Phi,pi,valid_Omega_c,opts);
        
 
        z = sum(Z.*Z,1);
        col_idx = z == 0 ;
        if sum(col_idx)==0
            score(iter,i) = -inf;
        else
            valid_Omega_c = valid_Omega_c_save;
            valid_Omega_b = valid_Omega_b_save;
  
            valid_Omega_c(:,~col_idx) = 0;
            valid_Omega_b(:,~col_idx) = 0;
            score_find_outlier = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);score_find_outlier = sum(score_find_outlier);
            score_all = scorer(valid_X,valid_Y,W_all,Phi_all,pi_all,[],valid_Omega_c,valid_Omega_b,opts);score_all = sum(score_all);
            opts_tmp = opts; opts_tmp.k=1;
            score_k1 = scorer(valid_X,valid_Y,W_k1,Phi_k1,pi_k1,[],valid_Omega_c,valid_Omega_b,opts_tmp);score_k1 = sum(score_k1);
            score(iter,i) = score_find_outlier - max(score_all,score_k1);
        end
    end
end
end

score(isnan(score)) = -inf;

score = max(score,[],1);
 

[~,idx] = max(score);
best_param = param_range(idx);

s = score;
    
s = s - min(s(s>(-inf)));
idx = find(s>=0.95*max(s),1);
best_param = param_range(idx(1));


end