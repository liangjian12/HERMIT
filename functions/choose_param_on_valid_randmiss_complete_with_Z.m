function [best_param,score] = choose_param_on_valid_randmiss_complete_with_Z(train_X,train_Y,valid_X,valid_Y,observe_rate_train,observe_rate_c,observe_rate_b,...
    k,opts_save,prox_param,param_range,solver,scorer,parfor_flag,iter_num)

 

n = length(param_range);
score = zeros(iter_num,n);

if parfor_flag
parfor i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.tau = param;
    for iter = 1:iter_num
        train_Omega = rand(size(train_Y))<observe_rate_train;
        valid_Omega_c = double(rand(size(train_Y))<observe_rate_c);
        valid_Omega_b = double(rand(size(train_Y))<observe_rate_b);
        [W,~,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
%         score(iter,i) = scorer(valid_X,valid_Y,W,Phi,pi,valid_Omega_c,opts);
        
        score_tmp = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
        score(iter,i) = sum(score_tmp);
        
%         score_tmp(i) = 2*score_tmp(i)*size(train_X_i,1) - (sum(sum(sum(abs(W)>0)))+2*k -1) * (log(size(train_X_i,1))+1);
%         disp(score(iter,i))
    end
end
else
for i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    for iter = 1:iter_num
        [train_Y,train_Omega] = randmiss(train_Y,observe_rate_train);
        [valid_Y,valid_Omega] = randmiss(valid_Y,observe_rate_train);
        [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(iter,i) = scorer(valid_X,valid_Y,W,Phi,pi,valid_Omega,opts);
%         disp(score(iter,i))
    end
end
end

score(isnan(score)) = -inf;

score = max(score,[],1);
 

[~,idx] = max(score);
best_param = param_range(idx);

% [~,idx] = sort(score,'descend');
% idx = min(idx(1:3));
% best_param = param_range(idx);


end