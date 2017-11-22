function [best_param,score] = choose_param_on_valid_randmiss(train_X,train_Y_save,valid_X,valid_Y_save,observe_rate,...
    k,opts_save,prox_param,param_range,solver,scorer,parfor_flag,iter_num)

 

n = length(param_range);
score = zeros(iter_num,n);

if parfor_flag
parfor i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    for iter = 1:iter_num
        [train_Y,train_Omega] = randmiss(train_Y_save,observe_rate);
        [valid_Y,valid_Omega] = randmiss(valid_Y_save,observe_rate);
        [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(iter,i) = scorer(valid_X,valid_Y,W,Phi,pi,valid_Omega,opts);
        
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
        [train_Y,train_Omega] = randmiss(train_Y_save,observe_rate);
        [valid_Y,valid_Omega] = randmiss(valid_Y_save,observe_rate);
        [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(iter,i) = scorer(valid_X,valid_Y,W,Phi,pi,valid_Omega,opts);
%         disp(score(iter,i))
    end
end
end

score(isnan(score)) = -inf;

score = max(score,[],1);

% score = median(score,1);

[~,idx] = max(score);
best_param = param_range(idx);


end