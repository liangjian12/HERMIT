function [best_param,score] = choose_param_on_valid_with_Z(train_X,train_Y,train_Omega,valid_X,valid_Y,valid_Omega,...
    k,opts_save,prox_param,param_range,solver,scorer,parfor_flag,iter_num)

 

n = length(param_range);
score = zeros(iter_num,n);

if parfor_flag
parfor i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    opts.tau = param*3;
    for iter = 1:iter_num
        [W,Z,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(iter,i) = scorer(valid_X,valid_Y,W,Z,Phi,pi,valid_Omega,opts);
%         disp(score(iter,i))
    end
end
else
for i = 1:n
    param = param_range(i);
    opts = opts_save;
    opts.lambda = param;
    opts.tau = param*3;
    for iter = 1:iter_num
        [W,Z,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(iter,i) = scorer(valid_X,valid_Y,W,Z,Phi,pi,valid_Omega,opts);
%         disp(score(iter,i))
    end
end
end

score = max(score,[],1);

[~,idx] = max(score);
best_param = param_range(idx);


end