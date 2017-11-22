function [best_param1,best_param2,score] = choose_two_param_on_valid_with_Z(train_X,train_Y,train_Omega,valid_X,valid_Y,valid_Omega,...
    k,opts,prox_param,param_range,param2_range,solver,scorer)

iter_num = 1;

n = length(param_range);
score = zeros(iter_num,n);
for i = 1:n
    param = param_range(i);
    param2 = param2_range(i);
    opts.lambda = param;
    opts.tau = param2;
    for iter = 1:iter_num
        [W,Z,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(iter,i) = scorer(valid_X,valid_Y,W,Z,Phi,pi,valid_Omega,opts);
    end
    disp(score(iter,i))
end

score = mean(score,1);

[~,idx] = max(score);
best_param1 = param_range(idx);
best_param2 = param2_range(idx);


end