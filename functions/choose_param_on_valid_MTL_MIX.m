function [best_param,score] = choose_param_on_valid_MTL_MIX(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    k,opts_save,prox_param,param_range,solver,scorer,sep_num,parfor_flag,iter_num)

 
 
n=length(param_range); 
score_record = zeros(n,sep_num,iter_num);
for iter = 1:iter_num

n = length(param_range);
score = zeros(n,sep_num);
best_param = zeros(1,sep_num);

 
% valid_Omega_c = double(rand(size(valid_Y))<observe_rate_c);
 

if parfor_flag
    parfor i = 1:n
        param = param_range(i);
        opts = opts_save;
        opts.lambda_set = param*ones(1,sep_num);
        [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
    end
else

    for i = 1:n
        param = param_range(i);
        opts = opts_save;
        opts.lambda_set = param*ones(1,sep_num);

        [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,[],valid_Omega,opts);

    end

end

score_record(:,:,iter) = score;

end

score_record(isnan(score_record)) = -inf;

score = max(score_record,[],3);

% score = median(score_record,3);
  
[~,idx] = max(score,[],1);

for j = 1:sep_num
    best_param(j) = param_range(idx(j));
end

best_param = [];
best_idx = [];
for j = 1:sep_num
    s = score(:,j);
    s = s - median(s);
    idx = find(s>=0.9*max(s),1);
    best_idx(j) = idx(1);
    best_param(j) = param_range(idx(1));
end
 
 

end
 