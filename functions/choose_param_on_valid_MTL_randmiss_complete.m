function [best_param,score] = choose_param_on_valid_MTL_randmiss_complete(train_X,train_Y,valid_X,valid_Y,observe_rate_train,observe_rate_c,observe_rate_b,...
    k,opts_save,prox_param,param_range,solver,scorer,sep_num,parfor_flag,iter_num)

record_best_param = [];
for iter_out = 1:5
n=length(param_range); 
score_record = zeros(n,sep_num,iter_num);
for iter = 1:iter_num

n = length(param_range);
score = zeros(n,sep_num);
best_param = zeros(1,sep_num);

train_Omega = rand(size(train_Y))<observe_rate_train;
valid_Omega_c = double(rand(size(train_Y))<observe_rate_c);
valid_Omega_b = double(rand(size(train_Y))<observe_rate_b);


if parfor_flag
    parfor i = 1:n
        param = param_range(i);
        opts = opts_save;
        opts.lambda_set = param*ones(1,sep_num);
        [W,Phi,pi,pi_each] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
%         score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi_each,valid_Omega_c,opts);
        
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi,pi_each,valid_Omega_c,valid_Omega_b,opts);
        
 
%         score(i,:) = scorer(train_X,train_Y,W,Phi,pi_each,train_Omega,opts);
%         score(i,:) = 2*score(i,:) -  sum(abs(W)>0)*(log(size(train_X,1))+1);
    end
else

    for i = 1:n
        param = param_range(i);
        opts = opts_save;
        opts.lambda_set = param*ones(1,sep_num);

        [W,Phi,pi,pi_each] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi_each,valid_Omega,opts);

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
for j = 1:sep_num
    s = score(:,j);
    s = s - min(s(s>(-inf)));
    idx = find(s>=1*max(s),1);
    best_param(j) = param_range(idx(1));
end

record_best_param = [record_best_param;best_param];
 

end

best_param = median(record_best_param,1);


end