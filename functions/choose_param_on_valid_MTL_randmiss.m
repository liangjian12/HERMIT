function [best_param,score] = choose_param_on_valid_MTL_randmiss(train_X,train_Y_save,valid_X,valid_Y_save,observe_rate,...
    k,opts_save,prox_param,param_range,solver,scorer,sep_num,parfor_flag,iter_num)

n=length(param_range); 
score_record = zeros(n,sep_num,iter_num);
for iter = 1:iter_num

n = length(param_range);
score = zeros(n,sep_num);
best_param = zeros(1,sep_num);

[train_Y,train_Omega] = randmiss(train_Y_save,observe_rate);
[valid_Y,valid_Omega] = randmiss(valid_Y_save,observe_rate);

if parfor_flag
    parfor i = 1:n
        param = param_range(i);
        opts = opts_save;
        opts.lambda_set = param*ones(1,sep_num);
        [W,Phi,pi,pi_each] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(i,:) = scorer(valid_X,valid_Y,W,Phi,pi_each,valid_Omega,opts);
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

% best_param = [];
% for j = 1:sep_num
%     [~,idx] = findpeaks(score(:,j));
%     best_param(j) = param_range(idx(1));
% end
 


end