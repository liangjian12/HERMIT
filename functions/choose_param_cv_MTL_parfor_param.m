function [best_param,score] = choose_param_cv_MTL_parfor_param(train_X,train_Y,train_Omega,...
    k,opts_save,prox_param,param_range,solver,scorer,sep_num,parfor_flag,fold_num)

n=length(param_range); 
score_record = zeros(n,sep_num,fold_num);
idx = crossvalind('Kfold',size(train_X,1),fold_num);
 

for i_fold = 1:fold_num

n = length(param_range);
score = zeros(n,sep_num);
best_param = zeros(1,sep_num);

idx_test = idx == i_fold;
idx_train = ~idx_test;

train_X_i = train_X(idx_train,:); 
train_Y_i = train_Y(idx_train,:); 
train_Omega_i = train_Omega(idx_train,:); 
test_X_i = train_X(idx_test,:); 
test_Y_i = train_Y(idx_test,:); 
test_Omega_i = train_Omega(idx_test,:); 

% opts = opts_save;

if parfor_flag
    parfor i = 1:n
%         if i == 1
%             opts.warm_start.flag = false;
%         else
%             opts.warm_start.flag = true;
%         end
        opts = opts_save;
        opts.warm_start.flag = false;
        param = param_range(i);

        opts.lambda_set = param*ones(1,sep_num);
        [W,Phi,pi,pi_each] = solver(train_X_i,train_Y_i,train_Omega_i,k,opts,prox_param);
        score(i,:) = scorer(test_X_i,test_Y_i,W,Phi,pi_each,test_Omega_i,opts);
%         opts.init.W = W;
%         opts.init.Phi = Phi;
    end
else

    for i = 1:n
        param = param_range(i);
        opts = opts_save;
        opts.lambda_set = param*ones(1,sep_num);

        [W,Phi,pi,pi_each] = solver(train_X_i,train_Y_i,train_Omega_i,k,opts,prox_param);
        score(i,:) = scorer(test_X_i,test_Y_i,W,Phi,pi_each,test_Omega_i,opts);
    end

end

score_record(:,:,i_fold) = score;

end

score = mean(score_record,3);
 
[~,idx] = max(score,[],1);

best_param = [];
for j = 1:sep_num
    best_param(j) = param_range(idx(j));
end
 


end