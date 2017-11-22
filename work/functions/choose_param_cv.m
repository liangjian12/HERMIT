function [best_param,score] = choose_param_cv(train_X,train_Y,train_Omega,...
    k,opts_save,prox_param,param_range,solver,scorer,parfor_flag,fold_num)

 

n = length(param_range);
score = zeros(fold_num,n);
idx = crossvalind('Kfold',size(train_X,1),fold_num);

parfor i_fold = 1:fold_num
    
score_tmp = zeros(1,n);
idx_test = idx == i_fold;
idx_train = ~idx_test;

train_X_i = train_X(idx_train,:); 
train_Y_i = train_Y(idx_train,:); 
train_Omega_i = train_Omega(idx_train,:); 
test_X_i = train_X(idx_test,:); 
test_Y_i = train_Y(idx_test,:); 
test_Omega_i = train_Omega(idx_test,:);

opts = opts_save;
if parfor_flag
    for i = 1:n
        if i == 1
            opts.warm_start.flag = false;
        else
            opts.warm_start.flag = true;
        end
        
        param = param_range(i);
%         opts = opts_save;
        opts.lambda = param;
%         [n_train,d_train] = size(train_X_i);
%         m = size(Y,2)
%         
%         opts.sep_pen.lambda_set = param * sqrt(m) * [sqrt(log(n_train)) 1 log(n_train)]*(log(n_train)*sqrt(log(max(n_train,d_train))/n_train));
        opts.sep_pen.lambda_set = param * ones(1,3);
        [W,Phi,pi] = solver(train_X_i,train_Y_i,train_Omega_i,k,opts,prox_param);
        score_tmp(i) = scorer(test_X_i,test_Y_i,W,Phi,pi,test_Omega_i,opts);
       
%         if strcmp(prox_param.W.method,'L1')
            score_tmp(i) = 2*score_tmp(i)*size(train_X_i,1) - (sum(sum(sum(abs(W)>0)))+2*k -1) * (log(size(train_X_i,1))+1);
%         elseif strcmp(prox_param.W.method,'GS')
%             W_tmp = sum(W.*W,2);
%             score_tmp(i) = 2*score_tmp(i)*size(train_X_i,1) - (sum(sum(sum(abs(W_tmp)>0)))+2*k -1) * (log(size(train_X_i,1))+1);
%         end
%         
        opts.init.W = W;
        opts.init.Phi = Phi;
%         disp(score(iter,i))
    end
 
else
    for i = 1:n
        if i == 1
            opts.warm_start.flag = false;
        else
            opts.warm_start.flag = true;
        end
        
        param = param_range(i);
%         opts = opts_save;
        opts.lambda = param;
        %         [n_train,d_train] = size(train_X_i);
%         m = size(Y,2)
%         
%         opts.sep_pen.lambda_set = param * sqrt(m) * [sqrt(log(n_train)) 1 log(n_train)]*(log(n_train)*sqrt(log(max(n_train,d_train))/n_train));
        opts.sep_pen.lambda_set = param * ones(1,3);
        [W,Phi,pi] = solver(train_X_i,train_Y_i,train_Omega_i,k,opts,prox_param);
        score_tmp(i) = scorer(test_X_i,test_Y_i,W,Phi,pi,test_Omega_i,opts);
        opts.init.W = W;
        opts.init.Phi = Phi;
%         disp(score(iter,i))
    end
end

score(i_fold,:) = score_tmp;
    
end

score = median(score,1);

[~,idx] = max(score);
best_param = param_range(idx);


end