function [best_k,score] = decide_k_for_compare(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    opts_save,prox_param,solver,scorer,iter_num,best_param)

score_record = zeros(1,8);
for k = 1:8
    disp(k)
opts_save.k = 1;
opts_save.warmFlag =false;
opts_save.lambda = best_param{1};
opts_save.lambda_set = best_param{1};
opts_save.sep_pen.lambda_set = best_param{1} ;
opts_save.lambda2 = best_param{2};
opts_save.lambda2_set = best_param{2};
opts_save.sep_pen.lambda2_set = best_param{2} ;
score = zeros(1, iter_num);
parfor it = 1:iter_num
%     disp(it)
%     tic
opts = opts_save;
[W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param); 
[score_tmp] = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
score(it) = sum(score_tmp);
% toc
end
score = sort(score,'descend');
[score_record(k)] = mean(score(1:12));
end
score = score_record;
[~,best_k] = max(score);
 
end