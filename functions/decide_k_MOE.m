function [best_k,score,score_vec] = decide_k_MOE(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    opts_save,prox_param,solver,scorer,iter_num,best_param,j_method,flag_more_pen)

score_record = zeros(1,8);


 
for k = 1:8
    if flag_more_pen
    if j_method == 4 || j_method == 19
        ratio = k  ;
    elseif j_method == 8
        ratio = sqrt(k)  ;
    end
    else
    ratio = 1;
    end
    
    disp(k)
opts_save.k = k;
opts_save.warmFlag =false;
opts_save.lambda = best_param{1}*ratio;
opts_save.lambda_set = best_param{1}*ratio;
opts_save.sep_pen.lambda_set = best_param{1}*ratio;
opts_save.lambda2 = best_param{2};
opts_save.lambda2_set = best_param{2};
opts_save.sep_pen.lambda2_set = best_param{2} ;
opts_save.theta = best_param{2} ;
score = zeros(1, iter_num);
score_vec_record = zeros(  iter_num,3);
parfor it = 1:iter_num
%     disp([k it])
opts = opts_save;
% [W,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param); 
% [score_tmp] = scorer(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_c,valid_Omega_b,opts);
% score(it) = sum(score_tmp);


[W,Phi,W_softmax] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
rho_train = softmax_pred(train_X,W_softmax);
pi = mean(rho_train);
rho_valid = softmax_pred(valid_X,W_softmax);
[score_tmp] = scorer_without_Z_real_GL_cv_giv_rho(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_b,opts,rho_valid);
score(it) = sum(score_tmp);


score_vec_record(it,:) = score_tmp;
end
[score,idx] = sort(score,'descend');
[score_record(k)] = mean(score(1:12));
disp(score_record(k))
disp(score_vec_record(idx(1),:))
end
score = score_record;
[~,best_k] = max(score);
 
end