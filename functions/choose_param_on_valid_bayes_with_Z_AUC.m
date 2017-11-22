function [opts_save,x,y,record_y_out,record_x_out] = choose_param_on_valid_bayes_with_Z_AUC(train_X,train_Y,train_Y_outlier,train_Z_idx,train_Omega,valid_X,valid_Y,valid_Z_idx,valid_Omega,...
    k,opts_save,prox_param,lb,ub,solver,scorer,iter_out_num,iter_num_find,iter_num_rep,parfor_flag,bayes_rho)

 
 
d = length(lb);
 
record_x_out = zeros(iter_out_num,d);
record_y_out = zeros(iter_out_num,1);

if parfor_flag

parfor iter_out = 1:iter_out_num
 
opts = opts_save;    
opts_bayes = [];    
opts_bayes.kappa = 3;
opts_bayes.sigma_0 = 1e-4;
opts_bayes.mcmc_rho = bayes_rho;
opts_bayes.K=1;
    
X = [];
Y = [];

  
x = rand(1,d);
x = x.*lb +(1-x).*ub;
 

record_x = [];
record_y = [];
record_a = [];
 

best_y = -inf;
best_x = [];

for iter = 1:iter_num_find
    
 
    
    cont = 1;
    opts.lambda = 10^x(cont);cont = cont + 1;
    opts.tau = zeros(1,opts.k);
    for r = 1:k
        opts.tau(r) = 10^x(cont);cont = cont + 1;
    end
%     opts.gaussClose.lambda = 10^x(1);
%     opts.lambda_gauss_close = 10^x(1);
%     opts.initial_scale = 10^x(2);
%     opts.maxIter_out = round(20*x(4));
%     opts.stop_param.fun = 10^x(5);
%     opts.stop_param.param = 10^x(6);
%     opts.STEP_SIZE = 10^x(2);
    
    score = [];
    for iter_in = 1:iter_num_rep
        [W,Z,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(iter_in) = scorer(valid_X,valid_Y,W,Z,Phi,pi,valid_Omega,opts,prox_param);
    end
    
    y = max(score);
     
    if y>=best_y
        best_y = y;
        best_x = x;
    end
    
    record_x = [record_x;x];
    record_y = [record_y;y];
    
    if iter == 1 && y < 0.51
        break
    end
    
    if iter > 5 && length(unique(record_y))<2
        break
    end
     
    if iter>10 && max(record_y) < 0.7
        break
    end
  
    X = [X;x];
    Y = [Y;y];
        
    opts_bayes.X=X;
    opts_bayes.Y=Y;
     
    
    [x] = gllc_my(lb,ub,opts_bayes);
    
    [a,K_new] = acq_fun(x,opts_bayes);
    opts_bayes.K = K_new;
     
    record_a = [record_a;a];
    
    
end

 
record_x_out(iter_out,:)=best_x;
record_y_out(iter_out,:)=best_y;

 
 
end

else
    
opts = opts_save;  
    
for iter_out = 1:iter_out_num
 
opts_bayes.kappa = 3;
opts_bayes.sigma_0 = 1e-2;
opts_bayes.mcmc_rho = bayes_rho;
opts_bayes.K=1;
    
X = [];
Y = [];

  
x = rand(1,d);
x = x.*lb +(1-x).*ub;
 

record_x = [];
record_y = [];
record_a = [];
 

best_y = -inf;
best_x = [];

for iter = 1:iter_num_find
    
 
    cont = 1;
    opts.lambda = 10^x(cont);cont = cont + 1;
    opts.tau = zeros(1,opts.k);
    for r = 1:k
        opts.tau(r) = 10^x(cont);cont = cont + 1;
    end
%     opts.gaussClose.lambda = 10^x(1);
%     opts.lambda_gauss_close = 10^x(1);
%     opts.initial_scale = 10^x(2);
%     opts.maxIter_out = round(20*x(4));
%     opts.stop_param.fun = 10^x(5);
%     opts.stop_param.param = 10^x(6);
%     opts.STEP_SIZE = 10^x(2);
    
    score = [];
    for iter_in = 1:iter_num_rep
        [W,Z,Phi,pi] = solver(train_X,train_Y,train_Omega,k,opts,prox_param);
        score(iter_in) = scorer(train_X,train_Y,valid_X,valid_Y,train_Y_outlier,train_Z_idx,valid_Z_idx,...
W,Z,Phi,pi,train_Omega,valid_Omega,opts);
    end
    
    y = max(score);
    
    disp([iter y])
     
    if y>=best_y
        best_y = y;
        best_x = x;
    end
    
    record_x = [record_x;x];
    record_y = [record_y;y];
    
    if iter == 1 && y < 1.5
        break
    end
    
    if iter > 5 && length(unique(record_y))<2
        break
    end
     
    if iter>10 && max(record_y) < 0.7*2
        break
    end
  
    X = [X;x];
    Y = [Y;y];
        
    opts_bayes.X=X;
    opts_bayes.Y=Y;
     
    
    [x] = gllc_my(lb,ub,opts_bayes);
    
    [a,K_new] = acq_fun(x,opts_bayes);
    opts_bayes.K = K_new;
     
    record_a = [record_a;a];
    
    
end

 
record_x_out(iter_out,:)=best_x;
record_y_out(iter_out,:)=best_y;

if best_y >= 2
    break
end

 
end    
    
    
end

[~,idx] = max(record_y_out);
x = record_x_out(idx,:);
 

cont = 1;
opts_save.lambda = 10^x(cont);cont = cont + 1;
opts_save.tau = zeros(1,opts_save.k);
for r = 1:k
    opts_save.tau(r) = 10^x(cont);cont = cont + 1;
end
% opts_save.gaussClose.lambda = 10^x(1);
% opts_save.initial_scale = 10^x(2);
% opts_save.maxIter_out = round(20*x(4));
% opts_save.stop_param.fun = 10^x(5);
% opts_save.stop_param.param = 10^x(6);
% opts_save.STEP_SIZE = 10^x(2);

end