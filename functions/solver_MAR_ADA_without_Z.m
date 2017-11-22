function [W,Phi,pi,pi_each] = solver_MAR_ADA_without_Z(X,Y,Omega,k,opts,prox_param)


[n,d] = size(X);
m = size(Y,2);

task_type_name = opts.task_type_name;
task_num_each_type = opts.task_num_each_type;
task_type = opts.task_type;
lambda_set = opts.lambda_set;



if opts.warm_start.flag
    
W = opts.init.W;
Phi = opts.init.Phi;

else
W = zeros(d,m,k);
Phi = zeros(m,m,k);

for r = 1:k 
Phi(:,:,r) =  eye(m);
end

end

pi = zeros(1,k);
pi_each = {};

for j = 1:m
    
%     disp(sprintf('%d / %d',j,m))





   
    task_num_each_type_j = task_num_each_type * 0;    
    task_num_each_type_j(task_type(j)) = 1;
    
    if task_type(j)>1
        opts.gaussClose.flag = false;
    end
    
    opts_tmp = opts;
       
    [n_train,d_train] = size(X);
    lambda_0 = sqrt(1)  * (log(n_train)*sqrt(log(max(n_train,d_train))/n_train));
    
    if task_type(j) == 1
    
        opts_tmp.lambda = lambda_set(j) * lambda_0 * sqrt(log(n_train)) ;
        opts_tmp.max_lambda = opts.max_lambda * lambda_0 * sqrt(log(n_train)) ;
    elseif task_type(j) == 2
        opts_tmp.lambda = lambda_set(j) * lambda_0 * 1   ;
        opts_tmp.max_lambda = opts.max_lambda * lambda_0 * 1 ;
    else
        opts_tmp.lambda = lambda_set(j) * lambda_0 * log(n_train); 
        opts_tmp.max_lambda = opts.max_lambda * lambda_0 * (log(n_train)) ;
    end
    
    
    opts_tmp.initial_scale = opts_tmp.initial_scale_scale;
    opts_tmp.task_num_each_type = task_num_each_type_j;
    opts_tmp.task_type = task_type(j);
    opts_tmp.Omega = Omega(:,j);
    
    opts_tmp.init.W = W(:,j,:);
    opts_tmp.init.Phi = Phi(j,j,:);
    
   
%     [W_j,Phi_j,pi_j] = solver_sub_without_Z(X,Y(:,j),k,opts_tmp,prox_param);
    [W_j,Phi_j,pi_j] = solver_ADA_without_Z(X,Y(:,j),k,opts_tmp,prox_param);
%     [W,Phi,pi] = solver_ADA_without_Z(X,Y,k,opts,prox_param)
    
    pi = pi + pi_j;
    
    
    W(:,j,:) = W_j;
    Phi(j,j,:) = Phi_j;
    
    pi_each{j} = pi_j;
    
end

pi = pi/sum(pi);



end