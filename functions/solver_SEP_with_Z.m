function [W,Z,Phi,pi,pi_each] = solver_SEP_with_Z(X,Y,Omega,k,opts,prox_param)


[n,d] = size(X);
m = size(Y,2);

task_type_name = opts.task_type_name;
task_num_each_type = opts.task_num_each_type;
task_type = opts.task_type;
lambda_set = opts.lambda_set;
tau_set = opts.tau_set;
 

if opts.warm_start.flag
    
W = opts.init.W;
Z = opts.init.Z;
Phi = opts.init.Phi;

else
W = zeros(d,m,k);
Z = zeros(n,m,k);
Phi = zeros(m,m,k);

for r = 1:k 
Phi(:,:,r) =  eye(m);
end

end

pi = zeros(1,k);
pi_each = {};

for i_type = 1:3
    
    idx = opts.task_type == i_type;
    if sum(idx) == 0
        continue
    end 
    
%     disp(sprintf('%d / %d',i_type,3))
   
    task_num_each_type_j = task_num_each_type * 0;    
    task_num_each_type_j(i_type) = sum(idx);
    
    if i_type>1
        opts.gaussClose.flag = false;
    end
    
    opts_tmp = opts;
    
    opts_tmp.init.W = W(:,idx,:);
    opts_tmp.init.Phi = Phi(idx,idx,:);
 
    
    [n_train,d_train] = size(X);
    m_tmp = sum(idx);
    lambda_0 = sqrt(m_tmp )  * (log(n_train)*sqrt(log(max(n_train,d_train))/n_train));
    tau_0 = lambda_0; 
    
    if i_type == 1
        opts_tmp.lambda = lambda_set(i_type) * lambda_0 * sqrt(log(n_train)) ;
        opts_tmp.max_lambda = opts.max_lambda * lambda_0 * sqrt(log(n_train)) ;
    elseif i_type == 2
        opts_tmp.lambda = lambda_set(i_type) * lambda_0 *  1  ;
        opts_tmp.max_lambda = opts.max_lambda * lambda_0 * 1   ;
    else
        opts_tmp.lambda = lambda_set(i_type) * lambda_0 * log(n_train);
        opts_tmp.max_lambda = opts.max_lambda * lambda_0 * log(n_train);
    end
    
    if i_type == 1
        opts_tmp.tau = tau_set(i_type) * tau_0 * sqrt(log(n_train)) ;
        opts_tmp.max_tau = opts.max_tau * tau_0 * sqrt(log(n_train)) ;
    elseif i_type == 2
        opts_tmp.tau = tau_set(i_type) * tau_0 *  1  ;
        opts_tmp.max_tau = opts.max_tau * tau_0 * 1   ;
    else
        opts_tmp.tau = tau_set(i_type) * tau_0 * log(n_train);
        opts_tmp.max_tau = opts.max_tau * tau_0 * log(n_train);
    end
    
    

    opts_tmp.initial_scale = opts_tmp.initial_scale_scale/sum(idx);
   
    opts_tmp.task_num_each_type = task_num_each_type_j;
    opts_tmp.task_type = ones(1,m_tmp) * i_type;
    opts_tmp.Omega = Omega(:,idx);
   
    [W_j,Z_j,Phi_j,pi_j] = solver_sub_with_Z(X,Y(:,idx),k,opts_tmp,prox_param);
    
    pi = pi + pi_j;
    pi_each{i_type} = pi_j;
    
    W(:,idx,:) = W_j;
    Z(:,idx,:) = Z_j;
    Phi(idx,idx,:) = Phi_j;
 
    
end

pi = pi/sum(pi);



end