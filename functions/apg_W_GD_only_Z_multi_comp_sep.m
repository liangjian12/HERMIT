function [Z] = apg_W_GD_only_Z_multi_comp_sep(X,Y,opts)
% Input:

%Output:
 

% store data
opts.X = X;
opts.Y = Y;

% super parameters
k = opts.k;         % number of columns of Fi, i = 1,2,3; 
[n,d] = size(X);
[n,m] = size(Y);
  


for i_type = 1:3
    
    idx = opts.task_type == i_type;
    if sum(idx) == 0
        continue
    end 
 
    task_num_each_type_j = opts.task_num_each_type * 0;    
    task_num_each_type_j(i_type) = sum(idx);
    
    task_type_j = i_type * ones(1,sum(idx));
   
    
    opts_tmp = opts;
    opts_tmp.task_num_each_type = task_num_each_type_j;
    opts_tmp.task_type = task_type_j;
    
    
    opts_tmp.Y = Y(:,idx);
    opts_tmp.Omega = opts.Omega(:,idx);
%     opts_tmp.Phi = opts.Phi(idx,idx);
    
     
    opts_tmp.prox_param.Z.fix_feature = opts.prox_param.W.fix_feature(:,idx,:);
    
    
    m_tmp = sum(idx);
    
    [n_train,d_train] = size(X);
    lambda_0 = sqrt(m_tmp)  * (log(n_train)*sqrt(log(max(n_train,d_train))/n_train));
    
    if i_type == 1   
        opts_tmp.prox_param.Z.param = opts.prox_param.Z.sep.param_set(i_type,:) * lambda_0 * sqrt(log(n_train)) ;    
    elseif i_type == 2
       opts_tmp.prox_param.Z.param = opts.prox_param.Z.sep.param_set(i_type,:) * lambda_0 * 1  ;
    else
       opts_tmp.prox_param.Z.param = opts.prox_param.Z.sep.param_set(i_type,:) * lambda_0 * log(n_train); 
    end
     
    %dimension of all parameters
    dim_x =  n * m_tmp *k  ;

    %initialization of all parameters    
    if  opts.warm_start.flag || opts.warmFlag %
        Z_tmp = opts.init.Z(:,idx,:);
        opts_tmp.X_INIT = [Z_tmp(:);];
    else
        opts_tmp.X_INIT = opts.initial_scale *  randn(dim_x,1);
    end

    %APG
%     [x,opts_tmp] = apg_plus(@grad_W_GD_with_Z_multi_comp, @soft_thresh_W_GD_with_Z_multi_comp, @opti_close_Phi_with_Z_multi_comp, dim_x, opts_tmp); % x is a vector, containing all the parameters   
    [x] = apg(@grad_W_GD_only_Z_multi_comp, @soft_thresh_W_GD_only_Z_multi_comp_sep, dim_x, opts_tmp); % x is a vector, containing all the parameters   

    
    % re-organize the parameters into the forms of (W,Z,Phi) from the vector x.
    for r = 1:k
        Z(:,idx,r) = reshape(x(1:n*m_tmp),[n m_tmp]); x(1:n*m_tmp) = []; 
    end
end

 

 
end