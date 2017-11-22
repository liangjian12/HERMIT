function [fun_c,nmse,auc,aupr,nmse_poiss] = estimate_fun_with_Z_SEP_complete(X,Y,W,Z,Phi,pi,pi_each,Omega_c,Omega_b,opts)

Omega_p = 1 - Omega_c;    

Omega_c = Omega_c .* Omega_b;
Omega_p = Omega_p .* Omega_b;

[n,d] = size(X);
m = size(Y,2);

 
task_num_each_type = opts.task_num_each_type;
  

fun_c = zeros(1,3);

nmse = 100;
auc = -1;
aupr = -1;
nmse_poiss = 100;
  
for i_type = 1:3
    
    idx = opts.task_type == i_type;
    if sum(idx) == 0
        continue
    end 
    
%     disp(sprintf('%d / %d',i_type,3))
   
    task_num_each_type_idx = task_num_each_type * 0;    
    task_num_each_type_idx(i_type) = sum(idx);
    
    task_type = [];
    for i = 1:3
        if task_num_each_type_idx(i) == 0
            continue
        end
        task_type = [task_type i * ones(1,task_num_each_type_idx(i))];
    end
     
    opts_tmp = opts;
    opts_tmp.task_num_each_type = task_num_each_type_idx;
    opts_tmp.task_type = task_type;
    
    W_idx = W(:,idx,:);
    Phi_idx = Phi(idx,idx,:);
    Z_idx = Z(:,idx,:);
 
    [fun_c(i_type),rho] = estimate_fun_without_Z(X,Y(:,idx),W_idx,Phi_idx,pi_each{i_type},Omega_c(:,idx),opts_tmp);
   % [fun_c(i_type)] = estimate_fun_giv_rho_without_Z(X,Y(:,idx),W_idx,Phi_idx,pi_each{i_type},rho,Omega_p(:,idx),opts_tmp);
    [fun_c(i_type)] = estimate_fun_without_Z(X,Y(:,idx),W_idx,Phi_idx,pi_each{i_type},Omega_b(:,idx),opts_tmp);
    
    
    if opts.k > 1
    [nmse_tmp,auc_tmp,aupr_tmp,nmse_poiss_tmp] = estimate_fun_with_Z_kn_softmax_nmse_auc(X,Y(:,idx),W_idx,Z_idx,Phi_idx,rho,pi_each{i_type},Omega_p(:,idx),opts_tmp);
    else
    [nmse_tmp,auc_tmp,aupr_tmp,nmse_poiss_tmp] = estimate_fun_with_Z_k1_nmse_auc(X,Y(:,idx),W_idx,Z_idx,Phi_idx,pi_each{i_type},Omega_p(:,idx),opts_tmp);
    end
    
    if i_type == 1
        nmse = nmse_tmp;
    elseif i_type == 2
        auc  = auc_tmp;
        aupr = aupr_tmp;
    else
        nmse_poiss = nmse_poiss_tmp;
    end
    
 end

 


end