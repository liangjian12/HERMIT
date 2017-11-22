function [score] = scorer_without_Z_real_int_complete_cv_MAR(X,Y,W,Phi,pi,pi_each,Omega_c,Omega_b,opts)

Omega_p = 1 - Omega_c;

Omega_c = Omega_c .* Omega_b;
Omega_p = Omega_p .* Omega_b;

m = size(Y,2);

task_num_each_type = opts.task_num_each_type;
task_type = opts.task_type;
 

fun_c = zeros(1,m);
  
weight = sum(Omega_p) > 0;
score = zeros(1,m);
for j = 1:m
    
%     disp(sprintf('%d / %d',j,m))
   
    task_num_each_type_j = task_num_each_type * 0;    
    task_num_each_type_j(task_type(j)) = 1;
    
     
     
    opts_tmp = opts;
    opts_tmp.task_num_each_type = task_num_each_type_j;
    opts_tmp.task_type = task_type(j);
 
    
    W_j = W(:,j,:);
    Phi_j = Phi(j,j,:);

%     [fun_c(j)] = estimate_fun_without_Z(X,Y(:,j),W_j,Phi_j,pi{j},Omega(:,j),opts_tmp);

    if opts.k > 1
        [fun_c,rho] = estimate_fun_without_Z_MAR(X,Y(:,j),W_j,Phi_j,pi_each{j},Omega_c(:,j),opts_tmp);
        [nmse(j),auc(j),aupr(j),nmse_poiss(j)] = estimate_fun_without_Z_kn_softmax_nmse_auc_MAR(X,Y(:,j),W_j,Phi_j,rho,pi_each{j},Omega_p(:,j),opts_tmp);

    else
        [nmse(j),auc(j),aupr(j),nmse_poiss(j)] = estimate_fun_without_Z_k1_nmse_auc_MAR(X,Y(:,j),W_j,Phi_j,pi_each{j},Omega_p(:,j),opts_tmp);
    end
    if task_type(j) == 1
        score(j) = -nmse(j);
    elseif task_type(j) == 2
        score(j) = auc(j);
    elseif task_type(j) == 3
        score(j) = -nmse_poiss(j);
    end
       
    

end

% nmse = nmse.*weight/sum(weight);
% auc = auc.*weight/sum(weight);
% aupr = aupr.*weight/sum(weight);
% nmse_poiss = nmse_poiss.*weight/sum(weight);

% score = [- nmse, auc , -nmse_poiss];

end