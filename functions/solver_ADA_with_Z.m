function [W,Z,Phi,pi] = solver_ADA_with_Z(X,Y,Omega,k,opts,prox_param)

 
task_type_name = opts.task_type_name;
task_num_each_type = opts.task_num_each_type;
 
prox_param.W.ada_lasso_flag = false; 
prox_param.W.ada_lasso_weight = {}; 

prox_param.Z.ada_lasso_flag = false; 
prox_param.Z.ada_lasso_weight = {};
   
[W,Z,Phi,pi] = solver_sub_with_Z(X,Y,Omega,k,task_num_each_type,task_type_name,opts,prox_param);
    
prox_param.W.ada_lasso_flag = true; 
for r = 1:k
prox_param.W.ada_lasso_weight{r} = 1./(abs(W{r})+eps);
end

prox_param.Z.ada_lasso_flag = true; 
for r = 1:k
prox_param.Z.ada_lasso_weight{r} = 1./(abs(Z{r})+eps);
end

[W,Z,Phi,pi] = solver_sub_with_Z(X,Y,Omega,k,task_num_each_type,task_type_name,opts,prox_param);

end