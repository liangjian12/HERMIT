function [fun_c] = estimate_auc_without_Z_MAR(X,Y,W,Phi,pi,Omega,opts)

    
[n,d] = size(X);
m = size(Y,2);

 
task_num_each_type = opts.task_num_each_type;
task_type_save = opts.task_type;
 

fun_c = zeros(1,m);
  
for j = 1:m
    
%     disp(sprintf('%d / %d',j,m))
   
    task_num_each_type_j = task_num_each_type * 0;    
    task_num_each_type_j(task_type_save(j)) = 1;
    
    task_type = [];
    for i = 1:3
        if task_num_each_type_j(i) == 0
            continue
        end
        task_type = [task_type i * ones(1,task_num_each_type_j(i))];
    end
     
    opts_tmp = opts;
    opts_tmp.task_num_each_type = task_num_each_type_j;
    opts_tmp.task_type = task_type;
 
    W_j = {};
    Phi_j = {};
    for r = 1:opts.k
        W_j{r} =  W{r}(:,j);          
        Phi_j{r} = Phi{r}(j,j);
        opts_tmp.true_W{r} = opts.true_W{r}(:,j);
    end
     
    [fun_c(j)] = estimate_auc_without_Z(X,Y(:,j),W_j,Phi_j,pi{j},Omega(:,j),opts_tmp);
    
    
end

 


end