function [fun_c] = estimate_fun_with_Z_MAR(X,Y,W,Z,Phi,pi,Omega,opts)

    
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
 
    W_j = W(:,j,:);
    Z_j = Z(:,j,:);
    Phi_j = Phi(j,j,:);
     
    [fun_c(j)] = estimate_fun_with_Z(X,Y(:,j),W_j,Z_j,Phi_j,pi{j},Omega(:,j),opts_tmp);
    
    
end

 


end