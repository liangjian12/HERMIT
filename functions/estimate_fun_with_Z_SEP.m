function [fun_c] = estimate_fun_with_Z_SEP(X,Y,W,Z,Phi,pi,Omega,opts)

    
[n,d] = size(X);
m = size(Y,2);

 
task_num_each_type = opts.task_num_each_type;
  

fun_c = zeros(1,3);
  
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
    Z_idx = Z(:,idx,:);
    Phi_idx = Phi(idx,idx,:);
 
     
     
    [fun_c(i_type)] = estimate_fun_with_Z(X,Y(:,idx),W_idx,Z_idx,Phi_idx,pi{i_type},Omega(:,idx),opts_tmp);
     
      
    
end

 


end