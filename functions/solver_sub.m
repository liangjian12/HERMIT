function solver_sub(X,Y,k,task_num_each_type,task_type_name,opts,prox_param)


[n,d] = size(X);
m = size(Y,2);

task_type = [];
for i = 1:3
    if task_num_each_type(i) == 0
        continue
    end
    task_type = [task_type i * ones(1,task_num_each_type(i))];
end





end