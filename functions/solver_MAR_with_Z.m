function [W,Z,Phi,pi,pi_each] = solver_MAR_with_Z(X,Y,Omega,k,opts,prox_param)


[n,d] = size(X);
m = size(Y,2);

task_type_name = opts.task_type_name;
task_num_each_type = opts.task_num_each_type;
task_type = opts.task_type;
lambda_set = opts.lambda_set;
tau_set = opts.tau_set;


W = {};
Z = {};
Phi = {};
pi_each= {};
for r = 1:k 
W{r} = zeros(d,m)  ; 
Z{r} = zeros(n,m);
tmp = ones(1,m); 
Phi{r} =  diag(tmp) ;
end

pi = zeros(1,k);

for j = 1:m
    
%     disp(sprintf('%d / %d',j,m))
   
    task_num_each_type_j = task_num_each_type * 0;    
    task_num_each_type_j(task_type(j)) = 1;
    opts.lambda = lambda_set(j);
    opts.tau = tau_set(j,:);
   
    [W_j,Z_j,Phi_j,pi_j] = solver_sub_with_Z(X,Y(:,j),Omega(:,j),k,task_num_each_type_j,task_type_name,opts,prox_param);
    
    pi = pi + pi_j;
    
    for r = 1:k 
        W{r}(:,j) = W_j{r}  ; 
        Z{r}(:,j) = Z_j{r}  ; 
        Phi{r}(j,j) = Phi_j{r} ;
    end
    pi_each{j} = pi_j;
end

pi = pi/sum(pi);



end