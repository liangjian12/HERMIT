function [sum_L,L] = compute_rho_r_without_Z_MAR(X,Y,W,Phi,Omega,opts)

G = X*W;

% G = minmaxbound(G,'user',1e-40,1e40);

L = zeros(size(Y));
 
label_type = opts.task_type  ;

[L] = link_fun(Y,G,Phi,Omega,opts.task_type_name{label_type});    

L = L.*Omega;

% L = L/size(Y,2);
 
sum_L = sum(L,2); 


end