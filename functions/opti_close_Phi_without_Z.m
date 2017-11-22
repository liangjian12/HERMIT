function opts = opti_close_Phi_without_Z(x,opts)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
  
 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Phi = opts.Phi; 
Omega = opts.Omega;
rho = opts.rho;
 
 
for label_type = 1:1
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    
    if label_type == 1
        X_tmp = bsxfun(@times,X,sqrt(rho));
        Y_tmp = bsxfun(@times,Y,sqrt(rho));
    else
        X_tmp = X;
        Y_tmp = Y;
    end
    
    G = X_tmp*W(:,idx);
    
    if   opts.opti_Phi_flag
        [Phi(idx,idx)] = link_fun_phi_rho(Y_tmp(:,idx),G,Omega(:,idx),rho,opts.task_type_name{label_type});
    else 
        [Phi(idx,idx)] = link_fun_phi_rho_fix(Y_tmp(:,idx),G,Omega(:,idx),rho,opts.task_type_name{label_type});
    end
 
end
 
opts.Phi = Phi;


end