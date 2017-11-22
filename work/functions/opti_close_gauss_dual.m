function  [W,Phi] = opti_close_gauss_dual(W,opts)

%  data

X = opts.X;
Y = opts.Y;

% super parameters
k = opts.k;         % number of columns of Fi, i = 1,2,3; 
[n,d] = size(X);
 
m =  opts.task_num_each_type(1);
rho = opts.rho;
Omega = opts.Omega;
 
 

idx = opts.task_type == 1;
Y = Y(:,idx);
X = bsxfun(@times,X,sqrt(rho));
Y = bsxfun(@times,Y,sqrt(rho));
iter_num = opts.gaussClose.maxIter;
 
n_vis = sum(Omega,1);
th =    n_vis .*opts.gaussClose.lambda;
th = th(idx);
% th = opts.prox_param.W.param;
for it = 1:iter_num
    
    G=X*W;
    
    [Phi] = link_fun_phi_rho(Y,G,Omega,rho,'gauss');
   
    S = zeros(d,m);
   
  
    for j = 1:d
       
        if j == 1
            S(j,:) = -  X(:,j)' * ( Y * Phi) + X(:,j)'*X(:,j+1:end) * W(j+1:end,:);
        elseif j == d
            S(j,:) = -  X(:,j)' * ( Y * Phi) + X(:,j)'*X(:,1:j-1) * W(1:j-1,:);
        else
            S(j,:) = -  X(:,j)' * ( Y * Phi) + X(:,j)'*X(:,1:j-1) * W(1:j-1,:) + X(:,j)'*X(:,j+1:end) * W(j+1:end,:);
        end
     
        norm2 = sum(X(:,j).*X(:,j));
        if j == 1
            W(j,:) =  - S(j,:)./max(norm2,eps);   
        else
            W(j,:) =  - sign(S(j,:)).*max(0, abs(S(j,:)) - th)./max(norm2,eps);    
        end
    end
        
    
    
end
 
 


end