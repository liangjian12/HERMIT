function [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_kn_softmax_nmse_auc_MAR(X,Y,W,Phi,rho,pi,Omega,opts)
 
k = opts.k;

 
G = zeros([size(Y,1) k]);

for r = 1:k
G(:,r) = X*W(:,r); 
end

nmse = 100;
auc = -1;
aupr = -1;
nmse_poiss = 100;

  
label_type = opts.task_type;

mu = link_fun_mu(G,opts.task_type_name{label_type});
mu = bsxfun(@times,mu,rho);
mu = sum(mu,2);

y_true = Y;
omega = Omega;

if label_type == 1
    y_pred = mu;

    idx = omega>0;
    if sum(idx)>0
        n = sum(idx);
        nmse =  sum(((y_pred(idx) - y_true(idx)).^2),1)  ./(n * (var(y_true(idx))+eps));
    end


elseif label_type == 2
    y_pred = mu;
     
    idx = omega(:)>0;
    if sum(idx)>0

         
        if length(unique(y_true(idx)))<=1
            auc = 0.5;
            aupr = 0.5;
        else
            [~,~,~,auc] = perfcurve(y_true(idx),y_pred(idx),1);  
            [~,~,~,aupr] = perfcurve(y_true(idx),y_pred(idx),1, 'xCrit', 'reca', 'yCrit', 'prec');  

        end

    end
         
else
    y_pred = mu;
     
    idx = omega(:)>0;
    if sum(idx)>0

        n = sum(idx);
        nmse_poiss =  sum(((log(y_pred(idx)+1) -log( y_true(idx) + 1)).^2),1)  ./(n * (var(log(y_true(idx)+1))+eps));

    end
end

end

  