function [nmse,auc,nmse_poiss] = estimate_fun_without_Z_kn_nmse_auc(X,Y_save,W,Phi,rho,pi,Omega_save,opts)

[~,idx_comp] = max(rho,[],2);
k = opts.k;

nmse_record = [];
auc_record = [];

for r = 1:k
    
    idx_r = idx_comp == r;

G = X(idx_r,:)*W(:,:,r);
n = sum(idx_r);
Y = Y_save(idx_r,:);
Omega = Omega_save(idx_r,:);

nmse = 100;
auc = -1;
nmse_poiss = 100;

for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    mu = link_fun_mu(G(:,idx),opts.task_type_name{label_type});
%     y_pred = link_fun_rnd_mean_gauss_no_err(mu,Phi{1}(idx,idx),opts.task_type_name{label_type});
    y_true = Y(:,idx);
    omega = Omega(:,idx);
    if label_type == 1
        y_pred = mu;
        for i = 1:size(y_true,2)
            idx_i = omega(:,i)>0;
            mse(i) =  sum(((y_pred(idx_i,i) - y_true(idx_i,i)).^2),1)  ./(n * (var(y_true(idx_i,i))+eps));
        end
        nmse = mean(mse);
    elseif label_type == 2
        y_pred = mu;
        for i = 1:size(y_true,2)
            idx_i = omega(:,i)>0;
%             auc(i) = scoreAUC(y_true(idx_i,i)>0,y_pred(idx_i,i)>0);  
            if length(unique(y_true(idx_i,i)))<=1
                auc(i) = 0.5;
            else
            [~,~,~,auc(i)] = perfcurve(y_true(idx_i,i),y_pred(idx_i,i),1);  
            end
            
            
        end
        auc = mean(auc);
    else
        y_pred = mu;
        for i = 1:size(y_true,2)
            idx_i = omega(:,i)>0;
            mse(i) =  sum(((log(y_pred(idx_i,i)+1) -log( y_true(idx_i,i) + 1)).^2),1)  ./(n * (var(log(y_true(idx_i,i)+1))+eps));
        end
        nmse_poiss = mean(mse);
        
    end

end

nmse_record(r) = nmse;
auc_record(r) = auc;
nmse_poiss_record(r) = nmse_poiss;

end

nmse = sum(nmse_record.*pi);
auc = sum(auc_record.*pi);
nmse_poiss = sum(nmse_poiss_record.*pi);

end