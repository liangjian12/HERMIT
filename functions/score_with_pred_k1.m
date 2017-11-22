function [nmse,auc,aupr,nmse_poiss] = score_with_pred_k1(Y_true,Y_pred,Omega,opts) 
nmse = 100;
auc = -1;
aupr = -1;
nmse_poiss = 100;
 
weight_save = opts.task_pred_weight;
for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    mu = Y_pred(:,idx);
%     y_pred = link_fun_rnd_mean_gauss_no_err(mu,Phi{1}(idx,idx),opts.task_type_name{label_type});
    y_true = Y_true(:,idx);
    omega = Omega(:,idx);
    weight = weight_save(idx);
    if label_type == 1
        y_pred = mu;
        mse = zeros(1,size(y_true,2));
        for i = 1:size(y_true,2)
            idx_i = omega(:,i)>0;
            if sum(idx_i)==0
                weight(i) = 0;
                continue
            end
            n = sum(idx_i);
            mse_i =  sum(((y_pred(idx_i,i) - y_true(idx_i,i)).^2),1)  ./(n * (var(y_true(idx_i,i))+eps));
%             mse_i =  sum(((y_pred(idx_i,i) - y_true(idx_i,i)).^2),1)  ./n;
            mse(i) = mse_i;
        end
        nmse = sum(weight.*mse)/(max(eps,sum(weight)));
    elseif label_type == 2
        y_pred = mu;
        auc = zeros(1,size(y_true,2));
        aupr = zeros(1,size(y_true,2));
        for i = 1:size(y_true,2)
            idx_i = omega(:,i)>0;
            if sum(idx_i)==0
                weight(i) = 0;
                continue
            end
            n = sum(idx_i);
%             auc(i) = scoreAUC(y_true(idx_i,i)>0,y_pred(idx_i,i)>0);  
            if length(unique(y_true(idx_i,i)))<=1
                auc_i = 0.5;
                aupr_i = 0.5;
            else
            [~,~,~,auc_i] = perfcurve(y_true(idx_i,i),y_pred(idx_i,i),1);  
            [~,~,~,aupr_i] = perfcurve(y_true(idx_i,i),y_pred(idx_i,i),1, 'xCrit', 'reca', 'yCrit', 'prec');  
            end
            auc(i) = auc_i;
            aupr(i) = aupr_i;
        end
        auc = sum(weight.*auc)/(max(eps,sum(weight)));
        aupr = sum(weight.*aupr)/(max(eps,sum(weight)));
    else
        y_pred = mu;
        mse = zeros(1,size(y_true,2));
        for i = 1:size(y_true,2)
            idx_i = omega(:,i)>0;
            if sum(idx_i)==0
                weight(i) = 0;
                continue
            end
            n = sum(idx_i);
            mse_i =  sum(((log(y_pred(idx_i,i)+1) -log( y_true(idx_i,i) + 1)).^2),1)  ./(n * (var(log(y_true(idx_i,i)+1))+eps));
%             mse_i =  sum(((log(y_pred(idx_i,i)+1) -log( y_true(idx_i,i) + 1)).^2),1)  ./(n );
            mse(i) = mse_i;
        end
        nmse_poiss = sum(weight.*mse)/(max(eps,sum(weight)));
        
    end

end

end