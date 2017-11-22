function [auc1_matlab,auc1_approx,auc_approx_k,auc_mean,auc_unimode,auc_max_loss,auc_adaboost,auc_softmax] = approx_AUC(Y,X,W,rho,pi,weight)
k = size(rho,2);
[~,idx_comp] = max(rho,[],2);
G = zeros(size(Y));
for r = 1:k
idx_r = idx_comp == r;
G(idx_r) = X(idx_r,:)*W{r}; 
end
mu = 1./(1+exp(-G));
[~,~,~,auc1_matlab] = perfcurve(Y,mu,1);
[auc1_approx] = approx_AUC_sub(Y,mu);

G = zeros([size(Y,1) k]);
for r = 1:k
G(:,r) = X*W{r}; 
end
mu = 1./(1+exp(-G));

sum_L = zeros([size(Y,1) k]); 
for r = 1:k
sum_L(:,r) = loss_bnl(Y,G(:,r));
end

mean_L =  mean(sum_L,2);
sum_L = bsxfun(@minus,sum_L,mean_L);
rho = exp(sum_L);
rho = bsxfun(@times,rho,pi);    
rho = bsxfun(@rdivide,rho,sum(rho,2)); 
if any(isnan(rho(:))) || any(isinf(rho(:)))
    rho(isnan(rho)) = 1/m;
    rho(isinf(rho)) = 1/m;
    rho = bsxfun(@rdivide,rho,sum(rho,2)); 
end
rho = max(rho,0);
rho = min(rho,1);     

mu_softmax = sum(mu.*rho,2);
[~,~,~,auc_softmax] = perfcurve(Y,mu_softmax,1);

[~,~,~,auc_adaboost] = perfcurve(Y,mu*weight,1);



l1 = zeros([size(Y,1) k]); 
for r = 1:k
l1(:,r) = loss_bnl(ones(size(Y)),G(:,r));
end
l0 = zeros([size(Y,1) k]); 
for r = 1:k
l0(:,r) = loss_bnl(zeros(size(Y)),G(:,r));
end
l1 = exp(l1) * pi(:);
l0 = exp(l0) * pi(:);
[~,idx] = max([l1 l0],[],2);
idx(idx==2)=0;
[~,~,~,auc_max_loss] = perfcurve(Y,idx,1);

auc_approx_k = 0.5;% [auc_approx_k] = approx_AUC_sub_k(Y,mu,pi);

mu = mu * pi(:);
[~,~,~,auc_mean] = perfcurve(Y,mu,1);

y=Y;
y(y==0)=-1;
c_set = 2.^[-9:9];method_num = 0;
% [model,max_auc,param_c,record_aAUC] = Tree_10foldCV(X,y,c_set,method_num);
[model,auc_unimode,param_c,record_aAUC] = liblinear_10foldCV(X,y,c_set,method_num);
% w = model.w(:);
% G = X*w;
% mu = 1./(1+exp(-G));
% [~,~,~,auc_unimode] = perfcurve(Y,mu,1);

end