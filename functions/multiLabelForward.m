function [data, para] = multiLabelForward(data, para)

nBatch = size(data.x,2);

data.G = para.theta0*[data.x;ones(1,nBatch)];
 
G = X*W;
 
L = zeros(size(data.G));

for label_type = 1:3
    idx = para.task_type == label_type;
    if sum(idx) == 0
        continue
    end    
    [L(idx,:)] = link_fun(Y(idx,:),G(idx,:),Phi(idx,idx),Omega(idx,:),label_type);    
end

L = L.*Omega;
 
sum_L = sum(L,2); 
 
data.y = ex./(ones(size(ex,1),1)*sum(ex)) + single(1e-40);

 
data.J = sum(data.q_eq.*log(data.y));
 