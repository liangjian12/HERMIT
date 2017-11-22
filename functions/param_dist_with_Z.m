function d =  param_dist_with_Z(param_pre,param,method)

w_pre = [];
w = [];
z_pre = [];
z = [];
eta = [];
eta_pre = [];
k = length(param.W);

 w_pre = [param_pre.W(:)];
 w = [w;param.W(:)];


for r = 1:k
 z_pre = [z_pre;param_pre.Z{r}(:)];
 z = [z;param.Z{r}(:)];
end

for r = 1:k
 eta_pre = [eta_pre;diag(param_pre.Phi{r})];
 eta = [eta;diag(param.Phi{r})];
end

eta_pre = [eta_pre;param_pre.pi(:)];
eta = [eta;param.pi(:)];



d1 = [w_pre - w;z_pre - z];
d2 = eta - eta_pre;



if strcmp(method,'L1+L2')
d1 = sum(abs(d1));
d2 = sum(d2.*d2).^0.5;
d = d1+d2;
elseif strcmp(method,'max')
d_now = 1+abs([w;z;eta]);
d = abs([d1(:);d2(:)]);
d = max(d./d_now);
end


end