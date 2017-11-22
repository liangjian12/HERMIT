function d =  param_norm_without_Z(param)

 
w = [];
eta = [];
k = length(param.W);
for r = 1:k
 w = [w;param.W{r}(:)];
end

for r = 1:k
 eta = [eta;diag(param.Phi{r})];
end

 
eta = [eta;param.pi(:)];

d1 = sum(abs(w));
d2 = eta ;
d2 = sum(d2.*d2).^0.5;

d = d1+d2;

end