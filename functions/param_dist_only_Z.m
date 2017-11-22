function d =  param_dist_only_Z(param_pre,param,method)

 
z_pre = [];
z = [];
 
k = length(param.Z);
 

for r = 1:k
 z_pre = [z_pre;param_pre.Z{r}(:)];
 z = [z;param.Z{r}(:)];
end

  


d1 = [ z_pre - z];
 



if strcmp(method,'L1')
d1 = sum(abs(d1));
 
d = d1 ;
elseif strcmp(method,'max')
d_now = 1+abs([ z; ]);
d = abs([d1(:); ]);
d = max(d./d_now);
end


end