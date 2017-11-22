function [Z,pi] = solver_sub_only_Z(X,Y,k,opts,prox_param) 
 
if opts.k > 1     
[~,Z,pi] = GEM_only_Z(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);
else
opts.fix.pi = 1;
[~,Z] = GEM_only_Z_k1(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);
pi = 1;
end

if opts.plot_GEM
figure;plot(fun);
end


end