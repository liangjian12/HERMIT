function [W,Phi,pi,W_k1,Phi_k1,pi_outlier] = solver_sub_without_Z_outlier(X,Y,k,opts,prox_param)
 



% if opts.k > 1
%     if opts.warmFlag
% [~,pi,W,Phi,fun] = GEM_without_Z_multi_comp_warm_start(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);
%     else
% [~,pi,W,Phi,fun] = GEM_without_Z_multi_comp(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);        
%     end
% else
% [~,pi,W,Phi] = GEM_without_Z_k1(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);
% end

% 
if opts.k > 1
 [~,pi,W,Phi,W_k1,Phi_k1,pi_outlier] = GEM_without_Z_outlier(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);
else
[~,pi,W,Phi] = GEM_without_Z_k1(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);
W_k1 = W;
Phi_k1 = Phi;
pi_outlier = ones(1,size(Y,2));
end



if opts.plot_GEM
figure;plot(fun);
end


end