function [W,Z,Phi,pi] = solver_sub_with_Z(X,Y,k,opts,prox_param)
 

if strcmp(opts.opti_method_with_Z,'multi_comp')
    if opts.warmFlag
        [~,pi,W,Z,Phi,fun] = GEM_with_Z_multi_comp_warm_start(X,Y,k,opts.lambda,opts.tau,opts.gamma,prox_param,opts);   
    else
        [~,pi,W,Z,Phi,fun] = GEM_with_Z_multi_comp(X,Y,k,opts.lambda,opts.tau,opts.gamma,prox_param,opts);   
    end
elseif strcmp(opts.opti_method_with_Z,'multi_comp_AD') 
    if opts.warmFlag
        [~,pi,W,Z,Phi,fun] = GEM_with_Z_multi_comp_AD_warm_start(X,Y,k,opts.lambda,opts.tau,opts.gamma,prox_param,opts);   
    else
        [~,pi,W,Z,Phi,fun] = GEM_with_Z_multi_comp_AD(X,Y,k,opts.lambda,opts.tau,opts.gamma,prox_param,opts);   
    end
elseif strcmp(opts.opti_method_with_Z,'AD') 
    if opts.warmFlag
        [~,pi,W,Z,Phi,fun] = GEM_with_Z_AD_warm_start(X,Y,k,opts.lambda,opts.tau,opts.gamma,prox_param,opts);
    else
        [~,pi,W,Z,Phi,fun] = GEM_with_Z_AD(X,Y,k,opts.lambda,opts.tau,opts.gamma,prox_param,opts);
    end
end

if opts.plot_GEM
figure;plot(fun);
end



end