function [W,Phi,pi] = solver_sub_without_Z(X,Y,k,opts,prox_param)
 



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
    
    if isfield(opts,'flag_MOE')
        if opts.flag_MOE
            [~,pi,W,Phi,fun] = GEM_without_Z_MOE(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);  
        else
            [~,pi,W,Phi,fun] = GEM_without_Z(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);     
        end
        
    else

            
            if opts.warmFlag
        [~,pi,W,Phi,fun] = GEM_without_Z_warm_start(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);
            else
        [~,pi,W,Phi,fun] = GEM_without_Z(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);     
            end
    
    end
else
[~,pi,W,Phi] = GEM_without_Z_k1(X,Y,k,opts.lambda,opts.gamma,prox_param,opts);
end



if opts.plot_GEM
figure;plot(fun);
end


end