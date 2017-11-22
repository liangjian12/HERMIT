function [fun_c,Z,rho] = eval_with_Z(X,Y,W,Phi,pi,Omega,opts,prox_param)

     
     
opts.Omega = Omega;
opts.fix.W=W;
opts.fix.Phi=Phi;
opts.fix.pi=pi;
opts.Phi = Phi;
  
[~,Z,fun] = GEM_only_Z(X,Y,opts.k,opts.tau,opts.gamma,prox_param,opts);
if opts.plot_GEM
figure;plot(fun)
end

[fun_c,rho] = estimate_fun_with_Z(X,Y,W,Z,Phi,pi,Omega,opts);



end