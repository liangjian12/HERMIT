function W_out = sover_softmax_regression(X,rho,lambda,method)


opts.prox_param.W_softmax.method = method;
opts.prox_param.W_softmax.param = lambda;
opts.prox_param.W_softmax.fix_feature_flag = 0;
opts.prox_param.W_softmax.fix_feature = ones(size(X,2),size(rho,2));
opts.GEN_PLOTS = 0;
opts.QUIET = 1;


[W_out] = apg_softmax_regression(X,rho,opts);

% disp(norm(W(:)-W_out(:)));
% 
% G = X*W_out;
% rho_out = softmax(G')';




end