function [x, t] = apg_decomp(grad_f, prox_h, dim_x, opts)
%
% apg v0.1b (@author bodonoghue)
%
% Implements an Accelerated Proximal Gradient method
% (Nesterov 2007, Beck and Teboulle 2009)
%
% solves: min_x (f(x) + h(x)), x \in R^dim_x
%
% where f is smooth, convex and h is non-smooth, convex but simple
% in that we can easily evaluate the proximal operator of h
%
% returns solution and last-used step-size (the step-size is useful
% if you're solving a similar problem many times serially, you can
% initialize apg with the last use step-size
%
% this takes in two function handles:
% grad_f(v,opts) = df(v)/dv (gradient of f)
% prox_h(v,t,opts) = argmin_x (t*h(x) + 1/2 * norm(x-v)^2)
%                       where t is the step size at that iteration
% if h = 0, then use prox_h = [] or prox_h = @(x,t,opts)(x)
% put the necessary function data in opts fields
%
% implements something similar to TFOCS step-size adaptation (Becker, Candes and Grant 2010)
% and gradient-scheme adaptive restarting (O'Donoghue and Candes 2013)
%
% quits when norm(y(k) - x(k+1)) < EPS * max(1, norm(x(k+1))
%
% optional opts fields defined are below (with defaults)
% to use defaults simply call apg with opts = []
X_INIT1 = zeros(dim_x,1); % initial starting point
X_INIT2 = zeros(dim_x,1); % initial starting point
USE_RESTART = true; % use adaptive restart scheme
MAX_ITERS = 2000; % maximum iterations before termination
EPS = 1e-6; % tolerance for termination
ALPHA = 1.01; % step-size growth factor
BETA = 0.5; % step-size shrinkage factor
QUIET = false; % if false writes out information every 100 iters
GEN_PLOTS = true; % if true generates plots of norm of proximal gradient
USE_GRA = false; % if true uses UN-accelerated proximal gradient descent (typically slower)
STEP_SIZE = []; % starting step-size estimate, if not set then apg makes initial guess
FIXED_STEP_SIZE = false; % don't change step-size (forward or back tracking), uses initial
% step-size throughout, only useful if good
% STEP_SIZE set

if (~isempty(opts))
    if isfield(opts,'X_INIT');X_INIT1 = opts.initial_scale *  randn(dim_x,1);X_INIT2 = opts.initial_scale *  randn(dim_x,1);end
    if isfield(opts,'USE_RESTART');USE_RESTART = opts.USE_RESTART;end
    if isfield(opts,'MAX_ITERS');MAX_ITERS = opts.MAX_ITERS;end
    if isfield(opts,'EPS');EPS = opts.EPS;end
    if isfield(opts,'ALPHA');ALPHA = opts.ALPHA;end
    if isfield(opts,'BETA');BETA = opts.BETA;end
    if isfield(opts,'QUIET');QUIET = opts.QUIET;end
    if isfield(opts,'GEN_PLOTS');GEN_PLOTS = opts.GEN_PLOTS;end
    if isfield(opts,'USE_GRA');USE_GRA = opts.USE_GRA;end
    if isfield(opts,'STEP_SIZE');STEP_SIZE = opts.STEP_SIZE;end
    if isfield(opts,'FIXED_STEP_SIZE');FIXED_STEP_SIZE = opts.FIXED_STEP_SIZE;end
end

% if quiet don't generate plots
GEN_PLOTS = GEN_PLOTS & ~QUIET;

if (GEN_PLOTS); errs = zeros(MAX_ITERS,2);end

x1 = X_INIT1; y1=x1;
x2 = X_INIT2; y2=x2;
g = grad_f(y1+y2,opts);
theta = 1;

if (isempty(STEP_SIZE) || isnan(STEP_SIZE))
    if(false)
        % perturbation for first step-size estimate:
        T = 10; dx = T*ones(dim_x,1); g_hat = nan;
        while any(isnan(g_hat))
            dx = dx/T;
            x1_hat = x1 + dx;
            x2_hat = x2 + dx;
            g_hat = grad_f(x1_hat+x2_hat,opts);
        end
        t = norm(x1+x2 - x1_hat-x2_hat)/norm(g - g_hat);
    else
        % Barzilai-Borwein step-size initialization:
        t = 1 / norm(g);
        x1_hat = x1 - t*g;
        x2_hat = x2 - t*g;
        g_hat = grad_f(x1_hat+x2_hat,opts);
        err = g - g_hat;
        t = abs(( x1+x2 - x1_hat-x2_hat )'*err / (sum(err.*err))) ;
    end
    clear x_hat g_hat
else
    t = STEP_SIZE;
end

for k=1:MAX_ITERS
    
    if (~QUIET && mod(k,100)==0)
        fprintf('iter num %i, norm(tGk): %1.2e, step-size: %1.2e\n',k,err1,t);
    end
    
    x1_old = x1;x2_old = x2;
    y1_old = y1;y2_old = y2;
    
    x1 = y1 - t*g;
    x2 = y2 - t*g;
    
    if ~isempty(prox_h)
        x1 = prox_h(x1,t,opts,1);
        x2 = prox_h(x2,t,opts,2);
    end
    
    err1 = norm(y1+y2-x1-x2)/max(1,norm(x1+x2));
   
    
    if (GEN_PLOTS)
        errs(k,1) = err1;
        %err2 = norm(x-x_old)/max(1,norm(x));
        %errs(k,2) = err2;
    end
    
    if (err1 < EPS)
        break;
    end
    
    if(~USE_GRA)
        theta = 2/(1 + sqrt(1+4/(theta^2)));
    else
        theta = 1;
    end
    
    if (USE_RESTART && (y1+y2-x1-x2)'*(x1+x2-x1_old-x2_old)>0)
        x1 = x1_old;x2 = x2_old;
        y1 = x1;y2 = x2;
        theta = 1;
    else
        y1 = x1 + (1-theta)*(x1-x1_old);
        y2 = x2 + (1-theta)*(x2-x2_old);
    end
    
    g_old = g;
    g = grad_f(y1+y2,opts);
 
    % TFOCS-style backtracking:
    if (~FIXED_STEP_SIZE)
        err = y1+y2-y1_old-y2_old;
        t_hat = 0.5*(sum(err.*err))/abs(err'*(g_old - g));
        t = min( ALPHA*t, max( BETA*t, t_hat ));
    end
	if any(isnan(x1(:))) || any(isnan(x2(:)))
	    x1=x1_old;
        x2=x2_old;
		break;
	end
    
end

x = x1+x2;

% if ~isempty(prox_h)
%     x = prox_h(x,t,opts);
% end

if (~QUIET)
    fprintf('iter num %i, norm(tGk): %1.2e, step-size: %1.2e\n',k,err1,t);
    fprintf('Terminated\n');
end
if (GEN_PLOTS)
    errs = errs(1:k,:);
    figure();semilogy(errs(:,1));
    xlabel('iters');title('norm(tGk)')
    %figure();semilogy(errs(:,2));
    %xlabel('iters');title('norm(Dxk)')
end

end