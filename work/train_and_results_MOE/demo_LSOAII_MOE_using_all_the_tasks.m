clc;clear;close all
%% super parameter
rng(0);
parfor_flag = 1; %1: use 'parfor' command to tune hyper-parameters parallel; 0: do not tune hyper-parameters parallel
rep_num = 100; %the number of replications to report results
param_num = 40; %the number of parameters to tune
tune_num = 1; %the number of replications to tune each setting of hyper-parameter
param_start = 0; %logarithm of the first hyper-parameter to tune
param_end = -4; %logarithm of the last hyper-parameter to tune
gamma = 1; %\gamma in the paper
flag_use_pre_tuned_best_k = true; %false: tune k on validation set  

method_name = {'Mix','Mix GS'}; 
method_set = [1 2]; %methods to run, 1:Mix, 2:Mix GS
 
% coefficient of regularization
prox_param = []; 
prox_param.W.method = 'L1';  % regularization method, options are : 'L2' for L2 norm; 'L1' for L1 norm;  'EN' for Elastic Net;  'GS' for Group Sparsity, which is only valid for matrix; 
prox_param.W.fix_feature_flag = true; %true: only perform proximal operator at every 11th (St?dler et al, 2010); false: perform proximal operator in every GEM iteration.
 
% optimization setting
opts = [];
opts.initial_scale_scale = 1.0e-5;     % initial_scale for the paramter for optimization 
opts.maxIter_out = 50;    %the maximum number of GEM iteration

opts.gamma = gamma; %\gamma in the paper
  
opts.sep_pen.flag = false;      %seprate regularization for each type of tasks for method Mix
opts.sep_pen.lambda_set = [];   %if seprate regularization, specify each regularization parameter for each type, e.g., opts.sep_pen.lambda_set = [1e-3, 1e-3, 1e-3]
opts.flag_decomp = false;       %true : decompose W = L + S, where L and S are all matrices

opts.opti_Phi_flag =false;      %true: optimize Phi for Gaussian targets.

opts.stop_param.fun  = 1e-6;    %stop criterion by loglikelihood 
opts.stop_param.param = 1e-3;   %stop criterion by parameters

opts.warm_start.flag = false;   %warm start inside GEM
opts.max_lambda = 10;           %max regularization parameter when performing warm start inside GEM
opts.warmFlag = false;          %warm start when tuning hyper-parameters

opts.max_norm_regu.flag = true; %true: restrain paramter space by max norm when there are Poisson targets.
opts.max_norm_regu.W.param = [10 1]; %the first element is max norm restraint for bias, the second element is for the rest of parameter

opts.plot_GEM = false; %true: plot GEM loglikelihood change of all iterations.

%optimization settings of the original APG matlab function
opts.MAX_ITERS = 200;          % the maximum number of iteration of the APG optimization.
opts.EPS = 1e-6;               % used for the stopping critierion, which is based on the relative error of two successive parameter vectors.
opts.QUIET = true;             % true: do not show the loglikelihood change of all iterations of APG. 
 
%% data
data_name = {'data_processed_LSOAII.mat',...
             'data_processed_easySHARE.mat'};
data_keyword = {'LSOAII','easySHARE'};           
for i_data =1:1  %choose data

str=data_name{i_data}; 
load(str);       %load data

d = size(train_X,2); %dimension of features
m = size(train_Y,2); %number of tasks 
n = size(train_X,1); %number of training samples
 
opts.d_total = d;

opts.task_type = task_type; %task type for each task, 1: Gaussian task, 2: Bernoulli task, 3: Poisson task
opts.task_type_name = task_type_name; %name of each type of task, e.g., {'gauss','bnl','poiss'} mean Gaussian, Bernoulli and Poisson
opts.task_num_each_type = [task_num_each_type]; %number of tasks for each type, e.g., [2,3,4] means 2 Gaussian tasks, 3 Bernoulli tasks, 4 Poisson tasks.
 
valid_Omega_b = valid_Omega; %indices of observed targets of validation set
test_Omega_b = test_Omega;   %indices of observed targets of testing set

%no effect for feature-based prediction.
observe_rate_c = 0.5; %how much percentage of targets are allowed to use for testing
observe_rate_p = 1 - observe_rate_c; %how much percentage of targets are allowed to use for testing
valid_Omega_c = double(rand(size(valid_Y))<observe_rate_c); % randomly generate indices of targets that are allowed to use for testing for validation set
test_Omega_c = double(rand(size(test_Y))<observe_rate_c);   % randomly generate indices of targets that are allowed to use for testing for testing set
 
opts.task_pred_weight = ones(1,m);  %select tasks to compute averaged performance, 1: selected, 0: not selected.
  
%save settings
opts_save = opts;
prox_param_save = prox_param;
train_Omega_save = train_Omega;
valid_Omega_b_save = valid_Omega_b;
test_Omega_b_save = test_Omega_b;
valid_Omega_c_save = valid_Omega_c;
test_Omega_c_save = test_Omega_c;  
opts_save_save = opts_save;
prox_param_save_save = prox_param_save;
save(sprintf('Omega_data_%s.mat',data_keyword{i_data}),'valid_Omega_c','test_Omega_c');



%% process
record = zeros(rep_num, 7, length(method_set)); %record scores, such nMSE, aAUC for each replications
record_param = {};     %record hyper-parameters
record_best_k = {};    %record the number of clusters, which is also tuned
record_nmi = {};       %record how much heterogeneity can be learned from only features, i.e., concordance between p(\delta|x,\alpha) and p(\delta|y,x,\theta)
for i_method = 1:length(method_set)  % for each method

j_method = method_set(i_method);
disp(method_name{j_method}) %display the name of the method
disp('**************************************************************')

idx_task = ones(1,size(test_Y,2));
 
Y_pred_all_groups = zeros([size(test_Y),rep_num]);  %record predict mean value of targets for each replication.
Omega_p_all_groups = zeros([size(test_Y),rep_num]); %record indices of targets to predict for each replication.
W_all_groups = zeros([d,m,rep_num]);
nmi_method = {};  %record concordance between p(\delta|x,\alpha) and p(\delta|y,x,\theta) for each method

uni = unique(idx_task); %unique value of cluster label of task clustering
num_group = length(uni);%number of groups of tasks
best_k_all_groups = zeros(length(uni),1);      %record the number of clusters
best_param_all_groups = cell(length(uni),1);   %record the hyper-parameters
record_theta_all_groups = cell(length(uni),1); %record regression parameters 

for i_group = 1:num_group %for each group of tasks
disp('--------------------------')    
fprintf('The %d-th group of tasks.\n',i_group)
opts_save = opts_save_save;
prox_param_save = prox_param_save_save;
opts = opts_save;
prox_param = prox_param_save;

str = data_name{i_data}; %load data
load(str);

idx =  find(idx_task==uni(i_group));
disp('The indices of tasks in this group')
disp(idx)
idx_task_this = idx;

%select the tasks in this group
train_Y = train_Y(:,idx); 
valid_Y = valid_Y(:,idx);
test_Y = test_Y(:,idx);
task_type = task_type(idx);
task_num_each_type = [sum(task_type == 1) sum(task_type == 2) sum(task_type == 3)];
train_Omega = train_Omega_save(:,idx);
valid_Omega_b = valid_Omega_b_save(:,idx);
test_Omega_b = test_Omega_b_save(:,idx);
valid_Omega_c = valid_Omega_c_save(:,idx);
test_Omega_c = test_Omega_c_save(:,idx);

d = size(train_X,2);
m = size(train_Y,2);
n = size(train_X,1);

opts.d_total = d;
opts.task_type = task_type;
opts.task_type_name = task_type_name;
opts.task_num_each_type = [task_num_each_type];

opts.task_pred_weight = ones(1,m);  

opts_save = opts;
prox_param_save = prox_param;
    
    
record_method = zeros(rep_num,7); %record scores, such nMSE, aAUC for each replications
opts_save.sep_pen.flag = false;   %true: seprate regularization for each type of tasks for method Mix
opts_save.flag_MOE = false;           %true: use MOE.

%solver
if j_method == 1
solver = @solver_MIX_without_Z; % Mix
elseif j_method == 2
solver = @solver_ORE_without_Z; % Mix GS
end

%change bias of range of hyper-parameter tuning
if ismember(j_method,[2]) 
   param_ratio = 10;  
else    
   param_ratio = 1; 
end

%solver for hyper-parameter tuning
solver_cv = solver;

flag_cv = 1; %1: tuning regularization parameters

 
if ismember(j_method,1) 
    flag_cv_sep = 1; % seprate regularization for each type of tasks for method Mix
else
    flag_cv_sep = 0;
end
 
if i_data == 1
    th = 1;
elseif i_data == 2 %use more penalty
    if ismember(j_method,1)
        th = 0.5;
    elseif ismember(j_method,2)
        th = 0.7;
    end
end
  
tic
best_param = cell(1,2);
param_range = cell(1,2);
    
param_range{1} = logspace(param_start,param_end,param_num) * param_ratio;
param_range_save = param_range;

if ismember(j_method,1)
opts_save.sep_pen.flag = true;
else
opts_save.sep_pen.flag = false;
end
 
disp('choose param')
opts_save.k=1; %tuning hyper-parameter using pre-fixed k = 1
if flag_cv_sep %for Mix, seprate regularization for each type of tasks 
    [best_param{1},score] = choose_param_on_valid_MTL_randmiss_complete_real(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
    opts_save.k,opts_save,prox_param_save,param_range{1},solver_cv,@scorer_without_Z_real_int_complete_cv,3,parfor_flag,tune_num,th);    
else %for Mix GS
    [best_param{1},score] = choose_param_on_valid_randmiss_complete_real(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
            opts_save.k,opts_save,prox_param_save,param_range{1},solver_cv,@scorer_without_Z_real_int_complete_cv,parfor_flag,tune_num,th);
end
 
disp('tuning k, the number of clusters')
 
if ~flag_use_pre_tuned_best_k %tune k 
    [best_k,score] = decide_k(train_X,train_Y,valid_X,valid_Y,train_Omega,valid_Omega_c,valid_Omega_b,...
        opts_save,prox_param_save,solver,@scorer_without_Z_real_int_complete_cv,500,best_param,j_method,1);   
else %use pre-tuned k
    load(sprintf('tuned_best_k_data_%s_using_all_the_tasks.mat',data_keyword{i_data}));
    best_k = record_tuned_best_k(i_group);
end
 
disp('best regularization parameters:')
disp(log10(best_param{1}))
disp('range of regularization parameters:')
disp(log10([param_range_save{1}(1) param_range_save{1}(end)]))
  
disp('best k:')
disp(best_k)

 
k = best_k; 
opts_save.k = best_k;
%since the hyper-parameter is tuned using pre-fixed k = 1, use more penalization when k > 1
if  ismember(j_method,1)
    best_param{1} = best_param{1} * k; %according to our theoretical results in Section 5
elseif ismember(j_method,2)
    best_param{1} = best_param{1} * sqrt(k); %according to our theoretical results in Section 5
end

%set hyper-parameters
opts_save.warmFlag =false;
opts_save.lambda = best_param{1};
opts_save.lambda_set = best_param{1};
opts_save.sep_pen.lambda_set = best_param{1} ;
  
%tuning regularization parameter for MOE
opts_save.flag_MOE = 1; %use MOE
% lambda_softmax = 1e-3;
if best_k == 1          %if k == 1, no concept of FMR or MOE
    lambda_softmax = 1;
else %k > 1
lambda = logspace(-6,0,24); % range of regularization 
record_score = zeros(size(lambda)); 
parfor i = 1:length(lambda)
    opts = opts_save; %load setting
    prox_param = prox_param_save;
    opts.lambda_MOE = lambda(i);
    score_tmp = zeros(10,1);
    for it = 1:24
        [W,Phi,W_softmax] = solver(train_X,train_Y,train_Omega,k,opts,prox_param); %train the model on training data, W is \Beta, W_softmax is \alpha
        rho_train = softmax_pred(train_X,W_softmax); %compute cluster soft label using x and \alpha, i.e., p(\delta|x,\alpha)
        pi = mean(rho_train); %compute pi
        rho_valid = softmax_pred(valid_X,W_softmax); %use \alpha to compute cluster label of validation data
        [score] = scorer_without_Z_real_GL_cv_giv_rho(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_b,opts,rho_valid); %use cluster label and \Beta to predict Y, then compute score
        score_tmp(it) = sum(score); 
    end
    score_tmp = sort(score_tmp,'descend'); %maxmize score to select regularization parameter for \alpha
    record_score(i) = mean(score_tmp(1:5));%reduce variance
end
[c_min,idx] = max(record_score); %maxmize score to select regularization parameter for \alpha
lambda_softmax = lambda(idx);
end
 
opts_save.lambda_MOE = lambda_softmax ; %set regularization parameter
W_record_all = {}; %record W, i.e., \Beta
pi_record = {}; %record pi
W_softmax_record = {}; %record \alpha
 
Y_pred_record = zeros([size(test_Y),rep_num]); %record predicted mean value of targets
Omega_p_record = zeros([size(test_Y),rep_num]);%record indices of targets to predict
W_record = zeros([d,m,rep_num]);
score_record = zeros(rep_num,1);
nmi_group = zeros(rep_num,4);
parfor it = 1:rep_num
opts = opts_save;
prox_param = prox_param_save;
[W,Phi,W_softmax] = solver(train_X,train_Y,train_Omega,k,opts,prox_param); %%train the model on training data, W is \Beta, W_softmax is \alpha
W_record(:,:,it) = W(:,:,1);
 
rho_train = softmax_pred(train_X,W_softmax); %compute cluster soft label using x and \alpha, i.e., p(\delta|x,\alpha)
pi = mean(rho_train); %compute pi
     
rho_test = softmax_pred(test_X,W_softmax); %use \alpha to compute cluster label of testing data
[Y_pred_record(:,:,it),Omega_p_record(:,:,it)] = FMR_predict_GL_giv_rho(test_X,test_Y,W,Phi,pi,[], test_Omega_b,opts,[],rho_test); %use cluster label and \Beta to predict Y

rho_valid = softmax_pred(valid_X,W_softmax); %use \alpha to compute cluster label of validation data
[score] = scorer_without_Z_real_GL_cv_giv_rho(valid_X,valid_Y,W,Phi,pi,[],valid_Omega_b,opts,rho_valid); %use cluster label and \Beta to predict Y, then compute score
score_record(it) = sum(score);

%compute how much heterogeneity can be learned from only features, i.e., concordance between p(\delta|x,\alpha) and p(\delta|y,x,\theta)
[~,rho_train_fmr] = estimate_fun_without_Z_MOE(train_X,train_Y,W,Phi,W_softmax,train_Omega,opts); %equation (24)
[~,rho_test_fmr] = estimate_fun_without_Z_MOE(test_X,test_Y,W,Phi,W_softmax,test_Omega_b,opts);   %equation (24)
nmi_train_true = 0;
nmi_train_fmr = nmi_prob(rho_train_fmr,rho_train);
nmi_test_true = 0;
nmi_test_fmr = nmi_prob(rho_test_fmr,rho_test);
nmi_group(it,:) = [nmi_train_true,nmi_train_fmr,nmi_test_true,nmi_test_fmr];

%record paramters
W_record_all{it} = W; %\Beta
W_softmax_record{it} = W_softmax; %\alpha
pi_record{it} = pi; %pi
end
[~,idx] = sort(score_record,'descend'); %sort replications by scores on the validation set

%reorder results by scores on the validation set
Y_pred_record = Y_pred_record(:,:,idx);
Omega_p_record = Omega_p_record(:,:,idx);
nmi_group = nmi_group(idx,:);
W_record = W_record(:,:,idx);
W_record_all = W_record_all(idx);
W_softmax_record = W_softmax_record(idx);
pi_record = pi_record(idx);

%record the results of this group
Y_pred_all_groups(:,idx_task_this,:) = Y_pred_record;
Omega_p_all_groups(:,idx_task_this,:) = Omega_p_record;
W_all_groups(:,idx_task_this,:) = W_record;
nmi_method{i_group} = nmi_group;

%record all the parameters and hyper-parameters
best_k_all_groups(i_group) = best_k;
best_param_all_groups{i_group} = best_param;
record_theta_all_groups{i_group}.W_record = W_record_all;
record_theta_all_groups{i_group}.pi_record = pi_record;
record_theta_all_groups{i_group}.W_softmax_record = W_softmax_record;
record_lambda_softmax_all_groups{i_group} = lambda_softmax;
end

%load setting to compute averaged performance of all the tasks (from all the groups)
opts_save = opts_save_save;
prox_param_save = prox_param_save_save;

%load data
str = data_name{i_data};
load(str);
 
parfor it = 1:rep_num %for each replication
    Y_pred = Y_pred_all_groups(:,:,it); 
    Omega_p = Omega_p_all_groups(:,:,it);
    W = W_all_groups(:,:,it); 
    
    opts = opts_save;
    prox_param = prox_param_save;
    
    [sparse,share,specific,nmse,auc,aupr,nmse_poiss] = scorer_without_Z_real_with_pred(test_Y,Y_pred,W,Omega_p,opts,[]); %compute nMSE for Gaussian tasks, aAUC for Bernoulli tasks, and nMSE for Poisson tasks
    record_method(it,:) = [sparse,share,specific,nmse,auc,aupr,nmse_poiss] ;
end

%show the sparseness, nMSE of Gaussian tasks, aAUC of Bernoulli tasks
for it = 1:rep_num
fprintf('sparse = %.4f, nmse = %.4f, auc =%.4f\n',...
    record_method(it,1),record_method(it,4),record_method(it,5))
end

%record all the results and parameters
best_k = best_k_all_groups;
best_param = best_param_all_groups;
record_theta{j_method} = record_theta_all_groups;
record_nmi{j_method} = nmi_method;
record_param{j_method} = best_param;
record_Y_pred_all_groups{j_method} = Y_pred_all_groups;
record_Omega_p_all_groups{j_method} = Omega_p_all_groups;
record_best_k{j_method} = best_k;
record(:,:,i_method) = record_method;
record_lambda_softmax{j_method} = record_lambda_softmax_all_groups;

report = mean(record_method(1:20,:));
nMSE = report(4);
aAUC = report(5);
disp('##############################')
fprintf('mean value of nMSE: %.4f, mean value of aAUC: %.4f\n',nMSE,aAUC)
disp('##############################')
save(sprintf('result_nMSE_aAUC_MOE_using_all_the_tasks_data_%s_method_%s.mat',data_keyword{i_data},method_name{j_method})); 

toc
 
end

str = sprintf('result_MOE_using_all_the_tasks_data_%s.mat',data_keyword{i_data});
save(str,'record','record_param','record_best_k','method_set','record_nmi','record_theta','record_Y_pred_all_groups','record_Omega_p_all_groups','record_lambda_softmax','-v7.3');
end
