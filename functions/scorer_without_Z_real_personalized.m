function [fun_c,mean_sparse_rate,share,mean_spec_rate,nmse,auc,aupr,nmse_poiss] = scorer_without_Z_real_personalized(X,Y,patno,W,Phi,pi,pi_each,Omega_c,Omega_b,opts,method)

Omega_c_save = Omega_c;
Omega_p = 1 - Omega_c;
Omega_c =   Omega_b;
Omega_p =   Omega_b;


uni = unique(patno);
X_all = [];
Y_all = [];
Omega_all = [];
rho_all = [];
fun_all = [];
for i_pat = 1:length(uni)
    idx = find(patno == uni(i_pat));
    n_rec = length(idx);
    if n_rec == 1
        continue
    end
%     idx1 = idx(1:fix(n_rec/2));
%     idx2 = idx(fix(n_rec/2)+1:end);
    idx1 = idx(1:end-1);
    idx2 = idx(2:end);
    
    [fun_c,rho] = estimate_fun_without_Z(X(idx1,:),Y(idx1,:),W,Phi,pi,Omega_c(idx1,:),opts);
    X_all = [ X_all ;X(idx2,:)];
    Y_all = [ Y_all ;Y(idx2,:)];
%     rho_all = [rho_all; repmat(mean(rho,1),[length(idx2),1])];
    rho = historical_sum(rho,[0.2 0.8]);
    rho_all = [rho_all; rho];
    Omega_all = [Omega_all; Omega_c(idx2,:)];         
end
    

if opts.k > 1
    
    [mean_sparse_rate,share,mean_spec_rate] = eval_W(W);
    
    [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_kn_softmax_nmse_auc(X_all,Y_all,W,Phi,rho_all,pi,Omega_all,opts);
    %[ fun_c ] = estimate_fun_giv_rho_without_Z(X,Y,W,Phi,pi,rho,Omega_p,opts);
    [fun_c] = estimate_fun_without_Z(X_all,Y_all,W,Phi,pi,Omega_all,opts);
    
else
    [fun_c] = estimate_fun_without_Z_k1(X_all,Y_all,W,Phi,pi,Omega_all,opts);
    [mean_sparse_rate] = eval_W_k1(W);
    share=1;
    mean_spec_rate=1;
    [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_k1_nmse_auc(X_all,Y_all,W,Phi,pi,Omega_all,opts);
end

% if strcmp(method,'SEP')
%     
%     [fun_c] = estimate_fun_without_Z_SEP_complete(X_all,Y_all,W,Phi,pi,pi_each,Omega_c_save,Omega_b,opts);
%     fun_c = sum(fun_c);
%     
%  
% end
    

end