function D = dist_fun_AUC(w_true,w_out)

D = - roc_curve_my(w_out,w_true);

end