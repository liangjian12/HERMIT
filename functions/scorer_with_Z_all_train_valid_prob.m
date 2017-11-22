function [score] = scorer_with_Z_all_train_valid_prob(train_X,train_Y,valid_X,valid_Y,train_Y_outlier,train_Z_idx,valid_Z_idx,...
W,Z,Phi,pi,train_Omega,valid_Omega,opts)

    [~,~,~,~,~,AUC_W,~,~,~,~,AUC_Z_train,~] = scorer_with_Z_all_train(train_X,train_Y,train_Y_outlier,train_Z_idx,W,Z,Phi,pi,train_Omega,opts);
    [~,~,~,~,~,~,~,~,~,~,AUC_Z_valid] = scorer_with_Z_all_test_prob(valid_X,valid_Y,valid_Z_idx,W,Phi,pi,valid_Omega,opts);

    score = AUC_W+AUC_Z_train;
    

end