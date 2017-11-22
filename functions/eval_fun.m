function mse = eval_fun(Y_out,Y_true)

err = Y_true(:) - Y_out(:);
mse = sum(err.*err)/numel(err);

end