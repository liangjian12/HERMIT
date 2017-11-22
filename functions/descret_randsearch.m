function param_range = descret_randsearch(param_num,param_range,output_param_num)

idx_order = [];
for i = 1:param_num
    tmp = [i * ones(param_num,1) [1:param_num]'];
    idx_order = [idx_order;tmp];
end
param1_record = [];
param2_record = []; 
u = randperm(param_num^2,output_param_num);
for i = 1:output_param_num
    param1_record(i) = param_range{1}(idx_order(u(i),1));
    param2_record(i) = param_range{2}(idx_order(u(i),2));
end
param_range{1} = param1_record;
param_range{2} = param2_record;

end