clc;clear;close all
data_keyword = {'LSOAII','easySHARE'};  
folder_name = 'fig';
mkdir(folder_name);

for i_data = 1:2
disp(data_keyword{i_data})
%load similarity matrices 
str = sprintf('result_task_clustering_similarity_NMI_data_%s.mat',data_keyword{i_data});
load(str,'record_similarity');
sim_mat = mean(record_similarity{1}{1},3); %use mean of the matrices 

%show the similarity matrix
figure;
imagesc(sim_mat)
colorbar

save_flag = 1;
xlabel_name = 'task ID';
ylabel_name = 'task ID';
if i_data == 1
save_name = sprintf('Fig10a');
else
save_name = sprintf('Fig11a');    
end
xlabel(xlabel_name,'fontsize',30,'interpreter','latex');
ylabel(ylabel_name,'fontsize',25,'interpreter','latex');
set(gca,'FontSize',16);
if save_flag
   mkdir(folder_name);
   str = sprintf('%s/%s.png',folder_name,save_name);
   saveas(gcf,str);
end


if i_data == 1
%use Kernel PCA to reduce dimension to 2
[c] = kernel_pca_my(sim_mat, 2);
%use kmeans to cluster the tasks into 4 groups
idx_task = kmeans(c,4);
%save the results if required
flag_save_task_clustering_results = false;
if flag_save_task_clustering_results
save(sprintf('cluster_idx_task_data_%s.mat',data_keyword{i_data}),'idx_task');
end
%show descriptions of each group of tasks
load task_description_LSOAII
cluster_num = length(unique(idx_task));
idx = {};
for i = 1:length(unique(idx_task))
    idx{i} = find(idx_task == i);
    fprintf('descriptions of the %d-th group of tasks:\n',i);
    disp(lower(task_description(idx{i})))
end
%plot the tasks in 2D space
figure
hold on
color_set = {'hr','pb','dm','sc'};
name_set = {'group 1','group 2','group 3','group 4'};
H = [];
for i = 1:cluster_num 
    h =  plot(c(idx{i},1),c(idx{i},2),color_set{i},'markersize', 10, 'linewidth',2);
    H = [H h];
end 
legend(H, name_set{1},name_set{2},name_set{3},name_set{4},'Location','best')
save_flag = 1;
xlabel_name = 'projection dimension 1';
ylabel_name = 'projection dimension 2';
save_name = sprintf('Fig10b');
xlabel(xlabel_name,'fontsize',30,'interpreter','latex');
ylabel(ylabel_name,'fontsize',25,'interpreter','latex');
set(gca,'FontSize',16);
if save_flag
   mkdir(folder_name);
   str = sprintf('%s/%s.png',folder_name,save_name);
   saveas(gcf,str);
end

end
     

if i_data == 2
%use Kernel PCA to reduce dimension to 2    
[c] = kernel_pca_my(sim_mat, 2);
%use kmeans to cluster the tasks into 4 groups
 idx_task = kmeans(c,4);
%save the results if required
flag_save_task_clustering_results = false;
if flag_save_task_clustering_results
save(sprintf('cluster_idx_task_data_%s.mat',data_keyword{i_data}),'idx_task');
end
%show descriptions of each group of tasks  
load task_description_easySHARE
load interview_module_description_easySHARE
cluster_num = length(unique(idx_task));
idx = {};
for i = 1:length(unique(idx_task))
    idx{i} = find(idx_task == i);
    fprintf('descriptions of the %d-th group of tasks:\n',i);
    disp(lower(task_description(idx{i})))
    fprintf('descriptions of interview module of the %d-th group of tasks:\n',i);
    disp(lower(interview_module_description(idx{i})))
end
%plot the tasks in 2D space
figure
hold on
color_set = {'hr','pb','dm','sc'};
name_set = {'group 1','group 2','group 3','group 4'};
H = [];
for i = 1:cluster_num 
    h =  plot(c(idx{i},1),c(idx{i},2),color_set{i},'markersize', 10, 'linewidth',2);
    H = [H h];
end 
legend(H, name_set{1},name_set{2},name_set{3},name_set{4},'Location','best')
save_flag = 1;
xlabel_name = 'projection dimension 1';
ylabel_name = 'projection dimension 2';
save_name = sprintf('Fig11b');
xlabel(xlabel_name,'fontsize',30,'interpreter','latex');
ylabel(ylabel_name,'fontsize',25,'interpreter','latex');
set(gca,'FontSize',16);
if save_flag
   mkdir(folder_name);
   str = sprintf('%s/%s.png',folder_name,save_name);
   saveas(gcf,str);
end

end

end
  