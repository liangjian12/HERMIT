clc;clear;close all
data_keyword = {'LSOAII','easySHARE'};  
folder_name = 'fig'; %folder to save figures
mkdir(folder_name);

for i_data = 1:2
disp(data_keyword{i_data})
%load concordant scores
str = sprintf('result_detect_anomaly_tasks_data_%s.mat',data_keyword{i_data});
load(str,'record_concordant_score');
record = record_concordant_score{2}{1}; %use the results by Mix GS
record =  record(1:10,:); %select results with high scores on validation data
mean_miss_record = squeeze(mean(record,1)); 
[mean_miss_record,idx] = sort(mean_miss_record,'descend'); %sort the concordant scores

%display descriptions of the sorted tasks by concordant scores
load task_description_LSOAII
fprintf('descriptions of the sorted tasks by concordant scores:\n');
disp(lower(task_description(idx)))

%display sorted concordant scores
figure;
x_axis = 1:size(mean_miss_record,2);
plot(x_axis,mean_miss_record,'.b','linewidth',4);
xlabel('reordered task ID','fontsize',30);
ylabel('concordant score','fontsize',25);
xlim([x_axis(1) x_axis(end)])
set(gca,'FontSize',25);
mkdir(folder_name);
if i_data == 1
str = sprintf('%s/%s.png',folder_name,'Fig9a');
else
str = sprintf('%s/%s.png',folder_name,'Fig9b');    
end
saveas(gcf,str);

end