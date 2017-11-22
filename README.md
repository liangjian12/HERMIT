# HERMIT
HEterogeneous-target Robust MIxTure regression

Two real-world data sets are in folder data.
LSOA II data set is in data/LSOAII
easySHARE data set is in data/easySHARE

Descriptions of data sets are in respective folders.

Installation: add all the files in HERMIT in your search path.

To see imputation performances of FMR models, learning all the tasks:
run demo_LSOAII_FMR_using_all_the_tasks.m
or demo_easySHARE_FMR_using_all_the_tasks.m

To see imputation performances of FMR models, handling anomaly tasks:
run demo_LSOAII_FMR_by_Handling_anomaly_tasks.m
or demo_easySHARE_FMR_by_Handling_anomaly_tasks.m

To see imputation performances of FMR models, clustering tasks:
run demo_LSOAII_FMR_by_Task_clustering.m
or demo_easySHARE_FMR_by_Task_clustering.m

To see feature-based prediction performances of MOE models, run similar .m files in data/train_and_results_MOE. 
File names differentiate on the keyword "FMR" and "MOE".

To see concordant scores of tasks:
open folder detect_anomaly_tasks
firstly run demo_LSOAII_detect_anomaly_tasks.m or demo_easySHARE_detect_anomaly_tasks.m,
then run demo_show_concordant_scores_handling_anomaly_tasks.m

To see task clustering results:
open folder task_clustering
firstly run demo_task_clustering_similarity_by_NMI_LSOAII.m or demo_task_clustering_similarity_by_NMI_easySHARE.m,
then run demo_task_clustering_KernelPCA_kmeans.m
