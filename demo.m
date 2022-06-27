
clear,clc
close all

addpath('data');

% set parameters
save_path = 'results/';
param.data_name = 'FLICKR25K';
param.bits = [8,16,32,64,96,128];
param.dataset = 'flickr_db_test_relu7';

% load data
load(param.dataset);
Xtrain = double(data_set);
Ytrain = double(dataset_L);
Xtest = double(test_data);
Ytest = double(test_L);

% test HMH model
[mAP,Rec,Pre,Precision,Recall,Precision100,Fmeasure] = test_model(Xtrain,Ytrain,Xtest,Ytest,param);
save([save_path,param.data_name,'.mat'],'mAP','Rec','Pre','Precision','Recall','Precision100','Fmeasure');







