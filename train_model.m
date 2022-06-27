function [Btrain,Btest] = train_model(Xtrain,Xtest,param)

addpath('model');

% set parameter
param.ks = 7;
param.kz = 7;
param.center = 500;
param.alpha = 1;
param.beta = 1;
param.maxItr = 10;

% normalize
Xtest = [Xtest, ones(size(Xtest,1),1)];
Xtrain = [Xtrain, ones(size(Xtrain,1),1)];
data = [Xtrain;Xtest];
sampleMean = mean(data,1);
Xtest = (Xtest - repmat(sampleMean,size(Xtest,1),1));
Xtrain = (Xtrain - repmat(sampleMean,size(Xtrain,1),1));

% get center points
center_nm = ['center_' num2str(param.center)];
eval(['[~,' center_nm '] = litekmeans(Xtrain, param.center, ''MaxIter'', 10);']);
eval(['center_set.' center_nm '= ' center_nm, ';']);
eval(['center = center_set.' center_nm ';'])

% get similarity matrices S and Z
S = getSimilarMatrixS(Xtrain, center, param.ks);
Z = getSimilarMatrixZ(center', param.kz);


% training the HMH model
W = HMH(Xtrain', S, Z, param.bit, param.alpha, param.beta, param.maxItr);
H = Xtrain*W > 0;
tH = Xtest*W > 0;
Btrain = compactbit(H);
Btest = compactbit(tH);

end