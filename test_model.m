function [mAP,Rec,Pre,Precision,Recall,Precision100,Fmeasure] = test_model(Xtrain,Ytrain,Xtest,Ytest,param)

addpath('utils')
runtimes = 10;

for m = 1:length(param.bits)
    param.bit = param.bits(m);
    for n = 1:runtimes
        [Btrain,Btest] = train_model(Xtrain,Xtest,param);
        [mAP{m,n},Rec{m,n},Pre{m,n},Precision{m,n},Recall{m,n},Fmeasure{m,n},Precision100{m,n}]...
            = test_traditional_mAP_Pre_Rec_Fmeasure(Ytrain,Ytest,Btrain,Btest,param);
    end
end

end