% Load the data
load('caltech101_silhouettes_28.mat');

% Number of classes
n_classes = 101;

% One-hot encoding scheme
Y_one_hot = full(ind2vec(Y, n_classes));

% data splittings to test
splitting = [0.80 0.10 0.10; 0.40 0.20 0.40; 0.10 0.10 0.80];
out_and_cost = ["logsig" "softmax"; "mse" "crossentropy"];

% table to save configurations and accuracies achieved
Size = [];
TransOut = [];
PerformFct = [];
TrainRatio = [];
ValRatio = [];
TestRatio = [];
Accuracy = [];

% set parameters for nnet
params.max_fail = 6; 
params.epochs = 2000; 
params.min_grad = 1e-5;
params.transHidden = 'logsig'; 
params.trainFcn = 'trainrp';
params.mc = 0.8; 
params.lr = 0.01;

% test with different number of hidden layers
for i = 50:50:500

    params.size = i;
    

    for j = 1:size(splitting, 1)

        params.trainRatio = splitting(j, 1); 
        params.valRatio = splitting(j, 2); 
        params.testRatio = splitting(j, 3); 

        for z = 1:size(out_and_cost, 1)

            params.transOut = out_and_cost(1, z); 
            params.performFcn = out_and_cost(2, z);
            acc = evaluator(X, Y_one_hot, params, 3);

            Size = [Size; params.size];
            TransOut = [TransOut; params.transOut];
            PerformFct = [PerformFct; params.performFcn];
            TrainRatio = [TrainRatio; params.trainRatio];
            ValRatio = [ValRatio; params.valRatio];
            TestRatio = [TestRatio; params.testRatio];
            Accuracy = [Accuracy; acc];
            
        end
    end
end
T = table(Size, TransOut, PerformFct, TrainRatio, ValRatio, TestRatio, Accuracy);

% function to calculate the mean of different runs
function accuracy = evaluator(X, Y, params, n_runs)
    accuracy = 0;
    for r = 1:n_runs
        accuracy = accuracy + trainer(X, Y, params);
    end
    accuracy = accuracy/n_runs;
end
