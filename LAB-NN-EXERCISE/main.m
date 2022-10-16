% Load the data
load('caltech101_silhouettes_28.mat');

% Number of classes
n_classes = 101;

% One-hot encoding scheme
Y_one_hot = full(ind2vec(Y, n_classes));

% set parameters for nnet
params.size = 50;
params.trainRatio = 0.8; 
params.valRatio = 0.1; 
params.testRatio = 0.1; 
params.max_fail = 6; 
params.epochs = 2000; 
params.min_grad = 1e-5;
params.transHidden = 'logsig'; 
params.transOut = 'logsig'; 
params.performFcn = 'mse';
params.trainFcn = 'trainrp';
params.mc = 0.8; 
params.lr = 0.01;

% test different functions
params.transHidden = 'logsig'; 
params.transOut = 'logsig'; 
params.performFcn = 'mse';
t1 = evaluator(X, Y_one_hot, params, 3)

params.transHidden = 'logsig'; 
params.transOut = 'softmax'; 
params.performFcn = 'crossentropy';
t2 = evaluator(X, Y_one_hot, params, 3)

% compare t1 vs t2, set the values of these 

% test different hidden layers
params.size = 50;
t3 = evaluator(X, Y_one_hot, params, 3)

params.size = 200;
t4 = evaluator(X, Y_one_hot, params, 3)

params.size = 500;
t5 = evaluator(X, Y_one_hot, params, 3)

% compare t3 vs t4 vs t5, set the values of these 

% test different train validation test ratios
params.trainRatio = 0.80; 
params.valRatio = 0.10; 
params.testRatio = 0.10; 
t6 = evaluator(X, Y_one_hot, params, 3)

params.trainRatio = 0.40; 
params.valRatio = 0.20; 
params.testRatio = 0.40; 
t7 = evaluator(X, Y_one_hot, params, 3)

params.trainRatio = 0.10; 
params.valRatio = 0.10; 
params.testRatio = 0.80; 
t8 = evaluator(X, Y_one_hot, params, 3)

% function to calculate the mean of different runs
function accuracy = evaluator(X, Y, params, n_runs)
    accuracy = 0;
    for r = 1:n_runs
        accuracy = accuracy + trainer(X, Y, params);
    end
    accuracy = accuracy/n_runs;
end
