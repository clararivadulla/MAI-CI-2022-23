%{
    This is the main function for the feedforwardnet network. Where it will
    read the caltech101 silhouettes 28x28 data and given a set of
    experiments, run them and saves the accuracies.
%}

clear

% Load the data
load('caltech101_silhouettes_28.mat');

% One-hot encoding scheme
Y_one_hot = onehotencode(categorical(Y'), 2)';

% Set CONSTANT parameters
params.max_fail = 1000; % this will cause overfitting, should be reduced in a real application
params.epochs = 1000; 
params.min_grad = 1e-20; % this will cause overfitting, should be reduced in a real application
params.trainFcn = 'trainscg';
params.mu = 0.0005; 
params.sigma = 5.0e-5;

% Set the parameters we will grid search
splitting = [0.8 0.1 0.1; 0.4 0.2 0.4; 0.1 0.1 0.8]; % Data splittings to test
in_func = ["logsig"; "poslin"]; % Different activation functions for the inner layer
out_and_cost = ["logsig" "softmax"; "mse" "crossentropy"]; % pairs of activation function outer and cost func
sizes = {50; 100; 200; 500;};  % [50 100]; [50 100 50] % since we save it in a table, we cannot put the multilayer with the others

% Table to save configurations and accuracies achieved
Size = [];
TransOut = [];
TransIn = [];
PerformFct = [];
TrainRatio = [];
ValRatio = [];
TestRatio = [];
Accuracy = [];
AccuracyVali = [];
AccuracyTrai = [];

% Test with different number of hidden layers
for i = 1:size(sizes, 1)
    params.size = sizes{i};
    
    % The different partition configurations
    for j = 1:size(splitting, 1)
        params.trainRatio = splitting(j, 1); 
        params.valRatio = splitting(j, 2); 
        params.testRatio = splitting(j, 3); 

        % The different in activation functions
        for k = 1:size(in_func, 1)
            params.transHidden = in_func(k, 1); % we could do different ones for each layer, but not done in this case 

            % The different in activation functions
            for z = 1:size(out_and_cost, 1)
                params.transOut = out_and_cost(1, z); 
                params.performFcn = out_and_cost(2, z);

                % train and get the mean accuracy for each partition
                [acc, accT, accV] = evaluator(X', Y_one_hot, params, 3)

                % Save the results, this could be optimized by
                % preallocating the space, but this is not the bottleneck
                % of the function by a long shot
                Size = [Size; params.size];
                TransOut = [TransOut; params.transOut];
                TransIn = [TransIn; params.transHidden];
                PerformFct = [PerformFct; params.performFcn];
                TrainRatio = [TrainRatio; params.trainRatio];
                ValRatio = [ValRatio; params.valRatio];
                TestRatio = [TestRatio; params.testRatio];
                Accuracy = [Accuracy; acc];
                AccuracyVali = [AccuracyVali; accV];   
                AccuracyTrai = [AccuracyTrai; accT];                
            end
        end
    end
end

% this has to be manually changed, if the size is an array remove Sizes
% from here:
T = table(Size, TransOut, TransIn, PerformFct, TrainRatio, ValRatio, TestRatio, Accuracy, AccuracyVali, AccuracyTrai);
save('results.mat', 'T')

%{
    Function that will call the trainer function from trainer.m, it will
    run it n_runs times with the given X, Y, params. It will return the
    mean accuracy from these n_runs, for the test, train and validation. To see if
    we are overfitting.
%}
function [accuracy, acc_trai, acc_vali] = evaluator(X, Y, params, n_runs)
    accuracy = 0;
    acc_vali = 0; % validation accuracy
    acc_trai = 0; % train accuracy
    for r = 1:n_runs
        [acc, acc_tra, acc_val] = trainer(X, Y, params);
        accuracy = accuracy + acc;
        acc_vali = acc_vali + acc_val;
        acc_trai = acc_trai + acc_tra;
    end
    accuracy = accuracy/n_runs;
    acc_vali = acc_vali/n_runs;
    acc_trai = acc_trai/n_runs;
end
