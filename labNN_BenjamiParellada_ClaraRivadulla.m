% Load the data
load('caltech101_silhouettes_28.mat');

% One-hot encoding scheme
Y_one_hot = full(ind2vec(Y, 8671));

% Create a Neural Network with 
% different number of hidden units: 50, 200 and 500
net = feedforwardnet(50);
% net = feedforwardnet(200);
% net = feedforwardnet(500);

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.1; % Ratio of data used as training set
net.divideParam.valRatio = 0.1; % Ratio of data used as validation set
net.divideParam.testRatio = 0.8; % Ratio of data used as test set

net.trainParam.max_fail = 7; % validation check parameter
net.trainParam.epochs=4000; % number of epochs parameter
net.trainParam.min_grad = 1e-6; % minimum performance gradient

% 1) and 2) logsig for the hidden layer
for i =1:(length(net.layers)-1)
    net.layers{i}.transferFcn = 'logsig'; 
end                

% 1) logsig the output layer transfer functions
net.layers{end}.transferFcn = 'logsig';

% 2) softmax the output layer transfer functions
% net.layers{end}.transferFcn = 'softmax

net.outputs{:}.processFcns={};

% 1) mean squared error cost function
net.performFcn = 'mse';

% 1) cross-entropy cost function
% 2) net.performFcn = 'crossentropy';

net.trainFcn= 'trainlm';

[net,tr,Y,E] = train(net, X.', Y_one_hot);

view(net)

fprintf('Accuracy: %f\n',100 - 100 * sum(abs((Y > 0.5) - T))/length(T))