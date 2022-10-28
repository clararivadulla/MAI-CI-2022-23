%{
    This is a wrapper function for the feedforwardnet network. Where given
    an X, Y, and a set of parameters, it will train a feed forward neural
    network with these parameters. Moreover, it will return the accuracy of
    the TEST dataset of the classifier.
%}

function [acc, acc_tra, acc_val] = trainer(X, Y, params)
    % Initialize the feedforward network with 
    %   Size of hidden layers = params.size
    %   Training function = params.trainFcn
    net = feedforwardnet(params.size, params.trainFcn);
    net = configure(net, X, Y);

    % Set the architecture of the network
    net.layers{1}.transferFcn = params.transHidden; % transfer function for hidden layer
    net.layers{2}.transferFcn = params.transOut; % transfer function for output layer
    net.input.processFcns = {'mapminmax'}; % we normalize each row to be [-1, 1], otherwise GPU complains

    % Set the cost function
    net.performFcn = params.performFcn; 

    % Divide targets into three sets using random indices
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = params.trainRatio; % Ratio of data used as training set
    net.divideParam.valRatio = params.valRatio; % Ratio of data used as validation set
    net.divideParam.testRatio = params.testRatio; % Ratio of data used as test set

    % Set the parameters for the scaled conjugate gradient backpropagation
    net.trainParam.epochs = params.epochs; % maximum epochs
    net.trainParam.max_fail = params.max_fail; % maximum validation errors
    net.trainParam.min_grad = params.min_grad; % minimum performance gradient
    net.trainParam.showWindow  = false; % no display

    % Set some specific parameters of the optimizer
    if strcmp(params.trainFcn, 'trainscg')
        net.trainParam.mu = params.mu; % Marquardt adjustment 
        net.trainParam.sigma = params.sigma; % Change in weight for second derivative approximation\
    elseif strcmp(params.trainFcn, 'traingdx')
        net.trainParam.lr = params.lr; % Learning Rate
        net.trainParam.mc = params.mc; % Momentum constant
    elseif strcmp(params.trainFcn, 'trainrp')
        net.trainParam.lr = params.lr; % Learning rate
    end


    % Train the network, using GPU and get the predictions and tr object
    [~, tr, Y_pred, ~] = train(net, X, Y, 'useGPU', 'yes');
    
    % Calcualte Accuracy
    % Test accuracy
    Y_test_pred = Y_pred(:, tr.testInd); % subset of only test data for predicted
    [~, pred_label] = max(Y_test_pred); % get the position (label)
    
    Y_test_true = Y(:, tr.testInd); % subset of only test data for the actual labels
    [~, true_label] = max(Y_test_true); % get the position (label)

    acc = 100*sum( pred_label == true_label  )/length(true_label);

    % Validation accuracy (to see if we overfit)
    [~, pred_label] = max(Y_pred(:, tr.valInd)); % get the position (label)
    [~, true_label] = max(Y(:, tr.valInd)); % get the position (label)
    acc_val = 100*sum( pred_label == true_label  )/length(true_label);

    % Train accuracy (to see if we overfit)
    [~, pred_label] = max(Y_pred(:, tr.trainInd)); % get the position (label)
    [~, true_label] = max(Y(:, tr.trainInd)); % get the position (label)
    acc_tra = 100*sum( pred_label == true_label  )/length(true_label);

end