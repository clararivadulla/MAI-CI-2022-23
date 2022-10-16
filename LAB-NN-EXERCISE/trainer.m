function acc = trainer(X, Y, params)
    net = feedforwardnet(params.size);

    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = params.trainRatio; % Ratio of data used as training set
    net.divideParam.valRatio = params.valRatio; % Ratio of data used as validation set
    net.divideParam.testRatio = params.testRatio; % Ratio of data used as test set

    net.trainParam.max_fail = params.max_fail; % validation check parameter
    net.trainParam.epochs = params.epochs; % number of epochs parameter
    net.trainParam.min_grad = params.min_grad; % minimum performance gradient

    net.layers{1}.transferFcn = params.transHidden; % transfer function for hidden layer
    net.layers{2}.transferFcn = params.transOut; % transfer function for output layer

    net.input.processFcns = {'mapminmax'};
    net.performFcn = params.performFcn; % cost function

    net.trainFcn = params.trainFcn;

    net.trainParam.mc = params.mc; % momentum parameter
    net.trainParam.lr = params.lr; % learning rate parameter

    [~, tr, Y_pred, ~] = train(net, X', Y, 'useGPU', 'yes');
    
    Y_test_pred = Y_pred .* tr.testMask{1}; % mask to get only test data
    Y_test_pred = Y_test_pred(:,~all(isnan(Y_test_pred))); % remove NA columns
    [~, pred_label] = max(Y_test_pred); % get the position (label)

    Y_test_true = Y .* tr.testMask{1}; % mask to get only test data
    Y_test_true = Y_test_true(:,~all(isnan(Y_test_true))); % remove NA columns
    [~, true_label] = max(Y_test_true); % get the position (label)

    acc = 100 - 100*sum( pred_label == true_label  )/length(true_label);
end