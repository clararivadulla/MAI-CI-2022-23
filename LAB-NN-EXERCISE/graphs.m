load('results.mat');
writetable(T,'results.xlsx',"WriteMode","append","AutoFitWidth",false);

accuracies1 = T.Accuracy(T.TransOut == "logsig" & T.TrainRatio == 0.80);
sizes1 = T.Size(T.TransOut == "logsig" & T.TrainRatio == 0.80);

accuracies2 = T.Accuracy(T.TransOut == "logsig" & T.TrainRatio == 0.40);
sizes2 = T.Size(T.TransOut == "logsig" & T.TrainRatio == 0.40);

accuracies3 = T.Accuracy(T.TransOut == "logsig" & T.TrainRatio == 0.10);
sizes3 = T.Size(T.TransOut == "logsig" & T.TrainRatio == 0.10);

figure
plot(sizes1,accuracies1,sizes2,accuracies2, sizes3,accuracies3)
title("'logsig' as output layers training function and 'mse' as the cost function")
legend('(0.80, 0.10, 0.10)','(0.40, 0.20, 0.40)', '(0.10, 0.10, 0.80)')
saveas(gcf,'figures/logsig.png')

accuracies4 = T.Accuracy(T.TransOut == "softmax" & T.TrainRatio == 0.80);
sizes4 = T.Size(T.TransOut == "softmax" & T.TrainRatio == 0.80);

accuracies5 = T.Accuracy(T.TransOut == "softmax" & T.TrainRatio == 0.40);
sizes5 = T.Size(T.TransOut == "softmax" & T.TrainRatio == 0.40);

accuracies6 = T.Accuracy(T.TransOut == "softmax" & T.TrainRatio == 0.10);
sizes6 = T.Size(T.TransOut == "softmax" & T.TrainRatio == 0.10);

figure
plot(sizes4,accuracies4,sizes5,accuracies5, sizes6,accuracies6)
title("'softmax' as output layers training function and 'crossentropy' as the cost function")
legend('(0.80, 0.10, 0.10)','(0.40, 0.20, 0.40)', '(0.10, 0.10, 0.80)')
saveas(gcf,'figures/softmax.png')

% Comparisons between 'logsig' and 'softmax'

figure
plot(sizes1,accuracies1,sizes4,accuracies4)
title("Comparison between configurations with trainRatio, valRatio, testRatio = (0.80, 0.10, 0.10)")
legend("'logsig' and 'mse'","'softmax' and 'crossentropy'")
saveas(gcf,'figures/comparison_0_80_0_10_0_10.png')

figure
plot(sizes2,accuracies2,sizes5,accuracies5)
title("Comparison between configurations with trainRatio, valRatio, testRatio = (0.40, 0.20, 0.40)")
legend("'logsig' and 'mse'","'softmax' and 'crossentropy'")
saveas(gcf,'figures/comparison_0_40_0_20_0_40.png')

figure
plot(sizes3,accuracies3,sizes6,accuracies6)
title("Comparison between configurations with trainRatio, valRatio, testRatio = (0.10, 0.10, 0.80)")
legend("'logsig' and 'mse'","'softmax' and 'crossentropy'")
saveas(gcf,'figures/comparison_0_10_0_10_0_80.png')