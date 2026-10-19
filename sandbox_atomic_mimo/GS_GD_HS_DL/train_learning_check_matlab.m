% CPU learning/speed diagnostic, not a final proposed estimator.
% Independent train/validation channels; test files are intentionally unused.
clear; clc;
rootFolder = fileparts(mfilename('fullpath'));
dataFolder = fullfile(rootFolder,'learning_check_data');
outputFolder = fullfile(rootFolder,'matlab_learning_check');
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end
pilotCounts = [10 30];
maximumEpochs = 400;
summary = table();
for pilotCase = 1:numel(pilotCounts)
    P = pilotCounts(pilotCase);
    rng(20261018 + P,'twister');
    training = load(fullfile(dataFolder,sprintf('train_P%d.mat',P)));
    validation = load(fullfile(dataFolder,sprintf('validation_P%d.mat',P)));
    [Xtrain,Ttrain] = makeExamples(training);
    [Xvalidation,Tvalidation] = makeExamples(validation);
    inputMean = mean(Xtrain,1);
    inputStd = max(std(Xtrain,0,1),single(1e-5));
    targetScale = sqrt(mean(Ttrain(:).^2));
    Xtrain = (Xtrain-inputMean)./inputStd;
    Xvalidation = (Xvalidation-inputMean)./inputStd;
    Ttrain = Ttrain/targetScale;
    Tvalidation = Tvalidation/targetScale;
    assert(all(isfinite(Xtrain(:))) && all(isfinite(Xvalidation(:))));
    layers = [featureInputLayer(size(Xtrain,2),Normalization='none')
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(48)];
    options = trainingOptions('adam', ...
        MaxEpochs=maximumEpochs,MiniBatchSize=size(Xtrain,1), ...
        InitialLearnRate=1e-3,Shuffle='every-epoch', ...
        ValidationData={Xvalidation,Tvalidation},ValidationFrequency=25, ...
        OutputNetwork='last-iteration',ExecutionEnvironment='cpu', ...
        Plots='none',Verbose=true,VerboseFrequency=100);
    % Last iteration is deliberate: this stage asks whether the training
    % samples can be fitted. Model selection is a separate validation task.
    fprintf('\nP=%d: %d training / %d validation channels; %d epochs\n', ...
        P,size(Xtrain,1),size(Xvalidation,1),maximumEpochs);
    timer = tic;
    [network,info] = trainnet(Xtrain,Ttrain,layers,'mse',options);
    trainingSeconds = toc(timer);
    predictionTrain = minibatchpredict(network,Xtrain);
    predictionValidation = minibatchpredict(network,Xvalidation);
    trainNMSE = sum((predictionTrain-Ttrain).^2,'all')/sum(Ttrain.^2,'all');
    validationNMSE = sum((predictionValidation-Tvalidation).^2,'all')/sum(Tvalidation.^2,'all');
    validationEnergyRatio = sum(predictionValidation.^2,'all')/sum(Tvalidation.^2,'all');
    correlation = sum(predictionValidation.*Tvalidation,'all') / ...
        sqrt(sum(predictionValidation.^2,'all')*sum(Tvalidation.^2,'all'));
    fitPass = trainNMSE < 1e-3; % Local -30 dB training-fit diagnostic only.
    row = table(P,trainingSeconds,trainingSeconds/maximumEpochs, ...
        10*log10(trainNMSE),10*log10(validationNMSE), ...
        validationEnergyRatio,correlation,fitPass, ...
        VariableNames={'P','TrainingSeconds','SecondsPerEpoch','TrainNMSEdB', ...
        'ValidationNMSEdB','ValidationPredictionEnergyRatio', ...
        'ValidationRealVectorCorrelation','TrainingFitPass'});
    summary = [summary;row]; %#ok<AGROW>
    disp(row);
    save(fullfile(outputFolder,sprintf('diagnostic_P%d.mat',P)), ...
        'network','info','inputMean','inputStd','targetScale','row', ...
        'predictionTrain','predictionValidation','maximumEpochs');
    writetable(summary,fullfile(outputFolder,'summary.csv'));
end
fprintf('Diagnostic finished. Final test data has not been loaded.\n');

function [X,T] = makeExamples(data)
    count = size(data.channels,3);
    P = size(data.pilots,2);
    X = zeros(count,8*P+2*3*P+2*8*P+1,'single');
    T = zeros(count,48,'single');
    for sample = 1:count
        Y = data.Ymag(:,:,sample);
        S = data.pilots(:,:,sample);
        B = data.references(:,:,sample);
        G = data.channels(:,:,sample);
        referenceRMS = sqrt(mean(abs(B(:)).^2));
        % Observable features only. Exact nonlinear Y is retained in data;
        % subtraction is input preprocessing, not an idealized measurement.
        centeredMagnitude = Y-abs(B);
        referenceDirection = B/referenceRMS;
        X(sample,:) = single([centeredMagnitude(:);real(S(:));imag(S(:)); ...
            real(referenceDirection(:));imag(referenceDirection(:)); ...
            log(referenceRMS)].');
        T(sample,:) = single([real(G(:));imag(G(:))].');
    end
end
