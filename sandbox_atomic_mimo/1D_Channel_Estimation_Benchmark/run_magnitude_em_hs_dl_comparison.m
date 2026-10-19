% Magnitude-only 1D comparison: EM-GS, HS-GS, and a plain MLP.
% Run this script directly. Outputs go to magnitude_em_hs_dl_results.
% Source attribution and all settings: magnitude_em_hs_dl_settings.md.
% This is an adapted Fig.3-style experiment, not a literal reproduction.
% Xu: https://arxiv.org/html/2503.08985 (Eqs.7-9 and Section V).
% Hankel: local ThesisProj__Copy__Haatim.pdf, simulation settings/Table I.
% No linearized measurement model is used anywhere in this script.
clear; clc; close all;

%% 1. Settings and where they come from
cfg.K = 3;                         % Xu Section V
cfg.N = 8;                         % Xu Fig.3
cfg.P = [10 30];                    % Xu Fig.3
cfg.SNRdB = -5:5:30;                % Xu Fig.3 sweep
cfg.TestTrials = 500;               % User choice, not paper trial count
cfg.RSRdB = 12;                     % Hankel simulation setting, not Xu
cfg.Spacing = 0.5;                  % Hankel half-wavelength ULA
cfg.FinalIterations = 100;          % Hankel fixed final-run budget
cfg.SelectionIterations = 25;       % Hankel candidate-rank budget
cfg.CadzowSweeps = 4;               % Hankel Cadzow budget per update
cfg.TrainFraction = 0.70;            % Hankel held-out rank selection
cfg.TrainingSamples = 12000;        % Local DL choice, per P
cfg.ValidationSamples = 2000;       % Local DL choice, per P
cfg.Epochs = 50;                    % Local DL choice
cfg.BatchSize = 256;                % Local DL choice
cfg.LearningRate = 1e-3;             % Local DL choice, Adam
cfg.HiddenUnits = [256 128];         % Plain fully connected MLP, local choice
cfg.TrainingSeed = 2026101401;
cfg.ValidationSeed = 2026101402;
cfg.TestSeed = 2026101403;
% L_k uniform integers 3..7; alpha_lk CN(0,1); S entries CN(0,1): Xu.
% Gaussian polarization components N(0,1/3): Xu distribution.
% Sharing each path polarization across cells: structural adaptation.
% User and reference polarization vectors are independently drawn; each is
% spatially shared within its own path, not the same draw for every path.
% Broadside range (-pi/2,pi/2): Hankel; uniform sampling: local choice.
% theta_axis=pi/2-beta; phi=2*pi*(d/lambda)*cos(theta_axis): Xu cosine.
% Raw reference gain CN(0,10): Xu; known reference symbols = 1: local.
% Reference rescaled PER realization so mean|B|^2 / average user power = RSR.
% SNR = mean|G*S|^2 / sigma2; excludes reference power. Hankel convention;
% using sample-averaged field power is an explicit implementation choice.
% Units: divide G,B,W,Z by the fixed dipole_y/hbar factor in Xu Eq.7/9.
% This changes no relative power, likelihood concentration, or channel NMSE.
% Set dipole=[0;1;0], hbar=1 to implement this fixed unit conversion.
assert(license('test','Neural_Network_Toolbox') == 1, ...
    'This benchmark requires MATLAB Deep Learning Toolbox.');
outputFolder = fullfile(fileparts(mfilename('fullpath')), ...
    'magnitude_em_hs_dl_results');
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end
methods = {'EM-GS','HS-GS','DL (MLP)'};
errors = zeros(cfg.TestTrials,numel(cfg.SNRdB),numel(cfg.P),3);
energies = zeros(cfg.TestTrials,numel(cfg.SNRdB),numel(cfg.P));
ranks = zeros(cfg.TestTrials,numel(cfg.SNRdB),numel(cfg.P));
runtimes = zeros(size(errors));
networks = cell(numel(cfg.P),1);
normalization = cell(numel(cfg.P),1);
trainingSeconds = zeros(numel(cfg.P),1);
resumeTrial = 0;
checkpointPath = fullfile(outputFolder,'checkpoint.mat');
if exist(checkpointPath,'file')
    previous = load(checkpointPath);
    % Resume only an identical experiment (test trial count may increase).
    oldCfg = rmfield(previous.cfg,'TestTrials');
    newCfg = rmfield(cfg,'TestTrials');
    if isequal(oldCfg,newCfg)
        resumeTrial = min(previous.completedTrials,cfg.TestTrials);
        errors(1:resumeTrial,:,:,:) = previous.errors(1:resumeTrial,:,:,:);
        energies(1:resumeTrial,:,:) = previous.energies(1:resumeTrial,:,:);
        ranks(1:resumeTrial,:,:) = previous.ranks(1:resumeTrial,:,:);
        runtimes(1:resumeTrial,:,:,:) = previous.runtimes(1:resumeTrial,:,:,:);
        trainingSeconds = previous.trainingSeconds;
        fprintf('Resuming from %d completed test trials.\n',resumeTrial);
    end
end

%% 2. Train a separate plain MLP for each pilot count
for pilotCase = 1:numel(cfg.P)
    P = cfg.P(pilotCase);
    modelPath = fullfile(outputFolder,sprintf('mlp_P%d.mat',P));
    if exist(modelPath,'file')
        cached = load(modelPath);
        if isequal(rmfield(cached.cfg,'TestTrials'),rmfield(cfg,'TestTrials'))
            networks{pilotCase} = cached.network;
            normalization{pilotCase} = cached.normalizer;
            if isfield(cached,'trainingElapsedSeconds')
                trainingSeconds(pilotCase) = cached.trainingElapsedSeconds;
            end
            fprintf('Reusing trained MLP for P=%d.\n',P);
            continue;
        end
    end
    fprintf('\nGenerating independent DL data for P=%d...\n',P);
    rng(cfg.TrainingSeed+pilotCase,'twister');
    [Xtrain,Ytrain] = makeLearningData(cfg,P,cfg.TrainingSamples);
    rng(cfg.ValidationSeed+pilotCase,'twister');
    [Xvalidation,Yvalidation] = makeLearningData(cfg,P,cfg.ValidationSamples);
    % Fit every normalization parameter on TRAINING data only.
    inputMean = mean(Xtrain,1);
    inputStd = std(Xtrain,0,1);
    inputStd(inputStd < 1e-6) = 1;
    targetScale = sqrt(mean(Ytrain(:).^2));
    Xtrain = (Xtrain-inputMean)./inputStd;
    Xvalidation = (Xvalidation-inputMean)./inputStd;
    Ytrain = Ytrain/targetScale;
    Yvalidation = Yvalidation/targetScale;
    layers = [featureInputLayer(size(Xtrain,2),'Normalization','none')
        fullyConnectedLayer(cfg.HiddenUnits(1))
        reluLayer
        fullyConnectedLayer(cfg.HiddenUnits(2))
        reluLayer
        fullyConnectedLayer(2*cfg.N*cfg.K)
        regressionLayer];
    options = trainingOptions('adam', ...
        'InitialLearnRate',cfg.LearningRate, ...
        'MaxEpochs',cfg.Epochs,'MiniBatchSize',cfg.BatchSize, ...
        'Shuffle','every-epoch','ValidationData',{Xvalidation,Yvalidation}, ...
        'ValidationFrequency',ceil(cfg.TrainingSamples/cfg.BatchSize), ...
        'OutputNetwork','best-validation-loss', ...
        'ExecutionEnvironment','cpu','Plots','none','Verbose',false);
    fprintf('Training plain MLP for P=%d (%d epochs maximum)...\n',P,cfg.Epochs);
    timer = tic;
    [networks{pilotCase},trainingInfo] = trainNetwork(Xtrain,Ytrain,layers,options);
    trainingSeconds(pilotCase) = toc(timer);
    normalization{pilotCase} = struct('Mean',inputMean,'Std',inputStd, ...
        'TargetScale',targetScale);
    network = networks{pilotCase};
    normalizer = normalization{pilotCase};
    trainingElapsedSeconds = trainingSeconds(pilotCase);
    save(modelPath,'network','normalizer','trainingInfo','cfg','trainingElapsedSeconds');
    fprintf('P=%d training finished in %.1f seconds.\n',P,trainingSeconds(pilotCase));
end

%% 3. Test every method on the exact same held-out observations
rng(cfg.TestSeed,'twister');
for trial = 1:cfg.TestTrials
    [G,~] = generateUserChannel(cfg.N,cfg.K,cfg.Spacing,[0;1;0],1);
    Sfull = (randn(cfg.K,max(cfg.P))+1j*randn(cfg.K,max(cfg.P)))/sqrt(2);
    Braw = generateReferencePath(cfg.N,max(cfg.P),1,cfg.Spacing,[0;1;0],1);
    for pilotCase = 1:numel(cfg.P)
        P = cfg.P(pilotCase);
        S = Sfull(:,1:P);
        B = scaleReference(G,S,Braw(:,1:P),cfg.RSRdB);
        userField = G*S;
        signalPower = mean(abs(userField(:)).^2);
        order = randperm(P);
        T = order(1:round(cfg.TrainFraction*P));
        V = order(round(cfg.TrainFraction*P)+1:end);
        for snrIndex = 1:numel(cfg.SNRdB)
            sigma2 = signalPower/10^(cfg.SNRdB(snrIndex)/10);
            W = sqrt(sigma2/2)*(randn(cfg.N,P)+1j*randn(cfg.N,P));
            Z = abs(userField+B+W);
            % Replay generator draws for completed trials, without fitting.
            % This preserves exactly the original seeded test sequence.
            if trial <= resumeTrial, continue; end
            timer = tic;
            Gem = runEMGS(Z,S,B,sigma2,cfg.FinalIterations);
            runtimes(trial,snrIndex,pilotCase,1) = toc(timer);
            % True G never enters HS fitting or rank selection.
            timer = tic;
            p = floor((cfg.N-1)/2);
            maximumRank = min(cfg.N-p,p+1);
            validationError = zeros(maximumRank,1);
            for rank = 1:maximumRank
                Gcandidate = runRankHS(Z(:,T),S(:,T),B(:,T),rank,p, ...
                    cfg.SelectionIterations,cfg.CadzowSweeps,sigma2);
                validationError(rank) = norm(Z(:,V)-abs(Gcandidate*S(:,V)+B(:,V)),'fro')^2;
            end
            [~,selectedRank] = min(validationError);
            Ghs = runRankHS(Z,S,B,selectedRank,p,cfg.FinalIterations, ...
                cfg.CadzowSweeps,sigma2);
            ranks(trial,snrIndex,pilotCase) = selectedRank;
            runtimes(trial,snrIndex,pilotCase,2) = toc(timer);
            timer = tic;
            x = makeFeatures(Z,S,B,sigma2);
            normalizer = normalization{pilotCase};
            x = (x-normalizer.Mean)./normalizer.Std;
            y = predict(networks{pilotCase},x,'ExecutionEnvironment','cpu');
            y = double(y)*normalizer.TargetScale;
            Gdl = reshape(y(1:cfg.N*cfg.K),cfg.N,cfg.K) + ...
                1j*reshape(y(cfg.N*cfg.K+1:end),cfg.N,cfg.K);
            runtimes(trial,snrIndex,pilotCase,3) = toc(timer);
            estimates = {Gem,Ghs,Gdl};
            for method = 1:3
                errors(trial,snrIndex,pilotCase,method) = norm(estimates{method}-G,'fro')^2;
            end
            energies(trial,snrIndex,pilotCase) = norm(G,'fro')^2;
        end
    end
    if trial > resumeTrial && (mod(trial,10)==0 || trial==cfg.TestTrials)
        fprintf('Completed independent test trial %d/%d\n',trial,cfg.TestTrials);
        completedTrials = trial;
        save(fullfile(outputFolder,'checkpoint.mat'),'cfg','errors', ...
            'energies','ranks','runtimes','completedTrials','trainingSeconds');
    end
end

%% 4. NMSE = sum of squared errors / sum of true channel energies
nmseDB = zeros(numel(cfg.SNRdB),numel(cfg.P),3);
for pilotCase=1:numel(cfg.P)
    for snrIndex=1:numel(cfg.SNRdB)
        for method=1:3
            nmseDB(snrIndex,pilotCase,method) = 10*log10( ...
                sum(errors(:,snrIndex,pilotCase,method))/sum(energies(:,snrIndex,pilotCase)));
        end
    end
end
% One Fig.3-style plot: six curves, two P cases, no vertical error bars.
f = figure('Color','w','Position',[100 100 900 650]); hold on;
colors = lines(3); markers = {'o','s','^'}; styles = {'--','-'};
labels = cell(6,1); row=0;
for pilotCase=1:numel(cfg.P)
    for method=1:3
        row=row+1;
        plot(cfg.SNRdB,nmseDB(:,pilotCase,method), ...
            'Color',colors(method,:),'Marker',markers{method}, ...
            'LineStyle',styles{pilotCase},'LineWidth',1.7);
        labels{row}=sprintf('%s, P=%d',methods{method},cfg.P(pilotCase));
    end
end
grid on; xlabel('Aggregate user-field SNR (dB)'); ylabel('NMSE (dB)');
title(sprintf('Magnitude-only 1D estimation: N=%d, K=%d, RSR=%g dB',cfg.N,cfg.K,cfg.RSRdB));
legend(labels,'Location','southwest');
axesHandle = gca; axesHandle.Toolbar.Visible = 'off'; drawnow;
savefig(f,fullfile(outputFolder,'nmse_snr.fig'));
exportgraphics(f,fullfile(outputFolder,'nmse_snr.png'),'Resolution',200);
Pcolumn=[]; SNRcolumn=[]; MethodColumn={}; NMSEcolumn=[]; RuntimeColumn=[]; RankColumn=[];
for pilotCase=1:numel(cfg.P)
    for snrIndex=1:numel(cfg.SNRdB)
        for method=1:3
            Pcolumn(end+1,1)=cfg.P(pilotCase);
            SNRcolumn(end+1,1)=cfg.SNRdB(snrIndex);
            MethodColumn{end+1,1}=methods{method};
            NMSEcolumn(end+1,1)=nmseDB(snrIndex,pilotCase,method);
            RuntimeColumn(end+1,1)=mean(runtimes(:,snrIndex,pilotCase,method));
            RankColumn(end+1,1)=NaN;
            if method==2, RankColumn(end)=mean(ranks(:,snrIndex,pilotCase)); end
        end
    end
end
results=table(Pcolumn,SNRcolumn,MethodColumn,NMSEcolumn,RuntimeColumn,RankColumn, ...
    'VariableNames',{'P','SNR_dB','Method','NMSE_dB','MeanRuntimeSeconds','MeanHSRank'});
writetable(results,fullfile(outputFolder,'nmse_snr.csv'));
save(fullfile(outputFolder,'results.mat'),'cfg','errors','energies','ranks', ...
    'runtimes','nmseDB','trainingSeconds','normalization');
disp(results); fprintf('Saved results to %s\n',outputFolder);

%% Local functions: independent learning data, known-input features, estimators
function [X,Y] = makeLearningData(cfg,P,count)
    featureCount = cfg.N*P+2*cfg.K*P+2*cfg.N*P+1;
    X=zeros(count,featureCount,'single');
    Y=zeros(count,2*cfg.N*cfg.K,'single');
    for sample=1:count
        [G,~] = generateUserChannel(cfg.N,cfg.K,cfg.Spacing,[0;1;0],1);
        S=(randn(cfg.K,P)+1j*randn(cfg.K,P))/sqrt(2);
        Braw=generateReferencePath(cfg.N,P,1,cfg.Spacing,[0;1;0],1);
        B=scaleReference(G,S,Braw,cfg.RSRdB);
        snr=cfg.SNRdB(1)+(cfg.SNRdB(end)-cfg.SNRdB(1))*rand;
        field=G*S; sigma2=mean(abs(field(:)).^2)/10^(snr/10);
        W=sqrt(sigma2/2)*(randn(cfg.N,P)+1j*randn(cfg.N,P));
        Z=abs(field+B+W);
        X(sample,:)=makeFeatures(Z,S,B,sigma2);
        Y(sample,:)=[real(G(:));imag(G(:))].';
    end
end
function x = makeFeatures(Z,S,B,sigma2)
    % Only observable/known values enter inference. No G, rank, or true AoAs.
    x=single([Z(:);real(S(:));imag(S(:));real(B(:));imag(B(:));log10(sigma2)].');
end
function B = scaleReference(G,S,Braw,rsrDB)
    userPowers=zeros(size(G,2),1);
    for user=1:size(G,2)
        field=G(:,user)*S(user,:); userPowers(user)=mean(abs(field(:)).^2);
    end
    B=Braw*sqrt(10^(rsrDB/10)*mean(userPowers)/mean(abs(Braw(:)).^2));
end

function [G, pathCount] = generateUserChannel(N, K, dOverLambda, ...
        dipoleMoment, hbar)
    G = zeros(N, K);
    pathCount = zeros(K, 1);
    elementIndex = (1:N).';

    for user = 1:K
        pathCount(user) = randi([3, 7]);
        numberOfPaths = pathCount(user);

        % Broadside angle beta is uniform on (-pi/2, pi/2). Convert to
        % theta measured from the array axis so the spatial phase uses cos.
        beta = (rand(numberOfPaths, 1) - 0.5) * pi;
        thetaAxis = pi/2 - beta;
        phaseSlope = 2*pi*dOverLambda*cos(thetaAxis);

        pathGain = sqrt(1/2) * (randn(numberOfPaths, 1) + ...
            1j*randn(numberOfPaths, 1));

        % One polarization vector per path, shared across array elements.
        polarization = sqrt(1/3) * randn(3, numberOfPaths);
        coupling = (dipoleMoment.' * polarization).' / hbar;
        pathCoefficient = coupling .* pathGain;

        spatialResponse = exp(-1j * (elementIndex - 1) * phaseSlope.');
        G(:, user) = spatialResponse * pathCoefficient;
    end
end

function B = generateReferencePath(N, P, referenceSymbol, dOverLambda, ...
        dipoleMoment, hbar)
    referenceGain = sqrt(10/2) * (randn + 1j*randn); % CN(0,10), Xu
    betaReference = (rand - 0.5) * pi;
    thetaReferenceAxis = pi/2 - betaReference;
    referencePhaseSlope = 2*pi*dOverLambda*cos(thetaReferenceAxis);

    % One reference-path polarization vector shared across array elements.
    referencePolarization = sqrt(1/3) * randn(3, 1);
    referenceCoupling = (dipoleMoment.' * referencePolarization) / hbar;
    elementIndex = (1:N).';
    referenceVector = referenceCoupling * referenceGain * ...
        exp(-1j*(elementIndex-1)*referencePhaseSlope);
    B = referenceVector * (referenceSymbol * ones(1, P));
end

function G = runEMGS(Z, S, B, sigma2, iterationCount)
    [N, P] = size(Z);
    K = size(S, 1);
    G = zeros(N, K);

    % Spectral initialization for every receive element.
    for element = 1:N
        z = Z(element, :).';
        b = B(element, :).';
        augmentedPilotMatrix = [S.' b];
        spectralMatrix = augmentedPilotMatrix' * (z .* augmentedPilotMatrix);
        spectralMatrix = (spectralMatrix + spectralMatrix')/2;
        [vectors, values] = eig(spectralMatrix);
        [~, principalIndex] = max(real(diag(values)));
        v = vectors(:, principalIndex);
        v = v / norm(v);

        predictedMagnitude = abs(augmentedPilotMatrix*v);
        scaleDenominator = norm(predictedMagnitude)^2;
        if scaleDenominator > eps
            scale = max(0, (predictedMagnitude.'*z) / scaleDenominator);
        else
            scale = 0;
        end
        augmentedEstimate = scale * v;
        if abs(augmentedEstimate(end)) > eps
            augmentedEstimate = exp(-1j*angle(augmentedEstimate(end))) * ...
                augmentedEstimate;
        end
        G(element, :) = augmentedEstimate(1:K).';
    end

    leastSquaresMap = S' / (S*S');
    for iteration = 1:iterationCount
        predictedField = G*S + B;
        predictedPhase = angle(predictedField);
        concentration = (2/sigma2) * Z .* abs(predictedField);

        besselRatio = besseli(1, concentration, 1) ./ ...
            besseli(0, concentration, 1);
        expectedField = Z .* exp(1j*predictedPhase) .* besselRatio;
        G = (expectedField - B) * leastSquaresMap;
    end
end

function G = runRankHS(Z, S, B, rankValue, hankelParameter, ...
        iterationCount, cadzowSweeps, sigma2)
    G = runEMGS(Z, S, B, sigma2, 0);
    leastSquaresMap = S' / (S*S');
    K = size(S, 1);

    for iteration = 1:iterationCount
        predictedField = G*S + B;
        predictedPhase = angle(predictedField);
        concentration = (2/sigma2) * Z .* abs(predictedField);
        besselRatio = besseli(1, concentration, 1) ./ ...
            besseli(0, concentration, 1);
        expectedField = Z .* exp(1j*predictedPhase) .* besselRatio;
        G = (expectedField - B) * leastSquaresMap;

        for user = 1:K
            channel = G(:, user);
            for sweep = 1:cadzowSweeps
                channel = projectHankelRank(channel, rankValue, hankelParameter);
            end
            G(:, user) = channel;
        end
    end
end

function channel = projectHankelRank(channel, rankValue, hankelParameter)
    hankelMatrix = makeHankelMatrix(channel, hankelParameter);
    [leftVectors, singularValues, rightVectors] = svd(hankelMatrix, 'econ');
    rankValue = min(rankValue, min(size(hankelMatrix)));
    lowRankMatrix = leftVectors(:, 1:rankValue) * ...
        singularValues(1:rankValue, 1:rankValue) * ...
        rightVectors(:, 1:rankValue)';
    channel = averageHankelAntidiagonals(lowRankMatrix);
end

function H = makeHankelMatrix(channel, hankelParameter)
    N = numel(channel);
    rowCount = N - hankelParameter;
    columnCount = hankelParameter + 1;
    H = zeros(rowCount, columnCount, 'like', channel);
    for row = 1:rowCount
        H(row, :) = channel(row:row+columnCount-1).';
    end
end

function channel = averageHankelAntidiagonals(H)
    [rowCount, columnCount] = size(H);
    N = rowCount + columnCount - 1;
    channel = zeros(N, 1, 'like', H);
    counts = zeros(N, 1);

    for row = 1:rowCount
        for column = 1:columnCount
            sampleIndex = row + column - 1;
            channel(sampleIndex) = channel(sampleIndex) + H(row, column);
            counts(sampleIndex) = counts(sampleIndex) + 1;
        end
    end
    channel = channel ./ counts;
end
