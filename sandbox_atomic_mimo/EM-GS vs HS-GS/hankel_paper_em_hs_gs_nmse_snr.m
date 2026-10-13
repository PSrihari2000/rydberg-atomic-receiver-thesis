% Main NMSE-versus-SNR reproduction for the Hankel-structured EM-GS paper.
% It reproduces the paper's N=32, K=3, P=10 and P=30 experiments,
% including its main P=30 setting, with RSR=12 dB,
% SNR=-5:5:20 dB, 200 trials/point for this run, 100 final EM-GS iterations,
% 25 candidate-rank iterations, four Cadzow sweeps, and 70/30 pilot split.
% The observation is the exact magnitude-only model Z=abs(G*S+B+W).
%
% Paper-stated channel: L_k is selected from {3,...,7}, alpha_lk are
% independent circular complex Gaussian, and broadside AoAs are uniform
% on (-90,90) degrees. We set Var(alpha_lk)=1/L_k so users have equal
% average received power, as specified. The pilot distribution and the
% detailed reference realization are not fully specified; this script
% uses iid CN(0,1) user pilots and one known reference plane wave with
% s_b,p=1, then enforces the paper's RSR definition per trial.
% The paper defines RSR as reference power/single-user signal power and
% SNR as total multiuser signal power/sigma^2. Random seed is local.

clear;
clc;
close all;

%% Simulation settings
K = 3;
N = 32;
P_list = [10, 30];
SNRdB = -5:5:20;
MonteCarloTrials = 200;
FinalIterations = 100;
RankSelectionIterations = 25;
CadzowSweeps = 4;
BootstrapReplicates = 1000; % retained in result data; not shown as plot bars
ReferenceToSignalRatioDB = 12;
RandomSeed = 20261013; % implementation choice; paper does not specify a seed

rng(RandomSeed, 'twister');
outputFolder = fileparts(mfilename('fullpath'));

numberOfSNR = numel(SNRdB);
numberOfPilotCases = numel(P_list);
errorEM = zeros(MonteCarloTrials, numberOfSNR, numberOfPilotCases);
errorHS = zeros(MonteCarloTrials, numberOfSNR, numberOfPilotCases);
channelEnergy = zeros(MonteCarloTrials, numberOfSNR, numberOfPilotCases);
selectedRank = zeros(MonteCarloTrials, numberOfSNR, numberOfPilotCases);
trainingPilotCount = zeros(numberOfPilotCases, 1);
validationPilotCount = zeros(numberOfPilotCases, 1);
pathCounts = zeros(MonteCarloTrials, K);

fprintf('Hankel-paper NMSE-SNR run: N=%d, K=%d, %d trials/point\n', ...
    N, K, MonteCarloTrials);
fprintf('SNR points: %s dB; P values: %s\n', ...
    mat2str(SNRdB), mat2str(P_list));

%% Paired Monte Carlo experiment
for trial = 1:MonteCarloTrials
    [Gtrue, trialPathCounts] = generatePaper1DChannel(N, K);
    pathCounts(trial, :) = trialPathCounts(:).';

    % Generate 30 iid CN(0,1) pilots once; P=10 uses the first 10.
    SpilotFull = sqrt(1/2) * ...
        (randn(K, max(P_list)) + 1j*randn(K, max(P_list)));
    Bfull = generatePaperReference(N, max(P_list));
    % Enforce the Hankel paper's RSR = 12 dB per channel realization.
    % RSR is reference-field power divided by average single-user power.
    singleUserPower = mean(abs(Gtrue(:)).^2);
    referencePower = mean(abs(Bfull(:)).^2);
    targetReferencePower = 10^(ReferenceToSignalRatioDB/10) * ...
        singleUserPower;
    Bfull = Bfull * sqrt(targetReferencePower / referencePower);

    for pilotCase = 1:numberOfPilotCases
        P = P_list(pilotCase);
        S = SpilotFull(:, 1:P);
        trainCount = round(0.70 * P);
        pilotOrder = randperm(P);
        trainIndices = pilotOrder(1:trainCount);
        validationIndices = pilotOrder(trainCount+1:end);
        trainingPilotCount(pilotCase) = trainCount;
        validationPilotCount(pilotCase) = P - trainCount;

            % The paper does not specify the reference sequence. This run uses
            % a known all-ones reference pilot sequence as an explicit choice.
        B = Bfull(:, 1:P);

        noiselessUserField = Gtrue * S;
        totalSignalPower = mean(abs(noiselessUserField(:)).^2);
        channelEnergy(trial, :, pilotCase) = norm(Gtrue, 'fro')^2;

        for snrIndex = 1:numberOfSNR
            snrLinear = 10^(SNRdB(snrIndex)/10);
            % Aggregate SNR convention from the Hankel paper: total
            % user-signal power divided by sigma^2. Here total signal power
            % is estimated as mean(|G*S|^2) over cell/pilot entries.
            % W is circular complex Gaussian with E|W|^2 = sigma^2.
            sigma2 = totalSignalPower / snrLinear;
            W = sqrt(sigma2/2) * ...
                (randn(N, P) + 1j*randn(N, P));
            Z = abs(noiselessUserField + B + W);

            % EM-GS baseline: same full observations, no Hankel projection.
            GestEM = runEMGS(Z, S, B, sigma2, FinalIterations);

            % HS-GS: train on T, select rank using held-out V, then refit
            % the selected rank with all P pilots.
            Ztrain = Z(:, trainIndices);
            Strain = S(:, trainIndices);
            Btrain = B(:, trainIndices);
            Zvalidation = Z(:, validationIndices);
            Svalidation = S(:, validationIndices);
            Bvalidation = B(:, validationIndices);

            hankelParameter = floor((N-1)/2);
            maximumRank = ceil(N/2);
            validationError = zeros(maximumRank, 1);
            for candidateRank = 1:maximumRank
                Gcandidate = runRankHS( ...
                    Ztrain, Strain, Btrain, candidateRank, ...
                    hankelParameter, RankSelectionIterations, ...
                    CadzowSweeps, sigma2);
                validationResidual = Zvalidation - ...
                    abs(Gcandidate*Svalidation + Bvalidation);
                validationError(candidateRank) = ...
                    norm(validationResidual, 'fro')^2;
            end

            [~, bestRank] = min(validationError);
            selectedRank(trial, snrIndex, pilotCase) = bestRank;
            GestHS = runRankHS(Z, S, B, bestRank, hankelParameter, ...
                FinalIterations, CadzowSweeps, sigma2);

            errorEM(trial, snrIndex, pilotCase) = ...
                norm(GestEM - Gtrue, 'fro')^2;
            errorHS(trial, snrIndex, pilotCase) = ...
                norm(GestHS - Gtrue, 'fro')^2;
        end
    end

    if mod(trial, 10) == 0 || trial == MonteCarloTrials
        fprintf('Completed trial %d / %d\n', trial, MonteCarloTrials);
        completedTrials = trial;
        save(fullfile(outputFolder, 'hankel_paper_em_hs_gs_checkpoint.mat'), ...
            'K', 'N', 'P_list', 'SNRdB', 'MonteCarloTrials', ...
            'FinalIterations', 'RankSelectionIterations', 'CadzowSweeps', ...
            'RandomSeed', 'ReferenceToSignalRatioDB', ...
            'errorEM', 'errorHS', 'channelEnergy', ...
            'selectedRank', 'trainingPilotCount', 'validationPilotCount', ...
            'pathCounts', 'completedTrials');
    end
end

%% Ratio-of-sums NMSE and paired bootstrap confidence intervals
nmseEM = zeros(numberOfSNR, numberOfPilotCases);
nmseHS = zeros(numberOfSNR, numberOfPilotCases);
ciEM = zeros(2, numberOfSNR, numberOfPilotCases);
ciHS = zeros(2, numberOfSNR, numberOfPilotCases);

for pilotCase = 1:numberOfPilotCases
    for snrIndex = 1:numberOfSNR
        trialEnergy = channelEnergy(:, snrIndex, pilotCase);
        emErrors = errorEM(:, snrIndex, pilotCase);
        hsErrors = errorHS(:, snrIndex, pilotCase);
        nmseEM(snrIndex, pilotCase) = sum(emErrors) / sum(trialEnergy);
        nmseHS(snrIndex, pilotCase) = sum(hsErrors) / sum(trialEnergy);

        bootstrapEM = zeros(BootstrapReplicates, 1);
        bootstrapHS = zeros(BootstrapReplicates, 1);
        for bootstrapIndex = 1:BootstrapReplicates
            sampleIndices = randi(MonteCarloTrials, MonteCarloTrials, 1);
            denominator = sum(trialEnergy(sampleIndices));
            bootstrapEM(bootstrapIndex) = ...
                10*log10(sum(emErrors(sampleIndices)) / denominator);
            bootstrapHS(bootstrapIndex) = ...
                10*log10(sum(hsErrors(sampleIndices)) / denominator);
        end
        ciEM(:, snrIndex, pilotCase) = percentileInterval(bootstrapEM);
        ciHS(:, snrIndex, pilotCase) = percentileInterval(bootstrapHS);
    end
end

nmseEMdB = 10*log10(nmseEM);
nmseHSdB = 10*log10(nmseHS);

%% NMSE-versus-SNR plot for N=32 at P=10 and P=30
figureHandle = figure('Color', 'w', 'Position', [100, 100, 850, 560], ...
    'ToolBar', 'none', 'MenuBar', 'none');
axesHandle = axes(figureHandle);
hold(axesHandle, 'on');
emColor = [0, 0.4470, 0.7410];
hsColor = [0.8500, 0.3250, 0.0980];

plot(axesHandle, SNRdB, nmseEMdB(:, 1), ...
    '-o', 'Color', emColor, 'LineWidth', 1.5, 'MarkerSize', 5);
plot(axesHandle, SNRdB, nmseHSdB(:, 1), ...
    '-s', 'Color', hsColor, 'LineWidth', 1.5, 'MarkerSize', 5);
plot(axesHandle, SNRdB, nmseEMdB(:, 2), ...
    '--^', 'Color', emColor, 'LineWidth', 1.5, 'MarkerSize', 5);
plot(axesHandle, SNRdB, nmseHSdB(:, 2), ...
    '--d', 'Color', hsColor, 'LineWidth', 1.5, 'MarkerSize', 5);

grid(axesHandle, 'on');
xlim(axesHandle, [min(SNRdB), max(SNRdB)]);
xlabel(axesHandle, 'Aggregate SNR (dB)');
ylabel(axesHandle, 'NMSE (dB)');
title(axesHandle, sprintf('Hankel-paper settings: N = %d, K = %d', N, K));
legend(axesHandle, {'EM-GS, P = 10', 'HS-GS, P = 10', ...
    'EM-GS, P = 30', 'HS-GS, P = 30'}, 'Location', 'northeast');

simulationChoices = struct( ...
    'measurementModel', 'Z = abs(G*S + B + W), exact magnitude-only', ...
    'channelModel', 'g_k[n] = sum_l alpha_lk exp(-j(n-1) psi_lk)', ...
    'pathCount', 'L_k uniformly sampled from integers 3 through 7', ...
    'angleDistribution', 'broadside AoA uniform on (-pi/2, pi/2), Hankel paper', ...
    'arrayPhase', 'psi = pi*sin(theta), with d/lambda = 0.5', ...
    'pathGain', 'alpha_lk ~ CN(0,1/L_k), equal average user channel power', ...
    'userPilots', 'iid CN(0,1), implementation choice because entry law is unspecified', ...
    'referenceField', 'one known plane wave, all-ones reference pilot sequence; implementation choice', ...
    'referenceToSignalRatioDB', ReferenceToSignalRatioDB, ...
    'snrDefinition', 'aggregate mean user-signal power / sigma2, Hankel paper', ...
    'monteCarloTrials', MonteCarloTrials, ...
    'hsRankSelection', '70/30 split and validation residual', ...
    'hsIterations', '25 candidate, 100 final, 4 Cadzow sweeps');
save(fullfile(outputFolder, 'hankel_paper_em_hs_gs_nmse_snr.mat'), ...
    'K', 'N', 'P_list', 'SNRdB', 'MonteCarloTrials', ...
    'FinalIterations', 'RankSelectionIterations', 'CadzowSweeps', ...
    'BootstrapReplicates', 'RandomSeed', 'ReferenceToSignalRatioDB', ...
    'errorEM', 'errorHS', ...
    'channelEnergy', 'selectedRank', 'trainingPilotCount', ...
    'validationPilotCount', 'nmseEM', 'nmseHS', 'nmseEMdB', ...
    'nmseHSdB', 'ciEM', 'ciHS', 'pathCounts', 'simulationChoices');
savefig(fullfile(outputFolder, 'hankel_paper_em_hs_gs_nmse_snr.fig'));
exportgraphics(gcf, fullfile(outputFolder, 'hankel_paper_em_hs_gs_nmse_snr.png'), ...
    'Resolution', 200);

numberOfResultRows = numberOfSNR * numberOfPilotCases;
resultP = zeros(numberOfResultRows, 1);
resultSNR = zeros(numberOfResultRows, 1);
resultEM = zeros(numberOfResultRows, 1);
resultHS = zeros(numberOfResultRows, 1);
resultEMlow = zeros(numberOfResultRows, 1);
resultEMhigh = zeros(numberOfResultRows, 1);
resultHSlow = zeros(numberOfResultRows, 1);
resultHShigh = zeros(numberOfResultRows, 1);
resultMeanRank = zeros(numberOfResultRows, 1);
resultRow = 0;
for pilotCase = 1:numberOfPilotCases
    for snrIndex = 1:numberOfSNR
        resultRow = resultRow + 1;
        resultP(resultRow) = P_list(pilotCase);
        resultSNR(resultRow) = SNRdB(snrIndex);
        resultEM(resultRow) = nmseEMdB(snrIndex, pilotCase);
        resultHS(resultRow) = nmseHSdB(snrIndex, pilotCase);
        resultEMlow(resultRow) = ciEM(1, snrIndex, pilotCase);
        resultEMhigh(resultRow) = ciEM(2, snrIndex, pilotCase);
        resultHSlow(resultRow) = ciHS(1, snrIndex, pilotCase);
        resultHShigh(resultRow) = ciHS(2, snrIndex, pilotCase);
        resultMeanRank(resultRow) = ...
            mean(selectedRank(:, snrIndex, pilotCase));
    end
end
resultTable = table(resultP, resultSNR, resultEM, resultHS, ...
    resultEMlow, resultEMhigh, resultHSlow, resultHShigh, resultMeanRank, ...
    'VariableNames', {'P', 'SNR_dB', 'EMGS_NMSE_dB', 'HSGS_NMSE_dB', ...
    'EMGS_CI_low_dB', 'EMGS_CI_high_dB', 'HSGS_CI_low_dB', ...
    'HSGS_CI_high_dB', 'MeanSelectedRank'});
writetable(resultTable, fullfile(outputFolder, 'hankel_paper_em_hs_gs_nmse_snr.csv'));

fprintf('\nResults (Monte Carlo NMSE, dB):\n');
for pilotCase = 1:numberOfPilotCases
    fprintf('\nP = %d\n', P_list(pilotCase));
    for snrIndex = 1:numberOfSNR
        fprintf('SNR %3d dB: EM-GS %8.3f dB, HS-GS %8.3f dB, mean selected rank %.2f\n', ...
            SNRdB(snrIndex), nmseEMdB(snrIndex, pilotCase), ...
            nmseHSdB(snrIndex, pilotCase), ...
            mean(selectedRank(:, snrIndex, pilotCase)));
    end
end
fprintf('\nSaved plot, MAT data, and CSV in:\n%s\n', outputFolder);

%% Local functions
function [G, pathCount] = generatePaper1DChannel(N, K)
    % Hankel-paper broadside angle convention and half-wavelength ULA.
    G = zeros(N, K);
    pathCount = zeros(K, 1);
    elementIndex = (1:N).';
    for user = 1:K
        pathCount(user) = randi([3, 7]);
        aoaBroadside = pi*(rand(pathCount(user), 1) - 0.5);
        phaseSlope = pi*sin(aoaBroadside); % psi=2*pi*(d/lambda)*sin(theta)
        pathGain = sqrt(1/(2*pathCount(user))) * ...
            (randn(pathCount(user), 1) + 1j*randn(pathCount(user), 1));
        spatialResponse = exp(-1j * (elementIndex - 1) * phaseSlope.');
        G(:, user) = spatialResponse * pathGain;
    end
end

function B = generatePaperReference(N, P)
    % The detailed reference-field generation is not specified. Use one
    % known plane wave and all-ones pilot sequence, then scale to paper RSR.
    referenceGain = sqrt(10/2) * (randn + 1j*randn); % CN(0,10)
    referenceAOABroadside = pi*(rand - 0.5);
    referencePhaseSlope = pi*sin(referenceAOABroadside);
    elementIndex = (1:N).';
    referenceSpatialResponse = exp(-1j * (elementIndex - 1) * referencePhaseSlope);
    referenceVector = referenceGain * referenceSpatialResponse;
    B = referenceVector * ones(1, P);
end

function G = runEMGS(Z, S, B, sigma2, iterationCount)
    [N, P] = size(Z);
    K = size(S, 1);
    G = zeros(N, K);

    % Spectral initialization, one receive element at a time.
    for element = 1:N
        z = Z(element, :).';
        b = B(element, :).';
        Abar = [S.' b];
        M = Abar' * (z .* Abar);
        M = (M + M')/2;
        [eigenvectors, eigenvalues] = eig(M);
        [~, principalIndex] = max(real(diag(eigenvalues)));
        v = eigenvectors(:, principalIndex);
        v = v / norm(v);
        q = abs(Abar*v);
        qEnergy = norm(q)^2;
        if qEnergy > eps
            scale = max(0, (q.'*z) / qEnergy);
        else
            scale = 0;
        end
        augmentedEstimate = scale*v;
        if abs(augmentedEstimate(end)) > eps
            phaseCorrection = exp(-1j*angle(augmentedEstimate(end)));
            augmentedEstimate = phaseCorrection*augmentedEstimate;
        end
        G(element, :) = augmentedEstimate(1:K).';
    end

    leastSquaresMap = S' / (S*S');
    for iteration = 1:iterationCount
        predictedField = G*S + B;
        predictedPhase = angle(predictedField);
        concentration = (2/sigma2) * Z .* abs(predictedField);
        expectedPhaseWeight = besseli(1, concentration, 1) ./ ...
            besseli(0, concentration, 1);
        expectedField = Z .* exp(1j*predictedPhase) .* expectedPhaseWeight;
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
        expectedPhaseWeight = besseli(1, concentration, 1) ./ ...
            besseli(0, concentration, 1);
        expectedField = Z .* exp(1j*predictedPhase) .* expectedPhaseWeight;
        G = (expectedField - B) * leastSquaresMap;

        for user = 1:K
            channel = G(:, user);
            for sweep = 1:cadzowSweeps
                channel = cadzowProjection(channel, rankValue, hankelParameter);
            end
            G(:, user) = channel;
        end
    end
end

function channel = cadzowProjection(channel, rankValue, hankelParameter)
    hankelMatrix = makeHankel(channel, hankelParameter);
    [leftVectors, singularMatrix, rightVectors] = svd(hankelMatrix, 'econ');
    rankValue = min(rankValue, min(size(hankelMatrix)));
    lowRankMatrix = leftVectors(:, 1:rankValue) * ...
        singularMatrix(1:rankValue, 1:rankValue) * ...
        rightVectors(:, 1:rankValue)';
    channel = hankelizeByAveraging(lowRankMatrix);
end

function H = makeHankel(channel, hankelParameter)
    N = numel(channel);
    rowCount = N - hankelParameter;
    columnCount = hankelParameter + 1;
    H = zeros(rowCount, columnCount, 'like', channel);
    for row = 1:rowCount
        H(row, :) = channel(row:row+columnCount-1).';
    end
end

function channel = hankelizeByAveraging(H)
    [rowCount, columnCount] = size(H);
    N = rowCount + columnCount - 1;
    channel = zeros(N, 1, 'like', H);
    counts = zeros(N, 1);
    for row = 1:rowCount
        for column = 1:columnCount
            diagonalIndex = row + column - 1;
            channel(diagonalIndex) = channel(diagonalIndex) + H(row, column);
            counts(diagonalIndex) = counts(diagonalIndex) + 1;
        end
    end
    channel = channel ./ counts;
end

function interval = percentileInterval(samples)
    sortedSamples = sort(samples(:));
    lowerIndex = max(1, ceil(0.025*numel(sortedSamples)));
    upperIndex = min(numel(sortedSamples), ceil(0.975*numel(sortedSamples)));
    interval = [sortedSamples(lowerIndex); sortedSamples(upperIndex)];
end
