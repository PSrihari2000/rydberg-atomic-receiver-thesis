% Convergence audit for the 1D magnitude-only channel-estimation benchmark.
%
% This is a separate, self-contained audit. It leaves the fixed-iteration
% benchmark unchanged. Every estimator uses a maximum iteration count and
% an iterate-change stopping rule. The main model remains magnitude-only:
% Z = abs(G*S + B + W).
%
% The three reference cases are:
%   1) Xu reference draw with alpha_b ~ CN(0,10), without RSR rescaling;
%   2) the same reference shape rescaled to RSR = 12 dB;
%   3) the same reference shape rescaled to RSR = 50 dB.
%
% The Xu paper does not specify every simulation convention. See the
% comments in the settings block for which choices come from the Hankel
% paper and which are implementation choices.

clear;
clc;
close all;

%% Simulation settings
K = 3;
N = 8;
P_list = [10, 30];
FullSNRdB = -5:5:30;
FullMonteCarloTrials = 200;
RandomSeed = 20261013;

% The first case keeps the Xu-distributed reference draw unchanged.
% The other two cases rescale that same draw to the listed RSR.
FullRSR_dB_list = [NaN, 12, 50];
FullRSR_case_names = {'Xu_raw', 'RSR12dB', 'RSR50dB'};
AuditPilotMode = true;
if AuditPilotMode
    % Small diagnostic grid: low/middle/high SNR, both P values, RSR=50 dB.
    % The strong reference is needed for Xu's linearized GD baseline.
    MonteCarloTrials = 2;
    SNRdB = [-5, 10, 30];
    RSR_dB_list = 50;
    RSR_case_names = {'RSR50dB'};
    auditFolderName = 'convergence_audit_pilot';
else
    MonteCarloTrials = FullMonteCarloTrials;
    SNRdB = FullSNRdB;
    RSR_dB_list = FullRSR_dB_list;
    RSR_case_names = FullRSR_case_names;
    auditFolderName = 'convergence_audit_full';
end

% Convergence-audit limits. The current paper-setting run remains in the
% other script; these larger caps are for a separate accuracy audit.
GS_max_iterations = 1000;
GD_max_iterations = 5000;
EM_max_iterations = 1000;
HS_rank_max_iterations = 3000;
Cadzow_sweeps = 4;
Rank_train_fraction = 0.70;
Relative_update_tolerance = 1e-8;
Consecutive_stable_iterations = 5;

% The channel phase is phi = 2*pi*(d/lambda)*cos(theta_axis).
% d/lambda = 0.5 is taken from the Hankel paper. We draw broadside beta
% uniformly on (-90,90) degrees and set theta_axis = 90 degrees - beta.
d_over_lambda = 0.5;

% Known reference pilot symbol. Xu gives s_b,p in its equation but does
% not state a distribution for it; using one is an explicit choice.
reference_pilot_symbol = 1;

% Physical constants used in the Xu channel and reference equations.
hbar = 1.054571817e-34;
elementary_charge = 1.602176634e-19;
bohr_radius = 5.29177210903e-11;
transition_dipole = [0; 1785.9*elementary_charge*bohr_radius; 0];

rng(RandomSeed, 'twister');

numberOfSNR = numel(SNRdB);
numberOfP = numel(P_list);
numberOfRSR = numel(RSR_dB_list);
numberOfMethods = 4;
method_names = {'GS', 'GD', 'EM-GS', 'HS-GS'};

% errorByMethod dimensions: trial x SNR x P-case x RSR-case x method.
errorByMethod = zeros(MonteCarloTrials, numberOfSNR, ...
    numberOfP, numberOfRSR, numberOfMethods);
channelEnergy = zeros(MonteCarloTrials, numberOfSNR, numberOfP);
selectedRank = zeros(MonteCarloTrials, numberOfSNR, numberOfP, numberOfRSR);
iterationCounts = zeros(MonteCarloTrials, numberOfSNR, numberOfP, ...
    numberOfRSR, numberOfMethods);
selectedRankIterationCounts = zeros(MonteCarloTrials, numberOfSNR, ...
    numberOfP, numberOfRSR, ceil(N/2));
elapsedSeconds = zeros(MonteCarloTrials, numberOfSNR, numberOfP, ...
    numberOfRSR, numberOfMethods);
measuredRSR_dB = zeros(MonteCarloTrials, numberOfP, numberOfRSR);
pathCounts = zeros(MonteCarloTrials, K, numberOfP);
trainingPilotCount = zeros(numberOfP, 1);
validationPilotCount = zeros(numberOfP, 1);

outputFolder = fullfile(fileparts(mfilename('fullpath')), auditFolderName);
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
checkpointFile = fullfile(outputFolder, 'convergence_audit_checkpoint.mat');

fprintf('1D convergence audit pilot: K=%d, N=%d, trials=%d\n', ...
    K, N, MonteCarloTrials);
fprintf('Pilot counts: %s; SNR points: %s dB\n', ...
    mat2str(P_list), mat2str(SNRdB));
fprintf('Reference cases: %s\n', strjoin(RSR_case_names, ', '));

%% Paired Monte Carlo simulation
for trial = 1:MonteCarloTrials
    % Generate one channel, pilot matrix, and reference path per trial.
    % Every estimator receives the same realization in each comparison.
    [Gtrue, trialPathCounts] = generateUserChannel( ...
        N, K, d_over_lambda, transition_dipole, hbar);

    maximumPilotCount = max(P_list);
    Sfull = sqrt(1/2) * (randn(K, maximumPilotCount) + ...
        1j*randn(K, maximumPilotCount));

    BrawFull = generateReferencePath(N, maximumPilotCount, ...
        reference_pilot_symbol, d_over_lambda, transition_dipole, ...
        hbar);

    for pCase = 1:numberOfP
        P = P_list(pCase);
        S = Sfull(:, 1:P);
        Braw = BrawFull(:, 1:P);
        pathCounts(trial, :, pCase) = trialPathCounts(:).';

        % Randomly hold out 30 percent of pilots for rank selection.
        numberTrainingPilots = round(Rank_train_fraction * P);
        pilotOrder = randperm(P);
        trainingIndices = pilotOrder(1:numberTrainingPilots);
        validationIndices = pilotOrder(numberTrainingPilots+1:end);
        trainingPilotCount(pCase) = numel(trainingIndices);
        validationPilotCount(pCase) = numel(validationIndices);

        noiselessUserField = Gtrue * S;
        totalUserSignalPower = mean(abs(noiselessUserField(:)).^2);

        % Hankel-paper RSR is reference power divided by single-user power.
        singleUserPowers = zeros(K, 1);
        for user = 1:K
            userField = Gtrue(:, user) * S(user, :);
            singleUserPowers(user) = mean(abs(userField(:)).^2);
        end
        averageSingleUserPower = mean(singleUserPowers);

        for rsrCase = 1:numberOfRSR
            B = Braw;
            if ~isnan(RSR_dB_list(rsrCase))
                rawReferencePower = mean(abs(B(:)).^2);
                targetReferencePower = 10^(RSR_dB_list(rsrCase)/10) * ...
                    averageSingleUserPower;
                B = B * sqrt(targetReferencePower / rawReferencePower);
            end

            referencePower = mean(abs(B(:)).^2);
            measuredRSR_dB(trial, pCase, rsrCase) = ...
                10*log10(referencePower / averageSingleUserPower);

            for snrIndex = 1:numberOfSNR
                snrLinear = 10^(SNRdB(snrIndex)/10);
                sigma2 = totalUserSignalPower / snrLinear;

                W = sqrt(sigma2/2) * (randn(N, P) + 1j*randn(N, P));
                Z = abs(noiselessUserField + B + W);

                % Xu's biased Gerchberg-Saxton baseline.
                timer = tic;
                [GestGS, gsIterationCount] = runBiasedGS(Z, S, B, ...
                    GS_max_iterations, Relative_update_tolerance, ...
                    Consecutive_stable_iterations);
                elapsedSeconds(trial, snrIndex, pCase, rsrCase, 1) = toc(timer);
                iterationCounts(trial, snrIndex, pCase, rsrCase, 1) = ...
                    max(gsIterationCount);

                % Xu's gradient-descent baseline on the linearized model.
                timer = tic;
                [GestGD, gdIterationCount] = runGradientDescent(Z, S, B, ...
                    GD_max_iterations, Relative_update_tolerance, ...
                    Consecutive_stable_iterations, 0.1);
                elapsedSeconds(trial, snrIndex, pCase, rsrCase, 2) = toc(timer);
                iterationCounts(trial, snrIndex, pCase, rsrCase, 2) = ...
                    gdIterationCount;

                % Magnitude-likelihood EM-GS baseline.
                timer = tic;
                [GestEM, emIterationCount] = runEMGS(Z, S, B, sigma2, ...
                    EM_max_iterations, Relative_update_tolerance, ...
                    Consecutive_stable_iterations);
                elapsedSeconds(trial, snrIndex, pCase, rsrCase, 3) = toc(timer);
                iterationCounts(trial, snrIndex, pCase, rsrCase, 3) = ...
                    emIterationCount;

                % Rank selection uses only training/validation pilots.
                Ztrain = Z(:, trainingIndices);
                Strain = S(:, trainingIndices);
                Btrain = B(:, trainingIndices);
                Zvalidation = Z(:, validationIndices);
                Svalidation = S(:, validationIndices);
                Bvalidation = B(:, validationIndices);

                hankelParameter = floor((N-1)/2);
                maximumHankelRank = ceil(N/2);
                validationError = zeros(maximumHankelRank, 1);
                rankSelectionTimer = tic;

                for candidateRank = 1:maximumHankelRank
                    [Gcandidate, candidateIterations] = runRankHS( ...
                        Ztrain, Strain, Btrain, candidateRank, ...
                        hankelParameter, HS_rank_max_iterations, ...
                        Cadzow_sweeps, sigma2, Relative_update_tolerance, ...
                        Consecutive_stable_iterations);
                    selectedRankIterationCounts(trial, snrIndex, pCase, ...
                        rsrCase, candidateRank) = candidateIterations;
                    predictedValidationMagnitude = abs( ...
                        Gcandidate*Svalidation + Bvalidation);
                    validationError(candidateRank) = norm( ...
                        Zvalidation - predictedValidationMagnitude, 'fro')^2;
                end

                [~, bestRank] = min(validationError);
                selectedRank(trial, snrIndex, pCase, rsrCase) = bestRank;
                rankSelectionSeconds = toc(rankSelectionTimer);

                timer = tic;
                [GestHS, hsIterationCount] = runRankHS(Z, S, B, bestRank, ...
                    hankelParameter, HS_rank_max_iterations, Cadzow_sweeps, ...
                    sigma2, Relative_update_tolerance, ...
                    Consecutive_stable_iterations);
                elapsedSeconds(trial, snrIndex, pCase, rsrCase, 4) = ...
                    rankSelectionSeconds + toc(timer);
                iterationCounts(trial, snrIndex, pCase, rsrCase, 4) = ...
                    hsIterationCount;

                estimates = {GestGS, GestGD, GestEM, GestHS};
                for methodIndex = 1:numberOfMethods
                    errorByMethod(trial, snrIndex, pCase, rsrCase, methodIndex) = ...
                        norm(estimates{methodIndex} - Gtrue, 'fro')^2;
                end
                channelEnergy(trial, snrIndex, pCase) = norm(Gtrue, 'fro')^2;
            end
        end
    end

    if mod(trial, 10) == 0 || trial == MonteCarloTrials
        completedTrials = trial;
        save(checkpointFile, 'K', 'N', 'P_list', 'SNRdB', ...
            'MonteCarloTrials', 'RandomSeed', 'RSR_dB_list', ...
            'RSR_case_names', 'GS_max_iterations', 'GD_max_iterations', ...
            'EM_max_iterations', 'HS_rank_max_iterations', ...
            'Cadzow_sweeps', 'Rank_train_fraction', ...
            'Relative_update_tolerance', 'Consecutive_stable_iterations', ...
            'errorByMethod', 'channelEnergy', 'selectedRank', ...
            'iterationCounts', 'selectedRankIterationCounts', ...
            'elapsedSeconds', 'measuredRSR_dB', 'pathCounts', ...
            'trainingPilotCount', 'validationPilotCount', 'completedTrials');
        fprintf('Completed trial %d / %d\n', trial, MonteCarloTrials);
    end
end

%% Compute ratio-of-sums NMSE
methodIterationCaps = [GS_max_iterations, GD_max_iterations, ...
    EM_max_iterations, HS_rank_max_iterations];
capHitCountByMethod = zeros(numberOfMethods, 1);
for methodIndex = 1:numberOfMethods
    capHitCountByMethod(methodIndex) = sum( ...
        iterationCounts(:, :, :, :, methodIndex) >= methodIterationCaps(methodIndex), ...
        'all');
end
candidateRankCapHitCount = sum( ...
    selectedRankIterationCounts >= HS_rank_max_iterations, 'all');

nmseLinear = zeros(numberOfSNR, numberOfP, numberOfRSR, numberOfMethods);
nmse_dB = zeros(numberOfSNR, numberOfP, numberOfRSR, numberOfMethods);

for rsrCase = 1:numberOfRSR
    for pCase = 1:numberOfP
        for snrIndex = 1:numberOfSNR
            denominator = sum(channelEnergy(:, snrIndex, pCase));
            for methodIndex = 1:numberOfMethods
                numerator = sum(errorByMethod(:, snrIndex, pCase, ...
                    rsrCase, methodIndex));
                nmseLinear(snrIndex, pCase, rsrCase, methodIndex) = ...
                    numerator / denominator;
                nmse_dB(snrIndex, pCase, rsrCase, methodIndex) = ...
                    10*log10(nmseLinear(snrIndex, pCase, rsrCase, methodIndex));
            end
        end
    end
end

%% Save a tidy CSV table
numberOfRows = numberOfSNR * numberOfP * numberOfRSR * numberOfMethods;
resultRSR = cell(numberOfRows, 1);
resultP = zeros(numberOfRows, 1);
resultSNR = zeros(numberOfRows, 1);
resultMethod = cell(numberOfRows, 1);
resultNMSE = zeros(numberOfRows, 1);
resultMeanRank = nan(numberOfRows, 1);
resultMeanMeasuredRSR = zeros(numberOfRows, 1);
resultMeanIterations = zeros(numberOfRows, 1);
resultMeanSeconds = zeros(numberOfRows, 1);
row = 0;

for rsrCase = 1:numberOfRSR
    for pCase = 1:numberOfP
        for snrIndex = 1:numberOfSNR
            for methodIndex = 1:numberOfMethods
                row = row + 1;
                resultRSR{row} = RSR_case_names{rsrCase};
                resultP(row) = P_list(pCase);
                resultSNR(row) = SNRdB(snrIndex);
                resultMethod{row} = method_names{methodIndex};
                resultNMSE(row) = nmse_dB(snrIndex, pCase, rsrCase, methodIndex);
                resultMeanMeasuredRSR(row) = mean(measuredRSR_dB(:, pCase, rsrCase));
                resultMeanIterations(row) = mean(iterationCounts(:, snrIndex, ...
                    pCase, rsrCase, methodIndex));
                resultMeanSeconds(row) = mean(elapsedSeconds(:, snrIndex, ...
                    pCase, rsrCase, methodIndex));
                if methodIndex == 4
                    resultMeanRank(row) = mean(selectedRank(:, snrIndex, pCase, rsrCase));
                end
            end
        end
    end
end

resultTable = table(resultRSR, resultP, resultSNR, resultMethod, ...
    resultNMSE, resultMeanIterations, resultMeanSeconds, ...
    resultMeanRank, resultMeanMeasuredRSR, ...
    'VariableNames', {'ReferenceCase', 'P', 'SNR_dB', 'Method', ...
    'NMSE_dB', 'MeanIterations', 'MeanRuntimeSeconds', ...
    'MeanSelectedRank', 'MeanMeasuredRSR_dB'});
writetable(resultTable, fullfile(outputFolder, ...
    'convergence_audit_nmse_snr.csv'));

%% Plot a separate two-panel figure for each reference case
for rsrCase = 1:numberOfRSR
    figureHandle = figure('Color', 'w', 'Position', [80, 80, 1150, 480]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for pCase = 1:numberOfP
        axesHandle = nexttile;
        hold(axesHandle, 'on');
        for methodIndex = 1:numberOfMethods
            plot(axesHandle, SNRdB, ...
                nmse_dB(:, pCase, rsrCase, methodIndex), '-o', ...
                'LineWidth', 1.5, 'MarkerSize', 5);
        end
        grid(axesHandle, 'on');
        xlabel(axesHandle, 'Aggregate user-signal SNR (dB)');
        ylabel(axesHandle, 'NMSE (dB)');
        title(axesHandle, sprintf('N = %d, K = %d, P = %d', ...
            N, K, P_list(pCase)));
        legend(axesHandle, method_names, 'Location', 'southwest');
    end

    if isnan(RSR_dB_list(rsrCase))
        meanObservedRSR = mean(measuredRSR_dB(:, :, rsrCase), 'all');
        figureTitle = sprintf('Xu reference draw; mean measured RSR = %.1f dB', ...
            meanObservedRSR);
        outputStem = 'convergence_audit_xu_raw_reference';
    else
        figureTitle = sprintf('Reference rescaled to RSR = %g dB', ...
            RSR_dB_list(rsrCase));
        outputStem = sprintf('convergence_audit_RSR_%gdB', RSR_dB_list(rsrCase));
    end
    sgtitle(figureTitle);
    savefig(figureHandle, fullfile(outputFolder, [outputStem, '.fig']));
    exportgraphics(figureHandle, fullfile(outputFolder, [outputStem, '.png']), ...
        'Resolution', 200);
end

simulationChoices = struct( ...
    'measurementModel', 'Z = abs(G*S + B + W), exact magnitude observations', ...
    'channel', 'Xu multipath equation; path polarization shared over the array for a Hankel-compatible extension', ...
    'reference', 'Xu one-path equation; one polarization vector shared over the array', ...
    'referenceSymbol', 's_b,p = 1, known; implementation choice because its distribution is unspecified', ...
    'aoa', 'Independent broadside AoAs uniform on (-90,90) degrees, borrowed from Hankel paper', ...
    'arraySpacing', 'd/lambda = 0.5, borrowed from Hankel paper', ...
    'arrayPhase', 'Xu cosine convention after coordinate conversion: phi = pi*cos(theta_axis)', ...
    'snr', 'Mean combined user-field power mean(abs(G*S).^2) divided by complex noise variance', ...
    'rsrCases', 'Xu raw reference draw; and reference power normalized to 12 dB and 50 dB vs one-user power', ...
    'rankSelection', '70/30 pilot split; choose rank only by held-out magnitude prediction residual', ...
    'convergence', 'Stop after 5 consecutive iterations with squared relative update below 1e-8, or at method-specific maximum iteration cap', ...
    'finalScore', 'Ratio-of-sums channel NMSE; true channel used only for evaluation');

save(fullfile(outputFolder, 'convergence_audit_nmse_snr.mat'), ...
    'K', 'N', 'P_list', 'SNRdB', 'MonteCarloTrials', 'RandomSeed', ...
    'RSR_dB_list', 'RSR_case_names', 'GS_max_iterations', ...
    'GD_max_iterations', 'EM_max_iterations', 'HS_rank_max_iterations', ...
    'methodIterationCaps', 'capHitCountByMethod', ...
    'candidateRankCapHitCount', ...
    'Cadzow_sweeps', 'Rank_train_fraction', ...
    'Relative_update_tolerance', 'Consecutive_stable_iterations', ...
    'errorByMethod', 'channelEnergy', 'selectedRank', 'measuredRSR_dB', ...
    'pathCounts', 'trainingPilotCount', 'validationPilotCount', ...
    'iterationCounts', 'selectedRankIterationCounts', 'elapsedSeconds', ...
    'nmseLinear', 'nmse_dB', 'simulationChoices');

fprintf('\nFinal ratio-of-sums NMSE results (dB):\n');
for rsrCase = 1:numberOfRSR
    fprintf('\nReference case: %s\n', RSR_case_names{rsrCase});
    for pCase = 1:numberOfP
        fprintf('P = %d\n', P_list(pCase));
        for snrIndex = 1:numberOfSNR
            fprintf('  SNR %3d dB:', SNRdB(snrIndex));
            for methodIndex = 1:numberOfMethods
                fprintf(' %s %.2f dB;', method_names{methodIndex}, ...
                    nmse_dB(snrIndex, pCase, rsrCase, methodIndex));
            end
            fprintf('\n');
        end
    end
end
fprintf('\nMean iteration counts and runtimes are saved in the audit CSV.\n');
fprintf('Final-estimate cap hits by method [GS GD EM-GS HS-GS]: %s\n', ...
    mat2str(capHitCountByMethod.'));
fprintf('Candidate-rank HS-GS cap hits: %d\n', candidateRankCapHitCount);
fprintf('Audit pilot mode: %d (set AuditPilotMode=false for full 200-trial grid).\n', ...
    AuditPilotMode);
fprintf('Saved CSV, MAT data, checkpoints, and figures in:\n%s\n', outputFolder);

%% Local functions
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

function [G, iterationCount] = runBiasedGS(Z, S, B, maximumIterations, ...
        tolerance, stableIterationsRequired)
    [N, ~] = size(Z);
    K = size(S, 1);
    G = zeros(N, K);
    iterationCount = zeros(N, 1);

    for element = 1:N
        z = Z(element, :).';
        b = B(element, :).';
        augmentedPilotMatrix = [S.' b];
        spectralMatrix = augmentedPilotMatrix' * ...
            (z .* augmentedPilotMatrix);
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
        estimate = augmentedEstimate(1:K);

        stableCount = 0;
        iterationCount(element) = maximumIterations;
        for iteration = 1:maximumIterations
            previousEstimate = estimate;
            predictedPhase = angle(S.'*estimate + b);
            estimate = (conj(S)*S.') \ ...
                (conj(S)*(z.*exp(1j*predictedPhase) - b));

            relativeChange = norm(estimate-previousEstimate)^2 / ...
                max(norm(estimate)^2, eps);
            if relativeChange < tolerance
                stableCount = stableCount + 1;
            else
                stableCount = 0;
            end
            if stableCount >= stableIterationsRequired
                iterationCount(element) = iteration;
                break;
            end
        end
        G(element, :) = estimate.';
    end
end

function [G, iterationCount] = runGradientDescent(Z, S, B, ...
        maximumIterations, tolerance, stableIterationsRequired, initialVariance)
    [N, ~] = size(Z);
    K = size(S, 1);
    G = sqrt(initialVariance/2) * ...
        (randn(N, K) + 1j*randn(N, K));

    stepSize = 1 / max(real(eig(S*S')));
    referencePhase = exp(-1j*angle(B));
    iterationCount = maximumIterations;
    stableCount = 0;

    for iteration = 1:maximumIterations
        previousEstimate = G;
        residual = Z - abs(B) - real(referencePhase .* (G*S));
        gradient = -(residual .* conj(referencePhase)) * S';
        G = G - stepSize * gradient;

        relativeChange = norm(G-previousEstimate, 'fro')^2 / ...
            max(norm(G, 'fro')^2, eps);
        if relativeChange < tolerance
            stableCount = stableCount + 1;
        else
            stableCount = 0;
        end
        if stableCount >= stableIterationsRequired
            iterationCount = iteration;
            break;
        end
    end
end

function [G, iterationCount] = runEMGS(Z, S, B, sigma2, ...
        maximumIterations, tolerance, stableIterationsRequired)
    [N, ~] = size(Z);
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
    iterationCount = maximumIterations;
    stableCount = 0;
    for iteration = 1:maximumIterations
        previousEstimate = G;
        predictedField = G*S + B;
        predictedPhase = angle(predictedField);
        concentration = (2/sigma2) * Z .* abs(predictedField);

        besselRatio = besseli(1, concentration, 1) ./ ...
            besseli(0, concentration, 1);
        expectedField = Z .* exp(1j*predictedPhase) .* besselRatio;
        G = (expectedField - B) * leastSquaresMap;

        relativeChange = norm(G-previousEstimate, 'fro')^2 / ...
            max(norm(G, 'fro')^2, eps);
        if relativeChange < tolerance
            stableCount = stableCount + 1;
        else
            stableCount = 0;
        end
        if stableCount >= stableIterationsRequired
            iterationCount = iteration;
            break;
        end
    end
end

function [G, iterationCount] = runRankHS(Z, S, B, rankValue, ...
        hankelParameter, maximumIterations, cadzowSweeps, sigma2, ...
        tolerance, stableIterationsRequired)
    [G, ~] = runEMGS(Z, S, B, sigma2, 0, tolerance, ...
        stableIterationsRequired);
    leastSquaresMap = S' / (S*S');
    K = size(S, 1);

    iterationCount = maximumIterations;
    stableCount = 0;
    for iteration = 1:maximumIterations
        previousEstimate = G;
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

        relativeChange = norm(G-previousEstimate, 'fro')^2 / ...
            max(norm(G, 'fro')^2, eps);
        if relativeChange < tolerance
            stableCount = stableCount + 1;
        else
            stableCount = 0;
        end
        if stableCount >= stableIterationsRequired
            iterationCount = iteration;
            break;
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
