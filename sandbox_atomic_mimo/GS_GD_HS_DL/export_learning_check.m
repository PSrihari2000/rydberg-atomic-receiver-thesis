% Small noiseless learning check, not the final NMSE benchmark.
% Existing channel helpers below are copied unchanged from
% 1D_Channel_Estimation_Benchmark/run_classic_hankel_comparison.m.
clear; clc;
outputFolder = fullfile(fileparts(mfilename('fullpath')), 'learning_check_data');
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end
N = 8; K = 3;
pilotCounts = [10 30];
splitNames = {'train','validation','test'};
splitCounts = [64 32 32];
splitSeeds = [2026101801 2026101802 2026101803];
RSRdB = 50; % Local strong-reference choice; not a claimed Xu paper value.
% Ratio here is reference power / aggregate user-field power.
for pilotCase = 1:numel(pilotCounts)
    P = pilotCounts(pilotCase);
    for splitIndex = 1:numel(splitNames)
        rng(splitSeeds(splitIndex) + pilotCase*100,'twister');
        sampleCount = splitCounts(splitIndex);
        Ymag = zeros(N,P,sampleCount);
        channels = complex(zeros(N,K,sampleCount));
        pilots = complex(zeros(K,P,sampleCount));
        references = complex(zeros(N,P,sampleCount));
        noiseVariance = zeros(sampleCount,1); % Noiseless diagnostic only.
        for sample = 1:sampleCount
            [G,~] = generateUserChannel(N,K,0.5,[0;1;0],1);
            S = (randn(K,P)+1j*randn(K,P))/sqrt(2);
            B = generateReferencePath(N,P,1,0.5,[0;1;0],1);
            userField = G*S;
            B = B*sqrt(10^(RSRdB/10)*mean(abs(userField(:)).^2) ...
                / mean(abs(B(:)).^2));
            channels(:,:,sample) = G;
            pilots(:,:,sample) = S;
            references(:,:,sample) = B;
            Ymag(:,:,sample) = abs(userField+B);
        end
        splitName = splitNames{splitIndex};
        fileName = sprintf('%s_P%d.mat',splitName,P);
        save(fullfile(outputFolder,fileName),'Ymag','channels','pilots', ...
            'references','noiseVariance','N','K','P','RSRdB','splitName','-v7');
        fprintf('Saved %s: %d independent channels\n',fileName,sampleCount);
    end
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

