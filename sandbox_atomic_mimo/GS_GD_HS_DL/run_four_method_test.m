function run_four_method_test(mode,trialCount)
% Shared exact-magnitude observations; frozen DL; no training/test leakage.
% Call run_four_method_test('integration',5) before ('final',500).
if nargin<1,mode='integration';end
if nargin<2,trialCount=5;end
root=fileparts(mfilename('fullpath'));
output=fullfile(root,['four_method_' mode]);
if ~exist(output,'dir'),mkdir(output);end
pilotCounts=[10 30];snrValues=-5:5:30;
methods={'Biased GS','Linearized GD','HS-GS','Noise-aware unfolded DL'};
if strcmp(mode,'final'),seed=2026101850;else,seed=2026101840;end
models=cell(2,1);
for pc=1:2
    models{pc}=load(fullfile(root,'frozen_noise_aware_models',sprintf('model_P%d.mat',pilotCounts(pc))));
    models{pc}.parameters=extractdata(models{pc}.bestParameters);
end
errors=zeros(trialCount,8,2,4);energies=zeros(trialCount,8,2);
seconds=zeros(size(errors));ranks=zeros(trialCount,8,2);completed=0;
checkpoint=fullfile(output,'checkpoint.mat');
if exist(checkpoint,'file')
    saved=load(checkpoint);
    assert(saved.trialCount==trialCount,'Incompatible checkpoint trial count');
    errors=saved.errors;energies=saved.energies;seconds=saved.seconds;
    ranks=saved.ranks;completed=saved.completed;
end
timer=tic;
for trial=completed+1:trialCount
    % Each trial has its own seed: resume cannot change future observations.
    rng(seed+trial,'twister');
    [G,~]=generateUserChannel(8,3,.5,[0;1;0],1);
    Sfull=(randn(3,30)+1j*randn(3,30))/sqrt(2);
    Braw=generateReferencePath(8,30,1,.5,[0;1;0],1);
    observation=cell(2,1);
    % Generate all observations BEFORE algorithms consume any RNG draws.
    for pc=1:2
        P=pilotCounts(pc);S=Sfull(:,1:P);B=Braw(:,1:P);field=G*S;
        B=B*sqrt(1e5*mean(abs(field(:)).^2)/mean(abs(B(:)).^2));
        order=randperm(P);T=order(1:round(.7*P));V=order(round(.7*P)+1:end);
        Y=zeros(8,P,8);variance=zeros(8,1);
        for si=1:8
            variance(si)=mean(abs(field(:)).^2)/10^(snrValues(si)/10);
            W=sqrt(variance(si)/2)*(randn(8,P)+1j*randn(8,P));
            Y(:,:,si)=abs(field+B+W);
        end
        observation{pc}=struct('S',S,'B',B,'Ymag',Y,'sigma2',variance,'T',T,'V',V);
    end
    % Save actual shared test observations for reproducibility across languages.
    save(fullfile(output,sprintf('observations_%04d.mat',trial)),'G','observation','pilotCounts','snrValues');
    for pc=1:2
        o=observation{pc};S=o.S;B=o.B;
        for si=1:8
            Y=o.Ymag(:,:,si);sigma2=o.sigma2(si);
            clock=tic;estimates{1}=runBiasedGS(Y,S,B,1000,1e-10);seconds(trial,si,pc,1)=toc(clock);
            clock=tic;estimates{2}=runGradientDescent(Y,S,B,1000,1e-10,.1);seconds(trial,si,pc,2)=toc(clock);
            clock=tic;score=zeros(4,1);
            for r=1:4
                candidate=runRankHS(Y(:,o.T),S(:,o.T),B(:,o.T),r,3,25,4,sigma2);
                score(r)=norm(Y(:,o.V)-abs(candidate*S(:,o.V)+B(:,o.V)),'fro')^2;
            end
            [~,r]=min(score);ranks(trial,si,pc)=r;
            estimates{3}=runRankHS(Y,S,B,r,3,100,4,sigma2);seconds(trial,si,pc,3)=toc(clock);
            clock=tic;estimates{4}=inferFrozenDL(Y,S,B,sigma2,models{pc});seconds(trial,si,pc,4)=toc(clock);
            for a=1:4
                assert(all(isfinite(estimates{a}),'all'));
                errors(trial,si,pc,a)=norm(estimates{a}-G,'fro')^2;
            end
            energies(trial,si,pc)=norm(G,'fro')^2;
        end
    end
    completed=trial;
    save(checkpoint,'errors','energies','seconds','ranks','completed','trialCount','methods','pilotCounts','snrValues','seed');
    fprintf('%s: completed %d/%d trials; elapsed %.1f s\n',mode,trial,trialCount,toc(timer));
end
result=table();
for pc=1:2
    for si=1:8
        for a=1:4
            P=pilotCounts(pc);SNRdB=snrValues(si);Method=string(methods{a});
            NMSEdB=10*log10(sum(errors(:,si,pc,a))/sum(energies(:,si,pc)));
            MeanSeconds=mean(seconds(:,si,pc,a));
            result=[result;table(P,SNRdB,Method,NMSEdB,MeanSeconds)]; %#ok<AGROW>
        end
    end
end
writetable(result,fullfile(output,'nmse_snr.csv'));
fprintf('Finished %s. Output %s\n',mode,output);
end

function G=inferFrozenDL(Y,S,B,sigma2,model)
% This function has no true-channel argument or access to test labels.
data.ar=zeros(3,size(S,2),8,'single');data.ai=data.ar;
data.b=zeros(1,size(S,2),8,'single');data.y=data.b;
data.bound=repmat(single(max(real(eig(S*S')))),1,1,8);
data.penalty=repmat(single(sigma2/model.priorComplexVariance),1,1,8);
data.count=8;
for i=1:8
    rotated=S.*(conj(B(i,:))./abs(B(i,:)));
    data.ar(:,:,i)=single(real(rotated));data.ai(:,:,i)=single(imag(rotated));
    data.b(:,:,i)=single(abs(B(i,:)));data.y(:,:,i)=single(Y(i,:));
end
prediction=forwardModel(model.parameters,data);
G=double(prediction(1:3,:)+1j*prediction(4:6,:)).';
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

function G = runBiasedGS(Z, S, B, maximumIterations, tolerance)
    [N, ~] = size(Z);
    K = size(S, 1);
    G = zeros(N, K);

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

        for iteration = 1:maximumIterations
            previousEstimate = estimate;
            predictedPhase = angle(S.'*estimate + b);
            estimate = (conj(S)*S.') \ ...
                (conj(S)*(z.*exp(1j*predictedPhase) - b));

            relativeChange = norm(estimate-previousEstimate)^2 / ...
                max(norm(estimate)^2, eps);
            if relativeChange < tolerance
                break;
            end
        end
        G(element, :) = estimate.';
    end
end

function [G, iterationCount] = runGradientDescent(Z, S, B, ...
        maximumIterations, tolerance, initialVariance)
    [N, ~] = size(Z);
    K = size(S, 1);
    G = sqrt(initialVariance/2) * ...
        (randn(N, K) + 1j*randn(N, K));

    stepSize = 1 / max(real(eig(S*S')));
    referencePhase = exp(-1j*angle(B));
    iterationCount = maximumIterations;

    for iteration = 1:maximumIterations
        previousEstimate = G;
        residual = Z - abs(B) - real(referencePhase .* (G*S));
        gradient = -(residual .* conj(referencePhase)) * S';
        G = G - stepSize * gradient;

        relativeChange = norm(G-previousEstimate, 'fro')^2 / ...
            max(norm(G, 'fro')^2, eps);
        if relativeChange < tolerance
            iterationCount = iteration;
            break;
        end
    end
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

function prediction=forwardModel(parameters,data)
    count=data.count;
    realG=zeros(3,1,count,'single');imagG=realG;
    previousReal=realG;previousImag=imagG;
    for layer=1:size(parameters,1)
        noiseLevel=log(1+data.penalty./data.bound);
        step=1.9./(1+exp(-(parameters(layer,1)+parameters(layer,4).*noiseLevel)));
        momentum=0.95./(1+exp(-(parameters(layer,2)+parameters(layer,5).*noiseLevel)));
        ridgeWeight=log(1+exp(parameters(layer,3)));
        extrapReal=realG+momentum.*(realG-previousReal);
        extrapImag=imagG+momentum.*(imagG-previousImag);
        fieldReal=sum(data.ar.*extrapReal-data.ai.*extrapImag,1)+data.b;
        fieldImag=sum(data.ar.*extrapImag+data.ai.*extrapReal,1);
        amplitude=sqrt(fieldReal.^2+fieldImag.^2+single(1e-12));
        residualFactor=(amplitude-data.y)./amplitude;
        errorReal=residualFactor.*fieldReal;errorImag=residualFactor.*fieldImag;
        gradientReal=sum(data.ar.*errorReal+data.ai.*errorImag,2);
        gradientImag=sum(-data.ai.*errorReal+data.ar.*errorImag,2);
        previousReal=realG;previousImag=imagG;
        denominator=1+step.*ridgeWeight.*data.penalty./data.bound;
        realG=(extrapReal-step.*gradientReal./data.bound)./denominator;
        imagG=(extrapImag-step.*gradientImag./data.bound)./denominator;
    end
    prediction=reshape(cat(1,realG,imagG),6,count);
end

