% Direct supervised DL with complete exact-magnitude input information.
% A phase rotation makes the known reference real without changing magnitudes.
% Inputs include every Y, |B| and rotated complex pilot, plus noise variance.
% Linearized summary features are auxiliary, not the only observation input.
% No classical estimator output enters training or inference.
clear; clc;
rootFolder=fileparts(mfilename('fullpath'));
dataFolder=fullfile(rootFolder,'nonlinear_development_data');
outputFolder=fullfile(rootFolder,'nonlinear_development_results');
if ~exist(outputFolder,'dir'),mkdir(outputFolder);end
summary=table();
for P=[10 30]
    rng(20261018+P,'twister');
    train=load(fullfile(dataFolder,sprintf('train_P%d.mat',P)));
    validation=load(fullfile(dataFolder,sprintf('validation_P%d.mat',P)));
    [X,T]=makeConditionedExamples(train);
    [XV,TV]=makeConditionedExamples(validation);
    inputMean=mean(X,1);inputStd=max(std(X,0,1),single(1e-5));
    targetScale=sqrt(mean(T(:).^2));
    X=(X-inputMean)./inputStd;XV=(XV-inputMean)./inputStd;
    T=T/targetScale;TV=TV/targetScale;
    layers=[featureInputLayer(size(X,2),Normalization='none')
        fullyConnectedLayer(128);reluLayer
        fullyConnectedLayer(128);reluLayer
        fullyConnectedLayer(6)];
    options=trainingOptions('adam',MaxEpochs=80,MiniBatchSize=256, ...
        InitialLearnRate=1e-3,Shuffle='every-epoch', ...
        ValidationData={XV,TV},ValidationFrequency=128, ...
        OutputNetwork='best-validation-loss',ExecutionEnvironment='cpu', ...
        Plots='none',Verbose=true,VerboseFrequency=320);
    fprintf('P=%d: %d channel realizations, %d cell rows; 256 validation channels, exact-magnitude inputs\n', ...
        P,size(train.channels,3),size(X,1));
    timer=tic;[network,info]=trainnet(X,T,layers,'mse',options);
    trainingSeconds=toc(timer);
    predictionTrain=minibatchpredict(network,X);
    predictionValidation=minibatchpredict(network,XV);
    trainNMSE=sum((predictionTrain-T).^2,'all')/sum(T.^2,'all');
    validationNMSE=sum((predictionValidation-TV).^2,'all')/sum(TV.^2,'all');
    predictionEnergyRatio=sum(predictionValidation.^2,'all')/sum(TV.^2,'all');
    correlation=sum(predictionValidation.*TV,'all')/ ...
        sqrt(sum(predictionValidation.^2,'all')*sum(TV.^2,'all'));
    % All cell rows of one channel remain together in one partition.
    % Summing rowwise real/imaginary squared errors equals channel Frobenius error.
    row=table(P,trainingSeconds,10*log10(trainNMSE),10*log10(validationNMSE), ...
        predictionEnergyRatio,correlation,VariableNames={'P','TrainingSeconds', ...
        'TrainNMSEdB','ValidationNMSEdB','PredictionEnergyRatio','RealVectorCorrelation'});
    summary=[summary;row]; %#ok<AGROW>
    disp(row);
    save(fullfile(outputFolder,sprintf('model_P%d.mat',P)), ...
        'network','info','inputMean','inputStd','targetScale','row');
    writetable(summary,fullfile(outputFolder,'summary.csv'));
    grouped=table();
    for snr=-5:5:30
        channelSelection=find(validation.snrDB==snr);
        rows=reshape((channelSelection(:)-1)*8+(1:8),[],1);
        errorEnergy=sum((predictionValidation(rows,:)-TV(rows,:)).^2,'all');
        trueEnergy=sum(TV(rows,:).^2,'all');
        score=10*log10(errorEnergy/trueEnergy);
        grouped=[grouped;table(P,snr,score,VariableNames={'P','SNRdB','ValidationNMSEdB'})]; %#ok<AGROW>
    end
    writetable(grouped,fullfile(outputFolder,sprintf('validation_by_snr_P%d.csv',P)));
end
fprintf('Noisy nonlinear-input development run complete. Final test set untouched.\n');

function [X,T]=makeConditionedExamples(data)
    N=size(data.channels,1);K=size(data.channels,2);
    P=size(data.pilots,2);count=size(data.channels,3);
    triangle=triu(true(2*K));
    X=zeros(N*count,6+nnz(triangle)+8*P+1,'single');
    T=zeros(N*count,2*K,'single');
    for sample=1:count
        S=data.pilots(:,:,sample);
        for i=1:N
            row=(sample-1)*N+i;
            B=data.references(i,:,sample).';
            Y=data.Ymag(i,:,sample).';
            % Real linearization: Y-|B| approx D*[real(g);imag(g)].
            referencePhase=conj(B)./max(abs(B),eps);
            weightedPilots=referencePhase.*S.';
            D=[real(weightedPilots),-imag(weightedPilots)];
            gram=(D.'*D)/P;
            backprojection=(D.'*(Y-abs(B)))/P;
            % Invertible magnitude preprocessing: Y=(Y-|B|)+|B|.
            % |sum rotatedPilots*g + |B|| equals |sum S*g+B|.
            X(row,:)=single([backprojection;gram(triangle);Y-abs(B); ...
                real(weightedPilots(:));imag(weightedPilots(:)); ...
                abs(B);sqrt(data.noiseVariance(sample))].');
            g=data.channels(i,:,sample);
            T(row,:)=single([real(g),imag(g)]);
        end
    end
    assert(all(isfinite(X(:))));
end
