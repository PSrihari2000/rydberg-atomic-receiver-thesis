% Version-2 noise-conditioned unfolded magnitude estimator.
% Uses all 2048 training channels, per-channel normalized NMSE loss, and a
% deterministic strong-reference linearized initialization. Model selection
% uses validation data only. Frozen version-1 models and test data are never
% loaded or overwritten by this development script.
clear;clc;
root=fileparts(mfilename('fullpath'));
output=fullfile(root,'noise_aware_v2_development_results');
if ~exist(output,'dir'),mkdir(output);end

layers=30;
epochs=40;
batchSize=512;
learningRate=0.01;
earlyStoppingPatience=8;
summary=table();

for P=[10 30]
    rng(20261020+P,'twister');
    training=load(fullfile(root,'nonlinear_development_data',sprintf('train_P%d.mat',P)));
    validation=load(fullfile(root,'nonlinear_development_data',sprintf('validation_P%d.mat',P)));
    trainingCount=size(training.channels,3);
    validationCount=size(validation.channels,3);

    priorComplexVariance=mean(abs(training.channels(:)).^2);
    fprintf('P=%d: packing %d training and %d validation channels...\n', ...
        P,trainingCount,validationCount);
    train=packData(training,trainingCount,priorComplexVariance);
    valid=packData(validation,validationCount,priorComplexVariance);

    previous=load(fullfile(root,'noise_aware_development_results',sprintf('model_P%d.mat',P)));
    parameters=dlarray(single(extractdata(previous.bestParameters)));
    initialParameters=parameters;
    average=[];averageSquared=[];iteration=0;

    initialMetrics=evaluate(parameters,valid,batchSize);
    bestScore=initialMetrics.ratioOfSums;
    bestParameters=parameters;
    bestEpoch=0;
    epochsWithoutImprovement=0;
    history=nan(epochs,5);
    fprintf(['P=%d: linearized initialization plus %d nonlinear layers; ' ...
        'initial validation ratio-of-sums %.3f dB, mean channel NMSE %.3f dB\n'], ...
        P,layers,10*log10(initialMetrics.ratioOfSums), ...
        10*log10(initialMetrics.meanChannelNMSE));

    timer=tic;
    for epoch=1:epochs
        order=randperm(size(train.target,2));
        epochLoss=0;
        batchCount=0;
        for first=1:batchSize:numel(order)
            indices=order(first:min(first+batchSize-1,numel(order)));
            batch=takeBatch(train,indices);
            iteration=iteration+1;
            [loss,gradients]=dlfeval(@modelLoss,parameters,batch);
            assert(all(isfinite(extractdata(gradients)),'all'),'Nonfinite gradients');
            [parameters,average,averageSquared]=adamupdate(parameters,gradients, ...
                average,averageSquared,iteration,learningRate);
            epochLoss=epochLoss+double(extractdata(loss));
            batchCount=batchCount+1;
        end

        metrics=evaluate(parameters,valid,batchSize);
        history(epoch,:)=[epoch,epochLoss/batchCount, ...
            10*log10(metrics.ratioOfSums),10*log10(metrics.meanChannelNMSE), ...
            metrics.predictionEnergyRatio];
        if metrics.ratioOfSums < bestScore
            bestScore=metrics.ratioOfSums;
            bestParameters=parameters;
            bestEpoch=epoch;
            epochsWithoutImprovement=0;
        else
            epochsWithoutImprovement=epochsWithoutImprovement+1;
        end
        fprintf(['P=%d epoch %d/%d: loss %.5g, validation %.3f dB, ' ...
            'mean-channel %.3f dB%s\n'],P,epoch,epochs,history(epoch,2), ...
            history(epoch,3),history(epoch,4), ...
            improvementMarker(epoch,bestEpoch));

        latestParameters=parameters; %#ok<NASGU>
        save(fullfile(output,sprintf('training_checkpoint_P%d.mat',P)), ...
            'latestParameters','bestParameters','initialParameters','history', ...
            'bestEpoch','bestScore','epoch','iteration','layers','epochs', ...
            'batchSize','learningRate','priorComplexVariance');
        if epoch>=12 && epochsWithoutImprovement>=earlyStoppingPatience
            fprintf('P=%d: early stopping after %d epochs without improvement.\n', ...
                P,epochsWithoutImprovement);
            break;
        end
    end

    trainingSeconds=toc(timer);
    history=history(1:epoch,:);
    trained=evaluate(bestParameters,valid,batchSize);
    trainMetrics=evaluate(bestParameters,train,batchSize);
    row=table(P,trainingCount,validationCount,trainingSeconds,bestEpoch, ...
        10*log10(trainMetrics.ratioOfSums),10*log10(trained.ratioOfSums), ...
        10*log10(trained.meanChannelNMSE),10*log10(initialMetrics.ratioOfSums), ...
        trained.predictionEnergyRatio, ...
        'VariableNames',{'P','TrainingChannels','ValidationChannels', ...
        'TrainingSeconds','BestEpoch','TrainNMSEdB','ValidationNMSEdB', ...
        'ValidationMeanChannelNMSEdB','InitialValidationNMSEdB', ...
        'PredictionEnergyRatio'});
    summary=[summary;row];disp(row); %#ok<AGROW>
    save(fullfile(output,sprintf('model_P%d.mat',P)), ...
        'bestParameters','initialParameters','history','row','layers','epochs', ...
        'batchSize','learningRate','priorComplexVariance','bestEpoch');
    writetable(summary,fullfile(output,'summary.csv'));

    bySNR=table();
    for snr=-5:5:30
        selected=find(valid.snr==snr);
        metrics=evaluate(bestParameters,takeBatch(valid,selected),batchSize);
        bySNR=[bySNR;table(P,snr,10*log10(metrics.ratioOfSums), ...
            10*log10(metrics.meanChannelNMSE), ...
            'VariableNames',{'P','SNRdB','ValidationNMSEdB', ...
            'ValidationMeanChannelNMSEdB'})]; %#ok<AGROW>
    end
    writetable(bySNR,fullfile(output,sprintf('validation_by_snr_P%d.csv',P)));
end
fprintf('Version-2 development training finished; frozen test files were never loaded.\n');

function data=packData(source,count,priorComplexVariance)
    N=8;K=3;P=size(source.pilots,2);rows=N*count;
    data.ar=zeros(K,P,rows,'single');data.ai=data.ar;
    data.b=zeros(1,P,rows,'single');data.y=data.b;
    data.bound=zeros(1,1,rows,'single');
    data.target=zeros(6,rows,'single');
    data.initial=data.target;
    data.snr=zeros(rows,1);
    data.penalty=zeros(1,1,rows,'single');
    data.channelIndex=zeros(1,rows,'uint32');
    data.channelEnergy=zeros(1,1,rows,'single');
    for m=1:count
        S=source.pilots(:,:,m);
        bound=max(real(eig(S*S')));
        channelEnergy=sum(abs(source.channels(:,:,m)).^2,'all');
        for i=1:N
            n=(m-1)*N+i;
            B=source.references(i,:,m);
            rotated=S.*(conj(B)./abs(B));
            ar=real(rotated);ai=imag(rotated);
            y=source.Ymag(i,:,m);
            b=abs(B);
            design=[ar.',-ai.'];
            initial=pinv(design)*(y-b).';
            data.ar(:,:,n)=single(ar);data.ai(:,:,n)=single(ai);
            data.b(:,:,n)=single(b);data.y(:,:,n)=single(y);
            data.bound(:,:,n)=single(bound);
            g=source.channels(i,:,m);
            data.target(:,n)=single([real(g),imag(g)].');
            data.initial(:,n)=single(initial);
            data.snr(n)=source.snrDB(m);
            data.penalty(:,:,n)=single(source.noiseVariance(m)/priorComplexVariance);
            data.channelIndex(n)=uint32(m);
            data.channelEnergy(:,:,n)=single(channelEnergy);
        end
    end
end

function out=takeBatch(data,index)
    out.ar=data.ar(:,:,index);out.ai=data.ai(:,:,index);
    out.b=data.b(:,:,index);out.y=data.y(:,:,index);
    out.bound=data.bound(:,:,index);out.target=data.target(:,index);
    out.initial=data.initial(:,index);
    out.snr=data.snr(index);out.penalty=data.penalty(:,:,index);
    out.channelIndex=data.channelIndex(index);
    out.channelEnergy=data.channelEnergy(:,:,index);
end

function [loss,gradients]=modelLoss(parameters,data)
    prediction=forwardModel(parameters,data);
    squaredError=sum((prediction-data.target).^2,1);
    loss=mean(squaredError./(reshape(data.channelEnergy,1,[])+single(1e-12)),'all');
    gradients=dlgradient(loss,parameters);
end

function prediction=forwardModel(parameters,data)
    count=size(data.target,2);
    realG=dlarray(reshape(data.initial(1:3,:),3,1,count));
    imagG=dlarray(reshape(data.initial(4:6,:),3,1,count));
    previousReal=realG;previousImag=imagG;
    noiseLevel=log(1+data.penalty./data.bound);
    for layer=1:size(parameters,1)
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

function metrics=evaluate(parameters,data,batchSize)
    error=0;energy=0;predictionEnergy=0;
    channelError=zeros(1,max(double(data.channelIndex)));
    channelEnergy=zeros(size(channelError));
    for first=1:batchSize:size(data.target,2)
        index=first:min(first+batchSize-1,size(data.target,2));
        batch=takeBatch(data,index);
        estimate=double(extractdata(forwardModel(parameters,batch)));
        target=double(batch.target);
        rowError=sum((estimate-target).^2,1);
        rowEnergy=sum(target.^2,1);
        error=error+sum(rowError);
        energy=energy+sum(rowEnergy);
        predictionEnergy=predictionEnergy+sum(estimate.^2,'all');
        for q=1:numel(index)
            m=double(batch.channelIndex(q));
            channelError(m)=channelError(m)+rowError(q);
            channelEnergy(m)=channelEnergy(m)+rowEnergy(q);
        end
    end
    metrics.ratioOfSums=error/energy;
    present=channelEnergy>0;
    metrics.meanChannelNMSE=mean(channelError(present)./channelEnergy(present));
    metrics.predictionEnergyRatio=predictionEnergy/energy;
end

function marker=improvementMarker(epoch,bestEpoch)
    if epoch==bestEpoch
        marker='  best';
    else
        marker='';
    end
end
