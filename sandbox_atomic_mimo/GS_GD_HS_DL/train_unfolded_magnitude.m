% Learned nonlinear magnitude-gradient unfolding; development experiment.
% Zero initialization, no GS/GD/EM prediction input, no Hankel projection.
% Thirty layers learn scalar step/momentum parameters by supervised channel MSE.
% The measurement gradient is exact for magnitude-residual least squares,
% not a linearized GD update and not a Rician likelihood update.
clear;clc;
root=fileparts(mfilename('fullpath'));
output=fullfile(root,'unfolded_development_results');
if ~exist(output,'dir'),mkdir(output);end
layers=30;epochs=20;batchSize=256;summary=table();
for P=[10 30]
    rng(20261019+P,'twister');
    training=load(fullfile(root,'nonlinear_development_data',sprintf('train_P%d.mat',P)));
    validation=load(fullfile(root,'nonlinear_development_data',sprintf('validation_P%d.mat',P)));
    % Small development run first. Channels, not cell rows, define the split.
    train=packData(training,256);
    valid=packData(validation,256);
    parameters=dlarray(single([repmat(-0.10536,layers,1),zeros(layers,1)]));
    initialParameters=parameters;
    average=[];averageSquared=[];iteration=0;
    initialMetrics=evaluate(parameters,valid,batchSize);
    bestScore=initialMetrics(1);bestParameters=parameters;
    history=zeros(epochs,3);
    fprintf('P=%d: zero-start %d-layer exact-magnitude unfolding, initial validation %.3f dB\n', ...
        P,layers,10*log10(bestScore));
    timer=tic;
    for epoch=1:epochs
        order=randperm(size(train.target,2));
        epochLoss=0;
        for first=1:batchSize:numel(order)
            indices=order(first:min(first+batchSize-1,numel(order)));
            batch=takeBatch(train,indices);
            iteration=iteration+1;
            [loss,gradients]=dlfeval(@modelLoss,parameters,batch);
            assert(all(isfinite(extractdata(gradients)),'all'),'Nonfinite gradients');
            [parameters,average,averageSquared]=adamupdate(parameters,gradients, ...
                average,averageSquared,iteration,0.03);
            epochLoss=epochLoss+double(extractdata(loss));
        end
        metrics=evaluate(parameters,valid,batchSize);
        if metrics(1)<bestScore
            bestScore=metrics(1);bestParameters=parameters;
        end
        history(epoch,:)=[epoch,epochLoss/ceil(numel(order)/batchSize),10*log10(metrics(1))];
        fprintf('P=%d epoch %d/%d validation %.3f dB\n',P,epoch,epochs,history(epoch,3));
    end
    trainingSeconds=toc(timer);
    trained=evaluate(bestParameters,valid,batchSize);
    trainMetrics=evaluate(bestParameters,train,batchSize);
    row=table(P,trainingSeconds,10*log10(trainMetrics(1)),10*log10(trained(1)), ...
        10*log10(initialMetrics(1)),trained(2), ...
        VariableNames={'P','TrainingSeconds','TrainNMSEdB','ValidationNMSEdB', ...
        'UntrainedUnfoldingNMSEdB','PredictionEnergyRatio'});
    summary=[summary;row];disp(row); %#ok<AGROW>
    save(fullfile(output,sprintf('model_P%d.mat',P)), ...
        'bestParameters','initialParameters','history','row','layers','epochs','batchSize');
    writetable(summary,fullfile(output,'summary.csv'));
    bySNR=table();
    for snr=-5:5:30
        selected=find(valid.snr==snr);
        metrics=evaluate(bestParameters,takeBatch(valid,selected),batchSize);
        score=10*log10(metrics(1));
        bySNR=[bySNR;table(P,snr,score,VariableNames={'P','SNRdB','ValidationNMSEdB'})]; %#ok<AGROW>
    end
    writetable(bySNR,fullfile(output,sprintf('validation_by_snr_P%d.csv',P)));
end
fprintf('Unfolded development experiment finished; test files never loaded.\n');

function data=packData(source,count)
    N=8;K=3;P=size(source.pilots,2);rows=N*count;
    data.ar=zeros(K,P,rows,'single');data.ai=data.ar;
    data.b=zeros(1,P,rows,'single');data.y=data.b;
    data.bound=zeros(1,1,rows,'single');data.target=zeros(6,rows,'single');
    data.snr=zeros(rows,1);
    for m=1:count
        S=source.pilots(:,:,m);
        bound=max(real(eig(S*S')));
        for i=1:N
            n=(m-1)*N+i;
            B=source.references(i,:,m);
            rotated=S.*(conj(B)./abs(B));
            data.ar(:,:,n)=single(real(rotated));data.ai(:,:,n)=single(imag(rotated));
            data.b(:,:,n)=single(abs(B));data.y(:,:,n)=single(source.Ymag(i,:,m));
            data.bound(:,:,n)=single(bound);
            g=source.channels(i,:,m);
            data.target(:,n)=single([real(g),imag(g)].');
            data.snr(n)=source.snrDB(m);
        end
    end
end

function out=takeBatch(data,index)
    out.ar=data.ar(:,:,index);out.ai=data.ai(:,:,index);
    out.b=data.b(:,:,index);out.y=data.y(:,:,index);
    out.bound=data.bound(:,:,index);out.target=data.target(:,index);
    out.snr=data.snr(index);
end

function [loss,gradients]=modelLoss(parameters,data)
    prediction=forwardModel(parameters,data);
    loss=mean((prediction-data.target).^2,'all');
    gradients=dlgradient(loss,parameters);
end

function prediction=forwardModel(parameters,data)
    count=size(data.target,2);
    realG=dlarray(zeros(3,1,count,'single'));imagG=realG;
    previousReal=realG;previousImag=imagG;
    for layer=1:size(parameters,1)
        step=1.9/(1+exp(-parameters(layer,1)));
        momentum=0.95/(1+exp(-parameters(layer,2)));
        extrapReal=realG+momentum*(realG-previousReal);
        extrapImag=imagG+momentum*(imagG-previousImag);
        fieldReal=sum(data.ar.*extrapReal-data.ai.*extrapImag,1)+data.b;
        fieldImag=sum(data.ar.*extrapImag+data.ai.*extrapReal,1);
        amplitude=sqrt(fieldReal.^2+fieldImag.^2+single(1e-12));
        residualFactor=(amplitude-data.y)./amplitude;
        errorReal=residualFactor.*fieldReal;errorImag=residualFactor.*fieldImag;
        gradientReal=sum(data.ar.*errorReal+data.ai.*errorImag,2);
        gradientImag=sum(-data.ai.*errorReal+data.ar.*errorImag,2);
        previousReal=realG;previousImag=imagG;
        realG=extrapReal-step*gradientReal./data.bound;
        imagG=extrapImag-step*gradientImag./data.bound;
    end
    prediction=reshape(cat(1,realG,imagG),6,count);
end

function metrics=evaluate(parameters,data,batchSize)
    error=0;energy=0;predictionEnergy=0;
    for first=1:batchSize:size(data.target,2)
        index=first:min(first+batchSize-1,size(data.target,2));
        batch=takeBatch(data,index);
        estimate=double(extractdata(forwardModel(parameters,batch)));
        target=double(batch.target);
        error=error+sum((estimate-target).^2,'all');
        energy=energy+sum(target.^2,'all');
        predictionEnergy=predictionEnergy+sum(estimate.^2,'all');
    end
    metrics=[error/energy,predictionEnergy/energy];
end
