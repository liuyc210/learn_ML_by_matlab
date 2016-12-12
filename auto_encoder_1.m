
clear all;
close all;
sampleNum  = 100;
sampleDimen = 10;
inputs = randn(sampleDimen, sampleNum)*.1;

for sampleIndex = 1: sampleNum
    if (unidrnd(2,1,1) == 1)
        inputs(:,sampleIndex) = inputs(:,sampleIndex) + [0 0 ones(1,5), 0 0 0]';
    end
end
targets = inputs;
%%this is a auto encoder for a seq of fixed pos/len. 1 hiden node is enough

net = network(1, ... %numInputs
    2 ... %numLayers
    );
net.layers{1}.size = 1;  %increasing size is very important
net.biasConnect = [1;1];
net.inputConnect = [1; 0];
net.layerConnect = [0 0;1 0];
net.outputConnect = [0 1];
net.inputConnect(1,1) = true;
% view(net);
% configure(net, inputs, targets);

net.dividefcn = 'dividerand';
net.trainFcn = 'trainlm';
net.performFcn = 'mse';

% net.adaptFn                    = 'adaptwb';   %their effects are tiny(maybe no effect)
% net.inputWeights{1,1}.learnFcn = 'learngdm';
% net.layerWeights{2,1}.learnFcn = 'learngdm';
% net.biases{:}.learnFcn         = 'learngdm';

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 25/100;
net.divideParam.testRatio = 15/100;
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression'};

net.layers{1}.transferFcn = 'purelin';% default is purelin
net.layers{2}.transferFcn = 'purelin';% default is purelin

%initialization
net.layers{1}.initFcn = 'initnw';        %initialization is important
net.layers{2}.initFcn = 'initnw';
net.biases{1}.initFcn = 'initnw';
net.biases{2}.initFcn = 'initnw';

%training settings 
net.trainParam.epochs = 100;
net.trainParam.min_grad = 1e-8;
net.trainParam.mu = 1e-3;
% net.trainParam.mu_dec = 1; 
% net.trainParam.mu_inc = 1;%can not turn them on if trainlm is selected, otherwise training% might not stop!!
% net.trainParam.time = 10;

% net = initlay(net);
disp('init');
% view(net)
% getwb(net)
[net,tr] = train(net,inputs,targets);
% view(net);
getwb(net)

outputs = net(inputs);
figure; plot(inputs, outputs,'r.'); title('trained net');
performance = perform(net,targets,outputs);
figure, plotperform(tr)
 