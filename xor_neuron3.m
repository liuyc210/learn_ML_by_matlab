clear all;
close all;
sampleNum  = 1000;
sampleDimen = 2;
inputs = unidrnd(2, sampleDimen, sampleNum)-1;   %
inputs = [repmat([0 0 1 1], 1, 1); repmat([0 1 0 1], 1, 1)];
targets = xor(inputs(1,:), inputs(2,:));

net = network(1, ... %numInputs
    3 ... %numLayers            a deeper network is better than xor_neuron2.m
    );
net.layers{1}.size = 2;  %increasing size is very important
net.layers{2}.size = 2;  %increasing size is very important
net.biasConnect = [1;1;1];
net.inputConnect = [1;0;0];
net.layerConnect = [0 0 0;1 0 0;0 1 0];
net.outputConnect = [0 0 1];
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

net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression'};

net.layers{1}.transferFcn = 'logsig';% default is purelin
net.layers{2}.transferFcn = 'purelin';% default is purelin

%initialization
net.layers{1}.initFcn = 'initnw';        %initialization is important
net.layers{2}.initFcn = 'initnw';
net.inputWeights{1}.initFcn = 'initnw';
net.layerWeights{2,1}.initFcn = 'initnw';
% net.layerWeights{3,2}.initFcn = 'initnw';
net.biases{1}.initFcn = 'initnw';
net.biases{2}.initFcn = 'initnw';

%training settings 
net.trainParam.epochs = 1000;
net.trainParam.min_grad = 1e-8;
net.trainParam.mu = 1e-3;
% net.trainParam.mu_dec = 1; 
% net.trainParam.mu_inc = 1;%can not turn them on if trainlm is selected, otherwise training% might not stop!!
% net.trainParam.time = 10;

net = init(net); configure(net, inputs, targets);
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

% break;
a = zeros(10,10);
for m=1:10 for n=1:10 a(m,n) = net([m*0.1; n*0.1]); end; end;
figure; mesh(a);

 