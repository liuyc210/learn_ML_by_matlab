
sampleNum  = 100;
sampleDimen = 1;
inputs = randn(sampleDimen, sampleNum);   %
targets = inputs>0.1;

net = network(1, ... %numInputs
    1, ... %numLayers
    [1], ...%biasConnet
    [1], ... %inputConnect
    [0], ... %layerConnect, no connect from layer 1 to itself
    [1]  ... %outputConnect to e layer
    );
% view(net);

net.dividefcn = 'dividerand';
net.trainFcn = 'traingdm';  %lm, gd, gda
net.performFcn = 'mse';

% net.adaptFcn = 'adaptwb';
% net.inputWeights{1,1}.learnFcn = 'learngdm';
% net.layerWeights{find(net.layerConnect)'}.learnFcn = 'learngdm';
% net.biases{:}.learnFcn = 'learngdm';

net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression'};

net.layers{1}.transferFcn = 'logsig';% default is purelin
net.layers{1}.initFcn = 'initnw';

net.trainParam.epochs = 100;
% net.trainParam.lr = 1000; %turn it on if using traingd

net = initlay(net);
disp('init');
view(net)
% getwb(net)
[net,tr] = train(net,inputs,targets);
view(net);
getwb(net)

outputs = net(inputs);
figure; plot(inputs, outputs,'r.'); title('trained net');
performance = perform(net,targets,outputs);
figure, plotperform(tr)


