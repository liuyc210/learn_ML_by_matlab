
sampleNum  = 100;
sampleDimen = 2;
inputs = unidrnd(2, sampleDimen, sampleNum)-1;   %
targets = xor(inputs(1,:), inputs(2,:));

net = network(1, ... %numInputs
    2, ... %numLayers
    [1; 1], ...%biasConnet
    [1 ; 0], ... %inputConnect, NL*Ni
    [0, 0; 1, 0], ... %layerConnect, no connect from layer 1 to itself NL*NL
    [0, 1]  ... %outputConnect to e layer
    );
% view(net);
% configure(net, inputs, targets);

net.dividefcn = 'dividerand';
net.trainFcn = 'trainlm';
net.performFcn = 'mse';

net.adaptFcn = 'adaptwb';
net.inputWeights{1,1}.learnFcn = 'learngdm';
net.layerWeights{2,1}.learnFcn = 'learngdm';
net.biases{:}.learnFcn = 'learngdm';

net.layers{1}.size = 2;
net.inputConnect(1,1) = true;

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression'};

net.layers{1}.transferFcn = 'logsig';% default is purelin
net.layers{2}.transferFcn = 'purelin';% default is purelin

net.trainParam.epochs = 100;
net.layers{1}.initFcn = 'initnw';  %try to comment it, the system will fail.
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


 