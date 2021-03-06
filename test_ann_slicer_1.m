 
inputs = random('unif', -10, 10,1,1e3);
targets = inputs > 0;

% Create a Fitting Network
hiddenLayerSize = 1;
net = fitnet(hiddenLayerSize);
% net = network;
% net.numInputs = 1;
net.numLayers = 1;
net.numOutputs = 1;
% net.biasConnect = 1;
% net.inputConnect = 1;
% net.layerConnect = 1;
% net.outputConnect = 1;


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

 
% Train the Network
[net,tr] = train(net,inputs,targets);
 
% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
 
% View the Network
% view(net)
v = -6:0.01:6;
plot(v, sim(net, v),'.'); title('sim');
 
% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotfit(targets,outputs)
%figure, plotregression(targets,outputs)
%figure, ploterrhist(errors)