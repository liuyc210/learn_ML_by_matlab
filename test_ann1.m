 sampleNum = 3000;
 inputs = unidrnd(4,6,sampleNum);
 for i=1:sampleNum targets(i) = prod(inputs(1:3,i)) > sum(inputs(4:6,i)); end

% Create a Fitting Network
hiddenLayerSize = 100;
net = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
 
net.iw{1,1}   %caution: in thie stage, net.inputs{1}.size is still 0, so there's no net.iw{1,1}
% Train the Network
[net,tr] = train(net,inputs,targets);
net.iw{1,1}
 
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
 
% View the Network
view(net)
% v = -6:0.01:6;
% plot(v, sim(net, v),'.'); title('sim');
 
% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotfit(targets,outputs)
%figure, plotregression(targets,outputs)
%figure, ploterrhist(errors)