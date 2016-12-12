
sampleNum  = 100;
vector_x = linspace(-1,1,100);
t1 = [vector_x; vector_x.^2; zeros(1,sampleNum)];
t2 = [vector_x; vector_x.^2+0.5; ones(1,sampleNum)];
inputs = [t1(1:2,:), t2(1:2,:)];   %
targets = [t1(3,:), t2(3,:)];

net = network(1, ... %numInputs
    2 ... %numLayers            a deeper network is better than xor_neuron2.m
    );
net.layers{1}.size = 2;  %increasing size is very important
%net.layers{2}.size = 2;  %increasing size is very important
net.biasConnect = [1;1];
net.inputConnect = [1;0];
net.layerConnect = [0 0;1 0];
net.outputConnect = [0 1];
net.inputConnect(1,1) = true;


net.dividefcn = 'dividerand';
net.trainFcn = 'trainlm';  %lm, gd, gda
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

nGrid = 10;
input_grid = zeros(2,nGrid^2);
ori = linspace(-1,1,nGrid);
for i=1:nGrid
    input_grid(:, (i-1)*nGrid+1:i*nGrid) = [linspace(-1,1,nGrid); ones(1,nGrid)*ori(i)];
end
output_grid = net(input_grid);





