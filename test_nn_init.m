nInput = 1;
nLayers = 2;
nHidden = 1;

L1 = zeros(nHidden, nInput+1);
L2 = ones(1, nHidden+1);

nSample = 1e3;
inputs = randn(1, nSample);
% inputs(inputs>0.4) = inputs(inputs>0.4) + 0.6;
target = (inputs > .5);

nIter = 12000;
lr = .01;
L2_1 = zeros(1,nIter); L2_2 = L2_1;
for i=1:nIter
    %forward
%     i_L1 = [inputs; ones(1,nSample)]; % bias
%     f_L1 = L1 * i_L1;
%     o_L1 = 1./(1+exp(-sf_L1);   % sigmoid
    
    i_L2 = [inputs; ones(1, nSample)];
    f_L2 = L2 * i_L2;
    o_L2 = 1./(1+exp(-f_L2)); 
    
    output = o_L2;

    %error
    error = output - target;
    
    %backward
%     delta2 = i_L2.*repmat((error.*output.*(1-output)) , 2,1);
%     L2 = L2 - lr*sum(delta2');
%     delta2 = repmat((error.*output.*(1-output)) , 2,1);
%     L2 = L2 - lr*sum( (i_L2.*delta2)' );
    delta2 = error.*output.*(1-output);
    L2 = L2 - lr* (delta2*i_L2');

    
    L2_1(i) = L2(1);
    L2_2(i) = L2(2);
end