clear all;
nInput = 1;
nLayers = 2;
nHidden = 2;

L1 = randn(nHidden, nInput+1);  % all one initialization lead to wrong result!
L2 = ones(1, nHidden+1);

nSample = 1e3;
inputs = randn(1, nSample);
% inputs(inputs>0.4) = inputs(inputs>0.4) + 0.6;
target = (abs(inputs) > .5);

nIter = 1200;
lr = .01;
L2_1 = zeros(1,nIter); L2_2 = L2_1;
for i=1:nIter
    %forward
    i_L1 = [inputs; ones(1,nSample)]; % bias
    f_L1 = L1 * i_L1;
    o_L1 = 1./(1+exp(-f_L1));   % sigmoid
    
    i_L2 = [o_L1; ones(1, nSample)];
    f_L2 = L2 * i_L2;
    o_L2 = 1./(1+exp(-f_L2)); 
    
    output = o_L2;

    %error
    error = output - target;
    
    %backward
%     delta2 = repmat((error.*output.*(1-output)) , nHidden+1,1);
%     delta1 = repmat(L2',1,nSample).*delta2.*i_L2.*(1-i_L2);
%     L2 = L2 - lr*sum( (i_L2.*delta2)' );
%     L1 = L1 - lr*( (i_L1*delta1(1:2,:)') );

    delta2 = error.*output.*(1-output);
    delta1 = repmat(L2',1,nSample).*repmat(delta2, nHidden+1,1).*i_L2.*(1-i_L2);
    L2 = L2 - lr*delta2*i_L2' ;
    L1 = L1 - lr*delta1(1:nHidden,:)*i_L1';
    
end
figure; plot(output,'.'); grid on;


