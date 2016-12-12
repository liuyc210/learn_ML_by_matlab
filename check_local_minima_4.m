clear all;
nInput = 2;
nHidden = 53;

L1 = randn(nHidden, nInput+1);
L2 = randn(1, nHidden+1);

nSample = 1e3;
%  difficult, may needs more nIter
freq = 5;                  % high freq need more hidden nodes, e.g. freq=3, nHidden = 8;
x = linspace(-1,1,nSample/2);
y1 = sin(freq*pi*x)*.5;
y2 = y1 + .5;
inputs = [ [x,x]; [y1,y2] ];
target = [zeros(1,nSample/2), ones(1,nSample/2)];

figure; plot(inputs(1,1:500), inputs(2,1:500),'.');hold on;
plot(inputs(1,501:1000), inputs(2,501:1000),'r.'); title('inputs'); 

%%
nIter = 47000;
lr = .005;
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
%     delta2 = error; % linear
    delta2 = error.*output.*(1-output);
    delta1 = repmat(L2',1,nSample).*repmat(delta2, nHidden+1,1).*i_L2.*(1-i_L2);
    L2 = L2 - lr*delta2*i_L2' ;
    L1 = L1 - lr*delta1(1:nHidden,:)*i_L1';
    
    if (mod(i, nIter/10) == 0)
        if (nHidden == 2)
            figure; plot(o_L1(1,1:500), o_L1(2,1:500),'.');
            hold on;
            plot(o_L1(1,501:1000), o_L1(2,501:1000),'r.');
            x_p = [-1,1]; y_p = (x_p*L2(1)+L2(3))/(-L2(2)); plot(x_p,y_p); %plot sepearte
            pause(.5);
        end
        if (nHidden == 3)
            figure; scatter3(o_L1(1,1:500), o_L1(2,1:500), o_L1(3, 1:500),'.');
            hold on;
            scatter3(o_L1(1,501:1000), o_L1(2,501:1000), o_L1(3,501:1000),'r.');
            [x y]=meshgrid(-1:0.03:1, -1:0.03:1); z = (L2(1)*x + L2(2)*y + L2(4))/(-L2(3)); surf(x,y,z); %plot seperate
            pause(.5);
        end
    end
end
figure; plot(output,'.'); grid on;

L1_bak = L1;
L2_bak = L2;
break;


%%
L1 = L1_bak;
L2 = L2_bak;
range = 50; 
res = 100;
[x_grid,y_grid] = meshgrid(linspace(-range,range,res));
x_grid_seq = reshape(x_grid', 1, res^2);
y_grid_seq = reshape(y_grid', 1, res^2);
net_in = [x_grid_seq; y_grid_seq];
cost = zeros(1,res^2);
for i=1:res^2
    L2(1,2) = net_in(1,i);
    L2(1,50) = net_in(2,i);
    i_L1 = [inputs; ones(1,nSample)]; % bias
    f_L1 = L1 * i_L1;
    o_L1 = 1./(1+exp(-f_L1));   % sigmoid
    i_L2 = [o_L1; ones(1, nSample)];
    f_L2 = L2 * i_L2;
    o_L2 = 1./(1+exp(-f_L2));
    output = o_L2;
    cost(i) = sum((output-target).^2);
end

cost = reshape(cost,res,res);
figure;
mesh(x_grid,y_grid,cost);
