clear all;
nInput = 2;
nHidden = 3;

L1 = randn(nHidden, nInput+1);
L2 = randn(1, nHidden+1);

nSample = 1e3;
caseID = 3;
if (caseID == 1)
    x = linspace(-1,1,nSample/2);
    y1 = x.^2;
    y2 = x.^2 + .5;
    inputs = [ [x,x]; [y1,y2] ];
    target = [zeros(1,nSample/2), ones(1,nSample/2)];
elseif (caseID == 2)
    theta = linspace(-pi, pi, nSample/2);
    inputs = [ [cos(theta), .15*cos(theta)+.3]; [sin(theta)+.04, .55*sin(theta)+.3] ];
    inputs = inputs + randn(size(inputs))*0.01;
    target = [zeros(1,nSample/2), ones(1,nSample/2)];
elseif (caseID == 3)
    x = linspace(-1,1,nSample/2);
    y1 = x.^2;
    y2 = -x.^2 + 1.5;
    inputs = [ [x,x-0.5]; [y1,y2]-1 ];
    target = [zeros(1,nSample/2), ones(1,nSample/2)];
elseif (caseID == 4)
    a1 = (0.5*randn(1,nSample/2)   + j*randn(1,nSample/2)) .* exp(-j*pi/4);
    a2 = (0.5*randn(1,nSample/2)+3 + j*randn(1,nSample/2)) .* exp(-j*pi/3);
    inputs = [ [real(a1), real(a2)]; [imag(a1), imag(a2)] ] * .3;
    target = [zeros(1,nSample/2), ones(1,nSample/2)];
elseif (caseID == 5)
    inputs = unidrnd(2,2,nSample)-1;
    target = xor(inputs(1,:), inputs(2,:));
elseif (caseID == 6)           %  difficult, may needs more nIter
    freq = 3;                  % high freq need more hidden nodes, e.g. freq=3, nHidden = 8;
    x = linspace(-1,1,nSample/2);
    y1 = sin(freq*pi*x)*.5;   
    y2 = y1 + .5;
    inputs = [ [x,x]; [y1,y2] ];
    target = [zeros(1,nSample/2), ones(1,nSample/2)];
end
figure; plot(inputs(1,1:500), inputs(2,1:500),'.');
hold on;
plot(inputs(1,501:1000), inputs(2,501:1000),'r.');
title('inputs'); 

nIter = 10000;
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
    %delta2 = error; % linear
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

nSample = 1600; ll = sqrt(nSample);
x = linspace(-1,1,ll); y = x;
grid_in = [repmat(x, 1, ll); reshape(repmat(y',1,ll)', 1, ll^2)];

i_L1 = [grid_in; ones(1,nSample)]; % bias
f_L1 = L1 * i_L1;
o_L1 = 1./(1+exp(-f_L1));   % sigmoid
i_L2 = [o_L1; ones(1, nSample)];
f_L2 = L2 * i_L2;
o_L2 = 1./(1+exp(-f_L2));
output = o_L2;

if (nHidden == 2)
    figure;
%     plot(o_L1(1,:), o_L1(2,:) ,'.');
    for i=1:ll
        plot(o_L1(1,(i-1)*ll+1:i*ll), o_L1(2,(i-1)*ll+1:i*ll)); hold on;
    end
end
if (nHidden == 3)
    figure;
    plot3(o_L1(1,:), o_L1(2,:), o_L1(3,:),'.');
end
 figure; surf(x,y,reshape(output, ll,ll)); title('input-output relation');