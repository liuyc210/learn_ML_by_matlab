clear all;
Na = 13; Nb = 7; Nc = 10;
L1 = randn(Na,3)*.03;   L1 = conv2(L1, [1,1;1,1]/4, 'same');
L2 = randn(Nb,Na+1)*.03;  L2 = conv2(L2, ones(3,3)/9, 'same');
L3 = randn(Nc, Nb+1)*.03; 
L4 = randn(1,  Nc+1)*.3;
% L2 = filter(ones(1,16)/16, 1, L2);

res = 200;
range = 1;
[x_grid,y_grid] = meshgrid(linspace(-range,range,res));

x_grid_seq = reshape(x_grid', 1, res^2);
y_grid_seq = reshape(y_grid', 1, res^2);
net_in = [x_grid_seq; y_grid_seq];

% nSample = 4;
% x_in = [1 0 1 0;1 1 0 0]; 
% y  = xor(x_in(1,:),x_in(2,:));
nSample = 30;
x_in = randn(2,nSample);
y = cos(3*x_in(1,:)).*sin(1*x_in(2,:));

cost = zeros(1,res^2);
for i = 1:res^2
    L1(1,1) = net_in(1,i);
    L2(4,2) = net_in(2,i);
    L1_out = L1*[x_in;ones(1,nSample)*.01];
    sig_L1 = 1./(1+exp(-L1_out));
    L2_out = L2 *[sig_L1; ones(1,nSample)*.01];
    sig_L2 = 1./(1+exp(-L2_out));
    L3_out = L3 *[sig_L2; ones(1,nSample)*.01];
    sig_L3 = 1./(1+exp(-L3_out));
    L4_out = L4 *[sig_L3; ones(1,nSample)*.01];
    out    = 1./(1+exp(-L4_out));
    cost(i) = sum((out - y).^2);
end

cost = reshape(cost,res,res);
figure;
mesh(x_grid,y_grid,cost);
figure;
contour(x_grid,y_grid,cost);