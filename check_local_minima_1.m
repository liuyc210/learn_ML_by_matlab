
L1 = randn(53,3)*5;
L2 = randn(1,54)*0.2;
L2 = filter(ones(1,16)/16, 1, L2);

res = 100;
[x_grid,y_grid] = meshgrid(linspace(-10,10,res));

x_grid_seq = reshape(x_grid', 1, res^2);
y_grid_seq = reshape(y_grid', 1, res^2);
net_in = [x_grid_seq; y_grid_seq; ones(1, res^2)];

L1_out = L1*net_in;
sig_L1 = 1./(1+exp(-L1_out));
L2_out = L2*[sig_L1; ones(1,res^2)];
sig_L2 = 1./(1+exp(-L2_out));

out = reshape(sig_L2,res,res);
figure;
mesh(x_grid,y_grid,out);
figure;
contour(x_grid,y_grid,out);