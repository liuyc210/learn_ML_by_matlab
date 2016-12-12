clear all;
Na = 3;
L1 = randn(Na,3)*0.3;
L2 = randn(1,Na+1)*0.3;
% L2 = filter(ones(1,16)/16, 1, L2);

res = 100;
[x_grid,y_grid] = meshgrid(linspace(-10,10,res));

x_grid_seq = reshape(x_grid', 1, res^2);
y_grid_seq = reshape(y_grid', 1, res^2);
net_in = [x_grid_seq; y_grid_seq];

for i = 1:res^2
    L1_out = L1*[1;1;1];
    sig_L1 = 1./(1+exp(-L1_out));
    L2_out = [L2(1:end-2), net_in(:,i)'] *[sig_L1; 1];
    out(i) = 1./(1+exp(-L2_out));
end

out = reshape(out,res,res);
figure;
mesh(x_grid,y_grid,out);
figure;
contour(x_grid,y_grid,out);