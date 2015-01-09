n = 1e2;

% generate (x,y)
RawData = 2*(unidrnd(2,1,n)-1); %random raw data
RawData = [zeros(1,n/2)*2, 2*ones(1,n/2)*2]; %definative raw data
x = (RawData -0.5) + normrnd(0,0.00, 1, n);

% RawData(1:10)
% figure;plot(x,'.');

% training perceptor
theta = ones(1,n)*0.2; theta(1) = 1;
mu = 0.01;
gradient = zeros(1,n);
h = zeros(1,n);
for i=2:n
    h(i) = sigmoid(theta(i)*x(i));
%     gradient(i) = RawData(i)*(1-h(i)) - (1-RawData(i))*h(i);
    gradient(i) = (RawData(i) - h(i))*x(i);  %be care of the gradient
    gradient(i) = h(i)*(1-h(i)) * (RawData(i) - h(i))*x(i);
    theta(i+1) = theta(i) + mu*gradient(i);
end

figure;
subplot(2,2,1); plot(theta,'.'); grid on; title('theta');
subplot(2,2,2); plot(gradient,'.'); grid on; title('gradient');
subplot(2,2,3); plot(h,'.'); grid on; title('h');
subplot(2,2,4); plot(x,'.'); grid on; title('x');

x_axis = -10:0.1:10;
H = zeros(1,length(x_axis));
for i=1:length(x_axis) H(i) = sigmoid(theta(n)*x_axis(i)); end;
figure; plot(x_axis, H); grid on; 



